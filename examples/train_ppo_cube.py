"""Custom JAX PPO for the OrcaHand cube reorientation task.

Self-contained, ~400 lines: actor-critic MLP, GAE, PPO clipped objective,
rollouts that reuse the underlying MJX env's jit-vmapped step. No brax / no
mujoco_playground dependencies.

Goal: learn a policy that rotates the in-hand cube so its red face points up.

Quick start (after `pip install -e '.[mjx]'`):
    python examples/train_ppo_cube.py --num-timesteps 50_000_000

After training:
    python examples/train_ppo_cube.py --eval --checkpoint runs/ppo_cube/params.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing runtime dependency '{exc.name}'. Install training extras with "
        "`pip install -e '.[mjx]'` (which now includes flax + optax)."
    ) from exc

from orca_sim.task_envs_mjx import OrcaHandRightCubeOrientationMjx


@dataclass
class Hyperparams:
    num_timesteps: int = 50_000_000
    num_envs: int = 2048
    rollout_steps: int = 32
    num_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 0
    log_interval: int = 1
    reset_every_rollouts: int = 0  # 0 = never; auto-reset inside the rollout handles per-env terminations


# -------- network -----------------------------------------------------------


class ActorCritic(nn.Module):
    action_size: int
    hidden_sizes: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = obs
        for h in self.hidden_sizes:
            x = nn.tanh(
                nn.Dense(
                    h,
                    kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
                    bias_init=nn.initializers.zeros,
                )(x)
            )
        mean = nn.Dense(
            self.action_size,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x).squeeze(-1)
        log_std = self.param(
            "log_std",
            lambda _: jnp.full((self.action_size,), -0.5, dtype=jnp.float32),
        )
        return mean, log_std, value


def _gauss_log_prob(u: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray) -> jnp.ndarray:
    var = jnp.exp(2.0 * log_std)
    return (
        -0.5 * jnp.sum((u - mean) ** 2 / var + 2.0 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    )


def _tanh_log_correction(u: jnp.ndarray) -> jnp.ndarray:
    # log|d tanh(u)/du| = log(1 - tanh(u)^2). Use the numerically-stable form.
    return jnp.sum(2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), axis=-1)


def sample_action(
    params,
    obs: jnp.ndarray,
    key: jax.Array,
    network: ActorCritic,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    deterministic: bool = False,
):
    """Returns (env_action, pre_squash_u, log_prob, value)."""
    mean, log_std, value = network.apply(params, obs)
    if deterministic:
        u = mean
    else:
        std = jnp.exp(log_std)
        eps = jax.random.normal(key, mean.shape)
        u = mean + std * eps
    squashed = jnp.tanh(u)
    env_action = action_low + 0.5 * (squashed + 1.0) * (action_high - action_low)
    log_prob = _gauss_log_prob(u, mean, log_std) - _tanh_log_correction(u)
    return env_action, u, log_prob, value


def recompute_log_prob_and_value(
    params,
    obs: jnp.ndarray,
    u: jnp.ndarray,
    network: ActorCritic,
):
    mean, log_std, value = network.apply(params, obs)
    log_prob = _gauss_log_prob(u, mean, log_std) - _tanh_log_correction(u)
    entropy = jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)
    return log_prob, value, entropy


# -------- GAE / advantages --------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
):
    """rewards/values/dones: (T, N). last_value: (N,). Returns advantages, returns: (T, N)."""

    def step(carry, transition):
        last_gae, next_value = carry
        reward, value, done = transition
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * last_gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(
        step, init, (rewards, values, dones), reverse=True
    )
    returns = advantages + values
    return advantages, returns


# -------- rollout (on-device) -----------------------------------------------


def make_rollout_fn(
    network: ActorCritic,
    env,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    rollout_steps: int,
    num_envs: int,
):
    """Build a jitted rollout function with per-env auto-reset on done.

    The reset state for a terminated/truncated env is sampled on-device via
    `env._sample_per_env_reset` and swapped into the next carry, so the
    rollout never leaves the GPU.
    """
    env_step_fn = env._jit_vstep
    obs_fn = env._jit_vobs
    sample_reset = env._sample_per_env_reset

    def step(carry, _):
        params, mjx_data, obs, key = carry
        key, action_key, reset_key = jax.random.split(key, 3)

        env_action, u, log_prob, value = sample_action(
            params, obs, action_key, network, action_low, action_high,
            deterministic=False,
        )
        new_data, _post_obs, reward, terminated, truncated = env_step_fn(
            mjx_data, env_action
        )
        done = terminated | truncated  # (num_envs,) bool

        # Per-env reset state (on-device).
        reset_keys = jax.random.split(reset_key, num_envs)
        reset_qpos, reset_qvel, reset_ctrl, reset_time = jax.vmap(sample_reset)(
            reset_keys
        )

        def swap(curr, fresh):
            mask = done.reshape((num_envs,) + (1,) * (curr.ndim - 1))
            return jnp.where(mask, fresh, curr)

        new_data = new_data.replace(
            qpos=swap(new_data.qpos, reset_qpos),
            qvel=swap(new_data.qvel, reset_qvel),
            ctrl=swap(new_data.ctrl, reset_ctrl),
            time=jnp.where(done, reset_time, new_data.time),
        )
        next_obs = obs_fn(new_data)

        transition = {
            "obs": obs,
            "u": u,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "done": done.astype(jnp.float32),
        }
        return (params, new_data, next_obs, key), transition

    def rollout(params, mjx_data, obs, key):
        (params, mjx_data, obs, key), transitions = jax.lax.scan(
            step, (params, mjx_data, obs, key), xs=None, length=rollout_steps
        )
        # bootstrap: value of the post-rollout obs.
        _, _, last_value = network.apply(params, obs)
        return mjx_data, obs, key, transitions, last_value

    return rollout


# -------- PPO update --------------------------------------------------------


def ppo_loss(
    params,
    batch: dict,
    network: ActorCritic,
    clip_eps: float,
    ent_coef: float,
    vf_coef: float,
):
    new_log_prob, new_value, entropy = recompute_log_prob_and_value(
        params, batch["obs"], batch["u"], network
    )
    ratio = jnp.exp(new_log_prob - batch["old_log_prob"])
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    value_loss = 0.5 * ((new_value - batch["return"]) ** 2).mean()
    entropy_mean = entropy.mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": ((batch["old_log_prob"] - new_log_prob).mean()),
        "clipfrac": (jnp.abs(ratio - 1.0) > clip_eps).mean(),
    }
    return loss, metrics


def make_update_step(network: ActorCritic, hp: Hyperparams):
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_step(state: TrainState, batch: dict):
        (loss, metrics), grads = grad_fn(
            state.params, batch, network, hp.clip_eps, hp.ent_coef, hp.vf_coef
        )
        state = state.apply_gradients(grads=grads)
        metrics = {**metrics, "loss": loss}
        return state, metrics

    return update_step


# -------- training loop -----------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom JAX PPO on the OrcaHand cube reorientation task."
    )
    parser.add_argument("--version", default=None, help="Embodiment version, e.g. v2.")
    parser.add_argument("--num-timesteps", type=int, default=50_000_000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=1e-2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/ppo_cube"),
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--render-mode", choices=["human", "rgb_array"], default="human"
    )
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument(
        "--initial-red-face",
        choices=["random", "down"],
        default="random",
        help=(
            "Cube starting orientation at every reset. 'random' samples a "
            "non-solved axis-aligned quat per env (default); 'down' fixes the "
            "red face down."
        ),
    )
    parser.add_argument(
        "--reset-every-rollouts",
        type=int,
        default=0,
        help=(
            "Optional periodic full host-side reset every N updates. Default "
            "0 (disabled) — the rollout's per-env auto-reset already handles "
            "terminated/truncated envs."
        ),
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help=(
            "Write params.pkl + history.pkl every N updates (so an interrupted "
            "run still has a usable checkpoint). 0 disables intermediate saves."
        ),
    )
    return parser.parse_args()


def hp_from_args(args: argparse.Namespace) -> Hyperparams:
    return Hyperparams(
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        num_epochs=args.num_epochs,
        num_minibatches=args.num_minibatches,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_interval=args.log_interval,
        reset_every_rollouts=args.reset_every_rollouts,
    )


def run_train(args: argparse.Namespace) -> None:
    hp = hp_from_args(args)

    print("[train] Building MJX env (this triggers JIT compile on first run)...")
    env = OrcaHandRightCubeOrientationMjx(
        num_envs=hp.num_envs,
        version=args.version,
        initial_red_face=args.initial_red_face,
    )
    print(f"[train] initial_red_face={args.initial_red_face} (per-env at every reset)")
    obs_size = env.single_observation_space.shape[0]
    action_size = env.single_action_space.shape[0]
    action_low = jnp.asarray(env.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(env.action_high, dtype=jnp.float32)

    network = ActorCritic(action_size=action_size, hidden_sizes=hp.hidden_sizes)
    key = jax.random.PRNGKey(hp.seed)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((obs_size,), dtype=jnp.float32)
    params = network.init(init_key, dummy_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(hp.max_grad_norm),
        optax.adam(hp.learning_rate),
    )
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    # Rollout does per-env auto-reset on done, all on-device (no host syncs).
    rollout_fn = jax.jit(
        make_rollout_fn(
            network, env, action_low, action_high, hp.rollout_steps, hp.num_envs
        )
    )
    update_step = make_update_step(network, hp)

    timesteps_per_update = hp.num_envs * hp.rollout_steps
    num_updates = max(1, hp.num_timesteps // timesteps_per_update)
    minibatch_size = (hp.num_envs * hp.rollout_steps) // hp.num_minibatches
    print(
        f"[train] num_updates={num_updates}  steps/update={timesteps_per_update}  "
        f"minibatch_size={minibatch_size}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "progress.log"
    params_path = args.output_dir / "params.pkl"
    history_path = args.output_dir / "history.pkl"
    history: list[dict[str, Any]] = []

    def save_checkpoint(tag: str) -> None:
        with params_path.open("wb") as f:
            pickle.dump(jax.device_get(state.params), f)
        with history_path.open("wb") as f:
            pickle.dump(history, f)
        print(f"[train] [{tag}] saved {params_path}")

    obs, _ = env.reset()
    mjx_data = env.mjx_data
    start = time.time()
    total_steps = 0

    for update in range(num_updates):
        key, rollout_key = jax.random.split(key)
        mjx_data, obs, _, transitions, last_value = rollout_fn(
            state.params, mjx_data, obs, rollout_key
        )
        env.mjx_data = mjx_data  # keep env in sync for any out-of-loop reads

        advantages, returns = compute_gae(
            transitions["reward"],
            transitions["value"],
            transitions["done"],
            last_value,
            hp.gamma,
            hp.gae_lambda,
        )

        # Flatten (T, N, ...) -> (T*N, ...)
        flat = {
            "obs": transitions["obs"].reshape(-1, obs_size),
            "u": transitions["u"].reshape(-1, action_size),
            "old_log_prob": transitions["log_prob"].reshape(-1),
            "advantage": advantages.reshape(-1),
            "return": returns.reshape(-1),
        }
        n_samples = flat["obs"].shape[0]

        epoch_metrics: list[dict[str, jnp.ndarray]] = []
        for _ in range(hp.num_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, n_samples)
            for mb in range(hp.num_minibatches):
                idx = perm[mb * minibatch_size : (mb + 1) * minibatch_size]
                batch = {k: v[idx] for k, v in flat.items()}
                state, m = update_step(state, batch)
                epoch_metrics.append(m)

        total_steps += timesteps_per_update

        if update % hp.log_interval == 0:
            ep_reward = float(transitions["reward"].mean())
            mean_align = float(transitions["obs"][:, :, -1].mean())
            avg_value_loss = float(jnp.mean(jnp.array([m["value_loss"] for m in epoch_metrics])))
            avg_pol_loss = float(jnp.mean(jnp.array([m["policy_loss"] for m in epoch_metrics])))
            avg_kl = float(jnp.mean(jnp.array([m["approx_kl"] for m in epoch_metrics])))
            avg_ent = float(jnp.mean(jnp.array([m["entropy"] for m in epoch_metrics])))
            elapsed = time.time() - start
            sps = total_steps / elapsed if elapsed > 0 else 0.0
            msg = (
                f"upd={update:>5d}  steps={total_steps:>10d}  "
                f"r/step={ep_reward:+.4f}  align={mean_align:+.3f}  "
                f"pol={avg_pol_loss:+.4f}  vf={avg_value_loss:.4f}  "
                f"kl={avg_kl:+.4f}  H={avg_ent:.3f}  "
                f"sps={sps:.0f}  elapsed={elapsed/60:.1f}m"
            )
            print(f"[train] {msg}")
            with log_path.open("a") as f:
                f.write(msg + "\n")
            history.append({
                "update": update,
                "steps": total_steps,
                "reward_per_step": ep_reward,
                "mean_alignment": mean_align,
            })

        # Optional full host-side reset. Auto-reset inside the rollout already
        # handles per-env termination, so this is off by default — set
        # --reset-every-rollouts > 0 only if you want a periodic re-randomize.
        if hp.reset_every_rollouts > 0 and (update + 1) % hp.reset_every_rollouts == 0:
            obs, _ = env.reset()
            mjx_data = env.mjx_data

        if args.save_interval > 0 and (update + 1) % args.save_interval == 0:
            save_checkpoint(f"upd={update}")

    save_checkpoint("final")


def run_eval(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required in --eval mode.")
    with args.checkpoint.open("rb") as f:
        params = pickle.load(f)

    env = OrcaHandRightCubeOrientationMjx(
        num_envs=1,
        version=args.version,
        render_mode=args.render_mode,
        initial_red_face=args.initial_red_face,
    )
    obs_size = env.single_observation_space.shape[0]
    action_size = env.single_action_space.shape[0]
    network = ActorCritic(action_size=action_size, hidden_sizes=(256, 256))
    action_low = jnp.asarray(env.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(env.action_high, dtype=jnp.float32)

    @jax.jit
    def policy(obs, key):
        action, _, _, _ = sample_action(
            params, obs, key, network, action_low, action_high, deterministic=True
        )
        return action

    obs, info = env.reset()
    rng = jax.random.PRNGKey(0)
    total_reward = 0.0
    print(f"[eval] running {args.eval_steps} steps with {args.render_mode} render")
    for step in range(args.eval_steps):
        rng, key = jax.random.split(rng)
        action = policy(obs[0], key)
        action_batch = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, term, trunc, info = env.step(action_batch)
        total_reward += float(reward[0])
        if args.render_mode == "human":
            env.render()
        if bool(term[0]) or bool(trunc[0]):
            print(
                f"[eval] step={step} done — alignment="
                f"{float(info['red_face_up_alignment'][0]):+.3f} "
                f"is_success={bool(info['is_success'][0])} "
                f"dropped={bool(info['dropped'][0])}"
            )
            obs, info = env.reset()
    print(f"[eval] total_reward={total_reward:.2f}")
    env.close()


def main() -> None:
    args = parse_args()
    if args.eval:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
