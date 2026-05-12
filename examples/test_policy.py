"""Run a trained PPO policy on the CPU MuJoCo cube env with a live viewer.

The policy (an ActorCritic flax module trained by `train_ppo_cube.py`) runs on
the GPU via JAX; observations and actions are shuttled to the CPU MuJoCo env
each step. One env, one viewer — meant for eyeballing whether a checkpoint
behaves sensibly, not for throughput.

Usage:
    python examples/test_policy.py --checkpoint runs/ppo_cube/params.pkl

Add `--no-match-physics` if you specifically want the XML's default solver
instead of the MJX-equivalent settings the policy was trained against.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = REPO_ROOT / "examples"
for p in (SRC_DIR, EXAMPLES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# Keep the viewer responsive: don't let JAX grab the whole GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


try:
    import jax
    import jax.numpy as jnp
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing runtime dependency '{exc.name}'. Install training extras "
        "with `pip install -e '.[mjx]'`."
    ) from exc

from orca_sim.envs_mjx import _prepare_mj_model_for_mjx
from orca_sim.task_envs import OrcaHandRightCubeOrientation
from train_ppo_cube import ActorCritic, sample_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO cube policy on the CPU env with a viewer."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to params.pkl saved by train_ppo_cube.py.",
    )
    parser.add_argument("--version", default=None, help="Embodiment version, e.g. v2.")
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "none"],
        default="human",
        help="'human' opens the MuJoCo viewer; 'none' runs headless.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to roll out before exiting (0 = forever).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Hard cap on env-steps per episode (matches training's truncation).",
    )
    parser.add_argument(
        "--initial-red-face",
        choices=["random", "down"],
        default="random",
        help="Cube starting orientation at reset.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use the policy mean (default). Disable with --stochastic.",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample from the policy distribution (otherwise mean is used).",
    )
    parser.add_argument(
        "--match-physics",
        action="store_true",
        default=True,
        help="Apply the MJX-equivalent solver settings to the CPU model (default).",
    )
    parser.add_argument(
        "--no-match-physics",
        dest="match_physics",
        action="store_false",
        help="Use the XML's solver settings as-is (dynamics will differ from training).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Substeps per env-step. Default 2 matches the MJX training config.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.005,
        help="MuJoCo timestep. Default 0.005 matches the MJX training config.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Actor-critic hidden layer sizes. Must match the training run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    print(f"[test] Loading params from {args.checkpoint}")
    with args.checkpoint.open("rb") as f:
        params = pickle.load(f)

    render_mode = None if args.render_mode == "none" else args.render_mode

    env = OrcaHandRightCubeOrientation(
        render_mode=render_mode,
        version=args.version,
        initial_red_face=args.initial_red_face,
        max_episode_steps=args.max_steps,
    )
    if args.match_physics:
        _prepare_mj_model_for_mjx(env.model)
    env.model.opt.timestep = float(args.timestep)
    env.frame_skip = int(args.frame_skip)
    print(
        f"[test] frame_skip={env.frame_skip}  timestep={env.model.opt.timestep:.4f}  "
        f"match_physics={args.match_physics}  deterministic={args.deterministic}"
    )

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    network = ActorCritic(
        action_size=action_size, hidden_sizes=tuple(args.hidden_sizes)
    )
    action_low = jnp.asarray(env.action_low, dtype=jnp.float32)
    action_high = jnp.asarray(env.action_high, dtype=jnp.float32)

    @jax.jit
    def policy(obs_jax: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        env_action, _, _, _ = sample_action(
            params,
            obs_jax,
            key,
            network,
            action_low,
            action_high,
            deterministic=args.deterministic,
        )
        return env_action

    # Warm up the jit so the first viewer frame isn't a multi-second hitch.
    print("[test] Compiling policy...")
    t_compile = time.time()
    dummy_obs = jnp.zeros((obs_size,), dtype=jnp.float32)
    _ = policy(dummy_obs, jax.random.PRNGKey(0)).block_until_ready()
    print(f"[test] Compile done in {time.time() - t_compile:.1f}s")

    rng = jax.random.PRNGKey(args.seed)
    target_dt = 1.0 / env.metadata["render_fps"]

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            obs, info = env.reset(seed=args.seed + episode if args.seed else None)
            ep_reward = 0.0
            ep_steps = 0
            best_align = -1.0
            success = False
            dropped = False

            while ep_steps < args.max_steps:
                rng, key = jax.random.split(rng)
                obs_jax = jnp.asarray(obs, dtype=jnp.float32)
                t0 = time.perf_counter()
                action = np.asarray(
                    policy(obs_jax, key).block_until_ready(), dtype=np.float32
                )

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                ep_steps += 1
                best_align = max(best_align, float(info["red_face_up_alignment"]))
                success = success or bool(info["is_success"])
                dropped = bool(info["dropped"])

                if render_mode == "human":
                    time.sleep(max(0.0, target_dt - (time.perf_counter() - t0)))

                if terminated or truncated:
                    break

            print(
                f"[test] ep={episode:>3d}  steps={ep_steps:>3d}  "
                f"return={ep_reward:+.2f}  best_align={best_align:+.3f}  reward/step={ep_reward/ep_steps} "
                f"success={success}  dropped={dropped}"
            )
            episode += 1
    except KeyboardInterrupt:
        print("[test] stopped by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
