<p align="center">
  <img src="https://huggingface.co/datasets/fracapuano/blogs/resolve/main/orca_sim.png" alt="orca_sim header" width="600"/>
</p>


`orca_sim` provides simulation environments for the ORCA hand.
You can start building your ORCA hand today at [orcahand.com](https://www.orcahand.com/).

## Install

We recommend using python 3.11 inside of a virtual environment.
You can create a virtual environment using `uv` ([how to install uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uv venv orca --python 3.11
source orca/bin/activate
uv pip install orca_sim
```

Alternatively, you can use `conda` ([how to install conda](https://www.anaconda.com/docs/getting-started/miniconda/install)):
```bash
conda create -n orca python=3.11 -y
conda activate orca
python -m pip install orca_sim
```
As we are continuously iterating on `orca_sim`, you can fetch the latest `main` building this package from source, so to be in the loop with the latest developments.

```bash
git clone https://github.com/orcahand/orca_sim
cd orca_sim && uv pip install -e .
```

> [!WARNING] 
We are still iterating (a lot!) on this package. If you need stability, consider sticking to the Pypi package (`pip install orca_sim`).

## Getting started

`orca_sim` follows the [Gymnasium](https://gymnasium.farama.org/) API, and uses [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html) for physics simulation and rendering.
You can instantiate an environment with one hand (or both) via:

```python
from orca_sim import OrcaHandRight  # or OrcaHandLeft, OrcaHandCombined

env = OrcaHandRight()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```
The 'extended' version of different hands contain additional bodies (incl. inertial properties) such as the camera mount, the U2D2 board and fans.

### Hands versioning

By default, any environment defaults to the latest fully-supported embodiment files.

```python
from orca_sim import OrcaHandRight, OrcaHandRightExtended

env = OrcaHandRight()  # latest version of the standard right hand
extended_env = OrcaHandRightExtended()  # latest version of the extended right hand
```

`orca_sim` stored different hand versions under versioned `scenes/` and `models/` directories. You can still pin an older version explicitly when needed:

```python
from orca_sim import OrcaHandCombinedExtended

env = OrcaHandCombinedExtended(version="v1")  # loads the v1 hand
```

See our [`random_policy.py`](random_policy.py) example to see how to instantiate and interface an ORCA hand.

### GPU-vectorized envs (MJX, experimental)

`orca_sim` ships an experimental [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)-backed
variant of every hand env (`OrcaHandLeftMjx`, `OrcaHandRightMjx`, …) plus a batched
`OrcaHandMjxVectorEnv` for running hundreds of hands in parallel on the GPU. Try it
with [`random_policy_mjx.py`](random_policy_mjx.py):

```bash
python random_policy_mjx.py --env left  --num-envs 1   --render-mode human
python random_policy_mjx.py --env left  --num-envs 256 --render-mode human
```

> [!NOTE]
> **First run is slow.** MJX has to JIT-compile the entire hand model (collisions,
> contacts, dynamics) into a single XLA graph. Expect **~5-10 min of "frozen" startup**
> on the first run for a given hand/model — the viewer will not appear until the
> compile finishes, and `nvidia-smi` will show the GPU near-100% memory used (JAX
> preallocates VRAM by default — this is not a leak).
>
> A persistent on-disk compile cache is enabled automatically at
> `~/.cache/orca_sim/jax`. **Subsequent runs reuse it and start in seconds.**
> Override / disable with:
>
> ```bash
> ORCA_SIM_JAX_CACHE_DIR=/some/path python random_policy_mjx.py ...   # custom dir
> ORCA_SIM_JAX_CACHE=0 python random_policy_mjx.py ...                 # disable
> ```
>
> The cache is invalidated automatically when JAX, CUDA, or the model XML changes.

## Sample task: in-hand cube orientation

`orca_sim` now also ships a task-level example that augments the right hand with a free-floating cube whose one target face is colored red:

```python
from orca_sim import OrcaHandRightCubeOrientation

env = OrcaHandRightCubeOrientation(version="v2", render_mode="human")
obs, info = env.reset(seed=0)
```

By default, the task resets to a palm-up open-hand pose with the cube resting on the palm and the red face pointing downward. You can also randomize the initial cube orientation while keeping it unsolved:

```python
env = OrcaHandRightCubeOrientation(
    version="v2",
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(seed=0)
```

If you want to keep the environment completely deterministic while you are still building it, you can use the nominal reset directly and add randomization later:

```python
env = OrcaHandRightCubeOrientation(version="v2")

nominal = env.nominal_reset_options()
obs, info = env.reset(options=nominal)

randomized = env.sample_randomized_reset_options(
    seed=0,
    initial_red_face="random",
    cube_pos_xy_jitter=0.01,
)
obs, info = env.reset(options=randomized)
```

The implementation is intentionally split so it doubles as a porting template:

- The task scene lives in [`src/orca_sim/scenes/v2/scene_right_cube_orientation.xml`](src/orca_sim/scenes/v2/scene_right_cube_orientation.xml) and composes the existing hand MJCF with a single task cube.
- The nominal palm-up hand pose and in-palm cube spawn are now authored into the task-specific scene/model files, so opening the XML directly in MuJoCo shows the intended setup.
- The task logic lives in [`src/orca_sim/task_envs.py`](src/orca_sim/task_envs.py), including reset-time cube randomization and optional hand-pose overrides for custom MJCF layouts.

