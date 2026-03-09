# JAXSIM

<p align="center">
  <img src="assets/drop_trimmed.gif" width="320" />
</p>

**jaxsim** is a complete JAX rewrite of [gradSim](https://github.com/gradsim/gradsim), a unified differentiable rendering and multiphysics framework for solving parameter estimation and visuomotor control tasks directly from images and video. It supports rigid bodies, deformable solids, and cloth — all fully differentiable end-to-end through simulation and rendering.

The original gradSim was built on PyTorch + NVIDIA warp/dflex. This rewrite replaces everything with JAX: autodiff, JIT compilation, and physics integration are all handled natively in JAX with no CUDA kernel compilation required.

---

## What's inside

| Module | Description |
|--------|-------------|
| `jaxsim.dflex` | JAX port of NVIDIA DFlex — cloth, FEM, rigid body physics with semi-implicit Euler integration |
| `jaxsim.renderutils` | Soft rasterizer (Liu et al., ICCV 2019) and DIBR renderer, both in pure JAX |
| `jaxsim.bodies` | Rigid body definitions with inertia and COM computation |
| `jaxsim.engines` | Physics integrators (Euler) |
| `jaxsim.utils` | Mesh utilities, quaternion math, GIF/image logging |

---

## Installation

Requires Python ≥ 3.9.

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate
# or: uv venv --python 3.11 .venv && source .venv/bin/activate

# 2. Install JAX (CPU)
pip install jax jaxlib

# For GPU (CUDA 12):
# pip install -U "jax[cuda12]"

# 3. Install jaxsim
pip install -e .
```

## Quick start

### Cloth falling on a sphere

```python
from jaxsim import dflex as df
from jaxsim.renderutils import SoftRenderer
from jaxsim.utils.logging import write_imglist_to_gif
import math, jax.numpy as jnp

# Build cloth
builder = df.sim.ModelBuilder()
builder.add_cloth_grid(
    pos=(0.0, 2.0, 0.0),
    rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
    vel=(0.0, 0.0, 0.0),
    dim_x=10, dim_y=10, cell_x=0.14, cell_y=0.14, mass=0.3,
)
model = builder.finalize("cpu")
model.ground = True

integrator = df.sim.SemiImplicitIntegrator()
state = model.state()

# Step simulation
for _ in range(100):
    state = integrator.forward(model, state, dt=1/960)
```

See [`examples/demo_cloth_sphere.py`](examples/demo_cloth_sphere.py) for the full demo with sphere collision and GIF export.

---

## Examples

Run any example from the repo root:

```bash
cd examples
../../jaxenv/bin/python3 <script>.py
```

### Physics demos

| Script | Description |
|--------|-------------|
| `hellodflex.py` | Spring-mass smoketest — 9-particle chain |
| `demo_pendulum.py` | Simple pendulum with parameter optimization |
| `demo_double_pendulum.py` | Double pendulum |
| `demo_bounce2d.py` | 2D bouncing ball, optimize restitution |
| `demo_tablecloth.py` | Flat cloth dropping onto a ground plane |
| `demo_cloth_sphere.py` | Cloth draped over a static sphere (GIF output) |

### Parameter estimation

| Script | Description |
|--------|-------------|
| `demo_mass_known_shape.py` | Estimate mass from rendered video |
| `demo_fem.py` | FEM material parameter optimization |
| `demo_cloth.py` | Cloth velocity optimization from images |

### Visuomotor control

| Script | Description |
|--------|-------------|
| `control_walker.py` | Deformable walker locomotion |
| `control_cloth.py` | Cloth manipulation |
| `control_fem.py` | Deformable gear control |

### Rendering

| Script | Description |
|--------|-------------|
| `softras_simple_render.py` | Soft rasterization forward pass |
| `softras_texture_optimization.py` | Optimize texture from target image |
| `dibr_forward_render.py` | DIBR forward render |

---

## Key differences from gradSim (PyTorch)

| | gradSim (original) | jaxsim (this repo) |
|--|--|--|
| Backend | PyTorch | JAX |
| Physics | NVIDIA warp/dflex (C++ kernels) | Pure Python JAX |
| Autodiff | `torch.autograd` | `jax.grad` / `jax.value_and_grad` |
| GPU compilation | CUDA kernel build required | XLA JIT, no compilation step |
| Install | Complex (CUDA toolkit, Kaolin) | `pip install jax jaxlib && pip install -e .` |
| Cloth/FEM | dflex warp kernels | dflex ported to JAX arrays |

---

