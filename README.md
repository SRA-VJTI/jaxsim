# gradsim

> gradSim: Differentiable simulation for system identification and visuomotor control

<p align="center">
	<img src="assets/walker.gif" />
</p>

**gradSim** is a unified differentiable rendering and multiphysics framework that allows solving a range of control and parameter estimation tasks (rigid bodies, deformable solids, and cloth) directly from images/video. Our unified computation graph — spanning from the dynamics and through the rendering process — enables learning in challenging visuomotor control tasks, without relying on state-based (3D) supervision, while obtaining performance competitive to or better than techniques that rely on precise 3D labels.


## Building the package

### Quick Start (Recommended)

#### Step 0: Create a virtual environment with uv

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python environment management. Tested with Python 3.9, 3.10, and 3.11.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11 .venv
source .venv/bin/activate
```

#### Step 1: Install PyTorch

Install PyTorch with CUDA support:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, see [pytorch.org](https://pytorch.org/).

#### Step 2: Install gradsim

```bash
uv pip install setuptools wheel ninja
uv pip install -e . --no-build-isolation
```

#### Step 3: Verify installation

```python
>>> import gradsim
>>> gradsim.__version__
'0.0.4'
```

#### Step 4 (Optional): Setup USD/pxr for mesh loading

If you need USD support for loading complex meshes:

**Option A**: Use the provided installer:
```bash
./buildusd.sh
source setenv.sh
```

**Option B**: If using Kaolin, configure paths:
```bash
cd path/to/kaolin/root/directory
export KAOLIN_HOME=$PWD
export PYTHONPATH=${KAOLIN_HOME}/build/target-deps/nv_usd/release/lib/python
export LD_LIBRARY_PATH=${KAOLIN_HOME}/build/target-deps/nv_usd/release/lib/
```

---

### Alternative: Using conda

<details>
<summary>Click to expand conda instructions</summary>

#### Step 0: Create a conda environment

```bash
conda create -n gradsim python=3.11
conda activate gradsim
```

#### Step 1: Install PyTorch

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### Step 2: Install gradsim

```bash
pip install -e .
```

#### Step 3: Install Ninja

```bash
conda install -c conda-forge ninja
```

</details>

---

## Python 3.11 Compatibility

This codebase has been updated for Python 3.11 compatibility:

- **dflex module**: Updated AST handling for Python 3.9+ changes (`ast.Constant` vs `ast.Num`, subscript handling)
- **Import system**: Replaced deprecated `imp` module with `importlib.util`
- **Dependencies**: `py3ode` is now optional (requires system ODE library)

### Known Limitations

- **Kaolin**: NVIDIA Kaolin may have its own Python version requirements. Check [Kaolin docs](https://kaolin.readthedocs.io/) for compatibility.
- **py3ode**: Optional dependency for ODE physics. Install separately if needed:
  ```bash
  # Requires libode-dev system package
  sudo apt-get install libode-dev
  uv pip install py3ode
  ```


## Examples

All examples are in the `examples` directory:

```bash
cd examples
```

#### Basic demos

```bash
# dflex smoketest (spring-mass system)
python hellodflex.py

# Simple physics simulation
python demo_pendulum.py
python demo_double_pendulum.py
python demo_bounce2d.py
```

#### Parameter estimation

```bash
# Estimate mass from video
python demo_mass_known_shape.py

# FEM parameter optimization
python demo_fem.py

# Cloth parameter optimization
python demo_cloth.py
```

#### Visuomotor control

```bash
# Walker locomotion
python control_walker.py

# Cloth manipulation
python control_cloth.py

# Deformable object control
python control_fem_gear.py
```

For command-line options:
```bash
python <script>.py --help
```


## Citing gradSim

```bibtex
@article{gradsim,
  title   = {gradSim: Differentiable simulation for system identification and visuomotor control},
  author  = {Krishna Murthy Jatavallabhula and Miles Macklin and Florian Golemo and Vikram Voleti and Linda Petrini and Martin Weiss and Breandan Considine and Jerome Parent-Levesque and Kevin Xie and Kenny Erleben and Liam Paull and Florian Shkurti and Derek Nowrouzezahrai and Sanja Fidler},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2021},
  url     = {https://openreview.net/forum?id=c_E8kFWfhp0},
  pdf     = {https://openreview.net/pdf?id=c_E8kFWfhp0},
}
```

If using Kaolin:
```bibtex
@misc{kaolin,
  author = {Jatavallabhula, Krishna Murthy and others},
  title = {Kaolin: A PyTorch Library for Accelerating 3D Deep Learning Research},
  howpublished = {\url{https://github.com/NVIDIAGameWorks/kaolin}},
  year = {2019}
}
```
