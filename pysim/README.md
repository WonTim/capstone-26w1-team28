# Simulation (WIP)
## Setup
### 1. Clone the repository

```bash
git clone https://github.com/WonTim/capstone-26w1-team28.git
cd capstone-26w1-team28/sim
```

### 2. Create a virtual environment

Create a Python virtual environment called .venv:

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows:

```bash
.venv\Scripts\activate
```

#### macOS/Linux:

```bash
source .venv/bin/activate
```

Once activated, you should see (.venv) at the beginning of your terminal prompt.

### 4. Install the project requirements

Make sure your virtual environment is activated, then run:

```bash
pip install -e .
```

The dependencies are defined in `pyproject.toml`.

The -e flag installs the project in editable mode, so changes to the source code under src/ are immediately reflected without reinstalling the project.

### Running the project

```bash
sim
```
