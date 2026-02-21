#!/bin/bash
# ============================================================
# SafeDisassemble — Setup & Run
# ============================================================
# Run this from the project root:
#   cd /Users/spartan/Desktop/"ML Project"
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh
# ============================================================

set -e  # exit on any error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "SafeDisassemble — Setup & Run"
echo "Project dir: $PROJECT_DIR"
echo "============================================================"

# ─── Step 1: Create virtual environment ───
echo ""
echo "[Step 1/7] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists, reusing"
fi

source .venv/bin/activate
echo "  Python: $(python --version)"
echo "  Location: $(which python)"

# ─── Step 2: Install dependencies ───
echo ""
echo "[Step 2/7] Installing dependencies (this takes a few minutes)..."
pip install --upgrade pip setuptools wheel -q

# Install the project in editable mode (pulls all deps from pyproject.toml)
pip install -e ".[dev]" -q 2>&1 | tail -3

echo "  Installed packages:"
echo "    mujoco    $(python -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null || echo 'FAILED')"
echo "    torch     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'FAILED')"
echo "    gymnasium $(python -c 'import gymnasium; print(gymnasium.__version__)' 2>/dev/null || echo 'FAILED')"
echo "    h5py      $(python -c 'import h5py; print(h5py.version.version)' 2>/dev/null || echo 'FAILED')"
echo "    pytest    $(python -c 'import pytest; print(pytest.__version__)' 2>/dev/null || echo 'FAILED')"

# ─── Step 3: Run tests ───
echo ""
echo "[Step 3/7] Running test suite..."
python -m pytest tests/ -v --tb=short 2>&1 | tail -40

# ─── Step 4: Inspect device models ───
echo ""
echo "[Step 4/7] Inspecting device models..."
python scripts/visualize_sim.py --device laptop_v1 --mode info
echo ""
python scripts/visualize_sim.py --device router_v1 --mode info

# ─── Step 5: Collect demonstration data ───
echo ""
echo "[Step 5/7] Collecting demonstration trajectories..."
echo "  Laptop demos (20 trajectories for quick test)..."
python scripts/collect_demos.py --device laptop_v1 --num-trajectories 20 --output-dir data/trajectories
echo ""
echo "  Router demos (20 trajectories for quick test)..."
python scripts/collect_demos.py --device router_v1 --num-trajectories 20 --output-dir data/trajectories

# ─── Step 6: Run full pipeline validation ───
echo ""
echo "[Step 6/7] Running full pipeline validation..."
python scripts/run_full_pipeline.py --quick

# ─── Step 7: Render simulation frames ───
echo ""
echo "[Step 7/7] Rendering simulation frames..."
python scripts/visualize_sim.py --device laptop_v1 --mode render --output-dir renders
echo ""

echo "============================================================"
echo "SETUP COMPLETE"
echo ""
echo "What to do next:"
echo ""
echo "  1. Activate the venv:    source .venv/bin/activate"
echo "  2. Collect full data:    python scripts/collect_demos.py --device laptop_v1 --num-trajectories 200 --randomize"
echo "  3. Train models:         python scripts/train.py --model both --data data/trajectories/laptop_v1_demos.h5 --device cuda"
echo "  4. Evaluate:             python scripts/evaluate.py --method all --output-dir results"
echo "  5. Run tests anytime:    python -m pytest tests/ -v"
echo "============================================================"
