#!/usr/bin/env bash
# Reproduces all numerical experiments of the paper.
# Run from anywhere:  bash code/run_all.sh
# Needs Python 3.9+ with numpy and matplotlib (pip install -r requirements.txt).
set -e
cd "$(dirname "$0")/.."
python code/exp_purity.py              # Experiment 1, about 1 minute
python code/plot_purity.py             # -> paper/figures/fig_purity_fidelity.{pdf,png}
python code/exp_recovery.py codim      # Experiment 2a, about 20-40 minutes on one core
python code/exp_recovery.py time       # Experiment 2b, estimated 10-30 minutes
python code/plot_recovery.py           # -> paper/figures/fig_recovery.{pdf,png}
