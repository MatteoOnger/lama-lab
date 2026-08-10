# LAMA-Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/MatteoOnger/lama-lab)


**LAMA-Lab** (**L**earning **A**gents in **M**arket **A**rchitectures) is a modular Python research framework designed for simulating, logging and analyzing multi-agent learning dynamics in complex economic environments.

## Architecture Overview

```text
lama-lab/
├── configs/                # Sample configuration files (YAML)
├── lama_lab/               # Core framework library
│   ├── agents/             # Autonomous agent implementations
│   ├── analysis/           # Functions and tools for evaluating and analyzing results
│   ├── envs/               # Market environment logic
│   ├── generators/         # Synthetic valuation generators
│   ├── plotting/           # Static and interactive visualization routines
│   ├── projectors/         # Bound-enforcement components
│   └── utils/              # ResultsManager, RingBuffer memory management, logging utils
├── notebooks/              # Jupyter & Google Colab analysis notebooks
├── results/                # Default directory for experiment outputs and saved artifacts
├── scripts/                # Standalone simulation execution scripts
└── pyproject.toml          # Project configuration and dependency specifications