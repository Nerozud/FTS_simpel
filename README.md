# Plant Simulation RL Agents for AGV Systems

## Overview

This repository contains Python scripts that demonstrate how to use Plant Simulation as a learning environment for Reinforcement Learning (RL) algorithms to tackle deadlock situations in Automated Guided Vehicle (AGV) systems. The code is compatible with the Gymnasium and Ray libraries.

## Files Included

### `main.py`

This file is the main entry point for training and testing multi-agent reinforcement learning models using Ray RLlib.

### `env_AGVsimple_gymnasium.py`

This is a single-agent environment that simulates an AGV system with deadlock capabilities. It is implemented using the Gymnasium framework.

### `env_AGVsimple_multiagent.py`

This is a multi-agent environment that simulates an AGV system with deadlock capabilities. It is implemented using the Gymnasium framework and is compatible with Ray's MultiAgentEnv.

### `env_agv_simple_pomdp.py`

This file contains another multi-agent environment for AGV systems, implemented using the Gymnasium framework and compatible with Ray's MultiAgentEnv.

### `simulation_models/`

This directory contains various simulation model files used by the environments.

### `experiments/`

This directory is used to store experiment data, including logs and results.

## Requirements

- Python 3.x
- Ray 2.35.0
- Gymnasium
- Plant Simulation > 2201
- PyTorch (for CUDA testing)



PS: Sorry, everything is still messy, I hopefully will organize it soon. 