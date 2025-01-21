# Plant Simulation RL Agents for AGV Systems

## Overview

This repository contains Python scripts that demonstrate how to use Plant Simulation as a learning environment for Reinforcement Learning (RL) algorithms to tackle deadlock situations in Automated Guided Vehicle (AGV) systems. The code is compatible with the Gymnasium and Ray libraries.

## Files Included

### `agent_RLlib.py`

This file contains the RL agent implemented using Ray's RLlib. The agent can be configured to work with one of the two environments provided in this repository.

### `env_AGVsimple_multiagent.py`

This is a multi-agent environment that simulates an AGV system with deadlock capabilities. It is implemented using the Gymnasium framework and is compatible with Ray's MultiAgentEnv.

### `env_AGVsimple_gymnasium.py`

This is a single-agent environment that also simulates an AGV system with deadlock capabilities. It is implemented using the Gymnasium framework.

## Requirements

- Python 3.x
- Ray 2.35.0 (recommended)
- Gymnasium
- Plant Simulation > 2201

## Contributing

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.
