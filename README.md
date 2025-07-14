# Multi-Agent Emergent Communication (MAC)

This repository provides a research framework for studying **emergent communication** in multi-agent reinforcement learning (MARL) environments. It supports a variety of on-policy algorithms and environments, enabling the exploration of how communication protocols arise, evolve, and can be interpreted in cooperative and competitive multi-agent settings.

## Key Features

- **On-Policy MARL Algorithms**: Includes implementations of algorithms such as RMAPPo, TMAPPo, MACPPO, MEMO_PPO, and more.
- **Emergent Communication**: Tools and environments for analyzing and interpreting communication between agents.
- **Diverse Environments**: Supports StarCraft II, Hanabi, MPE (Multi-Agent Particle Environments), Traffic Junction, MNIST-based tasks, and more.
- **Extensible Framework**: Modular design for easy addition of new algorithms, environments, and communication protocols.
- **Experiment Scripts**: Ready-to-use scripts for training, evaluation, and rendering across supported environments.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd MAC
   ```

2. **Install dependencies:**
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Or with conda:
     ```bash
     conda env create -f environment.yaml
     conda activate mac
     ```

## Repository Structure

- `onpolicy/algorithms/` — On-policy MARL algorithms (e.g., RMAPPo, MACPPO, MEMO_PPO)
- `onpolicy/envs/` — Supported environments (StarCraft2, Hanabi, MPE, Traffic Junction, MNIST, etc.)
- `onpolicy/runner/` — Training and evaluation runners
- `onpolicy/scripts/` — Shell scripts for launching experiments and evaluations
- `onpolicy/utils/` — Utility functions and helpers
- `onpolicy/config.py` — Centralized configuration and hyperparameter management

## Getting Started

To train a model in a supported environment, use one of the provided scripts. For example:
```bash
bash onpolicy/scripts/train/train_smac_8m.sh
```
Modify the scripts or use `onpolicy/config.py` to adjust hyperparameters and experiment settings.

## Citing This Work

If you use this repository or its components in your research, please cite the following works:

```bibtex
@article{karten2023interpretable,
  title={Interpretable learned emergent communication for human--agent teams},
  author={Karten, Seth and Tucker, Mycal and Li, Huao and Kailas, Siva and Lewis, Michael and Sycara, Katia},
  journal={IEEE Transactions on Cognitive and Developmental Systems},
  volume={15},
  number={4},
  pages={1801--1811},
  year={2023},
  publisher={IEEE}
}
@article{karten2023role,
  title={On the role of emergent communication for social learning in multi-agent reinforcement learning},
  author={Karten, Seth and Kailas, Siva and Li, Huao and Sycara, Katia},
  journal={arXiv preprint arXiv:2302.14276},
  year={2023}
}
@article{karten2022towards,
  title={Towards true lossless sparse communication in multi-agent systems},
  author={Karten, Seth and Tucker, Mycal and Kailas, Siva and Sycara, Katia},
  journal={arXiv preprint arXiv:2212.00115},
  year={2022}
}
@phdthesis{karten2023emergent,
  title={Emergent Communication and Decision-Making in Multi-Agent Teams},
  author={Karten, Seth},
  year={2023},
  school={Carnegie Mellon University Pittsburgh, PA}
}
```

## License

This project is licensed under the MIT License.
