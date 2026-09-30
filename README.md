# Metric-ProtoNet: Few-Shot Tobacco Leaf Maturity Classification

Official code and dataset repository for the paper:

**"Metric-ProtoNet: A Learnable Metric for Few-Shot Tobacco Leaf Maturity Classification"**
S. Manjunath, P. B. Mallikarjuna, D. S. Guru
*Submitted to Applied Soft Computing (Elsevier)*

## Overview

This repository accompanies a study on few-shot classification of tobacco leaf ripeness (**Unripe**, **Ripe**, **Overripe**) for low-cost mobile and robotic grading applications. We propose **Metric-ProtoNet**, a lightweight extension of Prototypical Networks that learns a diagonal distance metric (576 parameters) over a frozen MobileNetV3-Small backbone — achieving accuracy competitive with full-supervision baselines while using ~13× less labeled data than prior work on this task, and three orders of magnitude fewer trainable parameters than full backbone fine-tuning.

This work extends our group's earlier texture-based approaches to the same problem:
- Guru et al. (2012), *Machine Vision Based Classification of Tobacco Leaves for Automatic Harvesting*, Intelligent Automation & Soft Computing.
- Mallikarjuna & Guru (2022), *Selective Harvesting of Tobacco Leaves: An Approach Based on Texture Features*, Statistics and Applications.

## Repository Structure

├── code/
│ ├── config.py # Central configuration (all parameters)
│ ├── data_loader.py # Dataset loading, stratified split, episode sampling
│ ├── feature_extractor.py # Frozen MobileNetV3-Small backbone
│ ├── evaluation.py # Metrics, confusion matrices, statistical tests
│ ├── visualization.py # Publication-ready plots
│ ├── run_experiment.py # Main experiment runner
│ ├── run_ablation.py # Ablation study runner (A1–A4)
│ ├── run_complexity_analysis.py # Parameter/FLOPs/latency analysis
│ ├── generate_results_report.py # Excel report + summary plots
│ ├── requirements.txt
│ └── methods/
│ ├── transfer_learning.py # Baseline: frozen backbone + linear classifier
│ ├── protonet.py # Baseline: standard Prototypical Network
│ ├── metric_protonet.py # Proposed: learnable diagonal metric
│ └── full_finetune.py # Upper bound: full backbone fine-tuning
├── dataset/
│ ├── Unripe/ # 323 images
│ ├── Ripe/ # 667 images
│ └── Overripe/ # 300 images
├── results/ # CSVs and figures from the reported experiments
└── README.md


## Dataset

1,290 RGB images of tobacco leaves collected in real field conditions (sunny and cloudy illumination) at the Central Tobacco Research Institute (CTRI), Hunsur, Karnataka, India, across three ripeness classes:

| Class     | Images |
|-----------|--------|
| Unripe    | 323    |
| Ripe      | 667    |
| Overripe  | 300    |
| **Total** | **1,290** |

The dataset originates from field collection supporting our group's prior work (Guru et al., 2012; Mallikarjuna & Guru, 2022) and is released here for reproducibility of the present study.

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/code
pip install -r requirements.txt
```

Requires PyTorch with CUDA support for GPU-accelerated training (CPU also supported, slower).

## Usage

```bash
# Set dataset path
export TOBACCO_DATASET_PATH=/path/to/dataset

# Run the full experiment (5 k-shot settings × 10 runs × 4 methods)
python run_experiment.py

# Run the ablation study (A1–A4)
python run_ablation.py

# Run computational complexity analysis
python run_complexity_analysis.py

# Generate the Excel report and summary plots
python generate_results_report.py
```

Quick smoke test: `python run_experiment.py --quick`

## Results Summary

| Method | k=10 Accuracy | Trainable Params |
|---|---|---|
| Transfer Learning | 81.94% | 1,731 |
| ProtoNet | 81.09% | 0 |
| **Metric-ProtoNet (proposed)** | **83.70%** | **576** |
| Full Fine-Tune (upper bound) | 94.57%* | 1,520,931 |

*Full Fine-Tune uses all available training data, not the k-shot setting shown.

Full results, statistical significance tests, and ablation studies are reported in the paper.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{manjunath2026metricprotonet,
  title   = {Metric-ProtoNet: A Learnable Metric for Few-Shot Tobacco Leaf Maturity Classification},
  author  = {Manjunath, S. and Mallikarjuna, P. B. and Guru, D. S.},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Under review}
}
```

*(Citation will be updated with full volume/page/DOI details upon publication.)*

## License

The **code** in this repository is released under the [MIT License](LICENSE).

The **dataset** is provided for research and reproducibility purposes. If you plan to redistribute or reuse the dataset beyond replicating this study, please contact the authors.

## Contact

- S. Manjunath — manjunath.shantharamu@gmail.com
- P. B. Mallikarjuna — pbmalli2020@gmail.com

## Acknowledgements

We thank the Central Tobacco Research Institute (CTRI), Hunsur, Karnataka, for field access supporting the original dataset collection.
