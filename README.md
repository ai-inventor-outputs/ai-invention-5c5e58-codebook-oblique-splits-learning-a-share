# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/paper/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/tree/main/paper) [![Figures](https://img.shields.io/badge/Figures-5-blue)](https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/tree/main/figures)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_codebook_figs_t` | Codebook-FIGS Tabular Benchmark Suite (15 Datasets... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/dataset_iter1_codebook_figs_t/demo/data_code_demo.ipynb) |
| `experiment_iter2_unconstrained_o` | Unconstrained Oblique Baselines: SPORF + Oblique F... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/experiment_iter2_unconstrained_o/demo/method_code_demo.ipynb) |
| `experiment_iter2_codebook_figs_a` | Codebook-FIGS Ablation: Initialization, Refinement... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/experiment_iter2_codebook_figs_a/demo/method_code_demo.ipynb) |
| `experiment_iter2_codebook_figs_i` | Codebook-FIGS: Implementation, Benchmarking, and E... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/experiment_iter2_codebook_figs_i/demo/method_code_demo.ipynb) |
| `evaluation_iter3_comprehensive_s` | Comprehensive Statistical Evaluation of Codebook-F... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/evaluation_iter3_comprehensive_s/demo/eval_code_demo.ipynb) |
| `evaluation_iter3_codebook_figs_i` | Codebook-FIGS Interpretability Evaluation... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/evaluation_iter3_codebook_figs_i/demo/eval_code_demo.ipynb) |
| `experiment_iter3_definitive_code` | Definitive Codebook-FIGS Full K-Sweep Benchmark (I... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/experiment_iter3_definitive_code/demo/method_code_demo.ipynb) |
| `evaluation_iter4_definitive_code` | Definitive Codebook-FIGS Final Synthesis Evaluatio... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/evaluation_iter4_definitive_code/demo/eval_code_demo.ipynb) |
| `evaluation_iter4_diagnostic_gap` | Diagnostic Gap Decomposition of Codebook-FIGS Accu... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/evaluation_iter4_diagnostic_gap/demo/eval_code_demo.ipynb) |
| `experiment_iter4_codebook_figs_v` | Codebook-FIGS v2: Joint Gradient Refinement with E... | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/experiment_iter4_codebook_figs_v/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_codebook_figs` | Codebook-FIGS... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/research_iter1_codebook_figs/demo/research_demo.md) |
| `research_iter1_oblique_trees` | Oblique Trees... | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share/blob/main/research_iter1_oblique_trees/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper/                       # LaTeX paper and PDF
├── figures/                     # Visualizations
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/ai-inventor-outputs/ai-invention-5c5e58-codebook-oblique-splits-learning-a-share.git
cd ai-invention-5c5e58-codebook-oblique-splits-learning-a-share

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook exp_001/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
