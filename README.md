# 📸 CellFate: Automated Cell Event Recognition & Self-Supervised Temporal Representation Learning

*Temporal feature learning and event classification for live-cell imaging*

---

**Authors:**  
Cangxiong Chen, Guillermo Comesaña Cimadevila, Vinay P. Namboodiri, Julia E. Sero

---

## 🌟 Overview

**CellFate** is a fully automated, interactive pipeline for extracting dense cellular features and classifying cell events from microscopy time-lapse images.  
It uses self-supervised learning (Time Arrow Prediction, TAP) to train robust feature representations, followed by event classification, hyperparameter search, and publication-ready outputs.

- **No coding required:** Everything is menu-driven and beginner-friendly.
- **Optimised for cloud and GPU:** Runs beautifully on Vast.ai H200 France (H100), but also supports CPU or Mac (albeit slower).
- **Reproducible and modular:** All configurations, logs, and outputs are saved.
- **Comprehensive outputs:** Interactive HTML reports and figures are generated for every run.

---

## 🖥️ Computational Requirements

| Resource        | Recommended                               | Minimum        | Notes                              |
|-----------------|-------------------------------------------|---------------|-------------------------------------|
| **GPU**         | NVIDIA A100/H100/RTX, 16GB+ VRAM          | Any GPU/CPU   | Required for efficient training     |
| **System RAM**  | 32GB+                                     | 8GB           | More RAM speeds up data loading     |
| **Disk Space**  | 30GB+                                     | 10GB          | Storage for runs and outputs        |
| **OS**          | Linux, macOS (UNIX), WSL2                 | Linux/macOS   | Windows via WSL2 or Docker          |
| **Python**      | 3.8 or newer                              | 3.8+          |                                     |
| **Conda**       | Miniconda/Anaconda (auto-installed)       | Miniconda     |                                     |

> *Tested on vast.ai H200 France (recommended), Apple Silicon (CPU mode), and standard Linux systems.*

---

## 🚦 Getting Started

### 1. Clone the repository

**Before you add any data,** clone the repository to your machine:

```bash
git clone https://github.com/guillermocomesanacimadevila/CellFate.git
```

```bash
cd CellFate/
```

---

## 🗂️ Data Setup

**After cloning the repository,** add all your `.tif` movies and corresponding mask files into the `Data/` directory:

```bash
CellFate/
├── Data/
│   ├── movie1.tif
│   ├── movie2.tif
│   ├── mask1.tif
│   └── ...
├── Workflow/
│   └── ...
├── run_cellfate_conda.sh
├── environment.yml
└── ...
```

<img src="https://github.com/user-attachments/assets/71a2fda7-719f-4553-a92a-af6bff5344cd"
     width="420"
     alt="CellFate Pipeline Workflow"/>

- Masks should correspond to each movie as appropriate.
- Input files are selected interactively when running the pipeline.

---

## 🚀 Running the Pipeline

Start the **interactive pipeline**:

```bash
bash run_cellfate_conda.sh
```

---

## 🏗️ Pipeline Workflow

