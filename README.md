# 📸 EDITS: Automated Cell Event Recognition & Self-Supervised Temporal Representation Learning

> *Temporal feature learning and event classification for live-cell microscopy*

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c1c6875-df99-415a-b714-f5d9d6bec2c1" alt="composite_final_A" width="49%" />
  <img src="https://github.com/user-attachments/assets/03b4d94e-8475-4503-aa14-f3116e413e49" alt="T27-T42" width="49%" />
</p>

---

**Authors:**  
Guillermo Comesaña Cimadevila · Cangxiong Chen · Vinay P. Namboodiri · Julia E. Sero  

---

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20WSL2-lightgrey.svg)]()
[![Powered by PyTorch](https://img.shields.io/badge/pytorch-✅-ee4c2c.svg)]()
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Documentation](https://img.shields.io/badge/docs-auto_generated-green.svg)]()

---

## 🌟 Overview

**EDITS** is an automated, modular, and fully interactive pipeline for extracting dense cellular features and classifying cell events from microscopy time-lapse images.  
It combines **self-supervised temporal representation learning** (via *Time Arrow Prediction, TAP*) with **supervised event classification**, enabling reproducible workflows, without writing any code.

---

### 🧩 Highlights

- 🧠 **Self-supervised feature learning (TAP):** Learns temporal directionality in live-cell imaging data.  
- 🔍 **Event classification:** Detects and classifies dynamic cellular events (e.g., divisions, fusions).  
- 💡 **Zero-code workflow:** Fully interactive command-line interface.  
- 📊 **Comprehensive outputs:** Metrics, reports, Grad-CAMs, and interactive HTML summaries.  
- 🧾 **Reproducible:** Every run logs configs, seeds, model weights, and figures in a structured format.

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

> *Tested on vast.ai H200 France (recommended), and Apple Silicon (CPU mode).*

---

## 🚦 Getting Started

### 1. Clone the repository

Clone the repository to your machine:

```bash
git clone https://github.com/guillermocomesanacimadevila/EDITS.git
```

```bash
cd EDITS/
```

---

## 🗂️ Data Setup

**After cloning the repository,** add all your `.tif` movies and corresponding mask files into the `Data/` directory:

```bash
EDITS/
└── Data/
    ├── movie1.tif
    ├── movie1_mask.tif
    ├── movie2.tif
    ├── movie2_mask.tif
    └── ...
```

<img src="https://github.com/user-attachments/assets/71a2fda7-719f-4553-a92a-af6bff5344cd"
     width="420"
     alt="EDITS Pipeline Workflow"/>

- Masks should correspond spatially to each movie.
- Example data are provided in `Data/toy_data/` for testing.

---

## 🚀 Running the Pipeline

Start the **interactive pipeline**:

```bash
chmod +x run_edits.sh && bash run_edits.sh
```

---

## 🧭 Workflow Overview

![ Instagram Facebook Ads - last chance (1080x1080px)-8](https://github.com/user-attachments/assets/6ef350b6-4a65-4d63-8fa5-e81880351e20)

The **EDITS** pipeline follows a modular five-phase structure, ensuring transparency, reproducibility, and easy debugging throughout the full training and analysis workflow:

| 🧩 **Phase** | 🧠 **Purpose** | 📂 **Output Directory** |
|:-------------|:---------------|:------------------------|
| **① TAP Pretraining** | Learns temporal directionality using self-supervised *Time Arrow Prediction* | `phase1_pretraining/` |
| **② Data Preparation** | Extracts balanced, labeled crops for training and validation | `phase2_data_prep/` |
| **③ Event Classification** | Trains supervised classifiers for event recognition | `phase3_classification/` |
| **④ Error Analysis** | Identifies and analyses false positives / negatives | `phase4_mistake_analysis/` |
| **⑤ Visualisation & Reports** | Generates Grad-CAMs, figures, and interactive HTML summaries | `phase5_grad_cam/` |

---

Each experiment automatically creates a uniquely timestamped and seed-tagged run folder:


---

## 📊 Outputs

Each run generates a structured **output folder** containing:

- **Trained models:** Self-supervised and supervised checkpoints  
- **Interactive HTML reports:** Explore results interactively in your browser  
- **Loss and accuracy curves:** Training diagnostics for reproducibility  
- **Grad-CAM visualisations:** Interpretability maps for event localisation  
- **Event detection summaries:** Tabular outputs for downstream analysis  

---

## 🧠 Citation

If you use **EDITS** in your research, please cite our momentary preprint:

Chen, C., Namboodiri, V. P., & Sero, J. E. (2024). *Self-supervised Representation Learning for Cell Event Recognition through Time Arrow Prediction*. arXiv preprint [arXiv:2411.03924](https://arxiv.org/abs/2411.03924).

---

## 📬 Contact

For questions, feedback, or support, reach out to:

Guillermo Comesaña Cimadevila – gcc46@bath.ac.uk
