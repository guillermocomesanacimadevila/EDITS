# 📸 EDITS: Automated Cell Event Recognition & Self-Supervised Temporal Representation Learning

*Temporal feature learning and event classification for live-cell imaging*

---

**Authors:**  
Cangxiong Chen, Guillermo Comesaña Cimadevila, Vinay P. Namboodiri, Julia E. Sero  

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c1c6875-df99-415a-b714-f5d9d6bec2c1" alt="composite_final_A" width="49%" />
  <img src="https://github.com/user-attachments/assets/03b4d94e-8475-4503-aa14-f3116e413e49" alt="T27-T42" width="49%" />
</p>

---

## 🌟 Overview

**EDITS** is a fully automated, interactive pipeline for extracting dense cellular features and classifying cell events from microscopy time-lapse images.  
It leverages self-supervised learning (**Time Arrow Prediction**, TAP) to learn robust spatiotemporal feature representations, followed by event classification, hyperparameter tuning, and generation of publication-ready outputs.

**Key features:**
- **No coding required:** Beginner-friendly, menu-driven interface.
- **Optimised for cloud and GPU:** Seamless performance on `vast.ai` with H100/A100 GPUs; CPU and macOS supported (but slower).
- **Reproducible and modular:** All configurations, logs, and outputs are automatically saved.
- **Comprehensive outputs:** Interactive HTML reports, summary figures, and reproducible logs are generated for every run.

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
git clone https://github.com/guillermocomesanacimadevila/edits.git
```

```bash
cd EDITS/
```

---

## 🗂️ Data Setup

**After cloning the repository,** add all your `.tif` movies and corresponding mask files into the `Data/` directory:

```bash
EDITS/
├── Data/
│   ├── movie1.tif
│   ├── movie2.tif
│   ├── mask1.tif
│   └── ...
├── Workflow/
│   └── ...
├── run_edits_conda.sh
├── environment.yml
└── ...
```

<img src="https://github.com/user-attachments/assets/71a2fda7-719f-4553-a92a-af6bff5344cd"
     width="420"
     alt="EDITS Pipeline Workflow"/>

- Masks should correspond to each movie as appropriate.
- Input files are selected interactively when running the pipeline.

---

## 🚀 Running the Pipeline

Start the **interactive pipeline**:

```bash
bash run_edits_conda.sh
```

---

## 🏗️ Pipeline Workflow

EDITS follows a **modular workflow**:

![ Instagram Facebook Ads - last chance (1080x1080px)-8](https://github.com/user-attachments/assets/6ef350b6-4a65-4d63-8fa5-e81880351e20)

1. **Data ingestion:** Load and validate `.tif` time-lapse movies and corresponding masks  
2. **Time arrow prediction (TAP):** Learn robust spatiotemporal representations  
3. **Supervised fine-tuning:** Train classifiers for event detection  
4. **Hyperparameter optimisation:** Automatically search spatial and temporal configurations  
5. **Reporting:** Generate interactive HTML summaries, diagnostic plots, and logs

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
