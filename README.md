<div align="center">

<img src="https://raw.githubusercontent.com/GeorgakilasLab/CNNAMON/main/docs/img/logo_cnnamon.svg" alt="CNNAMON Logo" width="200" />

# CNNAMON  
### Convolutional Neural Network Analysis & Motif Discovery

**A modular, interpretability-first framework for deep learning in genomics.**

<p>
  <a href="https://pypi.org/project/cnnamon/">
    <img src="https://img.shields.io/pypi/v/cnnamon" alt="PyPI version" />
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" />
  </a>
  <a href="https://georgakilaslab.github.io/CNNAMON/">
    <img src="https://img.shields.io/badge/docs-online-green" alt="Documentation" />
  </a>
</p>

</div>

---

## 🧠 Overview

**CNNAMON** is a Python framework designed to bridge the gap between training high-performance  
**1D Convolutional Neural Networks (CNNs)** on DNA sequences and understanding *what* they learn.

It provides an end-to-end ecosystem for:

1. **Dataset Preparation** – Converting genomic intervals (BED3 + labels) to one-hot tensors  
2. **Modeling** – Building complex Keras models via simple JSON configuration files  
3. **Explainability** – Extracting learned motifs, clustering filters by activation profiles, assessing filter importance, and associating filters with prediction classes  

---

## ⚡ Key Features

| Module              | Functionality |
|---------------------|---------------|
| **🧬 PrepareData**      | Extract sequences from FASTA/BED files. Supports random, chromosome, or custom splits, and reverse-complement augmentation. |
| **🏗 KerasBuilder**     | Define model architectures, optimizers, and callbacks using **JSON** for reproducible experiments. |
| **🎨 FilterVisualize** | Extract learned motifs using **Top-Activating**, **Consensus**, or **Significant** (permutation-based) strategies. Export to **MEME** for TOMTOM validation. |
| **📉 FilterImportance**| Rank filters by their contribution to model loss using perturbation analysis. |
| **🌳 FilterClustering**| Group redundant or co-activated filters with hierarchical clustering and visualize relationships using circular dendrograms. |
| **🧪 Enrichment**       | Identify filters statistically enriched for prediction classes (e.g., Enhancer vs. Silencer). |

---

## 📦 Installation

We recommend installing CNNAMON in a fresh environment to manage dependencies (TensorFlow, BedTools).

```bash
# 1. Create environment
conda create -n cnnamon_env python=3.10
conda activate cnnamon_env

# 2. Install CNNAMON
pip install cnnamon

# 3. Install BedTools (required for sequence extraction)
conda install -c bioconda bedtools
```

---

## 🚀 Quick Start

Train a model and visualize motifs in four steps:

```python
import cnnamon as cn

# 1. Prepare data
preparer = cn.utility.PrepareData(
    intervalfile="peaks.bed", 
    genomefasta="hg38.fa", 
    outdir="data/",
    split_segmentation="random"
)
train, test, val = preparer.run()

# 2. Train model (from JSON config)
model = cn.utility.KerasModelBuilder.from_json("model_config.json")
model.train(train['x'], train['y'], val['x'], val['y'])

# 3. Extract significant motifs
motifs = cn.CNN1D.FilterVisualize.significant_activating(
    model, 
    data=test, 
    n_perturbations=1000,
    q_value_cutoff=0.05,
    n_cores=10
)

# 4. Plot sequence logos
motifs.to_motifs(savefig="learned_motifs.png")
```

---

## 📖 Documentation

Full documentation is available here:  
👉 **https://georgakilaslab.github.io/CNNAMON/**

---

## 📚 Citation

If you use CNNAMON in your research, please cite:

> *(Add paper reference / DOI here when available)*

---

<p align="center">
  <sub>Built by the Georgakilas Lab.</sub>
</p>
