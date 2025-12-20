# CNNAMON


<p align="center">
  <img src="docs/img/logo_cnnamon.svg" alt="CNNAMON Logo" height="180" />
</p>



**CNNAMON** is a modular Python framework for building, training, and interpreting **1D Convolutional Neural Networks (CNNs)** for **DNA sequence analysis**.  
It integrates data preparation, model construction, and rich explainability tools in a unified and flexible system tailored for genomics research.

---

## Features

### Modular Architecture
- JSON-based CNN model specification  
- Flexible Keras model building and training  
- Support for optimizers, callbacks, and training visualization  

### Data Preparation
- Extract sequences from BED intervals  
- One-hot encoding for DNA sequences  
- Multiple splitting strategies: random, chromosome-based, or custom  
- Optional reverse-complement augmentation  

### Interpretability & Explainability
- **Filter Visualization:** motif extraction using multiple strategies  
- **Filter Importance:** perturbation-based ranking  
- **Filter Clustering:** hierarchical circular clustering visualizations  
- **Filter Enrichment:** filter enrichment due to class-specific activation

---

## Installation

Clone and install in editable mode:

```bash
git clone https://gitlab.com/a5465/cnnamon.git
cd cnnamon
pip install -e .
