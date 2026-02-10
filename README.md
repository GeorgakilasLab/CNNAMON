<div align="center">

<img src="https://raw.githubusercontent.com/GeorgakilasLab/CNNAMON/main/docs/img/logo_cnnamon.svg" alt="CNNAMON Logo" width="200" />

<h1>CNNAMON: Convolutional Neural Network Analysis & Motif Discovery</h1>

<p><strong>A modular, interpretability-first framework for deep learning in genomics.</strong></p>

<p>
  <a href="https://badge.fury.io/py/cnnamon">
    <img src="https://badge.fury.io/py/cnnamon.svg" alt="PyPI version" />
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

<hr>

<p>
<strong>CNNAMON</strong> is a Python framework designed to bridge the gap between training high-performance <strong>1D Convolutional Neural Networks (CNNs)</strong> on DNA sequences and understanding <em>what</em> they actually learn.
</p>

<p>It provides an end-to-end ecosystem for:</p>
<ol>
  <li><strong>Dataset Preparation:</strong> Converting genomic intervals (BED3+1-labels) to One-Hot Tensors.</li>
  <li><strong>Modeling:</strong> Building complex Keras models via simple JSON config file.</li>
  <li><strong>Explainability:</strong> Extracting learned motifs, clustering filters based on activation profile, assesing the filter importance and associating filters to prediction classes.</li>
</ol>

<hr>

<h2>⚡️ Key Features</h2>

<table>
  <thead>
    <tr>
      <th>Module</th>
      <th>Functionality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>🧬 PrepareData</strong></td>
      <td>Extraction of sequences from FASTA/BED files. Supports Random, Chromosome, or Custom splits and reverse complement augmentation.</td>
    </tr>
    <tr>
      <td><strong>🏗 KerasBuilder</strong></td>
      <td>Define model architecture, optimizers, and callbacks in <strong>JSON</strong>. Ensures reproducibility and sharing of experiments.</td>
    </tr>
    <tr>
      <td><strong>🎨 FilterVisualize</strong></td>
      <td>Extract learned motifs using <strong>Top-Activating</strong>, <strong>Consensus</strong>, or <strong>Significant</strong> (permutation-based) strategies. Export to <strong>MEME</strong> for TOMTOM validation.</td>
    </tr>
    <tr>
      <td><strong>📉 FilterImportance</strong></td>
      <td>Rank filters by their contribution to model loss using perturbation analysis.</td>
    </tr>
    <tr>
      <td><strong>🌳 FilterClustering</strong></td>
      <td>Group redundant or co-activated filters with hierarchical clustering and visualize relationships with circular dendrograms.</td>
    </tr>
    <tr>
      <td><strong>🧪 Enrichment</strong></td>
      <td>Identify filters that are statistically enriched for the prediction classes (e.g., Enhancer vs. Silencer).</td>
    </tr>
  </tbody>
</table>

<hr>

<h2>📦 Installation</h2>

<p>We recommend installing CNNAMON in a fresh environment to manage dependencies (TensorFlow, BedTools).</p>

<h3>(Recommended)</h3>
<pre><code>
# 1. Create environment
conda create -n cnnamon_env python=3.10
conda activate cnnamon_env

# 2. Install library
pip install cnnamon 

# 3. Install BedTools (Required for sequence extraction)
conda install -c bioconda bedtools</code></pre>


<hr>

<h2>🚀 Quick Start</h2>
<p>Train a model and visualize motifs in 4 steps.</p>

<pre><code>import cnnamon as cn

# 1. Prepare Data
preparer = cn.utility.PrepareData(
    intervalfile="peaks.bed", 
    genomefasta="hg38.fa", 
    outdir="data/",
    split_segmentation="random"
)
train, test, val = preparer.run()

# 2. Train Model (from JSON config)
model = cn.utility.KerasModelBuilder.from_json("model_config.json")
model.train(train['x'], train['y'], val['x'], val['y'])

# 3. Extract & Visualize Motifs
# Extract the signifficant motifs
motifs = cn.CNN1D.FilterVisualize.significant_activating(model, 
                                                      data=test, 
                                                      n_perturbations=1000,
                                                      q_value_cutoff=0.05,
                                                      n_cores=10)

# 4. Plot Sequence Logos
motifs.to_motifs(savefig="learned_motifs.png")</code></pre>

<hr>

<h2>📖 Documentation</h2>
<p>Full documentation is available here:<br>
<a href="https://georgakilaslab.github.io/CNNAMON/"><strong>CNNAMON Documentation</strong></a></p>

<hr>

<h2>📚 Citation</h2>

<p>If you use CNNAMON in your research, please cite:</p>


<br>

<p align="center">
  <sub>Built by the Georgakilas Lab.</sub>
</p>