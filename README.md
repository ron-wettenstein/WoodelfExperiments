# WoodelfExperiments

This repository contains the experiments presented in the Woodelf papers, along with the original notebooks used to run them. These notebooks are the exact versions used in our experiments and include the reported runtime results.

### Included Experiments

* Experiments with the Woodelf algorithm from the paper 
“From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm” published at AAAI-26. 
  Processing [link](https://ojs.aaai.org/index.php/AAAI/article/view/39630), arXiv [link](https://arxiv.org/abs/2511.09376)  (full version).
* Experiments with the Woodelf and WoodelfHD algorithms from the paper "WOODELF-HD: Efficient Background SHAP for High-Depth Decision Trees". See in arXiv: TODO

### Reproducibility

All notebooks were executed on Google Colab. The runtime configuration (CPU, high-memory CPU, GPU, etc.) and additional technical details are documented within the notebooks and the corresponding papers.

### The Woodelf Package

We also published a Python package containing our latest code:  
**[WoodElf Python Package](https://github.com/ron-wettenstein/woodelf)**

Install it using:
```bash
pip install woodelf_explainer