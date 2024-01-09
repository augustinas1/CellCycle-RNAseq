# CellCycle-RNAseq

This repository contains the code for the paper
&nbsp;&nbsp;&nbsp; A. Sukys and R. Grima, "Transcriptome-wide analysis of cell cycle-dependent bursty gene expression from single-cell RNA-seq data using mechanistic model-based inference" (2024).

The code is used to perform mechanistic model-based inference on scRNA-seq data for a population of mouse embryonic stem cells (mESCs), where the cell-cycle phase (G1, S or G2/M) and cell age $\theta$ (position along the cell cycle) are known for each cell. The processed dataset used throughout is uploaded on Zenodo, and is based on the original work by Riba et al. [[1]](#1).

### Structure

- `src` - main Julia code used to build quantitative models of gene expression, perform maximum likelihood estimation and model selection, and compute the confidence intervals using profile likelihood.
- `analysis` - scripts for the mESC dataset-specific analysis, considering cell age-dependent ($\theta$-dependent) and cell age-independent ($\theta$-independent) mechanistic models using both cell-cycle-phase-specific (G1 or G2/M) data and merged (G1 + G2/M) data. In each case, the scripts are used to perform inference, model selection and confidence interval estimation.
- `notebooks` - Jupyter notebooks (written in Julia) used to explore the results and to generate the figures in the paper.
- `data` - directory to save the generated files, such as model-specific fits. 

### References:

<a id="1">[1]</a> Riba, A., Oravecz, A., Durik, M. et al. Cell cycle gene regulation dynamics revealed by RNA velocity and deep-learning. Nat Comm 13, 2865 (2022). [https://doi.org/10.1038/s41467-022-30545-8](https://doi.org/10.1038/s41467-022-30545-8).
