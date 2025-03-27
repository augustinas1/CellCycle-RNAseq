[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15096048.svg)](https://doi.org/10.5281/zenodo.15096048)

# CellCycle-RNAseq

This repository contains the code for the paper: &nbsp; A. Sukys and R. Grima, "Cell-cycle dependence of bursty gene expression: insights from fitting mechanistic models to single-cell RNA-seq data" (2025).

The code is used to perform mechanistic model-based inference on scRNA-seq data for a population of mouse embryonic stem cells (mESCs), where the cell-cycle phase (G1, S or G2/M) and cell age $\theta$ (position along the cell cycle) are known for each cell. The processed dataset used throughout is uploaded on [Zenodo](https://doi.org/10.5281/zenodo.10467234), and is based on the original work by Riba et al. [[1]](#1).

### Structure

- `src` - main Julia code used to build quantitative models of gene expression, perform maximum likelihood estimation and model selection, and compute the confidence intervals using profile likelihood.
- `analysis` - scripts for the mESC dataset-specific analysis, considering cell age-dependent and cell age-independent mechanistic models using cell-age-resolved mRNA count data. 
- `notebooks` - Jupyter notebooks (written in Julia) used to explore the results and to generate the figures in the paper.
- `Mathematica` - Mathematica notebooks used to solve the piecewise ODEs for the means and variances of the age-dependent models.
- `data` - directory to save the generated files, such as model-specific fits. 

### References:

<a id="1">[1]</a> Riba, A., Oravecz, A., Durik, M. et al. Cell cycle gene regulation dynamics revealed by RNA velocity and deep-learning. Nat Comm 13, 2865 (2022). [https://doi.org/10.1038/s41467-022-30545-8](https://doi.org/10.1038/s41467-022-30545-8).
