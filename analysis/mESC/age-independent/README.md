# Age-independent model analysis

The scripts include:
- `inference.jl` - Perform inference for all age-independent models -- Poisson, zero-inflated Poisson (ZIPoisson), negative binomial (NB), zero-inflated negative binomial (ZINB), telegraph (BP) and zero-inflated telegraph (ZIBP) -- using maximum likelihood and cell-cycle-phase-specific (G1 or G2/M) mRNA count data.
- `model_selection.jl` - perform model selection based on the Bayesian Information Criterion (BIC) for each gene in the dataset choosing the optimal age-independent model, and discard genes that are not found to be bursty.
- `confidence_intervals.jl` - compute confidence intervals on the parameter estimates for each best-fit bursty model using profile likelihood.
- `refit_BIC_A.jl` - perform age-independent model selection procedure validation test (related to Supplementary Text S1; Figure S1A).
- `refit_BIC_B.jl` - perform age-independent model selection procedure validation (related to Supplementary Text S2; Figure S1B)