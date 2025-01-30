# Age-dependent model analysis

Considering three age-dependent models and the associated scripts in each subdirectory:

- `main_model` - Main 8-parameter model with age-dependent burst size and fixed replication timing (presented in the main text)
    - `inference.jl` - Inference for the filtered dataset using the model
    - `refit_downsampled.jl` - Inference test with binomially downsampled count data (related to Fig. 4D)
    - `refit_synthetic.jl` - Validation of age-dependent model inference using synthetic data (Supplementary Text S2)
- `alt_model_1_10_params` - Alternative 10-parameter model with age-dependent burst size and variable replication timing (Supplementary Text S3)
    - `inference.jl` - Inference for the filtered dataset using the model
- `alt_model_2_age-dep_bf` - Alternative 8-parameter model with age-dependent burst frequency and fixed replication timing (Supplementary Text S4)
    - `inference.jl` - Inference for the filtered dataset using the model