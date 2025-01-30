dirpath = normpath(joinpath(@__DIR__, "../../../../"))
include(dirpath*"analysis/mESC/load_analysis.jl")
include(dirpath*"analysis/mESC/filter_prior.jl")

srcpath = normpath(srcpath*"../age-dependent/")
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")

fitpath = normpath(datapath*"fits_age-dependent/")
println("\nSaving fits in $fitpath\n")

# fix the replication point to be in the middle of the S phase
θᵣ = θ_G1_S + (θ_S_G2M - θ_G1_S)/2

# --- Inference ---

@time fits_mains = fit_dists(MainModel, counts_spliced, theta, T_cycle, decay_rates, θᵣ, θ_G1_S, θ_S_G2M; maxtime=60.0, n_repeats=10, error_check=false)
@save fitpath*"fits_main.jld2" fits_main