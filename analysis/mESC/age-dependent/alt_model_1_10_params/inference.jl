dirpath = normpath(joinpath(@__DIR__, "../../../../"))
include(dirpath*"analysis/mESC/load_analysis.jl")
include(dirpath*"analysis/mESC/filter_prior.jl")

srcpath = normpath(srcpath*"../age-dependent/")
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")

fitpath = normpath(datapath*"fits_age-dependent/")
println("\nSaving fits in $fitpath\n")

# fix the transition point from early to late S phase to occur when the 
# minimum total mRNA count per cell in S is observed at cell age θₘ
total_xS = vec(sum(hcat(xS...), dims=2))
counts_S = [total_xS[_inds] for _inds in inds_θs_S]
yS = mean.(counts_S)
min_ind = findmin(yS)[2]
θₘ = θs_S[min_ind]

# --- Inference ---

@time fits_alt1 = fit_dists(AltModel1, counts_spliced, theta, T_cycle, decay_rates, θₘ, θ_G1_S, θ_S_G2M; maxtime=60.0, n_repeats=10, error_check=false)
@save fitpath*"fits_alt_1_10_params.jld2" fits_alt1