# Inference in the typical case - we simply merge all the data together and ignore gene replication (using a convolution of two genes throughout)

apath = normpath(joinpath(@__DIR__, "../../"))
include(apath*"load_merged_data.jl")

srcpath = normpath(joinpath(@__DIR__, "../../../../src/"))
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")

fitpath = normpath(datapath*"fits_merged/")
println("\nSaving fits in $fitpath\n")

data = counts_merged
nconv = 2

@time fits_Poisson = fit_dists(Poisson, data; nconv)
@save fitpath*"fits_Poisson.jld2" fits_Poisson

@time fits_ZIPoisson = fit_dists(ZI{Poisson}, data; nconv, maxtime=10.0)
@save fitpath*"fits_ZIPoisson.jld2" fits_ZIPoisson

@time fits_NB = fit_dists(NegativeBinomial, data; nconv, maxtime=10.0)
@save fitpath*"fits_NB.jld2" fits_NB

@time fits_ZINB = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=10.0)
@save fitpath*"fits_ZINB.jld2" fits_ZINB

# BFGS() can take too long to completely converge in certain cases, usually when it reaches the 
# parameter bounds (bursty limit). Nevertheless, even then in generally reaches to the near-minimum (negligible difference in likelihood)
# and hence `error_check=false`
@time fits_BP = fit_dists(BetaPoisson, data; nconv, maxtime=30.0, error_check=false)
@save fitpath*"fits_BP.jld2" fits_BP

@time fits_ZIBP = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=30.0, error_check=false)
@save fitpath*"fits_ZIBP.jld2" fits_ZIBP