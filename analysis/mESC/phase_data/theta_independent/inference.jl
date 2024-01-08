apath = normpath(joinpath(@__DIR__, "../.."))
include(apath*"load_data.jl")

srcpath = normpath(joinpath(@__DIR__, "../../../../src/"))
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
fitpath = normpath(datapath*"fits_phase/")
println("\nSaving fits in $fitpath\n")

# --- G1 inference ---

data = xG1
theta = thetaG1
nconv = 2

@time fits_Poisson = fit_dists(Poisson, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_Poisson.jld2" fits_Poisson

@time fits_ZIPoisson = fit_dists(ZI{Poisson}, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_ZIPoisson.jld2" fits_ZIPoisson

@time fits_NB = fit_dists(NegativeBinomial, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_NB.jld2" fits_NB

@time fits_ZINB = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_ZINB.jld2" fits_ZINB

@time fits_BP = fit_dists(BetaPoisson, data; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G1_fits_BP.jld2" fits_BP

@time fits_ZIBP = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G1_fits_ZIBP.jld2" fits_ZIBP

# --- G2/M inference --- 

data = xG2M
theta = thetaG2M
nconv = 4

@time fits_Poisson = fit_dists(Poisson, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_Poisson.jld2" fits_Poisson

@time fits_ZIPoisson = fit_dists(ZI{Poisson}, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_ZIPoisson.jld2" fits_ZIPoisson

@time fits_NB = fit_dists(NegativeBinomial, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_NB.jld2" fits_NB

@time fits_ZINB = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_ZINB.jld2" fits_ZINB

@time fits_BP = fit_dists(BetaPoisson, data; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G2M_fits_BP.jld2" fits_BP

@time fits_ZIBP = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G2M_fits_ZIBP.jld2" fits_ZIBP