dirpath = normpath(joinpath(@__DIR__, "../../../."))
using Pkg; Pkg.activate(dirpath*"/.")

apath = normpath(joinpath(@__DIR__, "../"))
include(apath*"load_data.jl")

srcpath = dirpath*"src/age-independent/"
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
fitpath = normpath(datapath*"fits_age-independent/")
println("\nSaving fits in $fitpath\n")

# --- G1 inference ---

data = xG1
theta = thetaG1
nconv = 2

println("--- G1 ---")
println("Poisson")
@time fits_Poisson_G1 = fit_dists(Poisson, data; nconv, maxtime=20.0)
@save fitpath*"G1_fits_Poisson.jld2" fits_Poisson_G1

println("ZIPoisson")
@time fits_ZIPoisson_G1 = fit_dists(ZI{Poisson}, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G1_fits_ZIPoisson.jld2" fits_ZIPoisson_G1

println("NB")
@time fits_NB_G1 = fit_dists(NegativeBinomial, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G1_fits_NB.jld2" fits_NB_G1

println("ZINB")
@time fits_ZINB_G1 = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G1_fits_ZINB.jld2" fits_ZINB_G1

println("BP")
@time fits_BP_G1 = fit_dists(BetaPoisson, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
@save fitpath*"G1_fits_BP.jld2" fits_BP_G1

println("ZIBP")
@time fits_ZIBP_G1 = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
@save fitpath*"G1_fits_ZIBP.jld2" fits_ZIBP_G1

# --- G2/M inference --- 

data = xG2M
theta = thetaG2M
nconv = 4

println("--- G2/M ---")
println("Poisson")
@time fits_Poisson_G2M = fit_dists(Poisson, data; nconv, maxtime=20.0)
@save fitpath*"G2M_fits_Poisson.jld2" fits_Poisson_G2M

println("ZIPoisson")
@time fits_ZIPoisson_G2M = fit_dists(ZI{Poisson}, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G2M_fits_ZIPoisson.jld2" fits_ZIPoisson_G2M

println("NB")
@time fits_NB_G2M = fit_dists(NegativeBinomial, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G2M_fits_NB.jld2" fits_NB_G2M

println("ZINB")
@time fits_ZINB_G2M = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
@save fitpath*"G2M_fits_ZINB.jld2" fits_ZINB_G2M

println("BP")
@time fits_BP_G2M = fit_dists(BetaPoisson, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
@save fitpath*"G2M_fits_BP.jld2" fits_BP_G2M

println("ZIBP")
@time fits_ZIBP_G2M = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
@save fitpath*"G2M_fits_ZIBP.jld2" fits_ZIBP_G2M