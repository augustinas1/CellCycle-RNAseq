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

@time fits_PoissonTheta = fit_dists(PoissonTheta, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_PoissonTheta.jld2" fits_PoissonTheta

@time fits_ZIPoissonTheta = fit_dists(ZI{PoissonTheta}, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_ZIPoissonTheta.jld2" fits_ZIPoissonTheta

@time fits_NBTheta = fit_dists(NegativeBinomialTheta, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_NBTheta.jld2" fits_NBTheta

@time fits_ZINBTheta = fit_dists(ZI{NegativeBinomialTheta}, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G1_fits_ZINBTheta.jld2" fits_ZINBTheta

@time fits_BPTheta = fit_dists(BetaPoissonTheta, data, theta; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G1_fits_BPTheta.jld2" fits_BPTheta

@time fits_ZIBPTheta = fit_dists(ZI{BetaPoissonTheta}, data, theta; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G1_fits_ZIBPTheta.jld2" fits_ZIBPTheta

# --- G2/M inference --- 

data = xG2M
theta = thetaG2M
nconv = 4

@time fits_PoissonTheta = fit_dists(PoissonTheta, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_PoissonTheta.jld2" fits_PoissonTheta

@time fits_ZIPoissonTheta = fit_dists(ZI{PoissonTheta}, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_ZIPoissonTheta.jld2" fits_ZIPoissonTheta

@time fits_NBTheta = fit_dists(NegativeBinomialTheta, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_NBTheta.jld2" fits_NBTheta

@time fits_ZINBTheta = fit_dists(ZI{NegativeBinomialTheta}, data, theta; nconv, maxtime=20.0, error_check=false)
@save fitpath*"G2M_fits_ZINBTheta.jld2" fits_ZINBTheta

@time fits_BPTheta = fit_dists(BetaPoissonTheta, data, theta; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G2M_fits_BPTheta.jld2" fits_BPTheta

@time fits_ZIBPTheta = fit_dists(ZI{BetaPoissonTheta}, data, theta; nconv, maxtime=60.0, error_check=false)
@save fitpath*"G2M_fits_ZIBPTheta.jld2" fits_ZIBPTheta