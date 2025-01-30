# BIC model selection validation test 
# Supplementary Figure S1A

dirpath = normpath(joinpath(@__DIR__, "../../../."))
using Pkg; Pkg.activate(dirpath*"/.")

apath = normpath(joinpath(@__DIR__, "../"))
include(apath*"load_data.jl")

srcpath = dirpath*"src/age-independent/"
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")
include(srcpath*"bic.jl")
fitpath = normpath(datapath*"fits_age-independent/")
println("\nLoading fits from $fitpath\n")

# Load G1 age-independent model fits
fits_Poisson_G1 = load(fitpath*"G1_fits_Poisson.jld2", "fits_Poisson_G1")
fits_ZIPoisson_G1 = load(fitpath*"G1_fits_ZIPoisson.jld2", "fits_ZIPoisson_G1")
fits_NB_G1 = load(fitpath*"G1_fits_NB.jld2", "fits_NB_G1")
fits_ZINB_G1 = load(fitpath*"G1_fits_ZINB.jld2", "fits_ZINB_G1")
fits_BP_G1 = load(fitpath*"G1_fits_BP.jld2", "fits_BP_G1")
fits_ZIBP_G1 = load(fitpath*"G1_fits_ZIBP.jld2", "fits_ZIBP_G1")

nconv = 2
@time BICs_Poisson_G1 = get_BICs(fits_Poisson_G1, nconv, xG1)
@time BICs_ZIPoisson_G1 = get_BICs(fits_ZIPoisson_G1, nconv, xG1)
@time BICs_NB_G1 = get_BICs(fits_NB_G1, nconv, xG1)
@time BICs_ZINB_G1 = get_BICs(fits_ZINB_G1, nconv, xG1)
@time BICs_BP_G1 = get_BICs(fits_BP_G1, nconv, xG1)
@time BICs_ZIBP_G1 = get_BICs(fits_ZIBP_G1, nconv, xG1)

indsarr_G1 = sort_BICs(BICs_Poisson_G1, BICs_ZIPoisson_G1, BICs_NB_G1, BICs_ZINB_G1, BICs_BP_G1, BICs_ZIBP_G1, BICtol=10)
inds_Poisson_G1, inds_ZIPoisson_G1, inds_NB_G1, inds_ZINB_G1, inds_BP_G1, inds_ZIBP_G1 = indsarr_G1


# Functions to sample the distributions (specific age-independent models)
function Distributions.rand(rng::AbstractRNG, d::MixtureModel, n::Int)
    # otherwise default rand() tends to crash
    return [rand(rng, d) for i in 1:n]
end

function Distributions.rand(rng::AbstractRNG, d::BetaPoisson)
    return rand(rng, Poisson(d.ρ * rand(rng, Beta(d.σ_on, d.σ_off))))
end

function Distributions.rand(rng::AbstractRNG, d::ZI)
    return rand(rng, MixtureModel([Dirac(0.0), d.dist], [d.p0, one(d.p0)-d.p0]))
end

function count_generator(d::DiscreteUnivariateDistribution, ndata::Int, nconv::Int, rng::AbstractRNG)

    # Set up a mixture model for sampling in case we have a zero-inflated distribution 
    _d = d isa ZI ? MixtureModel([Dirac(0), d.dist], [d.p0, one(d.p0)-d.p0]) : d
    
    # Count generator for each cell
    function generate_counts()
        counts = sum(rand(rng, _d, ndata) for i in 1:nconv)
        counts
    end

    generate_counts

end

# Overloading the function in dists.jl to remove multithreading 
# (more useful to put multithreading in refit_dists for our purposers)
function fit_dists(T::Type{<:DiscreteUnivariateDistribution}, xdata::AbstractArray, theta; kwargs...)
    
    fits = Array{Any}(undef, length(xdata))
    
    for i in eachindex(xdata)
        try
            fits[i] = fit_mle(T, xdata[i], theta; kwargs...)
        catch e
            (e isa InterruptException) && break
            println("Failed to converge for ind = $i")
            println(e)
            println("-------------------------------------------")
        end
    end

    fits
end

# Resample dataset `n_rerun` times using the given distribution, run inference and model selection
# and save the most often selected model for each distribution (sample)
function refit_dists(dists)
    res = Vector{Int}(undef, length(dists))
    Threads.@threads for i in eachindex(dists)
        d = dists[i]
        generate_counts = count_generator(d, ncells, nconv, rngs[i])
        data = [generate_counts() for i in 1:n_rerun]
        fits_Poisson = fit_dists(Poisson, data; nconv, maxtime=20.0)
        fits_ZIPoisson = fit_dists(ZI{Poisson}, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
        fits_NB = fit_dists(NegativeBinomial, data; nconv, maxtime=20.0, n_repeats=5, error_check=false)
        fits_ZINB = fit_dists(ZI{NegativeBinomial}, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
        fits_BP = fit_dists(BetaPoisson, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
        fits_ZIBP = fit_dists(ZI{BetaPoisson}, data; nconv, maxtime=60.0, n_repeats=5, error_check=false)
        BICs_Poisson = get_BICs(fits_Poisson, nconv, data)
        BICs_ZIPoisson = get_BICs(fits_ZIPoisson, nconv, data)
        BICs_NB = get_BICs(fits_NB, nconv, data)
        BICs_ZINB = get_BICs(fits_ZINB, nconv, data)
        BICs_BP = get_BICs(fits_BP, nconv, data)
        BICs_ZIBP = get_BICs(fits_ZIBP, nconv, data)
        indsarr = sort_BICs(BICs_Poisson, BICs_ZIPoisson, BICs_NB, BICs_ZINB, BICs_BP, BICs_ZIBP, BICtol=10)
        res[i] = findmax(length.(indsarr))[2]
    end
    res
end

nconv = 2
ncells = length(thetaG1)
rng = MersenneTwister(1234) 
n_samples = 500  # number of best fit distributions (samples) to consider
n_rerun   = 10   # number of times to resample the dataset for each distribution

# Select a random subset of `n_samples` genes ouf of all genes that were 
# found to be optimally fit by the corresponding model
dists_Poisson = shuffle(rng, fits_Poisson_G1[inds_Poisson_G1])[1:n_samples]
dists_ZIPoisson = shuffle(rng, fits_ZIPoisson_G1[inds_ZIPoisson_G1])[1:n_samples]
dists_NB = shuffle(rng, fits_NB_G1[inds_NB_G1])[1:n_samples]
# separate rngs for each sample for thread safety and easy reproducibility
rngs = [MersenneTwister(i) for i in 1:n_samples]

println("Poisson")
GC.gc()
@time res_Poisson = refit_dists(dists_Poisson)
@save fitpath*"BIC_res_Poisson.jld2" res_Poisson

println("ZIPoisson")
GC.gc()
@time res_ZIPoisson = refit_dists(dists_ZIPoisson)
@save fitpath*"BIC_res_ZIPoisson.jld2" res_ZIPoisson

println("NB")
GC.gc()
@time res_NB = refit_dists(dists_NB)
@save fitpath*"BIC_res_NB.jld2" res_NB