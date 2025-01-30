# BIC model selection validation test 
# Supplementary Figure S1B

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

function Distributions.rand(rng::AbstractRNG, d::BetaPoisson)
    return rand(rng, Poisson(d.ρ * rand(rng, Beta(d.σ_on, d.σ_off))))
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


# Overloading the function in dists.jl to remove progress meter
function fit_dists(T::Type{<:DiscreteUnivariateDistribution}, xdata::AbstractArray; kwargs...)
    
    fits = Array{Any}(undef, length(xdata))
    
    Threads.@threads for i in eachindex(xdata)
        try
            fits[i] = fit_mle(T, xdata[i]; kwargs...)
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
# and save the times each age-independent model was selected as the optimal one
function refit_dists(dists)
    res = res = Vector{Vector{Int64}}(undef, length(dists))
    @showprogress for i in eachindex(dists)
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
        res[i] = length.(indsarr)
    end
    res
end

nconv = 2
ncells = length(thetaG1)
n_rerun = 100   # number of times to resample the dataset for each distribution

# Set up a telegraph (BetaPoisson) with varying σ_on rate with other parameters kept fixed
σ_off = 1.0
ρ = 50.0
n_points = 100 # number of σ_on values to consider in the given range
all_σ_on = exp10.(range(-2, 2, length=n_points))
dists_BP = [BetaPoisson(σ_on, σ_off, ρ) for σ_on in all_σ_on]
# separate rngs for each sample for reproducibility
rngs = [MersenneTwister(i) for i in 1:n_points]

@time res_BP = refit_dists(dists_BP)
@save fitpath*"BIC_res_BP.jld2" dists_BP res_BP;
