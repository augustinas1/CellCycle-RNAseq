dirpath = normpath(joinpath(@__DIR__, "../../../../."))
include(dirpath*"analysis/mESC/load_analysis.jl")
include(dirpath*"analysis/mESC/filter_prior.jl")

srcpath = normpath(srcpath*"../age-dependent/")
fitpath = datapath*"fits_age-dependent/"
include(srcpath*"dists.jl")
include(srcpath*"mle.jl")

# fix the replication point to be in the middle of the S phase
θᵣ = θ_G1_S + (θ_S_G2M - θ_G1_S)/2

fits_main = load(fitpath*"fits_main.jld2", "fits_main")
include(dirpath*"analysis/mESC/filter_post.jl")

# --- Compute burst parameter ratios ---

burst_freqs_G1 = get_burst_frequency_G1.(fits_main)
burst_freqs_G2M = get_burst_frequency_G2M.(fits_main)
burst_sizes_G1 = get_burst_size_G1.(fits_main, Ref(thetaG1))
burst_sizes_G2M = get_burst_size_G2M.(fits_main, Ref(thetaG2M))

ratio_f = burst_freqs_G2M ./ burst_freqs_G1 
ratio_b = burst_sizes_G2M ./ burst_sizes_G1

# --- Binomial capture ---

# Assuming that sampling efficiency is Beta distributed with CV=0.15
p_avg = 0.3
v = 100.0
α = p_avg*v
β = (1-p_avg)*v
d_p = Beta(α, β)
@show std(d_p) / mean(d_p)
rng = MersenneTwister(1234)

# p_avg, θs and n_θs are global variables

function rescale_transcription(m::MainModel)
    ps = collect(Distributions.params(m))
    ps[3:4] ./= p_avg
    MainModel(ps...)
end


function generate_downsampled_counts(m::MainModel, theta::AbstractArray, ps::AbstractArray, 
                                     T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, rng::AbstractRNG)
    
    # Given the count distribution m, the θ values associated with each cell, and the capture efficiency distribution d_p,
    # sample the mRNA counts per cell

    m = rescale_transcription(m)

    # generate the transcript counts
    counts = Int[]
    for i in eachindex(n_θs)
        θ = θs[i]
        n = n_θs[i]
        # transform into θ-independent model
        _d = NegativeBinomial(m, θ, T, d, θᵣ, θ_S_i, θ_S_f)
        _counts = rand(rng, _d, n)
        append!(counts, _counts)
    end
    
    # downsample counts
    binomial_ds = Binomial.(counts, ps)
    downsampled_counts = rand.(binomial_ds)
    
    downsampled_counts
    
end
    

function get_downsampled_ratios(m::MainModel, theta::AbstractArray, ps::AbstractArray, 
                                T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, rng::AbstractRNG; kwargs...)

    xcounts = generate_downsampled_counts(m, theta, ps, T, d, θᵣ, θ_S_i, θ_S_f, rng)
    xfit = fit_mle(MainModel, xcounts, theta, T, d, θᵣ, θ_S_i, θ_S_f; kwargs...)

    f1 = get_burst_frequency_G1(xfit)
    f2 = get_burst_frequency_G2M(xfit)
    b1 = get_burst_size_G1(xfit, theta[theta .< θ_S_i])
    b2 = get_burst_size_G2M(xfit, theta[theta .> θ_S_f])
    
    ratio_f = f2 / f1
    ratio_b = b2 / b1
    
    ratio_f, ratio_b
    
end

# Consider genes that are fairly standard (fall between 5 to 95 percentiles both in burst size and frequency ratios)
q005_rf = quantile(ratio_f, 0.05)
q095_rf = quantile(ratio_f, 0.95)
q005_rb = quantile(ratio_b, 0.05)
q095_rb = quantile(ratio_b, 0.95)

inds_rf = findall(q005_rf .<= ratio_f .<= q095_rf) 
inds_rb = findall(q005_rb .<= ratio_b .<= q095_rb)
inds = intersect(inds_rf, inds_rb)

# number of example genes considered 
n_examples = 200
ps = rand(rng, d_p, length(theta))
inds = shuffle(rng, inds)[1:n_examples]

# number of resamples per example
n_samples = 100
# separate rngs for each sample to for thread safety and easy reproducibility
rngs = [MersenneTwister(i) for i in 1:n_samples]

vec_ratios_f = []
vec_ratios_b = []

for ind in inds
    println("Gene $ind")
    m = fits_main[ind]
    dr = decay_rates[ind]
    ratios_f = zeros(n_samples)
    ratios_b = zeros(n_samples)
    
    @time Threads.@threads for i in 1:n_samples
        try
            rng = MersenneTwister(i)
            ratios_f[i], ratios_b[i] = get_downsampled_ratios(m, theta, ps, T_cycle, dr, θᵣ, θ_G1_S, θ_S_G2M,
                                                              rngs[i], maxtime=60.0, n_repeats=10, error_check=false)
        catch e
            println(e)
        end
    end
    
    push!(vec_ratios_f, ratios_f)
    push!(vec_ratios_b, ratios_b)
end

@save datapath*"vec_downsampling_ratios.jld2" inds vec_ratios_f vec_ratios_b;