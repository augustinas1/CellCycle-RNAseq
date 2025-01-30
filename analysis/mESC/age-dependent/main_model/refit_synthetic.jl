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

using Catalyst, JumpProcesses, DiffEqBase, DiffEqCallbacks

@register_symbolic Distributions.Geometric(b)
@variables t
@parameters f ρ β d k T

th = t/T - k
r = rand(Distributions.Geometric(1/(1 + ρ * exp(β * th))))

rn = @reaction_network rn_bursty begin
    @parameters f₁ f₂ ρ₁ ρ₂ β₁ β₂ β₃ β₄
    f, G → G + $r*M    # G -> G + rM, r ~ Geometric()
    d, M → 0           # M -> ∅
end

# convert the reaction network into a system of jump processes
jsys = convert(JumpSystem, rn; combinatoric_ratelaws=false)
jsys = complete(jsys)
u₀ = symmap_to_varmap(jsys, [:G => 2, :M => 0])

n_cycles = 10  # heuristically plenty to reach the steady state
tspan = (0., T_cycle*n_cycles) # time interval to solve over
ts = vcat((θs .* T_cycle .+ T_cycle*i for i in 0:1:n_cycles-1)... )
tθ = θs .* T_cycle .+ T_cycle*(n_cycles-1)
# otherwise takes pre-division state (due to extra timestep state saves added by callbacks)
ts[collect(1:length(tθ):length(ts))] .+= 0.000001
tθ[1] += 0.000001

n_examples = 200
n_samples = 100
rng = MersenneTwister(1234)
inds = shuffle(rng, 1:length(fits_main))[1:n_examples]

function affect_θ_S_i!(integrator)
    integrator.ps[:β] = integrator.ps[:β₂]
    reset_aggregated_jumps!(integrator)
    nothing
end

condition_θ_S_i = collect(0:1:n_cycles-1) .* T_cycle .+ (θ_G1_S * T_cycle)
cb_θ_S_i = PresetTimeCallback(condition_θ_S_i, affect_θ_S_i!)

function affect_replication!(integrator)
    integrator[:G] = 4
    integrator.ps[:f] = integrator.ps[:f₂]
    integrator.ps[:ρ] = integrator.ps[:ρ₂]
    integrator.ps[:β] = integrator.ps[:β₃]
    reset_aggregated_jumps!(integrator)
    nothing
end

condition_replication = collect(0:1:n_cycles-1) .* T_cycle .+ (θᵣ * T_cycle)
cb_replication = PresetTimeCallback(condition_replication, affect_replication!)

function affect_θ_S_f!(integrator)
    integrator.ps[:β] = integrator.ps[:β₄]
    reset_aggregated_jumps!(integrator)
    nothing
end

condition_θ_S_f = collect(0:1:n_cycles-1) .* T_cycle .+ (θ_S_G2M * T_cycle)
cb_θ_S_f = PresetTimeCallback(condition_θ_S_f, affect_θ_S_f!)

function affect_division!(integrator)
    integrator[:G] = 2
    integrator.ps[:k] += 1
    integrator[:M] = rand(Binomial(integrator[:M], 0.5))
    integrator.ps[:f] = integrator.ps[:f₁]
    integrator.ps[:ρ] = integrator.ps[:ρ₁]
    integrator.ps[:β] = integrator.ps[:β₁]
    reset_aggregated_jumps!(integrator)
    nothing
end

condition_division = collect(1:1:n_cycles-1) .* T_cycle
cb_division = PresetTimeCallback(condition_division, affect_division!)
cbs = CallbackSet(cb_θ_S_i, cb_replication, cb_θ_S_f, cb_division)

function construct_dataset(sol_SSA)
    xG1_SSA = Vector{Int}(undef, nG1)
    xS_SSA = Vector{Int}(undef, nS)
    xG2M_SSA = Vector{Int}(undef, nG2M)
    
    th_count = 0
    th_ind = 1

    for i in 1:nG1
        th_count += 1
        xG1_SSA[i] = sol_SSA[i](tθ[th_ind])[2]
        if th_count == n_θs[th_ind]
            th_count = 0
            th_ind += 1
        end
    end
    
    for i in 1:nS
        th_count += 1
        xS_SSA[i] = sol_SSA[nG1+i](tθ[th_ind])[2]
        if th_count == n_θs[th_ind]
            th_count = 0
            th_ind += 1
        end
    end
    
    for i in 1:nG2M
        th_count += 1
        xG2M_SSA[i] = sol_SSA[nG1+nS+i](tθ[th_ind])[2]
        if th_count == n_θs[th_ind]
            th_count = 0
            th_ind += 1
        end
    end

    xG1_SSA, xS_SSA, xG2M_SSA
end

function generate_counts(m::MainModel, dm::Real, n_samples::Int)
    
    f₁, f₂, ρ₁, ρ₂, β₁, β₂, β₃, β₄ = params(m)
    p = symmap_to_varmap(jsys, [:f => f₁, :ρ => ρ₁, :β => β₁, 
                                :k => 0.0, :T => T_cycle, :d => dm,
                                :f₁ => f₁, :f₂ => f₂,
                                :ρ₁ => ρ₁, :ρ₂ => ρ₂,
                                :β₁ => β₁, :β₂ => β₂, :β₃ => β₃, :β₄ => β₄]) 

    dprob = DiscreteProblem(jsys, u₀, tspan, p)
    jprob = JumpProblem(jsys, dprob, Direct(), save_positions=(false, false))
    ensembleprob = EnsembleProblem(jprob, safetycopy=true)    
    
    data_SSA = Vector{Vector{Int64}}(undef, n_samples)

    for i in 1:n_samples
        GC.gc()
        sol_SSA = solve(ensembleprob, SSAStepper(), saveat=ts, trajectories=n_cells, callback=cbs)
        xG1_SSA, xS_SSA, xG2M_SSA = construct_dataset(sol_SSA)
        data_SSA[i] = vcat(xG1_SSA, xS_SSA, xG2M_SSA)
    end

    data_SSA
end

println("--------------------------------------------")
println("Generating synthetic data")

synthetic_data = Vector{Vector{Vector{Int64}}}(undef, n_examples)
for i in 1:n_examples
    ind = inds[i]
    println("Iter $i: Gene $ind")
    m = fits_main[ind]
    dm = decay_rates[ind]
    @time data_SSA = generate_counts(m, dm, n_samples)
    synthetic_data[i] = data_SSA
end

@save datapath*"synthetic_data_gene_inds.jld2" inds
@save datapath*"synthetic_data.jld2" synthetic_data

println("--------------------------------------------")
println("Fitting models to synthetic data")

all_fits = Vector{Vector{MainModel{Float64}}}(undef, n_examples)
Threads.@threads for i in 1:n_examples
    ind = inds[i]
    println("Iter $i: Gene $ind")
    dm = decay_rates[ind]
    fits = Vector{MainModel{Float64}}(undef, n_samples)
    @time for j in 1:n_samples
        xcounts = synthetic_data[i][j]
        fits[j] = fit_mle(MainModel, xcounts, theta, T_cycle, dm, θᵣ, θ_G1_S, θ_S_G2M; 
                          maxtime=60.0, n_repeats=10, error_check=false)
    end
    all_fits[i] = fits

end

@save datapath*"synthetic_data_fits.jld2" all_fits
