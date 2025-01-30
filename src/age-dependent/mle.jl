using ProgressMeter
using Optimization, OptimizationOptimJL, OptimizationMOI, Ipopt, NaNMath

function construct_loss(_m::Union{MainModel, AltModel1, AltModel2}, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, y::AbstractArray, theta::AbstractArray)

    thetas = sort(unique(theta))
    yw = fit_hist.(y[theta .== th] for th in thetas)
   
    function loss(x)
        ll = zero(eltype(x))
        m = reconstruct(_m, x)
        
        for i in eachindex(thetas)
            prob = logpdf.(Ref(m), thetas[i], Ref(T), Ref(d), Ref(θᵣ), Ref(θ_S_i), Ref(θ_S_f), yw[i][1])
            ll += sum(prob .* yw[i][2])
        end
        -ll
    end
    
    return loss

end


### Maximum likelihood estimation ###

# Function to generate an initial parameter guess as input to the optimisation routine
# context: sometimes the randomly sampled numbers can be out of bounds and lead to errors
function generate_u0(::Type{<:Union{MainModel, AltModel2}}, lb::AbstractArray, ub::AbstractArray; nmax::Int=10)
    i = 1
    for i in 1:nmax
        u0 = vcat(randn(4), rand(4))
        if all(lb .<= u0 .<= ub)
            return u0
        end
    end
    error("Generated $nmax u0 guesses that were out of bounds.")    
end

function generate_u0(::Type{AltModel1}, lb::AbstractArray, ub::AbstractArray; nmax::Int=10)
    i = 1
    for i in 1:nmax
        u0 = vcat(randn(6), rand(4))
        if all(lb .<= u0 .<= ub)
            return u0
        end
    end
    error("Generated $nmax u0 guesses that were out of bounds.")    
end


function Distributions.fit_mle(dtype::Type{<:Union{MainModel, AltModel1, AltModel2}}, y::AbstractArray{<:Real}, theta::AbstractArray{<:Real}, 
                               T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                               n_repeats::Int=5, error_threshold::Int=10,
                               global_solver=:adaptive_de_rand_1_bin_radiuslimited, 
                               local_solver=BFGS(),
                               autodiff=:forward, maxiters = 10^4, maxtime::Real=10.0, 
                               error_check::Bool=true, verbose::Bool=false, kwargs...)

    m = dtype()
    loss = construct_loss(m, T, d, θᵣ, θ_S_i, θ_S_f, y, theta)
    lb, ub = get_bounds(dtype)
    n_ps = numparams(dtype)
    trace_mode = verbose ? :verbose : :silent
    vec_res = Vector{Optim.MultivariateOptimizationResults}(undef, n_repeats)
    
    i = 1
    n_errors = 0
    while (i <= n_repeats)
        verbose && println("-------------------- Fit $i --------------------")
        u0 = generate_u0(dtype, lb, ub)
        try
            res = bboptimize(loss, u0; SearchRange = collect(zip(lb, ub)), Method = global_solver,
            NumDimensions=n_ps, MaxTime=maxtime, TraceMode = trace_mode) 
            u0 = best_candidate(res)
            res = Optim.optimize(loss, lb, ub, u0, Fminbox(local_solver), 
                                Optim.Options(time_limit=maxtime, allow_f_increases=true, iterations=maxiters); 
                                autodiff, kwargs...)
            vec_res[i] = res
            verbose && println(res)
            i += 1
        catch e
            # Sometimes a DomainError can occur (certain parameter combinations lead to negative NB parameters)
            # Usually just generating a new initial condition is enough but added an extra counter to kill the code
            # if it continues to be unstable
            if !isa(e, DomainError)
                throw(e)
            else
                n_errors += 1
                if n_errors > error_threshold
                    throw(e)
                end
            end
        end
    end

    # Choose the best fit out of all repeats of the inference procedure
    res = vec_res[argmin(Optim.minimum.(vec_res))]
    error_check && check_convergence(res)
    p = Optim.minimizer(res)
    if verbose 
        println("-----------------------------------------------------------")
        println("Domain errors recorded: $n_errors")
        println("Final fit:\n")    
        println(res)
    end

    reconstruct(m, p)

end

### Parameter estimation utilities ###

function fit_dists(dtype::Type{<:Union{MainModel, AltModel1, AltModel2}}, xdata::AbstractArray{<:AbstractArray{<:Real}}, theta::AbstractArray{<:Real}, 
                   T::Real, ds::AbstractArray{<:Real}, θᵣ::Real, θ_S_i::Real, θ_S_f::Real; 
                   kwargs...)
    
    fits = Array{Any}(undef, length(xdata))
    
    @showprogress Threads.@threads for i in eachindex(xdata)
        try
            fits[i] = fit_mle(dtype, xdata[i], theta, T, ds[i], θᵣ, θ_S_i, θ_S_f; kwargs...)
        catch e
            (e isa InterruptException) && break
            println("Failed to converge for ind = $i")
            println(e)
            println("-------------------------------------------")
        end
    end

    fits
end