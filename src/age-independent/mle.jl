using Optim, BlackBoxOptim
using ProgressMeter

include("convolutions.jl")

### Utilities for histogram fitting 

function fit_hist(arr::AbstractArray)
    # fit histogram to a given array of samples
    # return the value range and the associated weights
    nmax = maximum(arr)
    # vec() in case we have N × 1 matrix
    ws = fit(Histogram, arr, 0:nmax+1, closed=:left)
    ws = FrequencyWeights(ws.weights)
    0:nmax, ws
end
    
function fit_hists(mat::Matrix)
    # fit histogram to each row of the given data matrix (nsamples × nvars)
    xdata = []
    for row in eachcol(mat)
        push!(xdata, fit_hist(row))
    end
    xdata
end

## Error handling

function check_convergence(res::Optim.MultivariateOptimizationResults)
    # can ignore the failure to converge if the objective increased between iterations
    # because usually this still ends up in a good local minimum
    # cannot ignore a long runtime however
    if !Optim.converged(res)
        if Optim.time_run(res) > Optim.time_limit(res)
            error("Optimiser took too long:\n\n$(res)")
        elseif isa(res.ls_success, Bool) && !res.ls_success
            error("Line search failed:\n\n$(res)")
        elseif !Optim.f_increased(res) && Optim.iteration_limit_reached(res)
            error("Not enough iterations to converge?:\n\n$(res)")
        end
    end
end

## Loss function

function construct_loss(_d::DiscreteUnivariateDistribution, nconv::Int, y::AbstractArray)

    yy, ww = fit_hist(y)
    convf(x) = resolve_logconv(x, nconv)

    function loss(x)
        l = zero(eltype(x))
        d = reconstruct(_d, x)
        ps = Vector{eltype(x)}(undef, length(yy))
        for i in eachindex(ps)
            ps[i] = logpdf(d, yy[i])
        end
        ps = convf(ps)
        for i in eachindex(ps)
            l += ps[i] * ww[i]
        end
        -l
    end
    
    return loss
    
end

## Maximum likelihood estimation

Distributions.fit_mle(T::Type{<:DiscreteUnivariateDistribution}, y::AbstractArray{<:Real}; kwargs...) = fit_mle(T, y; kwargs...)

function Distributions.fit_mle(::Type{Poisson}, y::AbstractArray{<:Real}; 
                               nconv::Int=1, kwargs...)
    yy, ww = fit_hist(y)
    λ = mean(yy, ww)
    Poisson(λ/nconv)
end


function Distributions.fit_mle(T::Type{<:DiscreteUnivariateDistribution}, y::AbstractArray{<:Real}; 
                               nconv::Int=1,
                               n_repeats::Int=5, 
                               global_solver=:adaptive_de_rand_1_bin_radiuslimited, 
                               local_solver=BFGS(),
                               autodiff=:forward, maxiters = 10^4, maxtime::Real=10.0, 
                               error_check::Bool=true, verbose::Bool=false, kwargs...)

    d = T()
    _nconv = isPoisson(d) || isNB(d) ? 1 : nconv
    _reconstruct(p) = isPoisson(d) || isNB(d) ? reconstruct(d, p, nconv) : reconstruct(d, p) 
    loss = construct_loss(d, _nconv, y)
    lb, ub = get_bounds(T)
    n_ps = numparams(T)
    trace_mode = verbose ? :verbose : :silent

    vec_res = Vector{Optim.MultivariateOptimizationResults}(undef, n_repeats)
    for i in 1:n_repeats
        verbose && println("-------------------- Fit $i --------------------")
        u0 = randn(n_ps)
        res = bboptimize(loss, u0; SearchRange = collect(zip(lb, ub)), Method = global_solver,
                         NumDimensions=n_ps, MaxTime=maxtime, TraceMode = trace_mode) 
        u0 = best_candidate(res)
        res = Optim.optimize(loss, lb, ub, u0, Fminbox(local_solver), 
                            Optim.Options(time_limit=maxtime, allow_f_increases=true, iterations=maxiters); 
                            autodiff, kwargs...)
        vec_res[i] = res
        verbose && println(res)
    end
    
    # Choose the best fit out of all repeats of the inference procedure
    res = vec_res[argmin(Optim.minimum.(vec_res))]
    error_check && check_convergence(res)      
    p = Optim.minimizer(res)
    verbose && println(res)
    _reconstruct(p)

end

### Parameter estimation utilities ###

function fit_dists(T::Type{<:DiscreteUnivariateDistribution}, xdata::AbstractArray; kwargs...)
    
    fits = Array{Any}(undef, length(xdata))
    
    @showprogress Threads.@threads for i in eachindex(xdata)
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