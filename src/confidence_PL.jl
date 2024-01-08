# ----- Confidence intervals using Profile Likelihood -----

get_CI_fns(::Type{<:NegativeBinomial}) = [log, logit], [exp, logistic]
get_CI_fns(::Type{<:BetaPoisson}) = fill(log, 3), fill(exp, 3)
get_CI_fns(::Type{<:NegativeBinomialTheta}) = fill(log, 3), fill(exp, 3)
get_CI_fns(::Type{<:BetaPoissonTheta}) = fill(log, 4), fill(exp, 4)

# Construct a distribution of the same type with new param values 
# (distributions are parametric types and the new values may be of different numerical type)
extract_dtype(d::ZI) = ZI{extract_dtype(d.dist)}
extract_dtype(::DT) where DT = DT.name.wrapper
construct_dist(::Type{<:ZI{DT}}, ps::AbstractArray{FT}) where {DT<:Distribution, FT<:Real} = ZI{FT, DT{FT}}(ps[1], DT{FT}(ps[2:end]...))
construct_dist(DT::Type{<:Distribution}, ps::AbstractArray{FT}) where FT<:Real = DT{FT}(ps...)

function get_CI_fns(::Type{<:ZI{DT}}) where DT
    lower_fns, upper_fns = get_CI_fns(DT)
    vcat(logit, lower_fns), vcat(logistic, upper_fns)
end

function check_CI_args(d, nconv)
    nconv > 0 || throw(DomainError(nconv, "nconv < 1"))
    isNB(d) || isZINB(d) || isBP(d) || isZIBP(d) || error("$d is not supported")
end

function find_root_PL_CI(xs, ls)
    
    # Some ideas using interpolation from ProfileLikelihood.jl
    if xs[1] > xs[end]
        xs = reverse(xs)
        ls = reverse(ls)
    end
    itp = interpolate(xs, ls, FritschCarlsonMonotonicInterpolation())
    itpf = (x, _) -> itp(x)
    interval = [xs[1], xs[end]]
    iprob = IntervalNonlinearProblem(itpf, interval)
    sol = solve(iprob, Falsi()).u
    sol
end

# Compute profile_likelihood using Optimisation.jl and allowing for inequality constraints  
function profile_likelihood_CI(lossf, loss_MLE, xs, dstep, tau, fn, lb, ub, cons; 
                               autodiff=Optimization.AutoForwardDiff(), error_check::Bool=true, time_limit::Real=20.0, kwargs...)
    
    # p is the parameter for which we compute the profile likelihood (PL parameter)
    # we are iterating in the transformed space as p0, p0+dstep, p0+2*dstep (where p0 is the original MLE estimate)
    # until we reach the point where MLE <= tau (otherwise the parameter is unidentifiable/hits the box boundaries)
    
    # if the initial parameter values are on the boundary of the box nudge them inside
    # (just to avoid Optim warnings)
    x2 = xs[2:end]
    lb2 = lb[2:end]; ub2 = ub[2:end]
    for i in eachindex(x2)
        if x2[i] == lb2[i]
            x2[i] *= 1.01
        elseif x2[i] == ub2[i]
            x2[i] *= 0.99
        end
    end
    
    # check whether the PL parameter goes over the box constraints (is practically unidentifiable)
    # doing this check on the physical parameter value (not the transformations used in optimisation)
    fun_lb = fn(lb[1]); fun_ub = fn(ub[1])
    function in_bounds(x)
        if dstep < 0
            x > lb[1] && !isapprox(fn(x), fun_lb, atol=1e-10)
        else
            x < ub[1]  && !isapprox(fn(x), fun_ub, atol=1e-10)
        end
    end
    
    p = xs[1]
    p_vals = [p]
    loss_vals = [loss_MLE]
    pCI = dstep < 0 ? lb[1] : ub[1]
    optf = OptimizationFunction(lossf, autodiff, cons=cons)
    oprob = Optimization.OptimizationProblem(optf, x2, p, lcons=lb2, ucons=ub2)
	solver = Ipopt.Optimizer()
    
    p += dstep
    _u0 = x2
    while in_bounds(p)
        oprob = remake(oprob, p=p, u0=_u0)
        res = solve(oprob, solver; print_level=0, max_cpu_time=time_limit)
        
        error_check && res.retcode == ReturnCode.MaxTime && 
            error("Optimiser took too long:\n\n$(res.original.solve_time) s")
        
        ploss = res.objective
        _u0 = res.u
        push!(p_vals, p)
        push!(loss_vals, ploss)
        
        if ploss - loss_MLE >= tau
            # reduce step size if it was too high (hard to fit a spline on very few points)
            if length(p_vals) < 5
                p = xs[1]; p_vals=[p]; loss_vals=[loss_MLE]; _u0 = x2
                dstep /= 5
            else
                pCI = find_root_PL_CI(p_vals, loss_vals .- loss_MLE .- tau)
                break
            end
        end
        
        p += dstep
        
    end    
    
    pCI
end


function init_PL_CI(d, y, theta, nconv)
    # initialise  optimisation utilities for the given distribution 
    # return the loss function, initial parameters in the transformed (optimisation) space,
    # the corresponding lower/upper bounds,
    # the parameter-specific functions to revert back to the original space,
    # and the lower/upper param bounds in the transformed space

    dtype = extract_dtype(d)
    lb, ub = get_bounds(dtype)
    invfns, fns = get_CI_fns(dtype)
    _nconv = isNB(d) ? 1 : nconv
    loss = construct_loss(d, _nconv, y, theta)

    ps = collect(Distributions.params(d))
    fun_lb = [fns[i](lb[i]) for i in eachindex(lb)]
    fun_ub = [fns[i](ub[i]) for i in eachindex(ub)] 
    xs = [invfns[i](ps[i]) for i in eachindex(ps)]
    inds_lb = findall(ps .== fun_lb)
    xs[inds_lb] .= lb[inds_lb]
    inds_ub = findall(ps .== fun_ub)
    xs[inds_ub] .= ub[inds_ub]
    xs[1] = isNB(d) ? xs[1]+log(nconv) : xs[1]
    
    loss, xs, lb, ub, fns, inds_lb, inds_ub, dtype
end


function get_confidence_intervals_PL(d::DiscreteUnivariateDistribution, y::AbstractArray, theta; 
                                     resolution::Int=100, alpha::Float64=0.05, nconv::Int=1, kwargs...)
    

    check_CI_args(d, nconv)
    loss, xs, lb, ub, fns, inds_lb, inds_ub, dtype = init_PL_CI(d, y, theta, nconv)
    reconstruct(p, i) = (isNB(d) && i == 1) ? p/nconv : p
    
    n = length(xs)
    loss_MLE = loss(xs)
    tau = cquantile(Chisq(1), alpha)/2

    ps_lb = similar(xs); ps_ub = similar(xs)
    cons(res, x, p) = (res .= x)
    
    for i in 1:n
        dstep = (ub[i] - lb[i])/resolution
        fn = fns[i]
        inds = circshift(1:n, n-i+1)
        srt_inds = sortperm(inds)
        lossf(x) = loss(x[srt_inds])
        lossf(x, p) = lossf(vcat(p, x))
        runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[inds], _dstep, tau, fn, lb[inds], ub[inds], cons; kwargs...)
        
        p_lower = i in inds_lb ? xs[i] : runf(-dstep)
        p_upper = i in inds_ub ? xs[i] : runf(dstep)
        
        ps_lb[i] = reconstruct(fn(p_lower), i)
        ps_ub[i] = reconstruct(fn(p_upper), i)
    end

    DistCI(dtype, ps_lb, ps_ub)
end


# --- Compare two MLE estimates (compute ratio confidence intervals) using Profile Likelihood

function init_ratio_CI_PL(d1, y1, theta1, nconv1, 
                          d2, y2, theta2, nconv2,
                          alpha)

    check_CI_args(d1, nconv1)
    check_CI_args(d2, nconv2)
    
    # Profile likelihood value: PLₘₗₑ-PL(v) <= tau
    tau = cquantile(Chisq(1), alpha)/2
    loss1, xs1, lb1, ub1, _, _, _, _ = init_PL_CI(d1, y1, theta1, nconv1)
    loss2, xs2, lb2, ub2, _, _, _, _ = init_PL_CI(d2, y2, theta2, nconv2)
    n1 = length(xs1); n2 = length(xs2)
    
    # Construct the optimisation such that the burst frequency ratio Δ is the first parameter
    xs = vcat(xs2, xs1)
    lb = vcat(lb2, lb1)
    ub = vcat(ub2, ub1)

    loss_joint(x2, x1) = loss2(x2) + loss1(x1)
    loss_joint(x) = loss_joint(x[1:n2], x[n2+1:end])
    
    loss_joint, n1, n2, xs, lb, ub, tau

end

# --- Burst frequency ratio between two fits --- #

bf_ind(d::ZI) = 1 + bf_ind(d.dist)
bf_ind(::Union{NegativeBinomial, NegativeBinomialTheta}) = 1
bf_ind(::Union{BetaPoisson, BetaPoissonTheta}) = 1

function get_burst_frequency_ratio_CI_PL(d1::DiscreteUnivariateDistribution, y1::AbstractArray, theta1, nconv1::Int,
                                         d2::DiscreteUnivariateDistribution, y2::AbstractArray, theta2, nconv2::Int; 
                                         resolution::Int=100, alpha::Float64=0.05, kwargs...)
    
    # Profile likelihood set up such that Δ = f₂ / f₁ ⟹ f₂ = Δ × f₁
    # We vary Δ and maximise the joint likelihood where f₂ is fixed

    loss_joint, n1, n2, xs, lb, ub, tau = init_ratio_CI_PL(d1, y1, theta1, nconv1, 
                                                           d2, y2, theta2, nconv2, alpha)
    loss_MLE = loss_joint(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = bf_ind(d1)
    i2 = bf_ind(d2)
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution 
    
    bf1 = get_burst_frequency(d1)
    bf2 = get_burst_frequency(d2)
    Δ = bf2 / bf1
    xs[i2] = log(Δ)
    fn = exp
    
    # compute bf₂ = Δ × bf₁ (in log space) including the corresponding convolutions for NBs
    # NOTE: this assumes that both burst frequencies are optimised in log space
    # (might need to be changed for different parameterisations)
    log_bf2(x) = x[i2] + x[n2+i1] - isNB(d1)*log(nconv1) + isNB(d2)*log(nconv2)
    
    # shift the parameters so that Δ is the first argument to the loss function
    n = n1+n2
    _inds = circshift(1:n, n-i2+1)
    srt_inds = sortperm(_inds)
    lossf(x) = loss_joint(vcat(log_bf2(x[srt_inds]), x[2:end])[srt_inds])
    lossf(x, p) = lossf(vcat(p, x))
    cons(res, x, p) = (res .= x)

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, lb[_inds], ub[_inds], cons; kwargs...)

    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub
    
end


# --- Burst size ratio between two fits ---

# return the index of the parameter which is fixed in the PL computation for burst size ratio 
# for NB this is simply burst size b, whereas for BP we take the transcription rate ρ
b_ind(d::ZI) = 1 + b_ind(d.dist)
b_ind(::NegativeBinomial) = 2
b_ind(::NegativeBinomialTheta) = 2
b_ind(::Union{BetaPoisson, BetaPoissonTheta}) = 3

function get_log_weighted_sum_ftheta(d, theta)
    
    θs, fθ = get_phase_dist(theta)
    β_ind = numparams(d) # β is always the last parameter for the defined dists
    
    function log_weighted_sum_ftheta(x)
        # NOTE: this assumes that 2^(βθ) is optimised in log space
        β = exp(x[β_ind]) 
        s = sum( 2^(β * θs[i]) * fθ[i] for i in eachindex(θs))
        log(s)
    end

    log_weighted_sum_ftheta
    
end

get_log_weighted_sum_ftheta(d, theta::Nothing) = x -> 0

# computed in log-space except for NB and ZINB:
# for NB p = sigmoid(x) => b = 1/p - 1 = e⁻ˣ => log(b) = -x
get_log_b(::NegativeBinomial) = x -> -x[2]
get_log_b(::ZI{<:NegativeBinomial}) = x -> -x[3]
get_log_b(::ZI{<:Real, <:NegativeBinomial}) = x -> -x[3]
get_log_b(::NegativeBinomialTheta) = x -> x[2]
get_log_b(::ZI{<:NegativeBinomialTheta}) = x -> x[3]
get_log_b(::ZI{<:Real, <:NegativeBinomialTheta}) = x -> x[3]
get_log_b(::Union{BetaPoisson, BetaPoissonTheta}) = x -> x[3] - x[2]
get_log_b(::ZI{<:Union{BetaPoisson, BetaPoissonTheta}}) = x -> x[4] - x[3]
get_log_b(::ZI{<:Real, Union{BetaPoisson, BetaPoissonTheta}}) = x -> x[4] - x[3]
is_p(d::Distribution) = 1
is_p(d::NegativeBinomial) = -1
is_p(d::ZI) = is_p(d.dist)

function get_burst_size_ratio_CI_PL(d1::DiscreteUnivariateDistribution, y1::AbstractArray, theta1, nconv1::Int,
                                    d2::DiscreteUnivariateDistribution, y2::AbstractArray, theta2, nconv2::Int; 
                                    resolution::Int=100, alpha::Float64=0.05, kwargs...)
    
    # Note that burst size is bs = ρ\σ_b (age-independent case)
    # OR, in the age-dependent case, bs = sum of (ρ/σ_b * 2^(βθ)) * f(θ)) over θ from θ₁ to θ₂, 
    #where θ₁ and θ₂ indicate the start and end θ of the specific phase (G1 or G2\M) 
    # and f(θ) is the phase distribution 

    # Profile likelihood set up such that Δ = bs₂ / bs₁ ⟹ bs₂ = (Δ × bs₁) / (sum of (ρ₂/σ_off₂ * 2^(β₂θ)) * f(θ))) / 
    # We vary Δ and maximise the joint likelihood where bs₂ is fixed
    # Note that the burst size is computed as bs = ρ/σ_b \sum_{θ}{2^(βθ))*f(θ)}

    loss_joint, n1, n2, xs, lb, ub, tau = init_ratio_CI_PL(d1, y1, theta1, nconv1, d2, y2, theta2, nconv2, alpha)
    loss_MLE = loss_joint(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = b_ind(d1)
    i2 = b_ind(d2)
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution 
    
    bs1 = get_burst_size(d1, theta1)
    bs2 = get_burst_size(d2, theta2)
    Δ = bs2 / bs1
    xs[i2] = log(Δ)
    fn = exp
    
    log_wsum_fth1 = get_log_weighted_sum_ftheta(d1, theta1)
    log_wsum_fth2 = get_log_weighted_sum_ftheta(d2, theta2)
    log_b1 = get_log_b(d1)
    
    function log_bs1(x)
        x2 = @view x[n2+1:end]
        log_b1(x2) + log_wsum_fth1(x2)
    end

    log_σ_off2(x) = isBP(d2) ? x[2] : isZIBP(d2) ? x[3] : 0
    
    # compute b₂ = Δ × bs₁ /  (in log space) including the corresponding convolutions for NBs) 
    log_b2(x) = x[i2] + log_bs1(x) - log_wsum_fth2(x) + log_σ_off2(x)

    # shift the parameters so that Δ is the first argument to the loss function
    n = n1+n2
    _inds = circshift(1:n, n-i2+1)
    srt_inds = sortperm(_inds)
    lossf(x) = loss_joint(vcat(is_p(d2)*log_b2(x[srt_inds]), x[2:end])[srt_inds])
    lossf(x, p) = lossf(vcat(p, x))
    cons(res, x, p) = (res .= x)
    
    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, lb[_inds], ub[_inds], cons; kwargs...)

    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub
end
