# Compute confidence intervals for ratios of weighted averages, i.e. ratios of form 
# Δ = (w₁x₁ + w₂x₂ + ⋯ + wᵢxᵢ) / (wᵢ₊₁xᵢ₊₁ + ⋯ + wₙxₙ), where n is the total number 
# of dists considered, i is the number of dists in the numerator and wᵢ denotes the 
# weight of i-th distribution.

using Optimization, OptimizationOptimJL, OptimizationMOI, Ipopt, NaNMath

function get_ratio_confidence_intervals_weighted_avg(inds::AbstractArray{<:Int}, fget1, fget2, funCI; printerr::Bool=true, kwargs...)

    xlen = length(inds)
    CIs = Vector{Tuple}(undef, length(inds))
    @views Threads.@threads for i in 1:xlen
        try
            CIs[i] = funCI(fget1(inds[i])..., fget2(inds[i])...; kwargs...)
        catch e
            printerr && println("ind $i: $e")
        end
    end
    
    CIs
end

# --- Profile likelihood ---#

function init_ratio_CI_PL_weighted_avg(ds, ys, thetas, nconvs, alpha)
    
    check_CI_args.(ds, nconvs)
    
    # Profile likelihood value: PLₘₗₑ-PL(v) <= tau
    tau = cquantile(Chisq(1), alpha)/2
    
    res = init_PL_CI.(ds, ys, thetas, nconvs)
    losses = [r[1] for r in res]
    in_xs = [r[2] for r in res]
    lbs = [r[3] for r in res]
    ubs = [r[4] for r in res]
    ns = length.(in_xs)
    nd = length(ns) 
    
    # Construct the optimisation such that the burst frequency ratio Δ is the first parameter
    xs = vcat(in_xs...)
    lb = vcat(lbs...)
    ub = vcat(ubs...)

    rinds = vcat([1:ns[1]], [ sum(ns[1:i-1])+1:sum(ns[1:i]) for i in 2:nd])
    loss_joint(x) = sum(losses[i](x[rinds[i]]) for i in 1:nd)
    
    loss_joint, ns, xs, lb, ub, tau

end

function get_burst_frequency_ratio_CI_PL_weighted_avg(ds1::AbstractArray, ys1::AbstractArray, thetas1::AbstractArray, nconvs1::AbstractArray, ws1::AbstractArray,
                                                      ds2::AbstractArray, ys2::AbstractArray, thetas2::AbstractArray, nconvs2::AbstractArray, ws2::AbstractArray;
                                                      resolution::Int=100, alpha::Float64=0.05, kwargs...)
            
    # Profile likelihood set up such that Δ = (w₁f₁ + w₂f₂ + ⋯ + wᵢfᵢ) / (wᵢ₊₁fᵢ₊₁ + ⋯ + wₙfₙ),
    # It follows that we can express f₁ as f₁ = Δ × (wᵢ₊₁fᵢ₊₁ + wₙfₙ) - (w₂f₂ + ... + wᵢfᵢ)
    # Performing profile likelihood by varying Δ and maximising the joint likelihood keeping f₁ accordingly.

    # construct the joint loss function
    nd1 = length(ds1); nd2 = length(ds2); nd = nd1+nd2
    ds = vcat(ds1, ds2); ys = vcat(ys1, ys2); thetas=vcat(thetas1, thetas2); nconvs = vcat(nconvs1, nconvs2); ws = vcat(ws1, ws2)
    loss_joint, ns, xs, lb, ub, tau = init_ratio_CI_PL_weighted_avg(ds, ys, thetas, nconvs, alpha)
    loss_MLE = loss_joint(xs)

    # parameter bounds
    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    is = bf_ind.(ds)
    lb[is[1]] = p_lb; ub[is[1]] = p_ub
    dstep = (p_ub - p_lb)/resolution

    # compute the ratio Δ
    bfs1 = get_burst_frequency.(ds1)
    bfs2 = get_burst_frequency.(ds2)
    bfs = vcat(bfs1, bfs2)
    Δ = sum(ws1[i] * bfs1[i] for i in 1:nd1) / sum(ws2[i] * bfs2[i] for i in 1:nd2)
    xs[is[1]] = log(Δ)
    fn = exp
    
    # get f₁ from w₁f₁/nconv₁ = Δ × (wᵢ₊₁fᵢ₊₁/nconvᵢ₊₁ + ⋯ + wₙfₙ/nconvₙ) - (w₂f₂/nconv₂ + ⋯ + wᵢfᵢ/nconvᵢ)
    # currently done in log-space (for the sake of not overcomplicating the rest of the code even further)
    # NOTE: assuming burst frequencies are optimised in log space (change for different dist parametrisations)
    
    nconvs = [isNB(ds[i]) ? nconvs[i] : 1 for i in 1:nd]
    xinds = vcat(is[1], [sum(ns[1:i-1]) + is[i] for i in 2:nd]...)

    term1(x) = exp(x[is[1]]) * sum(ws[i] / nconvs[i] * exp(x[xinds[i]]) for i in nd1+1:nd)
    term2(x) = sum(ws[i] / nconvs[i] * exp(x[xinds[i]]) for i in 2:nd1; init=0)
    term(x) = term1(x) - term2(x)
    
    log_bf(x) = log(nconvs[1]/ws[1]) + NaNMath.log(term(x))
    
    # shift the parameters so that Δ is the first argument to the loss function
    n = sum(ns)
    _inds = circshift(1:n, n-is[1]+1)
    srt_inds = sortperm(_inds)
    
    function lossf(x)
        _f = log_bf(x[srt_inds])
        if isnan(_f)
            return NaN
        else
            loss_joint(vcat(_f, x[2:end])[srt_inds])
        end
    end
    lossf(x2, p) = lossf(vcat(p, x2))

    # Add inequality constraint Δ × (wᵢ₊₁fᵢ₊₁/nconvᵢ₊₁ + ⋯ + wₙfₙ/nconvₙ) - (w₂f₂/nconv₂ + ⋯ + wᵢfᵢ/nconvᵢ) ≥ 0
    _lb = isone(nd1) ? lb[_inds] : vcat(lb[_inds], 0.0)
    _ub = isone(nd1) ? ub[_inds] : vcat(ub[_inds], Inf)
    cons(res, x2, p) = isone(nd1) ? (res .= x2) : (res .= vcat(x2, term1(vcat(p, x2)[srt_inds])))

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, _lb, _ub, cons; kwargs...)
    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub

end


function get_weighted_sum_ftheta(d, theta)
    
    θs, fθ = get_phase_dist(theta)
    β_ind = numparams(d) # β is always the last parameter for the defined dists
    
    function weighted_sum_ftheta(x)
        # NOTE: this assumes that the extrinsic noise parameter, 2^(βθ), is optimised in log space
        β = exp(x[β_ind]) 
        s = sum( 2^(β * θs[i]) * fθ[i] for i in eachindex(θs))
        s
    end

    weighted_sum_ftheta
    
end

get_weighted_sum_ftheta(d, theta::Nothing) = x -> 1


function get_burst_size_ratio_CI_PL_weighted_avg(ds1::AbstractArray, ys1::AbstractArray, thetas1::AbstractArray, nconvs1::AbstractArray, ws1::AbstractArray,
                                                 ds2::AbstractArray, ys2::AbstractArray, thetas2::AbstractArray, nconvs2::AbstractArray, ws2::AbstractArray;
                                                 resolution::Int=100, alpha::Float64=0.05, kwargs...)

    # Note that burst size is bs = ρ\σ_b (volume-independent case) OR, in the volume-dependent case,
    # bs = sum of (ρ/σ_b * 2^(βθ)) * f(θ)) over θ from θ₁ to θ₂, where θ₁ and θ₂ indicate the start 
    # and end θ of the specific phase (G1 or G2/M) and f(θ) is the phase distribution.    
    # Profile likelihood set up such that Δ = (w₁bs₁ + w₂bs₂ + ⋯ + wᵢbsᵢ) / (wᵢ₊₁bsᵢ₊₁ + ⋯ + wₙbsₙ),
    # It follows that we can express bs₁ as bs₁ = Δ × (wᵢ₊₁bsᵢ₊₁ + wₙbsₙ) - (w₂bs₂ + ... + wᵢbsᵢ)
    # Performing profile likelihood by varying Δ and maximising the joint likelihood keeping bs₁ fixed accordingly.

    # construct the joint loss function
    nd1 = length(ds1); nd2 = length(ds2); nd = nd1+nd2
    ds = vcat(ds1, ds2); ys = vcat(ys1, ys2); thetas=vcat(thetas1, thetas2); nconvs = vcat(nconvs1, nconvs2); ws = vcat(ws1, ws2)
    loss_joint, ns, xs, lb, ub, tau = init_ratio_CI_PL_weighted_avg(ds, ys, thetas, nconvs, alpha)
    loss_MLE = loss_joint(xs)
    
    # parameter bounds
    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    is = b_ind.(ds)
    lb[is[1]] = p_lb; ub[is[1]] = p_ub
    dstep = (p_ub - p_lb)/resolution 
    
    # compute the ratio Δ
    bs1 = get_burst_size.(ds1, thetas1)
    bs2 = get_burst_size.(ds2, thetas2)
    bs = vcat(bs1, bs2)
    Δ = sum(ws1[i] * bs1[i] for i in 1:nd1) / sum(ws2[i] * bs2[i] for i in 1:nd2)
    xs[is[1]] = log(Δ)
    fn = exp
   
    # get b₁ from w₁bs₁ = Δ × (wᵢ₊₁bsᵢ₊₁ + ⋯ + wₙbsₙ) - (w₂bs₂ + ⋯ + wᵢbsᵢ)    
    xinds = [1:ns[1]]
    for i in 2:nd
        j = xinds[end][end]
        push!(xinds, j+1:j+ns[i])
    end
    wsum_fths = get_weighted_sum_ftheta.(ds, thetas)
    log_bs = get_log_b.(ds)

    term1(x) = fn(x[is[1]]) * sum(ws[i] * exp(log_bs[i](x[xinds[i]])) * wsum_fths[i](x[xinds[i]]) for i in nd1+1:nd)
    term2(x) = sum(ws[i] * exp(log_bs[i](x[xinds[i]])) * wsum_fths[i](x[xinds[i]]) for i in 2:nd1; init=0)    
    term(x) = term1(x) - term2(x)

    log_σ_off(x) = isBP(ds[1]) ? x[2] : isZIBP(ds[1]) ? x[3] : 0
    log_b(x) = NaNMath.log(term(x)) + log_σ_off(x) - log(wsum_fths[1](x)) - log(ws[1])
    
    # shift the parameters so that Δ is the first argument to the loss function
    n = sum(ns)
    _inds = circshift(1:n, n-is[1]+1)
    srt_inds = sortperm(_inds)
    
    function lossf(x)
        _b = log_b(x[srt_inds])
        if isnan(_b)
            return NaN
        else
            loss_joint(vcat(is_p(ds[1])*_b, x[2:end])[srt_inds])
        end
    end
    lossf(x2, p) = lossf(vcat(p, x2))

    # Add inequality constraint Δ × (wᵢ₊₁bᵢ₊₁ + ⋯ + wₙbₙ) - (w₂b₂ + ⋯ + wᵢbᵢ) ≥ 0
    _lb = isone(nd1) ? lb[_inds] : vcat(lb[_inds], 0)
    _ub = isone(nd1) ? ub[_inds] : vcat(ub[_inds], Inf)
    cons(res, x2, p) = isone(nd1) ? (res .= x2) : (res .= vcat(x2, term(vcat(p, x2)[srt_inds])))

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, _lb, _ub, cons; kwargs...)
    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub

end
