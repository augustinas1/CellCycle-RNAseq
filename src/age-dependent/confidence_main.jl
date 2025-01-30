# ----- Confidence intervals for the age-dependent (main-text) model using Profile Likelihood -----

function init_PL_CI(m::MainModel, y::AbstractArray{<:Real}, theta::AbstractArray{<:Real},
                    T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real)
    # initialise  optimisation utilities for the given distribution 
    # return the loss function, initial parameters in the transformed (optimisation) space,
    # the corresponding lower/upper bounds,
    # the parameter-specific functions to revert back to the original space,
    # and the lower/upper param bounds in the transformed space

    lb, ub = get_bounds(MainModel)
    invfns, fns = get_CI_fns(MainModel)
    loss = construct_loss(m, T, d, θᵣ, θ_S_i, θ_S_f, y, theta)

    ps = collect(Distributions.params(m))
    fun_lb = [fns[i](lb[i]) for i in eachindex(lb)]
    fun_ub = [fns[i](ub[i]) for i in eachindex(ub)]
    xs = [invfns[i](ps[i]) for i in eachindex(ps)]
    inds_lb = findall(ps .== fun_lb)
    xs[inds_lb] .= lb[inds_lb]
    inds_ub = findall(ps .== fun_ub)
    xs[inds_ub] .= ub[inds_ub]
    
    loss, xs, lb, ub, fns, inds_lb, inds_ub
end


function get_confidence_intervals_PL(m::MainModel, y::AbstractArray{<:Real}, theta::AbstractArray{<:Real}, 
                                     T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                                     resolution::Int=100, alpha::Float64=0.05, kwargs...)
    
    loss, xs, lb, ub, fns, inds_lb, inds_ub = init_PL_CI(m, y, theta, T, d, θᵣ, θ_S_i, θ_S_f)
    
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
        
        ps_lb[i] = fn(p_lower)
        ps_ub[i] = fn(p_upper)
    end

    DistCI(MainModel, ps_lb, ps_ub)
end

function get_burst_frequency_ratio_CI_PL(m::MainModel, y::AbstractArray, theta::AbstractArray,
    T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
    resolution::Int=100, alpha::Float64=0.05, kwargs...)

    # Profile likelihood set up such that Δ = f₂ / f₁ ⟹ f₂ = Δ × f₁
    # We vary Δ and maximise the joint likelihood where f₂ is fixed

    loss, xs, lb, ub, _, _, _ = init_PL_CI(m, y, theta, T, d, θᵣ, θ_S_i, θ_S_f)
    tau = cquantile(Chisq(1), alpha)/2
    loss_MLE = loss(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = 1
    i2 = 3
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution

    bf1 = get_burst_frequency_G1(m)
    bf2 = get_burst_frequency_G2M(m)
    Δ = bf2 / bf1
    xs[i2] = log(Δ)
    fn = exp

    # compute bf₂ = Δ × bf₁ (in log space)
    # NOTE: this assumes that both burst frequencies are optimised in log space
    log_bf2(x) = x[i2] + x[i1]

    # shift the parameters so that Δ is the first argument to the loss function
    n = numparams(m)
    _inds = circshift(1:n, n-i2+1)
    srt_inds = sortperm(_inds)
    lossf(x) = loss(vcat(log_bf2(x[srt_inds]), x[2:end])[srt_inds])
    lossf(x, p) = lossf(vcat(p, x))
    cons(res, x, p) = (res .= x)

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, lb[_inds], ub[_inds], cons; kwargs...)

    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub

end


function get_log_weighted_sum_ftheta_G1(::MainModel, theta::AbstractArray)
    
    θs, fθ = get_phase_dist(theta)
    β_ind = 5 # β₁ index
    
    function log_weighted_sum_ftheta_G1(x)
        β = x[β_ind] # assuming identity fn
        s = sum( exp(β * θs[i]) * fθ[i] for i in eachindex(θs))
        log(s)
    end

    log_weighted_sum_ftheta_G1
    
end


function get_log_weighted_sum_ftheta_G2M(::MainModel, theta::AbstractArray)
    
    θs, fθ = get_phase_dist(theta)
    β_ind = 8 # β₄ index
    
    function log_weighted_sum_ftheta_G2M(x)
        β = x[β_ind] # assuming identity fn
        s = sum( exp(β * θs[i]) * fθ[i] for i in eachindex(θs))
        log(s)
    end

    log_weighted_sum_ftheta_G2M
    
end


function get_burst_size_ratio_CI_PL(m::MainModel, y::AbstractArray, theta::AbstractArray,
    T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
    resolution::Int=100, alpha::Float64=0.05, kwargs...)

    # Note that burst size bs = sum of (ρ * exp(βθ)) * f(θ)) over θ from θ₁ to θ₂, 
    # where θ₁ and θ₂ indicate the start and end θ of the specific phase (G1 or G2\M)
    # and f(θ) is the phase distribution. Depending on the cell cycle phase, we have 
    # ρ = ρ₁ (G1) or ρ = ρ₂ (G2/M)

    # Profile likelihood set up such that Δ = bs₂ / bs₁ ⟹ bs₂ = (Δ × bs₁) / (sum of (exp(β₂θ)) * f(θ))) / 
    # We vary Δ and maximise the joint likelihood where bs₂ is fixed
    # Note that the burst size is computed as bs = ρ \sum_{θ}{exp(βθ))*f(θ)}

    loss, xs, lb, ub, _, _, _ = init_PL_CI(m, y, theta, T, d, θᵣ, θ_S_i, θ_S_f)
    tau = cquantile(Chisq(1), alpha)/2
    loss_MLE = loss(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = b_ind(m, "G1")
    i2 = b_ind(m, "G2/M")
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution 

    theta1 = theta[theta .< θ_S_i]
    theta2 = theta[theta .> θ_S_f]
    bs1 = get_burst_size_G1(m, theta1)
    bs2 = get_burst_size_G2M(m, theta2)
    Δ = bs2 / bs1
    xs[i2] = log(Δ)
    fn = exp

    log_wsum_fth1 = get_log_weighted_sum_ftheta_G1(m, theta1)
    log_wsum_fth2 = get_log_weighted_sum_ftheta_G2M(m, theta2)
    log_b1 = x -> x[i1]
    log_bs1(x) = log_b1(x) + log_wsum_fth1(x)

    # compute b₂ = Δ × bs₁ / (in log space) including the corresponding convolutions for NBs) 
    log_b2(x) = x[i2] + log_bs1(x) - log_wsum_fth2(x) 

    # shift the parameters so that Δ is the first argument to the loss function
    n = numparams(m)
    _inds = circshift(1:n, n-i2+1)
    srt_inds = sortperm(_inds)
    lossf(x) = loss(vcat(log_b2(x[srt_inds]), x[2:end])[srt_inds])
    lossf(x, p) = lossf(vcat(p, x))
    cons(res, x, p) = (res .= x)

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, lb[_inds], ub[_inds], cons; kwargs...)
    
    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub
end

# Profile likelihood ratios for age-dependent model vs. age-independent

function init_ratio_CI_PL(m1::DiscreteUnivariateDistribution, nconv1::Int, m2::MainModel,
                          y::AbstractArray, theta::AbstractArray, phase::String,
                          T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, alpha::Real)

    check_CI_args(m1, nconv1)
    if phase == "G1"
        inds = findall(theta .< θ_S_i)
    elseif phase == "G2/M"
        inds = findall(theta .> θ_S_f)
    else
        error("phase must be \"G1\" or \"G2/M\"")
    end
    
    # Profile likelihood value: PLₘₗₑ-PL(v) <= tau
    tau = cquantile(Chisq(1), alpha)/2
    # leaving theta input as nothing but could generalise to our old θ-dep dists too 
    loss1, xs1, lb1, ub1, _, _, _, _ = init_PL_CI(m1, y[inds], nconv1)
    loss2, xs2, lb2, ub2, _, _, _ = init_PL_CI(m2, y, theta, T, d, θᵣ, θ_S_i, θ_S_f)
    n1 = length(xs1); n2 = length(xs2)

    # Construct the optimisation such that the burst frequency ratio Δ is the first parameter
    xs = vcat(xs2, xs1)
    lb = vcat(lb2, lb1)
    ub = vcat(ub2, ub1)

    loss_joint(x2, x1) = loss2(x2) + loss1(x1)
    loss_joint(x) = loss_joint(x[1:n2], x[n2+1:end])

    loss_joint, n1, n2, xs, lb, ub, tau, inds

end

bf_ind(::MainModel, phase::String) = phase == "G1" ? 1 : 
                                     phase == "G2/M" ? 2 : 
                                     error("phase must be \"G1\" or \"G2/M\"")

get_burst_frequency(m::MainModel, phase::String) = phase == "G1" ? get_burst_frequency_G1(m) : 
                                                   phase == "G2/M" ? get_burst_frequency_G2M(m) : 
                                                   error("phase must be \"G1\" or \"G2/M\"")

# compute confidence interval for the paramater ratio of the type p_{θ-dependent} / p_{θ-independent}
# due to the setup of the cell division model, need to explicitly define which cell-cycle phase we are considering (either "G1" or "G2/M")

function get_burst_frequency_ratio_CI_PL(m1::DiscreteUnivariateDistribution, nconv1::Int, m2::MainModel, 
                                         y::AbstractArray, theta::AbstractArray, phase::String,
                                         T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                                         resolution::Int=100, alpha::Float64=0.05, kwargs...)

    # Profile likelihood set up such that Δ = f₂ / f₁ ⟹ f₂ = Δ × f₁
    # We vary Δ and maximise the joint likelihood where f₂ is fixed
    
    loss_joint, n1, n2, xs, lb, ub, tau, _ = init_ratio_CI_PL(m1, nconv1, m2, y, theta, phase, T, d, θᵣ, θ_S_i, θ_S_f, alpha)
    loss_MLE = loss_joint(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = bf_ind(m1)
    i2 = bf_ind(m2, phase)
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution

    # times the decay rate to obtain the absolute value
    bf1 = get_burst_frequency(m1) * d
    bf2 = get_burst_frequency(m2, phase)
    Δ = bf2 / bf1
    xs[i2] = log(Δ)
    fn = exp

    # compute bf₂ = Δ × bf₁ (in log space) including the corresponding convolution for NB
    # NOTE: this assumes that both burst frequencies are optimised in log space
    log_bf2(x) = x[i2] + x[n2+i1] + log(d) - isNB(m1)*log(nconv1)

    # shift the parameters so that Δ is the first argument to the loss function
    n = n1 + n2
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

b_ind(::MainModel, phase::String) = phase == "G1" ? 3 :
                                    phase == "G2/M" ? 4 :
                                    error("phase must be \"G1\" or \"G2/M\"")

get_log_weighted_sum_ftheta(m::MainModel, theta::AbstractArray, phase::String) = phase == "G1" ? get_log_weighted_sum_ftheta_G1(m, theta) :
                                                                                 phase == "G2/M" ? get_log_weighted_sum_ftheta_G2M(m, theta) :   
                                                                                 error("phase must be \"G1\" or \"G2/M\"")     

function get_burst_size_ratio_CI_PL(m1::DiscreteUnivariateDistribution, nconv1::Int, m2::MainModel, 
                                    y::AbstractArray, theta::AbstractArray, phase::String,
                                    T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                                    resolution::Int=100, alpha::Float64=0.05, kwargs...)

    loss_joint, n1, n2, xs, lb, ub, tau, inds = init_ratio_CI_PL(m1, nconv1, m2, y, theta, phase, T, d, θᵣ, θ_S_i, θ_S_f, alpha)
    loss_MLE = loss_joint(xs)

    p_lb = -5; p_ub = 5 # fixed Δ bounds in the log-space
    i1 = b_ind(m1)
    i2 = b_ind(m2, phase)
    lb[i2] = p_lb; ub[i2] = p_ub
    dstep = (p_ub - p_lb)/resolution

    if phase == "G1"
        bf1 = get_burst_size(m1)
        bf2 = get_burst_size_G1(m2, theta[inds])
    elseif phase == "G2/M"
        bf1 = get_burst_size(m1)
        bf2 = get_burst_size_G2M(m2, theta[inds])
    else
        error("phase must be \"G1\" or \"G2/M\"")
    end
    Δ = bf2 / bf1
    xs[i2] = log(Δ)
    fn = exp

    log_wsum_fth2 = get_log_weighted_sum_ftheta(m2, theta[inds], phase)
    log_b1 = get_log_b(m1)
    
    function log_bs1(x)
        x2 = @view x[n2+1:end]
        log_b1(x2)
    end
    
    # compute b₂ = Δ × bs₁ (in log space) including the corresponding convolutions for NBs
    # note that we consider the burst size averaged over all cells for MainModel
    log_b2(x) = x[i2] + log_bs1(x) - log_wsum_fth2(x)

    # shift the parameters so that Δ is the first argument to the loss function
    n = n1 + n2
    _inds = circshift(1:n, n-i2+1)
    srt_inds = sortperm(_inds)
    lossf(x) = loss_joint(vcat(log_b2(x[srt_inds]), x[2:end])[srt_inds])
    lossf(x, p) = lossf(vcat(p, x))
    cons(res, x, p) = (res .= x)

    runf(_dstep) = profile_likelihood_CI(lossf, loss_MLE, xs[_inds], _dstep, tau, fn, lb[_inds], ub[_inds], cons; kwargs...)

    p_lower = runf(-dstep)
    p_upper = runf(dstep)
    Δ_lb = fn(p_lower)
    Δ_ub = fn(p_upper)

    Δ_lb, Δ_ub

end


function get_ratio_confidence_intervals(fits1::AbstractArray, nconv1::Int, fits2::AbstractArray,
                                        data::AbstractArray, theta::AbstractArray, phase::String, T_cycle::Real, decay_rates::AbstractArray, θᵣ::Real, θ_S_i::Real, θ_S_f::Real,
                                        funCI; printerr::Bool=true, kwargs...)
    
    xlen = length(data)
    @assert (xlen == length(fits1) == length(fits2)) 
        "Provided fits are inconsistent with the number of genes"

    CIs = Vector{Tuple}(undef, xlen)
    @showprogress Threads.@threads for i in 1:xlen
		try
            CIs[i] = funCI(fits1[i], nconv1, fits2[i], data[i], theta, phase, T_cycle, decay_rates[i], θᵣ, θ_S_i, θ_S_f; kwargs...)
        catch e
            printerr && println("ind $i: $e")
        end
    end
    
    CIs
end

function get_burst_frequency_ratios_CI(fits::AbstractArray, data::AbstractArray, theta::AbstractArray, T_cycle::Real, decay_rates::AbstractArray, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                                       printerr::Bool=true, kwargs...)
    
    xlen = length(data)
    @assert (xlen == length(fits)) 
        "Provided fits are inconsistent with the number of genes"

    CIs = Vector{Tuple}(undef, xlen)
    @showprogress Threads.@threads for i in 1:xlen
		try
            CIs[i] = get_burst_frequency_ratio_CI_PL(fits[i], data[i], theta, T_cycle, decay_rates[i], θᵣ, θ_S_i, θ_S_f; kwargs...)
        catch e
            printerr && println("ind $i: $e")
        end
    end
    
    CIs
end

function get_burst_size_ratios_CI(fits::AbstractArray, data::AbstractArray, theta::AbstractArray, T_cycle::Real, decay_rates::AbstractArray, θᵣ::Real, θ_S_i::Real, θ_S_f::Real;
                                  printerr::Bool=true, kwargs...)
    
    xlen = length(data)
    @assert (xlen == length(fits)) 
        "Provided fits are inconsistent with the number of genes"

    CIs = Vector{Tuple}(undef, xlen)
    @showprogress Threads.@threads for i in 1:xlen
		try
            CIs[i] = get_burst_size_ratio_CI_PL(fits[i], data[i], theta, T_cycle, decay_rates[i], θᵣ, θ_S_i, θ_S_f; kwargs...)
        catch e
            printerr && println("ind $i: $e")
        end
    end
    
    CIs
end