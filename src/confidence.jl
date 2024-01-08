using ForwardDiff, Optim, Interpolations, NonlinearSolve, StatsFuns

include("confidence_PL.jl")
include("confidence_wavg.jl")

# Data structure to store the resulting confidence intervals (as the corresponding distributions)
# note that these distributions by themselves have no meaning, they are just convenient for storage
struct DistCI{DT<:DiscreteUnivariateDistribution}
    lower::DT
    upper::DT
end

DistCI(lower::DT, upper::DT) where DT<:DiscreteUnivariateDistribution = DistCI{DT}(lower, upper)

DistCI(::Type{DT}, lower::AbstractArray{T}, upper::AbstractArray{T}) where {DT<:DiscreteUnivariateDistribution, T<:Real} = 
    DistCI(DT{T}(lower...), DT{T}(upper...)) 
DistCI(::Type{<:ZI{DT}}, lower::AbstractArray{T}, upper::AbstractArray{T}) where {DT<:DiscreteUnivariateDistribution, T<:Real} = 
    DistCI(ZI(max(zero(T), lower[1]), DT{T}(lower[2:end]...)), ZI(min(one(T), upper[1]), DT{T}(upper[2:end]...)))

# Compute confidence intervals for all given MLE fits
function get_confidence_intervals(fits::AbstractArray{Distribution}, xdata::AbstractArray, funCI, theta=nothing; kwargs...)
    
    xlen = length(xdata)
    @assert xlen == length(fits) "Provided fits are inconsistent with the number of genes"

    CIs = Vector{DistCI}(undef, xlen)
    @views Threads.@threads for i in 1:xlen
        try
            CIs[i] = funCI(fits[i], xdata[i], theta; kwargs...)
        catch e
            println("ind $i: $e")
        end
    end
    
    CIs
    
end

function get_ratio_confidence_intervals(fits1::AbstractArray, xdata1::AbstractArray, theta1, nconv1::Int,
                                        fits2::AbstractArray, xdata2::AbstractArray, theta2, nconv2::Int,
                                        funCI; printerr::Bool=true, kwargs...)
    
    xlen = length(xdata1)
    @assert (xlen == length(xdata2) == length(fits1) == length(fits2)) 
        "Provided fits are inconsistent with the number of genes"

    CIs = Vector{Tuple}(undef, xlen)
    @views Threads.@threads for i in 1:xlen
		try
            CIs[i] = funCI(fits1[i], xdata1[i], theta1, nconv1,
                           fits2[i], xdata2[i], theta2, nconv2; kwargs...)
        catch e
            printerr && println("ind $i: $e")
        end
    end
    
    CIs
end

### Lower/upper bound burst size/frequency given confidence intervals ###

lb_burst_frequency(dCI::DistCI{<:NegativeBinomial}) = dCI.lower.r
ub_burst_frequency(dCI::DistCI{<:NegativeBinomial}) = dCI.upper.r

lb_burst_frequency(dCI::DistCI{<:BetaPoisson}) = dCI.lower.σ_on
ub_burst_frequency(dCI::DistCI{<:BetaPoisson}) = dCI.upper.σ_on

lb_burst_frequency(dCI::DistCI{<:NegativeBinomialTheta}) = dCI.lower.r
ub_burst_frequency(dCI::DistCI{<:NegativeBinomialTheta}) = dCI.upper.r

lb_burst_frequency(dCI::DistCI{<:BetaPoissonTheta}) = dCI.lower.σ_on
ub_burst_frequency(dCI::DistCI{<:BetaPoissonTheta}) = dCI.upper.σ_on

lb_burst_frequency(dCI::DistCI{<:ZI}) = lb_burst_frequency(DistCI(dCI.lower.dist, dCI.upper.dist))
ub_burst_frequency(dCI::DistCI{<:ZI}) = ub_burst_frequency(DistCI(dCI.lower.dist, dCI.upper.dist))

lb_burst_size(dCI::DistCI, ::Nothing) = lb_burst_size(dCI)
ub_burst_size(dCI::DistCI, ::Nothing) = ub_burst_size(dCI)

lb_burst_size(dCI::DistCI{<:NegativeBinomial}) = 1/dCI.upper.p - 1
ub_burst_size(dCI::DistCI{<:NegativeBinomial}) = 1/dCI.lower.p - 1

lb_burst_size(dCI::DistCI{<:BetaPoisson}) = dCI.lower.ρ / dCI.upper.σ_off
ub_burst_size(dCI::DistCI{<:BetaPoisson}) = dCI.upper.ρ / dCI.lower.σ_off

lb_burst_size(dCI::DistCI{<:NegativeBinomialTheta}, theta::AbstractArray) = get_burst_size(dCI.lower, theta)
ub_burst_size(dCI::DistCI{<:NegativeBinomialTheta}, theta::AbstractArray) = get_burst_size(dCI.upper, theta)

lb_burst_size(dCI::DistCI{<:BetaPoissonTheta}, theta::AbstractArray) = get_burst_size(BetaPoisson, theta, dCI.lower.ρ, dCI.upper.σ_off, dCI.lower.β)
ub_burst_size(dCI::DistCI{<:BetaPoissonTheta}, theta::AbstractArray) = get_burst_size(BetaPoisson, theta, dCI.upper.ρ, dCI.lower.σ_off, dCI.upper.β)

lb_burst_size(dCI::DistCI{<:ZI}) = lb_burst_size(DistCI(dCI.lower.dist, dCI.upper.dist))
ub_burst_size(dCI::DistCI{<:ZI}) = ub_burst_size(DistCI(dCI.lower.dist, dCI.upper.dist))

lb_burst_size(dCI::DistCI{<:ZI}, theta::AbstractArray) = lb_burst_size(DistCI(dCI.lower.dist, dCI.upper.dist), theta)
ub_burst_size(dCI::DistCI{<:ZI}, theta::AbstractArray) = ub_burst_size(DistCI(dCI.lower.dist, dCI.upper.dist), theta)

lb_beta_param(dCI::DistCI) = dCI.lower.β
ub_beta_param(dCI::DistCI) = dCI.upper.β
lb_beta_param(dCI::DistCI{<:ZI}) = dCI.lower.dist.β
ub_beta_param(dCI::DistCI{<:ZI}) = dCI.upper.dist.β

function get_burst_frequency_ratio_CI_naive(dCI1::DistCI, dCI2::DistCI)
    lb1 = lb_burst_frequency(dCI1)
    ub1 = ub_burst_frequency(dCI1)
    lb2 = lb_burst_frequency(dCI2)
    ub2 = ub_burst_frequency(dCI2)
    lb2 / ub1, ub2 / lb1
end

function get_burst_size_ratio_CI_naive(dCI1::DistCI, dCI2::DistCI, theta1, theta2)
    lb1 = lb_burst_size(dCI1, theta1)
    ub1 = ub_burst_size(dCI1, theta1)
    lb2 = lb_burst_size(dCI2, theta2)
    ub2 = ub_burst_size(dCI2, theta2)
    lb2 / ub1, ub2 / lb1
end
