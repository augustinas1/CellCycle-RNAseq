using Distributions, StatsBase, LinearAlgebra, Random,
      SpecialFunctions, HypergeometricFunctions, LogExpFunctions

include("convolutions.jl")

### Poisson ###

numparams(d::T) where {T<:DiscreteUnivariateDistribution} = numparams(T)
numparams(::Type{<:Poisson}) = 1

reconstruct(::Poisson, ps::AbstractArray{<:Real}) = Poisson(exp(ps[1]))
get_bounds(::Type{<:Poisson}) = log(1e-6), log(1e4)

### Poisson model with cell age (θ) dependence ###
# Telegraph model in the limit of constitutive expression
# The model is Poisson(λ = ρ*2^(βθ)) where θ is the cell cycle phase (t/T) ∈ [0, 1]
# and β is the strength of cell age dependence 

struct PoissonTheta{T} <: DiscreteUnivariateDistribution
    ρ::T
    β::T

    function PoissonTheta{T}(ρ::T, β::T) where {T<:Real}
        return new{T}(ρ, β)
    end    
end

function PoissonTheta(ρ::T, β::T) where {T<:Real}
    return PoissonTheta{T}(ρ, β)
end

PoissonTheta() = PoissonTheta{Float64}(1.0, 0.1)
Poisson(d::PoissonTheta, θ::Real) = Poisson(d.ρ*2^(d.β*θ))

Distributions.params(d::PoissonTheta) = (d.ρ, d.β)
numparams(::Type{<:PoissonTheta}) = 2
get_th_param(d::PoissonTheta) = d.β

Distributions.logpdf(d::PoissonTheta, θ::Real, k::Real) = logpdf(Poisson(d, θ), k)
Distributions.pdf(d::PoissonTheta, θ::Real, k::Real) = exp(logpdf(d, θ, k))

# MLE functions
# seems that dispatching on instances rather than types themselves is more stable in general
# trying do that in performance-sensitive bits of the code at least
reconstruct(::PoissonTheta, ps::AbstractArray{<:Real}) = PoissonTheta(exp(ps[1]), exp(ps[2]))
reconstruct(::PoissonTheta, ps::AbstractArray{<:Real}, nconv::Int) = PoissonTheta(exp(ps[1])/nconv, exp(ps[2]))
get_bounds(::Type{<:PoissonTheta}) = vcat(log(1e-6), -20.0), vcat(log(1e3), 5.0)

### Negative Binomial ### 

numparams(::Type{<:NegativeBinomial}) = 2

reconstruct(::NegativeBinomial, ps::AbstractArray{<:Real}) = NegativeBinomial(exp(ps[1]), logistic(ps[2]))
reconstruct(::NegativeBinomial, ps::AbstractArray{<:Real}, nconv::Int) = NegativeBinomial(exp(ps[1]-log(nconv)), logistic(ps[2]))
get_bounds(::Type{<:NegativeBinomial}) = [-20.0, -30.0], [10.0, 30.0]

### NegativeBinomial model with cell age (θ) dependence
# Telegraph model in the limit of large gene inactivation rate σ_b
# The model is NB(r, p): r is the gene activation rate σ_u and p = 1/(1+b'), 
# b' denotes the burst size (ρ\σ_b) given by b' = b*2^(β*θ), 
# where θ is the cell cycle phase (t/T) ∈ [0, 1] and 
# β is the strength of balanced mRNA synthesis (cell age dependence)

struct NegativeBinomialTheta{T} <: DiscreteUnivariateDistribution
    r::T
    b::T
    β::T

    function NegativeBinomialTheta{T}(r::T, b::T, β::T) where {T<:Real}
        return new{T}(r, b, β)
    end    
end

function NegativeBinomialTheta(r::T, b::T, β::T) where {T<:Real}
    return NegativeBinomialTheta{T}(r, b, β)
end

NegativeBinomialTheta() = NegativeBinomialTheta{Float64}(1.0, 1.0, 0.1)
NegativeBinomial(d::NegativeBinomialTheta, θ::Real) = NegativeBinomial(d.r, 1/(1+d.b*2^(d.β*θ)))

Distributions.params(d::NegativeBinomialTheta) = (d.r, d.b, d.β)
numparams(::Type{<:NegativeBinomialTheta}) = 3
get_th_param(d::NegativeBinomialTheta) = d.β

# statistics/evaluation require the cell-cycle phase input
Distributions.mean(d::NegativeBinomialTheta, θ::Real) = mean(NegativeBinomial(d, θ))
Distributions.var(d::NegativeBinomialTheta, θ::Real) = var(NegativeBinomial(d, θ))
Distributions.logpdf(d::NegativeBinomialTheta, θ::Real, k::Real) = logpdf(NegativeBinomial(d, θ), k)
Distributions.pdf(d::NegativeBinomialTheta, θ::Real, k::Real) = exp(logpdf(d, θ, k))

# MLE functions
reconstruct(::NegativeBinomialTheta, ps::AbstractArray{<:Real}) = NegativeBinomialTheta(exp(ps[1]), exp(ps[2]), exp(ps[3]))
reconstruct(::NegativeBinomialTheta, ps::AbstractArray{<:Real}, nconv::Int) = NegativeBinomialTheta(exp(ps[1])/nconv, exp(ps[2]), exp(ps[3]))
get_bounds(::Type{<:NegativeBinomialTheta}) = vcat(log.([1e-4, 1e-4]), -20.0), vcat(log.([1e3, 1e3]), 5.0)

### Beta-Poisson distribution (telegraph model steady-state solution) ###

# Hypergeometric 1F1 computation
function stable_log₁F₁maclaurin(a::Number, b::Number, z::Number; maxiter=1e7)
    T = float(promote_type(typeof(a), typeof(b), typeof(z)))
    S₀, S₁, j = one(T), one(T)+a*z/b, 1
    logs = zero(T)
    while (HypergeometricFunctions.errcheck(S₀, S₁, 10eps(real(T))) || j ≤ 1) && (j <= maxiter)
        rⱼ = (a+j)/((b+j)*(j+1))
        S₀, S₁ = S₁, S₁+(S₁-S₀)*rⱼ*z
        if HypergeometricFunctions.norm2(S₀) > 1e30
            logs += log(S₀)
            S₁ /= S₀
            S₀ = one(T)
        end
        j += 1
    end
    return log(S₁) + logs
end

function log1F1(a, b, z)
    if z ≥ 0
        stable_log₁F₁maclaurin(a, b, z)
    else
        z + stable_log₁F₁maclaurin(b-a, b, -z)
    end
end

function logpoch(x::T, n::Integer) where T
    ret = zero(T)
    logx = log(x)
    for i = 0:n-1
        ret += log(1 + i/x) + logx
    end
    ret
end

struct BetaPoisson{T<:Real} <: DiscreteUnivariateDistribution
    σ_on::T
    σ_off::T
    ρ::T

    function BetaPoisson{T}(σ_on::T, σ_off::T, ρ::T) where {T<:Real}
        return new{T}(σ_on, σ_off, ρ)
    end
end

function BetaPoisson(σ_on::T, σ_off::T, ρ::T) where {T<:Real}
    return BetaPoisson{T}(σ_on, σ_off, ρ)
end

BetaPoisson() = BetaPoisson{Float64}(0.1, 0.1, 1.0)

numparams(::Type{<:BetaPoisson}) = 3
Distributions.params(d::BetaPoisson) = (d.σ_on, d.σ_off, d.ρ)

function Distributions.logpdf(d::BetaPoisson, k::Real)

    loghyp = log1F1(d.σ_on + k, d.σ_on + d.σ_off + k, -d.ρ)
    
    if !isfinite(loghyp)
        return -Inf
    end
    
    ret = k * log(d.ρ) - loggamma(k + 1) + logpoch(d.σ_on, k) - logpoch(d.σ_on + d.σ_off, k) + loghyp
    
    if !isfinite(ret)
        -Inf
    end

    ret
end

Distributions.pdf(bp::BetaPoisson, k::Real) = exp(logpdf(bp, k))
Distributions.mean(d::BetaPoisson) = d.ρ * d.σ_on / (d.σ_on + d.σ_off)
Distributions.var(d::BetaPoisson) = mean(d) + d.ρ^2 * d.σ_on * d.σ_off / (d.σ_on + d.σ_off)^2 / (d.σ_on + d.σ_off + 1)

# MLE functions
reconstruct(::BetaPoisson, ps::AbstractArray{<:Real}) = BetaPoisson(exp(ps[1]), exp(ps[2]), exp(ps[3]))
get_bounds(::Type{<:BetaPoisson}) = log.([1e-4, 1e-4, 1e-3]), log.([1e3, 1e3, 1e3])


### Telegraph model with θ dependence
# Transcription rate is now ρ*2^(βθ) where θ is the cell age (t/T) ∈ [0, 1]

struct BetaPoissonTheta{T} <: DiscreteUnivariateDistribution
    σ_on::T
    σ_off::T
    ρ::T
    β::T

    function BetaPoissonTheta{T}(σ_on::T, σ_off::T, ρ::T, β::T) where {T<:Real}
        return new{T}(σ_on, σ_off, ρ, β)
    end
end

function BetaPoissonTheta(σ_on::T, σ_off::T, ρ::T, β::T) where {T<:Real}
    return BetaPoissonTheta{T}(σ_on, σ_off, ρ, β)
end

BetaPoissonTheta() = BetaPoissonTheta(0.1, 0.1, 1.0, 0.1)
BetaPoisson(d::BetaPoissonTheta, θ::Real) = BetaPoisson(d.σ_on, d.σ_off, d.ρ*2^(d.β*θ))

Distributions.params(d::BetaPoissonTheta) = (d.σ_on, d.σ_off, d.ρ, d.β)
numparams(::Type{<:BetaPoissonTheta}) = 4
get_th_param(d::BetaPoissonTheta) = d.β

Distributions.mean(d::BetaPoissonTheta, θ::Real) = mean(BetaPoisson(d, θ))
Distributions.var(d::BetaPoissonTheta, θ::Real) = var(BetaPoisson(d, θ))

Distributions.logpdf(d::BetaPoissonTheta, θ::Real, k::Real) = logpdf(BetaPoisson(d, θ), k)
Distributions.pdf(d::BetaPoissonTheta, θ::Real, k::Real) = exp(logpdf(d, θ, k))

reconstruct(::BetaPoissonTheta, ps::AbstractArray{<:Real}) = BetaPoissonTheta(exp.(ps)...)
get_bounds(::Type{<:BetaPoissonTheta}) = vcat(log.([1e-4, 1e-4, 1e-3]), -20.0), vcat(log.([1e3, 1e3, 1e3]), 5.0)


### Zero-inflated (Beta-Poisson, Negative Binomial or Poisson) ###

struct ZI{FT, DT} <: DiscreteUnivariateDistribution where {FT<:Real, DT<:DiscreteUnivariateDistribution}
    p0::FT
    dist::DT

    function ZI{FT, DT}(p0::FT, dist::DT) where {FT<:Real, DT<:DiscreteUnivariateDistribution}
        zero(p0) <= p0 <= one(p0) || throw(DomainError(:p0, "0 <= p0 <= 1"))
        return new{FT, DT}(p0, dist)
    end    
end

function ZI(p0::FT, dist::DT) where {FT <:Real, DT<:DiscreteUnivariateDistribution}
    ZI{FT, DT}(p0, dist)
end

function ZI(::Type{DT}, p0::FT, args...; kwargs...) where {DT,FT}
    zero(p0) <= p0 <= one(p0) || throw(DomainError(:p0, "0 <= p0 <= 1"))
    ZI{DT,FT}(p0, DT(args...; kwargs...))
end

ZI() = ZI{Float64, NegativeBinomial}(0.1, NegativeBinomial())
ZI{DT}() where DT<:DiscreteUnivariateDistribution = ZI{Float64, DT}(0.1, DT()) 

function Distributions.logpdf(zi::ZI, x::Real)
    l = logpdf(zi.dist, x)
    if iszero(x)
        v1 = l+log(1-zi.p0)
        v2 = log(zi.p0)
        maxv = max(v1, v2)
        maxv + log(exp(v1-maxv) + exp(v2-maxv))    
        #log(exp(l)*(1-zi.p0) + zi.p0*iszero(x))
    else
        l+log(1-zi.p0)
    end
end

Distributions.pdf(d::ZI, x::Real) = exp(logpdf(d, x))
Distributions.mean(d::ZI) = (1 - d.p0) * mean(d.dist)
Distributions.var(d::ZI) = (1 - d.p0) * var(d.dist) + d.p0 * (1 - d.p0) * mean(d.dist)^2
Distributions.params(d::ZI) = (d.p0, Distributions.params(d.dist)...)
numparams(::Type{<:ZI{DT}}) where {DT} = 1+numparams(DT)
numparams(::Type{<:ZI{FT, DT}}) where {FT, DT} = 1+numparams(DT)

reconstruct(d::ZI, ps::AbstractArray{<:Real}) = ZI(logistic(ps[1]), reconstruct(d.dist, @view ps[2:end]))

function get_bounds(::Type{ZI{DT}}) where DT
    lb, ub = get_bounds(DT)
    vcat(-20, lb), vcat(30, ub)
end

### Zero-inflated θ-dependent distributions

function Distributions.logpdf(zi::ZI, θ::Real, x::Real)
    l = logpdf(zi.dist, θ, x)
    if iszero(x)
        v1 = l+log(1-zi.p0)
        v2 = log(zi.p0)
        maxv = max(v1, v2)
        maxv + log(exp(v1-maxv) + exp(v2-maxv)) 
    else
        l+log(1-zi.p0)
    end
end

Distributions.pdf(d::ZI, θ::Real, x::Real) = exp(logpdf(d, θ, x))
Distributions.mean(d::ZI, θ::Real) = (1 - d.p0) * mean(d.dist, θ)
Distributions.var(d::ZI, θ::Real) = (1 - d.p0) * var(d.dist, θ) + d.p0 * (1 - d.p0) * mean(d.dist, θ)^2
get_th_param(d::ZI) = get_th_param(d.dist)

### Burst size/frequency helper functions ###

get_burst_frequency(d::NegativeBinomial) = d.r
get_burst_frequency(d::BetaPoisson) = d.σ_on
get_burst_frequency(d::NegativeBinomialTheta) = d.r
get_burst_frequency(d::BetaPoissonTheta) = d.σ_on
get_burst_frequency(d::ZI) = get_burst_frequency(d.dist)

get_burst_size(d::NegativeBinomial) = 1/d.p - 1
get_burst_size(d::BetaPoisson) = d.ρ / d.σ_off
get_burst_size(d::ZI) = get_burst_size(d.dist)
get_burst_size(d, theta::Nothing) = get_burst_size(d)

# We averarage the age-dependent burst size bs = ρ\σ_b as bs = sum of (ρ/σ_b * 2^(βθ)) * f(θ)) 
# over θ from θ₁ to θ₂, where θ₁ and θ₂ indicate the start and end θ of the specific phase (G1 or G2/M)
# and f(θ) is the phase distribution 

function get_phase_dist(theta::AbstractArray)
    thetas = sort(unique(theta))
    dist = last.(sort(collect(countmap(theta)), by = x -> x[1])) ./ length(theta)
    thetas, dist
end

function get_burst_size(d::NegativeBinomialTheta, theta::AbstractArray)
    thetas, dist = get_phase_dist(theta)
    d.b * sum( 2^(d.β * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

function get_burst_size(d::BetaPoissonTheta, theta::AbstractArray)
    thetas, dist = get_phase_dist(theta)
    d.ρ / d.σ_off * sum( 2^(d.β * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

function get_burst_size(::Type{<:BetaPoisson}, theta::AbstractArray, ρ, σ_off, β)
    # useful functional form for lower/upper CI bounds
    thetas, dist = get_phase_dist(theta)
    ρ / σ_off * sum( 2^(β * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

get_burst_size(d::ZI, theta::AbstractArray) = get_burst_size(d.dist, theta)

### Helper functions to determine the distribution type

isPoisson(T::Type{<:DiscreteUnivariateDistribution}) = T <: Poisson || T <: PoissonTheta
isPoisson(::T) where T <: DiscreteUnivariateDistribution = isPoisson(T)

isNB(T::Type{<:DiscreteUnivariateDistribution}) = T <: NegativeBinomial || T <: NegativeBinomialTheta
isNB(::T) where T <: DiscreteUnivariateDistribution = isNB(T)
isBP(T::Type{<:DiscreteUnivariateDistribution}) = T <: BetaPoisson || T <: BetaPoissonTheta
isBP(::T) where T <: DiscreteUnivariateDistribution  = isBP(T)

isZINB(::Type{<:ZI{FT, DT}}) where {FT<:Real, DT<:DiscreteUnivariateDistribution} = isNB(DT)
isZINB(::Type{<:ZI{DT}}) where DT<:DiscreteUnivariateDistribution = isNB(DT)
isZINB(::Type{<:DT}) where DT<:DiscreteUnivariateDistribution = false
isZINB(::DT) where DT<:DiscreteUnivariateDistribution = isZINB(DT)

isZIBP(::Type{<:ZI{FT, DT}}) where {FT<:Real, DT<:DiscreteUnivariateDistribution} = isBP(DT)
isZIBP(::Type{<:ZI{DT}}) where DT<:DiscreteUnivariateDistribution = isBP(DT)
isZIBP(::Type{<:DT}) where DT<:DiscreteUnivariateDistribution = false
isZIBP(::DT) where DT<:DiscreteUnivariateDistribution = isZIBP(DT)

is_th_dep(d::DiscreteUnivariateDistribution) = (d isa ZI && is_th_dep(d.dist)) || d isa PoissonTheta || d isa NegativeBinomialTheta || d isa BetaPoissonTheta