# Stochastic model of bursty transcription in dividing cells -- age-dependent burst size and fixed replication timing (main text)
# transcription rate is proportional to cell age θ; 
# burst frequency in each cell-cycle phase (G1, S prior-replication, S post-replication, G2/M) is {f₁, f₁, f₂, f₂}
# burst size in each cell-cycle phase is {ρ₁*exp(β₁θ), ρ₁*exp(β₂θ), ρ₂*exp(β₃θ), ρ₂*exp(β₄θ)}
# gene doubling occurs at fixed cell age θᵣ and cell division at cell age θ=1

struct MainModel{T} <: DiscreteUnivariateDistribution
    f₁::T
    f₂::T
    ρ₁::T
    ρ₂::T
    β₁::T
    β₂::T
    β₃::T
    β₄::T
    
    function MainModel{T}(f₁::T, f₂::T, ρ₁::T, ρ₂::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
        return new{T}(f₁, f₂, ρ₁, ρ₂, β₁, β₂, β₃, β₄)
    end
end

function MainModel(f₁::T, f₂::T, ρ₁::T, ρ₂::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
    return MainModel{T}(f₁, f₂, ρ₁, ρ₂, β₁, β₂, β₃, β₄)
end

using Distributions: params
Distributions.params(m::MainModel) = (m.f₁, m.f₂, m.ρ₁, m.ρ₂, m.β₁, m.β₂, m.β₃, m.β₄)
numparams(::Type{<:MainModel}) = 8

get_a_1(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((2*(-1 + 4*exp(2*d*T))*T*(((-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (1/(-2 + exp((-d)*T)))*(((1/((d*T + β₁)*(d*T + β₂)))*(f₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + 
              exp(θ_S_i*(d*T + β₂))*β₁ - exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*
             ρ₁) + (1/((d*T + β₃)*(d*T + β₄)))*(2*f₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 
                exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + 
              (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄)*ρ₂))/exp(d*T)))^2)/
     ((1/((d*T + β₁)*(d*T + β₂)))*(f₁*(d*(4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 
           4*exp(2*d*T*(1 + θ) + 2*θ*β₁) - exp(2*θᵣ*(d*T + β₂)) + exp(2*θ_S_i*(d*T + β₂)))*T + 
         (-exp(2*θᵣ*(d*T + β₂)) + exp(2*θ_S_i*(d*T + β₂)))*β₁ + (4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 
           4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*β₂)*ρ₁^2) + (1/((d*T + β₃)*(d*T + β₄)))*
       (2*f₂*(d*exp(2*θᵣ*(d*T + β₃))*T - d*exp(2*θ_S_f*(d*T + β₃))*T - d*exp(2*(d*T + β₄))*T - exp(2*(d*T + β₄))*β₃ + 
         exp(2*θ_S_f*(d*T + β₄))*(d*T + β₃) + exp(2*θᵣ*(d*T + β₃))*β₄ - exp(2*θ_S_f*(d*T + β₃))*β₄)*ρ₂^2)))

get_b_1(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((exp(d*T*θ)*(1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        (((-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
         (1/(-2 + exp((-d)*T)))*(((f₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + exp(θ_S_i*(d*T + β₂))*β₁ - 
               exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*ρ₁)/
             ((d*T + β₁)*(d*T + β₂)) + (1/((d*T + β₃)*(d*T + β₄)))*
             (2*f₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + 
               (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄)*ρ₂))/
           exp(d*T))))/((-1 + 2*exp(d*T))*(f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*
          (β₂*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) + exp(θ*(d*T + β₁)) - exp(θ_S_i*(d*T + β₁)) - 2*exp(d*T*(1 + θ) + θ*β₁)) + 
             (4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*ρ₁) - 
           (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂)))*
              ρ₁) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) + exp(θ*(d*T + β₁)) - exp(θ_S_i*(d*T + β₁)) - 2*exp(d*T*(1 + θ) + θ*β₁) - 
               exp(θᵣ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂))) + (4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 
               4*exp(2*d*T*(1 + θ) + 2*θ*β₁) - exp(2*θᵣ*(d*T + β₂)) + exp(2*θ_S_i*(d*T + β₂)))*ρ₁)) + 
         2*f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*((exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*
            (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*
            β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₂) + 
           d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + 
             (exp(2*θᵣ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂)))))

get_a_2(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*(1 + 2*exp(d*T))*T*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((1/((d*T + β₁)*(d*T + β₂)))*(f₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + 
               exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + 
             (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 
             2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + (1/((d*T + β₃)*(d*T + β₄)))*
           (2*f₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
             exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂))^2)/
       ((1 - 2*exp(d*T))*(f₁*(d*(4*exp(2*d*T) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₁) + exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 
             4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*T + (exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 
             4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*β₁ - 4*exp(2*d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)))*β₂)*
          (d*T + β₃)*(d*T + β₄)*ρ₁^2 + 2*f₂*(d*T + β₁)*(d*T + β₂)*
          (d*exp(2*θᵣ*(d*T + β₃))*T - d*exp(2*θ_S_f*(d*T + β₃))*T - d*exp(2*(d*T + β₄))*T - exp(2*(d*T + β₄))*β₃ + 
           exp(2*θ_S_f*(d*T + β₄))*(d*T + β₃) + exp(2*θᵣ*(d*T + β₃))*β₄ - exp(2*θ_S_f*(d*T + β₃))*β₄)*ρ₂^2))

get_b_2(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((exp(d*T*θ)*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((1/((d*T + β₁)*(d*T + β₂)))*(f₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + 
              exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + 
            (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 
            2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + (1/((d*T + β₃)*(d*T + β₄)))*
          (2*f₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
            exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂)))/
       (f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*(-2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂*
           (exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
          β₁*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₂)) - exp(θᵣ*(d*T + β₂)) - 2*exp(d*T*(1 + θ) + θ*β₂) + 
              2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + (exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 
              4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₁) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) + exp(θ*(d*T + β₂)) - 
              exp(θᵣ*(d*T + β₂)) - 2*exp(d*T*(1 + θ) + θ*β₂) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + 
            (4*exp(2*d*T) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₁) + exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 
              4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₁)) + 2*f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
         ((exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*
             ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*
             ρ₂) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + 
            (exp(2*θᵣ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂))))

get_a_3(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (4*(1 + 2*exp(d*T))*T*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((1/((d*T + β₁)*(d*T + β₂)))*(exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
             (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + 
          (1/((d*T + β₃)*(d*T + β₄)))*(f₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
               2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*
              β₃ + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*
            ρ₂))^2)/((1 - 2*exp(d*T))*(-2*exp(2*d*T)*f₁*(d*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T + 
           (exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(2*θ_S_i*(d*T + β₁)))*β₂)*(d*T + β₃)*
          (d*T + β₄)*ρ₁^2 + f₂*(d*T + β₁)*(d*T + β₂)*
          (d*(exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃) - 
             exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*T + (-exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*β₃ + 
           (exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃))*β₄)*
          ρ₂^2))

get_b_3(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((exp(d*T*θ)*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((1/((d*T + β₁)*(d*T + β₂)))*(exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
            (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + 
         (1/((d*T + β₃)*(d*T + β₄)))*(f₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
              2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*
             β₃ + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*
           ρ₂)))/((-f₁)*(d*T + β₃)*(d*T + β₄)*ρ₁*
         (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
          exp(d*T)*(2*d*exp(d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T*ρ₁ + 
            (-1 + exp(θ_S_i*(d*T + β₁)))*β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
            (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
              2*(exp(d*T*(1 + θᵣ) + θᵣ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₁))) + f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
         (β₄*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
              2*exp(d*T*(1 + θᵣ) + θᵣ*β₃)) + (exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 
              4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃))*ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃*
           (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₂) + 
          d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
              exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + (exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 
              4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂))))

get_a_4(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        ((1 - 2*exp(d*T))*(1 + 2*exp(d*T))*T*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((2*exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
             (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
           ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + (1/(d*T + β₄))*(2*f₂*(exp(θ*(d*T + β₄)) + 
             (1/((-1 + 2*exp(d*T))*(d*T + β₃)))*(exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + d*exp(β₄)*T + 
                exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
                2*exp(θ_S_f*(d*T + β₃))*β₄)))*ρ₂))^2)/exp(2*d*T*θ)/
      ((-2*f₁*(d*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T + 
          (exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(2*θ_S_i*(d*T + β₁)))*β₂)*(d*T + β₃)*
         (d*T + β₄)*ρ₁^2)/exp(2*d*T*(-1 + θ)) + (f₂*(d*T + β₁)*(d*T + β₂)*
         (exp(2*θ*(d*T + β₄))*(d*T + β₃) + exp(2*d*T)*((-d)*(-4*exp(2*θᵣ*(d*T + β₃)) + 4*exp(2*θ_S_f*(d*T + β₃)) + exp(2*β₄) + 
              4*exp(2*θ*(d*T + β₄)) - 4*exp(2*θ_S_f*(d*T + β₄)))*T - (exp(2*β₄) + 4*exp(2*θ*(d*T + β₄)) - 4*exp(2*θ_S_f*(d*T + β₄)))*
             β₃ + 4*(exp(2*θᵣ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)))*β₄))*ρ₂^2)/exp(2*d*T*θ))

get_b_4(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)*
        ((2*exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
            (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
          ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + (1/(d*T + β₄))*
          (2*f₂*(exp(θ*(d*T + β₄)) + (1/((-1 + 2*exp(d*T))*(d*T + β₃)))*(exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + 
               d*exp(β₄)*T + exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
               2*exp(θ_S_f*(d*T + β₃))*β₄)))*ρ₂)))/
       (2*(-1 + 2*exp(d*T))*(f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*
          (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
           exp(d*T)*(2*d*exp(d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T*ρ₁ + 
             (-1 + exp(θ_S_i*(d*T + β₁)))*β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
             (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
               2*(exp(d*T*(1 + θᵣ) + θᵣ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₁))) + f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
          (-2*exp(d*T)*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
             2*(exp(d*T*(1 + θᵣ) + θᵣ*β₃) + exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*ρ₂) + 
           β₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄)) + 
             (exp(2*(d*T + β₄)) - exp(2*θ*(d*T + β₄)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) - 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₂) - 
           d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₄))*(1 - 2*exp(d*T)) + exp(d*T)*(2*exp(θᵣ*(d*T + β₃)) - 2*exp(θ_S_f*(d*T + β₃)) - exp(β₄) + 
                 2*exp(θ_S_f*(d*T + β₄)))) + (4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃) - 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₃) - exp(2*(d*T + β₄)) + 
               exp(2*θ*(d*T + β₄)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₂))))
    
MainModel() = MainModel{Float64}(ones(numparams(MainModel))...)

NegativeBinomial(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) =
    if θ <= θ_S_i
        NegativeBinomial(get_a_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    elseif θ_S_i < θ <= θᵣ
        NegativeBinomial(get_a_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    elseif θᵣ < θ <= θ_S_f
        NegativeBinomial(get_a_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    else
        NegativeBinomial(get_a_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    end

get_a_param(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_a_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_a_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_a_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_a_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

get_b_param(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_b_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_b_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_b_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_b_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

# statistics/evaluation require the cell-cycle phase input
Distributions.logpdf(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = logpdf(NegativeBinomial(m, θ, T, d, θᵣ, θ_S_i, θ_S_f), k)
Distributions.pdf(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = exp(logpdf(m, θ, T, d, θᵣ, θ_S_i, θ_S_f, k))

# MLE functions

get_bounds(::Type{<:MainModel}) = vcat(log.(fill(1e-3, 4)), fill(-100.0, 4)), vcat(log.(fill(1e3, 4)), fill(1e2, 4)) 
reconstruct(::MainModel, ps::AbstractArray{<:Real}) = MainModel(exp.(ps[1:4])..., ps[5:end]...)
get_CI_fns(::Type{<:MainModel}) = vcat(fill(log, 4), fill(identity, 4)), vcat(fill(exp, 4), fill(identity, 4))

get_burst_frequency_G1(m::MainModel) = m.f₁
get_burst_frequency_G2M(m::MainModel) = m.f₂

# Burst size averaged over all cells in the G1 phase
function get_burst_size_G1(m::MainModel, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₁ * sum( exp(m.β₁ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

# Burst size averaged over all cells in the G2/M phase
function get_burst_size_G2M(m::MainModel, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₂ * sum( exp(m.β₄ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

## --- Extra functions --- ##

get_m_1(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1,β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*(((-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (1/(-2 + exp((-d)*T)))*(((f₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + exp(θ_S_i*(d*T + β₂))*β₁ - 
              exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*ρ₁)/
            ((d*T + β₁)*(d*T + β₂)) + (1/((d*T + β₃)*(d*T + β₄)))*
            (2*f₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + 
              (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄)*ρ₂))/
          exp(d*T))))/exp(d*T*θ)

get_m_2(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 2*exp(d*T)))*
        ((2*T*((1/((d*T + β₁)*(d*T + β₂)))*(f₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + 
                exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + 
              (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 
              2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + (1/((d*T + β₃)*(d*T + β₄)))*
            (2*f₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
              exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂)))/exp(d*T*θ))

get_m_3(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 2*exp(d*T)))*((4*T*((exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((d*T + β₁)*(d*T + β₂)) + (1/((d*T + β₃)*(d*T + β₄)))*
      (f₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
          exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃ + 
        (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*ρ₂)))/
   exp(d*T*θ))

get_m_4(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*((2*exp(d*T)*f₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + (1/(d*T + β₄))*
      (2*f₂*(exp(θ*(d*T + β₄)) + (1/((-1 + 2*exp(d*T))*(d*T + β₃)))*(exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + 
           d*exp(β₄)*T + exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
           2*exp(θ_S_f*(d*T + β₃))*β₄)))*ρ₂)))/exp(d*T*θ)

get_v_1(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((2*(-1 + 2*exp(d*T))*T*(f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*
        (β₂*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) + exp(θ*(d*T + β₁)) - exp(θ_S_i*(d*T + β₁)) - 2*exp(d*T*(1 + θ) + θ*β₁)) + 
           (4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*ρ₁) - 
         (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂)))*
            ρ₁) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) + exp(θ*(d*T + β₁)) - exp(θ_S_i*(d*T + β₁)) - 2*exp(d*T*(1 + θ) + θ*β₁) - 
             exp(θᵣ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂))) + (4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 
             4*exp(2*d*T*(1 + θ) + 2*θ*β₁) - exp(2*θᵣ*(d*T + β₂)) + exp(2*θ_S_i*(d*T + β₂)))*ρ₁)) + 
       2*f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*((exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*
          (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*
          β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₂) + 
         d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + 
           (exp(2*θᵣ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂))))/exp(2*d*T*θ)/
    ((1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)))

get_v_2(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((2*(-1 + 2*exp(d*T))*T*(f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*(-2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂*
        (exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
       β₁*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₂)) - exp(θᵣ*(d*T + β₂)) - 2*exp(d*T*(1 + θ) + θ*β₂) + 
           2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + (exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 
           4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₁) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) + 
           exp(θ*(d*T + β₂)) - exp(θᵣ*(d*T + β₂)) - 2*exp(d*T*(1 + θ) + θ*β₂) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + 
         (4*exp(2*d*T) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₁) + exp(2*θ*(d*T + β₂)) - exp(2*θᵣ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 
           4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₁)) + 2*f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
      ((exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θᵣ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*
          ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
         (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₂) + 
       d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + 
         (exp(2*θᵣ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂))))/exp(2*d*T*θ)/
  ((1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)))

get_v_3(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((4*(-1 + 2*exp(d*T))*T*((-f₁)*(d*T + β₃)*(d*T + β₄)*ρ₁*
        (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
         exp(d*T)*(2*d*exp(d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T*ρ₁ + 
           (-1 + exp(θ_S_i*(d*T + β₁)))*β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
           (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*(exp(d*T*(1 + θᵣ) + θᵣ*β₂) + 
               exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₁))) + f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
        (β₄*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
             2*exp(d*T*(1 + θᵣ) + θᵣ*β₃)) + (exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 
             4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃))*ρ₂) - (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃*
          (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₂) + 
         d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
             exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄))) + (exp(2*θ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) + 
             4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃) - exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*ρ₂))))/exp(2*d*T*θ)/
    ((1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄)))

get_v_4(f₁::dT1, f₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (4*(-1 + 2*exp(d*T))*T*(f₁*(d*T + β₃)*(d*T + β₄)*ρ₁*
        (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
         exp(d*T)*(2*d*exp(d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)) + exp(2*θᵣ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)))*T*ρ₁ + 
           (-1 + exp(θ_S_i*(d*T + β₁)))*β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*exp(d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁) + 
           (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
             2*(exp(d*T*(1 + θᵣ) + θᵣ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₁))) + f₂*(d*T + β₁)*(d*T + β₂)*ρ₂*
        (-2*exp(d*T)*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
           2*(exp(d*T*(1 + θᵣ) + θᵣ*β₃) + exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*ρ₂) + 
         β₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
             2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄)) + (exp(2*(d*T + β₄)) - exp(2*θ*(d*T + β₄)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) - 
             4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₂) - d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θ*(d*T + β₄))*(1 - 2*exp(d*T)) + 
             exp(d*T)*(2*exp(θᵣ*(d*T + β₃)) - 2*exp(θ_S_f*(d*T + β₃)) - exp(β₄) + 2*exp(θ_S_f*(d*T + β₄)))) + 
           (4*exp(2*d*T*(1 + θᵣ) + 2*θᵣ*β₃) - 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₃) - exp(2*(d*T + β₄)) + exp(2*θ*(d*T + β₄)) - 
             4*exp(2*d*T*(1 + θ) + 2*θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₂))))/exp(2*d*T*θ)/
    ((1 - 2*exp(d*T))^2*(1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)*(d*T + β₃)*(d*T + β₄))

Distributions.mean(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_m_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_m_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_m_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_m_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

Distributions.var(m::MainModel, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_v_1(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_v_2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_v_3(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_v_4(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end
