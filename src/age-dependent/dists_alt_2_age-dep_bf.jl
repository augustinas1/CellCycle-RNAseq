# Stochastic model of bursty transcription in dividing cells -- age-dependent burst frequency and fixed replication timing (Supplementary Text S4)
# transcription rate is proportional to cell age θ; 
# burst size in each cell-cycle phase (G1, S prior-replication, S post-replication, G2/M) is {b₁, b₁, b₂, b₂}
# burst frequency in each cell-cycle phase {ρ₁*exp(β₁θ), ρ₁*exp(β₂θ), ρ₂*exp(β₃θ), ρ₂*exp(β₄θ)}
# gene doubling occurs at fixed cell age θᵣ and cell division at cell age θ=1

struct AltModel2{T} <: DiscreteUnivariateDistribution
    b₁::T
    b₂::T
    ρ₁::T
    ρ₂::T
    β₁::T
    β₂::T
    β₃::T
    β₄::T
    
    function AltModel2{T}(b₁::T, b₂::T, ρ₁::T, ρ₂::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
        return new{T}(b₁, b₂, ρ₁, ρ₂, β₁, β₂, β₃, β₄)
    end
end

function AltModel2(b₁::T, b₂::T, ρ₁::T, ρ₂::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
    return AltModel2{T}(b₁, b₂, ρ₁, ρ₂, β₁, β₂, β₃, β₄)
end

using Distributions: params
Distributions.params(m::AltModel2) = (m.b₁, m.b₂, m.ρ₁, m.ρ₂, m.β₁, m.β₂, m.β₃, m.β₄)
numparams(::Type{<:AltModel2}) = 8

get_a_1_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -(((-1 + 4*exp(2*d*T))*T*(((-1 + exp(θ*(d*T + β₁)))*b₁*ρ₁)/(d*T + β₁) + 
        ((b₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + exp(θ_S_i*(d*T + β₂))*β₁ - 
             exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*ρ₁)/
           ((d*T + β₁)*(d*T + β₂)) + (2*b₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + 
               exp(θ_S_f*(d*T + β₄)))*T + (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + 
             (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄)*ρ₂)/((d*T + β₃)*(d*T + β₄)))/(exp(d*T)*(-2 + exp((-d)*T))))^2)/
     ((b₁^2*(2*d*(4*exp(2*d*T) + exp(θ*(2*d*T + β₁)) - exp(θ_S_i*(2*d*T + β₁)) - 4*exp(2*d*T*(1 + θ) + θ*β₁) - exp(θᵣ*(2*d*T + β₂)) + 
           exp(θ_S_i*(2*d*T + β₂)))*T + (-exp(θᵣ*(2*d*T + β₂)) + exp(θ_S_i*(2*d*T + β₂)))*β₁ + 
         (4*exp(2*d*T) + exp(θ*(2*d*T + β₁)) - exp(θ_S_i*(2*d*T + β₁)) - 4*exp(2*d*T*(1 + θ) + θ*β₁))*β₂)*ρ₁)/
       ((2*d*T + β₁)*(2*d*T + β₂)) + 
      (2*b₂^2*(2*d*(exp(θᵣ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*T + 
         (-exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*β₃ + (exp(θᵣ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)))*β₄)*
        ρ₂)/((2*d*T + β₃)*(2*d*T + β₄))))

get_b_1_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(((-1 + exp(θ*(d*T + β₁)))*b₁*ρ₁)/(d*T + β₁) + 
        ((b₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + exp(θ_S_i*(d*T + β₂))*β₁ - 
             exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*ρ₁)/
           ((d*T + β₁)*(d*T + β₂)) + (2*b₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + 
               exp(θ_S_f*(d*T + β₄)))*T + (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*
              β₄)*ρ₂)/((d*T + β₃)*(d*T + β₄)))/(exp(d*T)*(-2 + exp((-d)*T)))))/
      ((exp(d*T*θ)*b₁*(d*(-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁) + exp(θᵣ*(d*T + β₂)) - 
            exp(θ_S_i*(d*T + β₂)))*T + (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + 
          (-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁))*β₂)*ρ₁)/
        ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
       (2*b₁^2*(2*d*(-4*exp(2*d*T) - exp(θ*(2*d*T + β₁)) + exp(θ_S_i*(2*d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + θ*β₁) + exp(θᵣ*(2*d*T + β₂)) - 
            exp(θ_S_i*(2*d*T + β₂)))*T + (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + 
          (-4*exp(2*d*T) - exp(θ*(2*d*T + β₁)) + exp(θ_S_i*(2*d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + θ*β₁))*β₂)*ρ₁)/
        ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
       (2*b₂*((exp(d*T*θ)*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
             exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄))/
           ((d*T + β₃)*(d*T + β₄)) + (2*b₂*(-2*d*exp(θᵣ*(2*d*T + β₃))*T + 2*d*exp(θ_S_f*(2*d*T + β₃))*T + 2*d*exp(2*d*T + β₄)*T + 
             exp(2*d*T + β₄)*β₃ - exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) - exp(θᵣ*(2*d*T + β₃))*β₄ + 
             exp(θ_S_f*(2*d*T + β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*ρ₂)/(-1 + 2*exp(d*T)))

get_a_2_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        ((1 - 4*exp(2*d*T))^2*T*(2*d*T + β₁)*(2*d*T + β₂)*(2*d*T + β₃)*(2*d*T + β₄)*
        ((b₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
               2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
               2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
           ((d*T + β₁)*(d*T + β₂)) + (2*b₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + 
             exp(d*T + β₄)*β₃ - exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + 
             exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂)/((d*T + β₃)*(d*T + β₄)))^2)/
       ((1 - 2*exp(d*T))^2*(-((-1 + 4*exp(2*d*T))*b₁^2*(2*d*(4*exp(2*d*T) - 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₁) + exp(θ*(2*d*T + β₂)) - exp(θᵣ*(2*d*T + β₂)) - 
              4*exp(2*d*T*(1 + θ) + θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (exp(θ*(2*d*T + β₂)) - exp(θᵣ*(2*d*T + β₂)) - 
              4*exp(2*d*T*(1 + θ) + θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ - 4*exp(2*d*T)*(-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*
           (2*d*T + β₃)*(2*d*T + β₄)*ρ₁) + 2*(-1 + 2*exp(d*T))*(1 + 2*exp(d*T))*b₂^2*(2*d*T + β₁)*(2*d*T + β₂)*
          (-2*d*exp(θᵣ*(2*d*T + β₃))*T + 2*d*exp(θ_S_f*(2*d*T + β₃))*T + 2*d*exp(2*d*T + β₄)*T + exp(2*d*T + β₄)*β₃ - 
           exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) - exp(θᵣ*(2*d*T + β₃))*β₄ + exp(θ_S_f*(2*d*T + β₃))*β₄)*ρ₂))

get_b_2_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*((b₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
        2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
        2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
    ((d*T + β₁)*(d*T + β₂)) + (2*b₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + 
      exp(d*T + β₄)*β₃ - exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + 
      exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂)/((d*T + β₃)*(d*T + β₄))))/
 ((-1 + 2*exp(d*T))*((exp(d*T*θ)*b₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 
        2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 
        2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
    ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
   (2*b₁^2*(2*d*(-4*exp(2*d*T) + 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(2*d*T + β₂)) + exp(θᵣ*(2*d*T + β₂)) + 
        4*exp(2*d*T*(1 + θ) + θ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(2*d*T + β₂)) + exp(θᵣ*(2*d*T + β₂)) + 
        4*exp(2*d*T*(1 + θ) + θ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 4*exp(2*d*T)*(-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*
     ρ₁)/((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
   (2*b₂*((exp(d*T*θ)*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
         exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄))/
       ((d*T + β₃)*(d*T + β₄)) + (2*b₂*(-2*d*exp(θᵣ*(2*d*T + β₃))*T + 2*d*exp(θ_S_f*(2*d*T + β₃))*T + 
         2*d*exp(2*d*T + β₄)*T + exp(2*d*T + β₄)*β₃ - exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) - 
         exp(θᵣ*(2*d*T + β₃))*β₄ + exp(θ_S_f*(2*d*T + β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*
     ρ₂)/(-1 + 2*exp(d*T))))

get_a_3_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*(-1 + 4*exp(2*d*T))*T*((exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((d*T + β₁)*(d*T + β₂)) + (b₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
          2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*
         β₃ + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*
       ρ₂)/((d*T + β₃)*(d*T + β₄)))^2)/
  ((1 - 2*exp(d*T))^2*((2*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
       (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
     ((2*d*T + β₁)*(2*d*T + β₂)) - 
    (b₂^2*(2*d*(exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - 
         exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*T + (-exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*β₃ + 
       (exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*
      ρ₂)/((2*d*T + β₃)*(2*d*T + β₄))))

get_b_3_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*((exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((d*T + β₁)*(d*T + β₂)) + 
     (b₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
          exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃ + 
        (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*ρ₂)/
      ((d*T + β₃)*(d*T + β₄))))/
   ((-1 + 2*exp(d*T))*((exp(d*T*(1 + θ))*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (4*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
        (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
     (1/(-1 + 2*exp(d*T)))*(b₂*(-((exp(d*T*θ)*(d*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
              2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*
             β₃ + (exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*
             β₄))/((d*T + β₃)*(d*T + β₄))) - 
        (2*b₂*(2*d*(exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - 
             exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*T + (-exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*β₃ + 
           (exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃))*β₄))/
         ((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*ρ₂)))

get_a_4_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -((2*T*((2*exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (2*b₂*(exp(θ*(d*T + β₄)) + (exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + d*exp(β₄)*T + 
           exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
           2*exp(θ_S_f*(d*T + β₃))*β₄))/((-1 + 2*exp(d*T))*(d*T + β₃)))*ρ₂)/(d*T + β₄))^2)/
  (-((8*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
       (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
     ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂))) + 
   (4*b₂^2*(-exp(θ*(2*d*T + β₄)) + (exp(2*d*T)*(8*d*exp(θᵣ*(2*d*T + β₃))*T - 8*d*exp(θ_S_f*(2*d*T + β₃))*T - 2*d*exp(β₄)*T - 
         exp(β₄)*β₃ + 4*exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) + 4*exp(θᵣ*(2*d*T + β₃))*β₄ - 
         4*exp(θ_S_f*(2*d*T + β₃))*β₄))/((-1 + 4*exp(2*d*T))*(2*d*T + β₃)))*ρ₂)/(2*d*T + β₄)))

get_b_4_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*((2*exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (2*b₂*(exp(θ*(d*T + β₄)) + (exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + d*exp(β₄)*T + 
           exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
           2*exp(θ_S_f*(d*T + β₃))*β₄))/((-1 + 2*exp(d*T))*(d*T + β₃)))*ρ₂)/(d*T + β₄)))/
   (2*((exp(d*T*(1 + θ))*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (4*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
        (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
     (1/(-1 + 2*exp(d*T)))*(b₂*(-((exp(d*T*θ)*(d*(2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₃) - exp(d*T + β₄) + 
              2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄) + exp(θ*(d*T + β₄))*(1 - 2*exp(d*T)))*T + (-exp(d*T + β₄) + exp(θ*(d*T + β₄)) - 
              2*exp(d*T*(1 + θ) + θ*β₄) + 2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄))*β₃ + 2*(exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
              exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*β₄))/((d*T + β₃)*(d*T + β₄))) - 
        (2*b₂*(2*d*(4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₃) - exp(2*d*T + β₄) + exp(θ*(2*d*T + β₄)) - 
             4*exp(2*d*T*(1 + θ) + θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₄))*T + (-exp(2*d*T + β₄) + exp(θ*(2*d*T + β₄)) - 
             4*exp(2*d*T*(1 + θ) + θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₄))*β₃ + 
           4*(exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*
       ρ₂)))
    
AltModel2() = AltModel2{Float64}(ones(numparams(AltModel2))...)

NegativeBinomial(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) =
    if θ <= θ_S_i
        NegativeBinomial(get_a_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    elseif θ_S_i < θ <= θᵣ
        NegativeBinomial(get_a_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    elseif θᵣ < θ <= θ_S_f
        NegativeBinomial(get_a_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    else
        NegativeBinomial(get_a_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f), 
                         get_b_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f))
    end

get_a_param(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_a_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_a_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_a_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_a_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

get_b_param(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_b_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_b_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_b_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_b_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

# statistics/evaluation require the cell-cycle phase input
Distributions.logpdf(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = logpdf(NegativeBinomial(m, θ, T, d, θᵣ, θ_S_i, θ_S_f), k)
Distributions.pdf(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = exp(logpdf(m, θ, T, d, θᵣ, θ_S_i, θ_S_f, k))

# MLE functions

get_bounds(::Type{<:AltModel2}) = vcat(log.(fill(1e-3, 4)), fill(-100.0, 4)), vcat(log.(fill(1e3, 4)), fill(1e2, 4)) 
reconstruct(::AltModel2, ps::AbstractArray{<:Real}) = AltModel2(exp.(ps[1:4])..., ps[5:end]...)
get_CI_fns(::Type{<:AltModel2}) = vcat(fill(log, 4), fill(identity, 4)), vcat(fill(exp, 4), fill(identity, 4))

get_burst_size_G1(m::AltModel2) = m.b₁
get_burst_size_G2M(m::AltModel2) = m.b₂

# Burst frequency averaged over all cells in the G1 phase
function get_burst_frequency_G1(m::AltModel2, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₁ * sum( exp(m.β₁ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

# Burst frequency averaged over all cells in the G2/M phase
function get_burst_frequency_G2M(m::AltModel2, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₂ * sum( exp(m.β₄ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

## --- Extra functions --- ##

get_m_1_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1,β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*(((-1 + exp(θ*(d*T + β₁)))*b₁*ρ₁)/(d*T + β₁) + 
        (1/(-2 + exp((-d)*T)))*(((b₁*(d*T - d*exp(θ_S_i*(d*T + β₁))*T + d*exp(θ_S_i*(d*T + β₂))*T + exp(θ_S_i*(d*T + β₂))*β₁ - 
              exp(θᵣ*(d*T + β₂))*(d*T + β₁) + β₂ - exp(θ_S_i*(d*T + β₁))*β₂)*ρ₁)/
            ((d*T + β₁)*(d*T + β₂)) + (1/((d*T + β₃)*(d*T + β₄)))*
            (2*b₂*(d*(exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + 
              (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*β₃ + (exp(θᵣ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₄)*ρ₂))/
          exp(d*T))))/exp(d*T*θ)

get_m_2_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 2*exp(d*T)))*
        ((2*T*((1/((d*T + β₁)*(d*T + β₂)))*(b₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + 
                exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + 
              (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 
              2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁) + (1/((d*T + β₃)*(d*T + β₄)))*
            (2*b₂*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - 
              exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + exp(θ_S_f*(d*T + β₃))*β₄)*ρ₂)))/exp(d*T*θ))

get_m_3_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 2*exp(d*T)))*((4*T*((exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((d*T + β₁)*(d*T + β₂)) + (1/((d*T + β₃)*(d*T + β₄)))*
      (b₂*((-d)*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
          exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*β₃ + 
        (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*β₄)*ρ₂)))/
   exp(d*T*θ))

get_m_4_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*((2*exp(d*T)*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + (1/(d*T + β₄))*
      (2*b₂*(exp(θ*(d*T + β₄)) + (1/((-1 + 2*exp(d*T))*(d*T + β₃)))*(exp(d*T)*(-2*d*exp(θᵣ*(d*T + β₃))*T + 2*d*exp(θ_S_f*(d*T + β₃))*T + 
           d*exp(β₄)*T + exp(β₄)*β₃ - 2*exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - 2*exp(θᵣ*(d*T + β₃))*β₄ + 
           2*exp(θ_S_f*(d*T + β₃))*β₄)))*ρ₂)))/exp(d*T*θ)

get_v_1_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*((exp(d*T*θ)*b₁*(d*(-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁) + exp(θᵣ*(d*T + β₂)) - 
        exp(θ_S_i*(d*T + β₂)))*T + (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + 
      (-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁))*β₂)*ρ₁)/
    ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
   (2*b₁^2*(2*d*(-4*exp(2*d*T) - exp(θ*(2*d*T + β₁)) + exp(θ_S_i*(2*d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + θ*β₁) + exp(θᵣ*(2*d*T + β₂)) - 
        exp(θ_S_i*(2*d*T + β₂)))*T + (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + 
      (-4*exp(2*d*T) - exp(θ*(2*d*T + β₁)) + exp(θ_S_i*(2*d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + θ*β₁))*β₂)*ρ₁)/
    ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
   (1/(-1 + 2*exp(d*T)))*(2*b₂*((1/((d*T + β₃)*(d*T + β₄)))*(exp(d*T*θ)*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + 
         d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + 
         exp(θ_S_f*(d*T + β₃))*β₄)) + (2*b₂*(-2*d*exp(θᵣ*(2*d*T + β₃))*T + 2*d*exp(θ_S_f*(2*d*T + β₃))*T + 
         2*d*exp(2*d*T + β₄)*T + exp(2*d*T + β₄)*β₃ - exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) - 
         exp(θᵣ*(2*d*T + β₃))*β₄ + exp(θ_S_f*(2*d*T + β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*
     ρ₂)))/exp(2*d*T*θ)

get_v_2_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*((exp(d*T*θ)*b₁*(d*(-2*exp(d*T) + 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 
        2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(d*T + β₂)) + exp(θᵣ*(d*T + β₂)) + 
        2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
    ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
   (2*b₁^2*(2*d*(-4*exp(2*d*T) + 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₁) - exp(θ*(2*d*T + β₂)) + exp(θᵣ*(2*d*T + β₂)) + 
        4*exp(2*d*T*(1 + θ) + θ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*T + (-exp(θ*(2*d*T + β₂)) + exp(θᵣ*(2*d*T + β₂)) + 
        4*exp(2*d*T*(1 + θ) + θ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₁ + 4*exp(2*d*T)*(-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*
     ρ₁)/((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
   (1/(-1 + 2*exp(d*T)))*(2*b₂*((1/((d*T + β₃)*(d*T + β₄)))*(exp(d*T*θ)*((-d)*exp(θᵣ*(d*T + β₃))*T + d*exp(θ_S_f*(d*T + β₃))*T + 
         d*exp(d*T + β₄)*T + exp(d*T + β₄)*β₃ - exp(θ_S_f*(d*T + β₄))*(d*T + β₃) - exp(θᵣ*(d*T + β₃))*β₄ + 
         exp(θ_S_f*(d*T + β₃))*β₄)) + (2*b₂*(-2*d*exp(θᵣ*(2*d*T + β₃))*T + 2*d*exp(θ_S_f*(2*d*T + β₃))*T + 
         2*d*exp(2*d*T + β₄)*T + exp(2*d*T + β₄)*β₃ - exp(θ_S_f*(2*d*T + β₄))*(2*d*T + β₃) - 
         exp(θᵣ*(2*d*T + β₃))*β₄ + exp(θ_S_f*(2*d*T + β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*
     ρ₂)))/exp(2*d*T*θ)

get_v_3_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (4*T*((exp(d*T*(1 + θ))*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (4*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
        (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
     (1/(-1 + 2*exp(d*T)))*(b₂*(-((exp(d*T*θ)*(d*(exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 
              2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*T + (-exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*
             β₃ + (exp(θ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)) - 2*exp(d*T*(1 + θ) + θ*β₃) + 2*exp(d*T*(1 + θᵣ) + θᵣ*β₃))*
             β₄))/((d*T + β₃)*(d*T + β₄))) - 
        (2*b₂*(2*d*(exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - 
             exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*T + (-exp(2*d*T + β₄) + exp(θ_S_f*(2*d*T + β₄)))*β₃ + 
           (exp(θ*(2*d*T + β₃)) - exp(θ_S_f*(2*d*T + β₃)) - 4*exp(2*d*T*(1 + θ) + θ*β₃) + 4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃))*β₄))/
         ((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*ρ₂)))/exp(2*d*T*θ)

get_v_4_alt2(b₁::dT1, b₂::dT1, ρ₁::dT1, ρ₂::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θᵣ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (4*T*((exp(d*T*(1 + θ))*b₁*(d*(-1 + exp(θ_S_i*(d*T + β₁)) + exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*T + 
        (exp(θᵣ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 2*exp(d*T))*(d*T + β₁)*(d*T + β₂)) + 
     (4*exp(2*d*T)*b₁^2*(2*d*(-1 + exp(θ_S_i*(2*d*T + β₁)) + exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*T + 
        (exp(θᵣ*(2*d*T + β₂)) - exp(θ_S_i*(2*d*T + β₂)))*β₁ + (-1 + exp(θ_S_i*(2*d*T + β₁)))*β₂)*ρ₁)/
      ((-1 + 4*exp(2*d*T))*(2*d*T + β₁)*(2*d*T + β₂)) + 
     (1/(-1 + 2*exp(d*T)))*(b₂*(-((exp(d*T*θ)*(d*(2*exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₃) - exp(d*T + β₄) + 
              2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄) + exp(θ*(d*T + β₄))*(1 - 2*exp(d*T)))*T + (-exp(d*T + β₄) + exp(θ*(d*T + β₄)) - 
              2*exp(d*T*(1 + θ) + θ*β₄) + 2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄))*β₃ + 2*(exp(d*T*(1 + θᵣ) + θᵣ*β₃) - 
              exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*β₄))/((d*T + β₃)*(d*T + β₄))) - 
        (2*b₂*(2*d*(4*exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₃) - exp(2*d*T + β₄) + exp(θ*(2*d*T + β₄)) - 
             4*exp(2*d*T*(1 + θ) + θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₄))*T + (-exp(2*d*T + β₄) + exp(θ*(2*d*T + β₄)) - 
             4*exp(2*d*T*(1 + θ) + θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₄))*β₃ + 
           4*(exp(2*d*T*(1 + θᵣ) + θᵣ*β₃) - exp(2*d*T*(1 + θ_S_f) + θ_S_f*β₃))*β₄))/((1 + 2*exp(d*T))*(2*d*T + β₃)*(2*d*T + β₄)))*
       ρ₂)))/exp(2*d*T*θ)

Distributions.mean(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_m_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_m_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_m_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_m_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

Distributions.var(m::AltModel2, θ::Real, T::Real, d::Real, θᵣ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_v_1_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θᵣ
        get_v_2_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    elseif θᵣ < θ <= θ_S_f
        get_v_3_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    else
        get_v_4_alt2(params(m)..., θ, T, d, θᵣ, θ_S_i, θ_S_f)
    end

