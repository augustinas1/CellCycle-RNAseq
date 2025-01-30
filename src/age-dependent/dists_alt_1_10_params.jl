# Stochastic model of bursty transcription in dividing cells -- age-dependent burst size and variable replication timing (Supplementary Text S3)
# transcription rate is proportional to cell age θ; 
# burst frequency in each cell-cycle phase (G1, early S, late S, G2/M) is {f₁, f₂, f₂, f₃}
# burst size in each cell-cycle phase is {ρ₁*exp(β₁θ), ρ₂*exp(β₂θ), ρ₂*exp(β₃θ), ρ₃*exp(β₄θ)}
# we consider 2 indepedent gene copies in G1, 1 copy in early and late S, and 4 copies in G2/M
# the transition from the early to late S phase is demarcated by cell age θₘ,
# and cell division occurs at cell age θ=1 

struct AltModel1{T} <: DiscreteUnivariateDistribution
    f₁::T
    f₂::T
    f₃::T
    ρ₁::T
    ρ₂::T
    ρ₃::T
    β₁::T
    β₂::T
    β₃::T
    β₄::T
    
    function AltModel1{T}(f₁::T, f₂::T, f₃::T, ρ₁::T, ρ₂::T, ρ₃::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
        return new{T}(f₁, f₂, f₃, ρ₁, ρ₂, ρ₃, β₁, β₂, β₃, β₄)
    end
end

function AltModel1(f₁::T, f₂::T, f₃::T, ρ₁::T, ρ₂::T, ρ₃::T, β₁::T, β₂::T, β₃::T, β₄::T) where {T<:Real}
    return AltModel1{T}(f₁, f₂, f₃, ρ₁, ρ₂, ρ₃, β₁, β₂, β₃, β₄)
end

using Distributions: params
Distributions.params(m::AltModel1) = (m.f₁, m.f₂, m.f₃, m.ρ₁, m.ρ₂, m.ρ₃, m.β₁, m.β₂, m.β₃, m.β₄)
numparams(::Type{<:AltModel1}) = 10

get_a_1_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -(((-1 + 4*exp(2*d*T))*T*(-((2*(-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁)) + 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T))))^2)/((2*(4*exp(2*d*T) + exp(2*θ*(d*T + β₁)) - exp(2*θ_S_i*(d*T + β₁)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*f₁*
        ρ₁^2)/(d*T + β₁) - (f₂*(d*exp(2*θₘ*(d*T + β₂))*T - d*exp(2*θ_S_i*(d*T + β₂))*T - d*exp(2*θₘ*(d*T + β₃))*T - 
         exp(2*θₘ*(d*T + β₃))*β₂ + exp(2*θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(2*θₘ*(d*T + β₂))*β₃ - 
         exp(2*θ_S_i*(d*T + β₂))*β₃)*ρ₂^2)/((d*T + β₂)*(d*T + β₃)) + 
      (8*(exp((1/2)*(3 + θ_S_f)*(d*T + β₄)) + exp((1/2)*(1 + 3*θ_S_f)*(d*T + β₄)))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃^2)/
       (d*T + β₄)))        

get_b_1_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(-1 + 4*exp(2*d*T))*((2*(-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) - 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T)))))/
      ((2*f₁*ρ₁*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁)) + 
          (-4*exp(2*d*T) - exp(2*θ*(d*T + β₁)) + exp(2*θ_S_i*(d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*ρ₁))/(d*T + β₁) + 
       (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*((exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*
           (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂)))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*
           β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) + 
          d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)) - exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃))) + 
            (exp(2*θₘ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*ρ₂))) - 
       (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
         (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄))        

get_a_2_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -(((-1 + 4*exp(2*d*T))*T*((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        ((exp(θ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*f₂*ρ₂)/(d*T + β₂) - 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T))))^2)/(-((8*exp(2*d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)))*f₁*ρ₁^2)/(d*T + β₁)) + 
      (f₂*(d*(exp(2*θ*(d*T + β₂)) - exp(2*θₘ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) + 
           exp(2*θₘ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)))*T + (exp(2*θₘ*(d*T + β₃)) - exp(2*θ_S_f*(d*T + β₃)))*β₂ + 
         (exp(2*θ*(d*T + β₂)) - exp(2*θₘ*(d*T + β₂)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) + 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*β₃)*
        ρ₂^2)/((d*T + β₂)*(d*T + β₃)) + (4*(-exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*f₃*ρ₃^2)/
       (d*T + β₄)))        

get_b_2_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(-1 + 4*exp(2*d*T))*((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        ((exp(θ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*f₂*ρ₂)/(d*T + β₂) - 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T)))))/((4*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 
          2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁))/(d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*
        (f₂*ρ₂*(β₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₂)) + exp(θₘ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
              2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + (-exp(2*θ*(d*T + β₂)) + exp(2*θₘ*(d*T + β₂)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) - 
              4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₂*
           (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) + 
          d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₂)) + exp(θₘ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - 
              exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃))) + (-exp(2*θ*(d*T + β₂)) + exp(2*θₘ*(d*T + β₂)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) - 
              4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*ρ₂))) - 
       (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
         (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄))        

get_a_3_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -(((-1 + 4*exp(2*d*T))*T*((4*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (f₂*(d*(2*exp(d*T*(1 + θₘ) + θₘ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*T + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*β₂ + 
           2*(exp(d*T*(1 + θₘ) + θₘ*β₂) - exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*β₃)*ρ₂)/((d*T + β₂)*(d*T + β₃)) + 
        (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))^2)/
     ((1 - 2*exp(d*T))^2*(-((8*exp(2*d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)))*f₁*ρ₁^2)/(d*T + β₁)) - 
       (f₂*(d*(4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) - exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 
            4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*T + (-exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 
            4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*β₂ + 
          4*(exp(2*d*T*(1 + θₘ) + 2*θₘ*β₂) - exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*β₃)*ρ₂^2)/((d*T + β₂)*(d*T + β₃)) + 
       (4*(-exp(2*(d*T + β₄)) + exp(2*θ_S_f*(d*T + β₄)))*f₃*ρ₃^2)/(d*T + β₄))))        

get_b_3_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(-1 + 4*exp(2*d*T))*((4*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (f₂*(d*(2*exp(d*T*(1 + θₘ) + θₘ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*T + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*β₂ + 2*(exp(d*T*(1 + θₘ) + θₘ*β₂) - exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*
            β₃)*ρ₂)/((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/
         (d*T + β₄)))/((-1 + 2*exp(d*T))*((4*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 
           2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁))/(d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*
         (f₂*ρ₂*(2*exp(d*T)*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*
            (exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*(exp(d*T*(1 + θₘ) + θₘ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₂) + 
           β₂*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 
               2*exp(d*T*(1 + θₘ) + θₘ*β₃)) + (-exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 
               4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*ρ₂) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T*(1 + θₘ) + θₘ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - 
               exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃)) + 
             (4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) - exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 
               4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*ρ₂))) - 
        (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
          (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄)))        

get_a_4_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        -(((-1 + 4*exp(2*d*T))*T*((2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (exp(d*T)*f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
           exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
         ((d*T + β₂)*(d*T + β₃)) + (2*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
           2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄))*f₃*ρ₃)/(d*T + β₄))^2)/
     ((1 - 2*exp(d*T))^2*(-((2*exp(2*d*T)*(-1 + exp(2*θ_S_i*(d*T + β₁)))*f₁*ρ₁^2)/(d*T + β₁)) - 
       (exp(2*d*T)*f₂*(d*exp(2*θₘ*(d*T + β₂))*T - d*exp(2*θ_S_i*(d*T + β₂))*T - d*exp(2*θₘ*(d*T + β₃))*T - 
          exp(2*θₘ*(d*T + β₃))*β₂ + exp(2*θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(2*θₘ*(d*T + β₂))*β₃ - 
          exp(2*θ_S_i*(d*T + β₂))*β₃)*ρ₂^2)/((d*T + β₂)*(d*T + β₃)) + 
       ((-exp(2*(d*T + β₄)) + exp(2*θ*(d*T + β₄)) - 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) + 4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*f₃*
         ρ₃^2)/(d*T + β₄))))        

get_b_4_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (exp(d*T*θ)*(-1 + 4*exp(2*d*T))*((2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (exp(d*T)*f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
           exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
         ((d*T + β₂)*(d*T + β₃)) + (2*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
           2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄))*f₃*ρ₃)/(d*T + β₄)))/
      ((-1 + 2*exp(d*T))*((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*
            ρ₁))/(d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*
          (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)) - exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*T + 
           exp(d*T)*(2*d*exp(d*T)*(exp(2*θₘ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*T*ρ₂ + 
             (exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
               2*(exp(d*T*(1 + θₘ) + θₘ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*
              β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*(exp(d*T*(1 + θₘ) + θₘ*β₃) + exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*ρ₂)))) + 
        (2*f₃*ρ₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
             2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄)) + (exp(2*(d*T + β₄)) - exp(2*θ*(d*T + β₄)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) - 
             4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₃))/(d*T + β₄)))        
    
AltModel1() = AltModel1{Float64}(ones(numparams(AltModel1))...)

NegativeBinomial(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real) =
    if θ <= θ_S_i
        NegativeBinomial(get_a_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f), 
                         get_b_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f))
    elseif θ_S_i < θ <= θₘ
        NegativeBinomial(get_a_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f), 
                         get_b_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f))
    elseif θₘ < θ <= θ_S_f
        NegativeBinomial(get_a_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f), 
                         get_b_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f))
    else
        NegativeBinomial(get_a_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f), 
                         get_b_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f))
    end

get_a_param(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_a_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θₘ
        get_a_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θₘ < θ <= θ_S_f
        get_a_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    else
        get_a_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    end

get_b_param(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_b_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θₘ
        get_b_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θₘ < θ <= θ_S_f
        get_b_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    else
        get_b_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    end

# statistics/evaluation require the cell-cycle phase input
Distributions.logpdf(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = logpdf(NegativeBinomial(m, θ, T, d, θₘ, θ_S_i, θ_S_f), k)
Distributions.pdf(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real, k::Real) = exp(logpdf(m, θ, T, d, θₘ, θ_S_i, θ_S_f, k))

# MLE functions

get_bounds(::Type{<:AltModel1}) = vcat(log.(fill(1e-4, 6)), fill(-100.0, 4)), vcat(log.(fill(1e3, 6)), fill(1e2, 4)) 
reconstruct(::AltModel1, ps::AbstractArray{<:Real}) = AltModel1(exp.(ps[1:6])..., ps[7:end]...)
get_CI_fns(::Type{<:AltModel1}) = vcat(fill(log, 6), fill(identity, 4)), vcat(fill(exp, 6), fill(identity, 4))

get_burst_frequency_G1(m::AltModel1) = m.f₁
get_burst_frequency_G2M(m::AltModel1) = m.f₃

# Burst size averaged over all cells in the G1 phase
function get_burst_size_G1(m::AltModel1, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₁ * sum( exp(m.β₁ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

# Burst size averaged over all cells in the G2/M phase
function get_burst_size_G2M(m::AltModel1, theta::AbstractArray) 
    thetas, dist = get_phase_dist(theta)
    m.ρ₃ * sum( exp(m.β₄ * thetas[i]) * dist[i] for i in eachindex(thetas) )
end

## --- Extra functions --- ##

get_m_1_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (T*((2*(-1 + exp(θ*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) - 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T)))))/exp(d*T*θ)      

get_m_2_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (T*((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        ((exp(θ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*f₂*ρ₂)/(d*T + β₂) - 
        ((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
          (f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
             exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
           ((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/(d*T + β₄))/
         (exp(d*T)*(-2 + exp((-d)*T)))))/exp(d*T*θ)    

get_m_3_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (T*((4*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (f₂*(d*(2*exp(d*T*(1 + θₘ) + θₘ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*T + (-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 
             2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃))*β₂ + 2*(exp(d*T*(1 + θₘ) + θₘ*β₂) - exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*
            β₃)*ρ₂)/((d*T + β₂)*(d*T + β₃)) + (4*(exp(d*T + β₄) - exp(θ_S_f*(d*T + β₄)))*f₃*ρ₃)/
         (d*T + β₄)))/(exp(d*T*θ)*(-1 + 2*exp(d*T)))        

get_m_4_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (2*T*((2*exp(d*T)*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁)/(d*T + β₁) + 
        (exp(d*T)*f₂*(d*exp(θₘ*(d*T + β₂))*T - d*exp(θ_S_i*(d*T + β₂))*T - d*exp(θₘ*(d*T + β₃))*T - exp(θₘ*(d*T + β₃))*β₂ + 
           exp(θ_S_f*(d*T + β₃))*(d*T + β₂) + exp(θₘ*(d*T + β₂))*β₃ - exp(θ_S_i*(d*T + β₂))*β₃)*ρ₂)/
         ((d*T + β₂)*(d*T + β₃)) + (2*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
           2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄))*f₃*ρ₃)/(d*T + β₄)))/(exp(d*T*θ)*(-1 + 2*exp(d*T)))        

get_v_1_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 4*exp(2*d*T)))*
        ((T*((2*f₁*ρ₁*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-2*exp(d*T) - exp(θ*(d*T + β₁)) + exp(θ_S_i*(d*T + β₁)) + 2*exp(d*T*(1 + θ) + θ*β₁)) + 
              (-4*exp(2*d*T) - exp(2*θ*(d*T + β₁)) + exp(2*θ_S_i*(d*T + β₁)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₁))*ρ₁))/(d*T + β₁) + 
           (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*((exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*
               (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₂)) + exp(θ_S_i*(d*T + β₂)))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*
               β₂*(exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) + 
              d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)) - exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃))) + 
                (exp(2*θₘ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*ρ₂))) - 
           (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
             (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄)))/exp(2*d*T*θ))        

get_v_2_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 4*exp(2*d*T)))*
        ((T*((4*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁))/
            (d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*
             (β₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₂)) + exp(θₘ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 
                  2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂)) + (-exp(2*θ*(d*T + β₂)) + exp(2*θₘ*(d*T + β₂)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) - 
                  4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₂*
               (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*ρ₂) + 
              d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₂)) + exp(θₘ*(d*T + β₂)) + 2*exp(d*T*(1 + θ) + θ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - 
                  exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃))) + (-exp(2*θ*(d*T + β₂)) + exp(2*θₘ*(d*T + β₂)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₂) - 
                  4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*ρ₂))) - 
           (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
             (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄)))/exp(2*d*T*θ))        

get_v_3_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 4*exp(2*d*T)))*
        ((T*((4*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁))/
            (d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*
             (2*exp(d*T)*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 
                2*(exp(d*T*(1 + θₘ) + θₘ*β₂) + exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₂) + 
              β₂*(exp(d*T*θ)*(1 + 2*exp(d*T))*(-exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 
                  2*exp(d*T*(1 + θₘ) + θₘ*β₃)) + (-exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 
                  4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*ρ₂) + d*T*(exp(d*T*θ)*(1 + 2*exp(d*T))*(2*exp(d*T*(1 + θₘ) + θₘ*β₂) - 2*exp(d*T*(1 + θ_S_i) + θ_S_i*β₂) - 
                  exp(θ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)) + 2*exp(d*T*(1 + θ) + θ*β₃) - 2*exp(d*T*(1 + θₘ) + θₘ*β₃)) + 
                (4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₂) - 4*exp(2*d*T*(1 + θ_S_i) + 2*θ_S_i*β₂) - exp(2*θ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)) + 
                  4*exp(2*d*T*(1 + θ) + 2*θ*β₃) - 4*exp(2*d*T*(1 + θₘ) + 2*θₘ*β₃))*ρ₂))) - 
           (8*exp((1/2)*(1 + θ_S_f)*(d*T + β₄))*sinh((1/2)*(-1 + θ_S_f)*(d*T + β₄))*f₃*ρ₃*
             (exp(d*T*θ)*(1 + 2*exp(d*T)) + (exp(d*T + β₄) + exp(θ_S_f*(d*T + β₄)))*ρ₃))/(d*T + β₄)))/exp(2*d*T*θ))        

get_v_4_alt1(f₁::dT1, f₂::dT1, f₃::dT1, ρ₁::dT1, ρ₂::dT1, ρ₃::dT1, β₁::dT1, β₂::dT1, β₃::dT1, β₄::dT1, 
        θ::dT2, T::dT2, d::dT2, θₘ::dT2, θ_S_i::dT2, θ_S_f::dT2) where {dT1<:Real, dT2<:Real} = 
        (1/(-1 + 4*exp(2*d*T)))*
        ((2*T*((2*(-1 + exp(θ_S_i*(d*T + β₁)))*f₁*ρ₁*(exp(d*T*(1 + θ)) + 2*exp(d*T*(2 + θ)) + 2*exp(2*d*T)*(1 + exp(θ_S_i*(d*T + β₁)))*ρ₁))/
            (d*T + β₁) + (1/((d*T + β₂)*(d*T + β₃)))*(f₂*ρ₂*
             (d*exp(d*T*(1 + θ))*(1 + 2*exp(d*T))*(exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)) - exp(θₘ*(d*T + β₃)) + exp(θ_S_f*(d*T + β₃)))*T + 
              exp(d*T)*(2*d*exp(d*T)*(exp(2*θₘ*(d*T + β₂)) - exp(2*θ_S_i*(d*T + β₂)) - exp(2*θₘ*(d*T + β₃)) + exp(2*θ_S_f*(d*T + β₃)))*T*ρ₂ + 
                (exp(θₘ*(d*T + β₂)) - exp(θ_S_i*(d*T + β₂)))*β₃*(exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*(exp(d*T*(1 + θₘ) + θₘ*β₂) + 
                    exp(d*T*(1 + θ_S_i) + θ_S_i*β₂))*ρ₂) - (exp(θₘ*(d*T + β₃)) - exp(θ_S_f*(d*T + β₃)))*β₂*
                 (exp(d*T*θ)*(1 + 2*exp(d*T)) + 2*(exp(d*T*(1 + θₘ) + θₘ*β₃) + exp(d*T*(1 + θ_S_f) + θ_S_f*β₃))*ρ₂)))) + 
           (2*f₃*ρ₃*(exp(d*T*θ)*(1 + 2*exp(d*T))*(exp(d*T + β₄) - exp(θ*(d*T + β₄)) + 2*exp(d*T*(1 + θ) + θ*β₄) - 
                2*exp(d*T*(1 + θ_S_f) + θ_S_f*β₄)) + (exp(2*(d*T + β₄)) - exp(2*θ*(d*T + β₄)) + 4*exp(2*d*T*(1 + θ) + 2*θ*β₄) - 
                4*exp(2*d*T*(1 + θ_S_f) + 2*θ_S_f*β₄))*ρ₃))/(d*T + β₄)))/exp(2*d*T*θ))        

Distributions.mean(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_m_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θₘ
        get_m_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θₘ < θ <= θ_S_f
        get_m_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    else
        get_m_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    end

Distributions.var(m::AltModel1, θ::Real, T::Real, d::Real, θₘ::Real, θ_S_i::Real, θ_S_f::Real) = 
    if θ <= θ_S_i
        get_v_1_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θ_S_i < θ <= θₘ
        get_v_2_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    elseif θₘ < θ <= θ_S_f
        get_v_3_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    else
        get_v_4_alt1(params(m)..., θ, T, d, θₘ, θ_S_i, θ_S_f)
    end

