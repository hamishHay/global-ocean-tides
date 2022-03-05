# Load the numerical LTE solver


module dyn_LN_functions

using LTE_solver

export get_LN2mq_analytical, get_LN2mq_numerical, get_Hres2mq_analytical

# Function to calculate dynamical Love numbers analytically
# Inputs: m = harmonic order
#         R = Radius 
#         g = Surface gravity 
#         Ω = Rotation rate 
#         ω = Forcing Frequency
#         Hₒ = Ocean thickness 
#         alpha = Ocean drag coefficient 
#         LNt = Tidal Love number at the surface
#         LNp = Pressure Love number at the surface
#         ups = Upsilon (see just after Eq. 13) 
#         beta = beta (see just after Eq. 13)
# Output: LN = Dynamic Love number
function get_LN2mq_analytical(m, R, g, Ω, ω, Hₒ, alpha, LNt, LNp, ups, beta, deg=2)    
    λ = ω./2Ω
    ϵ = 4Ω^2 .* R.^2 ./ (g .* Hₒ)
    fmag = 1.0

    if m == 2
        n = 2.0
        K2q = λ .+ m/ (n*(n+1.)) .- beta .* n*(n+1) ./ (ϵ.*λ) .+ alpha ./ (2Ω) *1im
        q2 = n*(n+1-m) / ((n+1)*(2*n+1))

        n = 3.0
        L3q = λ .+ m/ (n*(n+1.)) .+ alpha ./ (2Ω)*1im
        p3 = (n+1)*(n+m) / (n*(2n+1))

        # Eq A5
        Phi = ups ./ (2Ω) * fmag *1im  ./ (K2q .* (1 .- p3*q2./(K2q .* L3q)))
    
    elseif m == 0
        n = 2.0
        K2q = λ .+ m/ (n*(n+1.)) .- beta .* n*(n+1) ./ (ϵ.*λ) .+ alpha ./ (2Ω) *1im
        q2 = n*(n+1-m) / ((n+1)*(2n+1))
        p2 = (n+1)*(n+m) / (n*(2n+1))

        n = 3.0
        L3q = λ .+ m/ (n*(n+1.)) .+ alpha ./ (2Ω)*1im
        p3 = (n+1)*(n+m) / (n*(2n+1))

        n = 1.0
        L1q = λ .+ m/ (n*(n+1.)) .+ alpha ./ (2Ω)*1im
        q1 = n*(n+1-m) / ((n+1)*(2n+1))

        # Eq A10
        Phi = ups ./ (2Ω) * fmag *1im  ./ (K2q .* (1 .- p3*q2./(K2q.*L3q) .- p2*q1./(K2q.*L1q)))
    end

    # Eq 17
    eta = deg*(deg + 1) ./ R.^2 .* Hₒ ./ ω * 1im .* Phi  

    # Eq 6/7
    LN = LNt .+ LNp .* (beta .* g .* eta ./ fmag .- ups)
    
    return LN 

end

# Function to calculate dynamical Love numbers numerically, by solving the LTE
# Inputs: m = harmonic order
#         R = Radius 
#         g = Surface gravity 
#         Ω = Rotation rate 
#         ω = Forcing Frequency
#         Hₒ = Ocean thickness 
#         alpha = Ocean drag coefficient 
#         LNt = Tidal Love number at the surface
#         LNp = Pressure Love number at the surface
#         ups = Upsilon (see just after Eq. 13) 
#         beta = beta (see just after Eq. 13), must have length=nmax
#         nmax = maximum harmonic degree (higher is more accurate)
# Output: LN = Dynamic Love number
function get_LN2mq_numerical(m, R, g, Ω, ω, Hₒ, alpha, LNt, LNp, ups, beta; nmax=8)
    moon = Moon(Ω, R, g, Hₒ, alpha, nmax)   
    moon.βₙ[:] .= beta[:]
    
    fmag = 1.0*ups

    forcing = Forcing(ω, 2, m, fmag)
    lte_container = LTE(moon, forcing, nmax)

    # Get solution to LTE
    S = solve_LTE(lte_container) 

    # Get h22 using solution, S, computed above
    LN = get_LN2mq(lte_container, S, 2, LNt, LNp, ups, beta[2])

    return LN
end

# Function to calculate resonant ocean thickness, analytically
# Inputs: m = harmonic order
#         R = Radius 
#         g = Surface gravity 
#         Ω = Rotation rate 
#         ω = Forcing Frequency
#         beta = beta (see just after Eq. 13) at degree n=2
# Output: Hres = Resonant ocean thickness
function get_Hres2mq_analytical( m, R, g, Ω, ω, beta=1.0+0im, deg=2)
    λ = ω ./ 2Ω
    
    if m==2
        n = 3.0
        L3q = λ .+ m/ (n*(n+1.)) #+ alpha / (2Ω) *1im --> ignore imaginary terms
        p3 = (n+1)*(n+m) / (n*(2n+1))
        
        n = 2.0
        q2 = n*(n+1-m) / ((n+1)*(2*n+1))

        # Eq. A6
        Hres = 4Ω^2 .* R.^2 .* λ ./ (g .* real(beta) .* (deg*(deg+1))) .* (λ .+ m/(deg*(deg+1)) .- p3*q2 ./ L3q)
    elseif m==0
        n = 3.0
        L3q = λ #+ alpha / (2Ω) *1im --> ignore imaginary terms
        p3 = (n+1)*(n+m) / (n*(2n+1))

        n = 2.0
        q2 = n*(n+1-m) / ((n+1)*(2*n+1))
        p2 = (n+1)*(n+m) / (n*(2n+1))

        n = 1.0
        L1q = λ #+ alpha / (2Ω) *1im --> ignore imaginary terms
        q1 = n*(n+1-m) / ((n+1)*(2*n+1))

        # Eq. A11
        Hres = 4Ω^2 .* R.^2 .* λ ./ (g .* real(beta) .* (deg*(deg+1))) .* (λ .- p3*q2 ./ L3q .- q1*p2 ./ L1q)
    end

    return Hres 
end


end