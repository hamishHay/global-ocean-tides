using SparseArrays: sparse, spzeros, SparseMatrixCSC, SparseVector, dropzeros!, sparsevec
using BenchmarkTools
# using LinearAlgebra: 
mutable struct Moon
    Ω::Float64
    R::Float64 
    g::Float64 
    hₒ::Float64 
    α::Float64 
    nmax::Int64
    ρₒ::Float64
    βₙ::Array{ComplexF64}

    function Moon(Ω, R, g, hₒ, α, nmax)
        #Perform input checks here
        this = new(Ω, R, g, hₒ, α, nmax)
        this.ρₒ = 1e3
        this.βₙ = ones(nmax)
        return this
    end
end

mutable struct Forcing 
    ω::Float64
    n::Int64 
    m::Int64
    magnitude::ComplexF64 

    function Forcing(ω, n, m, magnitude)
        new(ω, n, m, magnitude)
    end
end 

mutable struct LTE
    moon::Moon
    forcing::Forcing
    nmax::Int

    # Constants defining the LTE problem
    ϵ::Float64
    b::Float64 
    λ::Float64
    Kₙ::Vector{ComplexF64}
    Lₙ::Vector{ComplexF64}
    pₙ::Vector{Float64}
    qₙ::Vector{Float64}

    lte_matrix::SparseMatrixCSC{ComplexF64}
    forcing_vec::Vector{ComplexF64}
    
    function LTE(moon::Moon, forcing::Forcing, nmax::Int)
        this = new(moon,
                   forcing,
                   nmax)

        # call functions that assign the constants
        # in the LTE problem, like lamb parameter,
        # kn, ln, etc.
        this.ϵ = 4moon.Ω^2 * moon.R^2 / (moon.g * moon.hₒ)
        this.b = moon.α / (2moon.Ω)
        this.λ = forcing.ω / (2moon.Ω)

        # βₙ = 1.0
        
        n = collect(1:1:nmax)
        m = forcing.m
        
        this.Lₙ = m ./ (n .* (n .+ 1)) .+ this.λ
        this.Kₙ = this.Lₙ .- moon.βₙ .* n .* (n .+ 1) ./ (this.ϵ * this.λ)
        this.pₙ = (n .+ 1) .* (n .+ m) ./ (n .* (2n .+ 1))
        this.qₙ = n .* (n .+ 1 .- m) ./ ( (n .+ 1) .* (2n .+ 1))
        if m>0
            this.pₙ[m] = 0.0
        end
        this.lte_matrix = spzeros(ComplexF64, 2nmax, 2nmax)
        get_lte_matrix(this)

        this.forcing_vec = zeros(2nmax)
        this.forcing_vec[2forcing.n] = forcing.magnitude / (2moon.Ω)
        return this
    end
    
end

function get_lte_matrix(container::LTE)
    nrows, ncols = size(container.lte_matrix)

    mat_temp = zeros(ComplexF64, nrows, ncols)
    for i in 1:2:nrows
        ni  = Int((i+1)/2)
        if i > 1
            mat_temp[i, i-1] = -container.qₙ[ni-1]
        end
        if i+3 <= nrows
            mat_temp[i,i+3] = -container.pₙ[ni+1]
        end

        j = i+1 
        if j-3 >= 1
            mat_temp[j, j-3] = container.qₙ[ni-1]
        end
        if j+1 <= nrows
            mat_temp[j, j+1] = container.pₙ[ni+1]
        end

        mat_temp[i, i] = container.b - container.Lₙ[ni]*im
        mat_temp[i+1, i+1] = container.b - container.Kₙ[ni]*im 
    end

    container.lte_matrix = sparse(mat_temp)
end

function get_ocean_diss(container::LTE, soln::Array{ComplexF64, 1})
    n = collect(1:1:container.nmax)
    m = container.forcing.m

    if n[1] - m >= 0
        Nnm = n .* (n .+ 1) ./ (2n .+ 1) .* factorial.(big.(n .+ m)) ./ factorial.(big.(n .- m))  
    else
        n0 = copy(n)
        n0[1] = 2
        Nnm = n .* (n .+ 1) ./ (2n .+ 1) .* factorial.(big.(n .+ m)) ./ factorial.(big.(n0 .- m))
        # Nnm[1] = 1
    end

    psi = soln[1:2:end]
    phi = soln[2:2:end]

    E =  2container.moon.ρₒ * abs(container.moon.hₒ) * container.moon.α * pi * sum(Nnm .* (abs.(phi).^2 + abs.(psi).^2 ))
    return E
end


h = 10 .^( range(0, stop=5, length=20001) )

nmax = 24

forcing = Forcing(5.31e-5, 2, 1, 1.0)
moon = Moon(5.31e-5, 252.1e3, 0.113, 100.0, 1e-5, nmax)
lte_container = LTE(moon, forcing, nmax)

S = zeros(ComplexF64, size(h)[1], 2nmax)
E = zeros(Float64, size(h)[1])

function runthis()
    Threads.@threads for i in 1:size(h)[1]
        moon = Moon(5.31e-5, 252.1e3, 0.113, h[i], 1e-7, nmax)

        lte_container = LTE(moon, forcing, nmax)
        
        S[i, :] .= lte_container.lte_matrix \ lte_container.forcing_vec 
        E[i] = get_ocean_diss(lte_container, S[i,:])
    end
    return S
end


S = runthis()

using PyPlot

fig = figure("test")
ax = PyPlot.axes()
p1 = loglog(h./1e3, E)

fig.canvas.draw()
show()
