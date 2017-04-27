


push!(LOAD_PATH, "/Users/ortner/gits/Atoms.jl")

using JuLIP, TightBinding
using JLD
import AtJuLIP
TB = TightBinding



nkpoints = (0,0,0)
beta = 10.0

# compute a configuration
at = bulk("Si", cubic = true) * 2
rattle!(at, 0.02)

# JuLIP TB model
tbj = TB.NRLTB.NRLTBModel(:Si, FermiDiracSmearing(beta), orbitals = :sp, cutoff=:energyshift)

# JulIP matrices
Hj, Mj = hamiltonian(tbj, at)
Hj = full(Hj)
Mj = full(Mj)

# Atoms.jl TB Model
tba = AtJuLIP.JuLIPTB(:Si, nkpoints = nkpoints)

# Atoms Matrices
Ha, Ma = hamiltonian(tba, at)
Ha = TB.NRLTB.half_eV * full(Ha)
Ma = full(Ma)

@show vecnorm(Hj - Ha, Inf)
@test vecnorm(Hj - Ha, Inf) < 1e-12
@show vecnorm(Mj - Ma, Inf)
@test vecnorm(Mj - Ma, Inf) < 1e-12
ae = full(Ha) |> eigvals |> real |> sort
je = full(Hj) |> eigvals |> real |> sort
@show vecnorm(ae - je, Inf)
