


push!(LOAD_PATH, "/Users/ortner/gits/Atoms.jl")

using JuLIP, SKTB
import AtJuLIP
TB = SKTB



nkpoints = (0,0,0)
beta = 10.0

# compute a configuration
at = bulk("Si", cubic = true) * 2
rattle!(at, 0.02)

# JuLIP TB model
tbj = TB.NRLTB.NRLTBModel(:Si, FermiDiracSmearing(beta), orbitals = :sp, cutoff=:energyshift)

# JulIP matrices
Hj, Mj = hamiltonian(tbj, at)
Hj = Array(Hj)
Mj = Array(Mj)

# Atoms.jl TB Model
tba = AtJuLIP.JuLIPTB(:Si, nkpoints = nkpoints)

# Atoms Matrices
Ha, Ma = hamiltonian(tba, at)
Ha = TB.NRLTB.half_eV * Array(Ha)
Ma = Array(Ma)

@show norm(Hj - Ha, Inf)
@test norm(Hj - Ha, Inf) < 1e-12
@show norm(Mj - Ma, Inf)
@test norm(Mj - Ma, Inf) < 1e-12
ae = Array(Ha) |> eigvals |> real |> sort
je = Array(Hj) |> eigvals |> real |> sort
@show norm(ae - je, Inf)
