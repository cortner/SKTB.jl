using LinearAlgebra
using JuLIP, JuLIP.Potentials, SKTB, SparseArrays
using JuLIP.Potentials: site_energy, site_energy_d
TB = SKTB

@testset "Virial" begin

# gamma point only
beta = 10.0        # temperature / smearing paramter: 10 to 50 for room temperature
DIM = (2,2,2)
# define the model
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(beta), bzquad = TB.GammaPoint() )
at = DIM * bulk(:Si, pbc=(true, true, true), cubic = true)
# JuLIP.rattle!(at, 0.02)
F = defm(at)
pert = zeros(Float64, 3, 3)
@show length(at)
println("Finite difference test for virial, with Gamma point only")
E = energy(tbm, at)
vir = virial(tbm, at)
errors = Float64[]
for p = 2:9
    h = 0.1^p
    dEh = zero(vir)
    for n = 1:length(F)
        pert[n] += h
        set_defm!(at, F+pert; updatepositions=true)
        Eh = energy(tbm, at)
        dEh[n] = (Eh - E) / h
        pert[n] -= h
    end
    push!(errors, norm(vir + dEh, Inf))
    println( " ", p, " | ", errors[end])
end
@test minimum(errors) < 1e-3 * maximum(errors)


# multiple k-points
nquad = 2
bzquad = TB.MPGrid(at, (nquad, nquad, nquad))
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(beta), bzquad = bzquad)
DIM = (2,2,2)
at = DIM * bulk(:Si, pbc=(true, true, true), cubic = true)
# JuLIP.rattle!(at, 0.02)
F = defm(at)
pert = zeros(Float64, 3, 3)
@show length(at)

println("Finite difference test for virial, with ", nquad, " points in each direction")
E = energy(tbm, at)
vir = virial(tbm, at)
errors = Float64[]
for p = 2:9
    h = 0.1^p
    dEh = zero(vir)
    for n = 1:length(F)
        pert[n] += h
        set_defm!(at, F+pert; updatepositions=true)
        Eh = energy(tbm, at)
        dEh[n] = (Eh - E) / h
        pert[n] -= h
    end
    push!(errors, norm(vir + dEh, Inf))
    println( " ", p, " | ", errors[end])
end
@test minimum(errors) < 1e-3 * maximum(errors)


end
