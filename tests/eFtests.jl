push!(LOAD_PATH, ".", "..")

# using PyPlot
using ASE
using MatSciPy
using PyCall
using TightBinding
import NRLTB
@pyimport ase


# number of cells each direction
# TODO: one has to revise the symbol of the vacancy if necll is changed
# 2 -> C63
# 3 -> C215
# 4 -> C511
# 5 -> C999
ncell = 5
# number of k-points each directiom
nkpoint = 4

println("\n Testing band structure of Carbon")

println("\n --- perfect system --- \n")

at = bulk("C"; cubic=true)
at = repeat(at, (ncell, ncell, ncell))
natom = length(at)
@printf(" number of atoms = %2d \n", length(at))

X = positions(at)


tbm = NRLTB.NRLTBModel(elem = NRLTB.C_sp)
set_pbc!(at, [true, true, true])
tbm.nkpoints = (nkpoint, nkpoint, nkpoint)

K, E = TightBinding.band_structure(at, tbm)

# take a very low temperature to approximate the fermi level at 0 temperature
tbm.fixed_eF = false
tbm.smearing.beta = 1000.0
TightBinding.update_eF!(at, tbm)
EF = tbm.eF

n = size(K,2)
kk = zeros(n)
for k = 1:n
    kk[k] = norm(K[:,k])
        @printf(" %1.7e  %1.7e  %1.7e  %1.7e \n", kk[k], E[1,k], E[2,k], E[3,k])
end

@printf(" Fermi level = %1.7e \n", EF)


println("\n --- system with a vacancy --- \n")

# construct aseAtom for system with a vacancy
Y = zeros(3, natom-1)
Y = X[:,1:natom-1]


at_vac = ASEAtoms( ase.Atoms("C999") )
set_cell!(at_vac, cell(at))
set_positions!(at_vac, Y)
set_pbc!(at_vac, [true, true, true])


X_vac = positions(at_vac)
@printf(" number of atoms (with vacancy) = %2d \n", length(at_vac))

K, E = TightBinding.band_structure(at_vac, tbm)

# take a very low temperature to approximate the fermi level at 0 temperature
# tbm.fixed_eF = false
# tbm.smearing.beta = 1000.0
TightBinding.update_eF!(at_vac, tbm)
EF = tbm.eF

n = size(K,2)
kk = zeros(n)
for k = 1:n
    kk[k] = norm(K[:,k])
        @printf(" %1.7e  %1.7e  %1.7e  %1.7e \n", kk[k], E[1,k], E[2,k], E[3,k])
end

@printf(" Fermi level (with vacancy) = %1.7e \n\n", EF)

