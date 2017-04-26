
push!(LOAD_PATH, "/Users/ortner/gits/Atoms.jl")

using JuLIP, TightBinding
using JLD
import AtJuLIP
TB = TightBinding

nkpoints = (0,0,0)

# compute a configuration
at = bulk("Si", cubic = true)
rattle!(at, 0.02)

# JuLIP TB model
tbj = TB.NRLTB.NRLTBModel(elem = TB.NRLTB.Si_sp, nkpoints = nkpoints)

# JulIP matrices
Hj, Mj = hamiltonian(tbj, at)
Hj = full(Hj)
Mj = full(Mj)

# Atoms.jl TB Model
tba = AtJuLIP.JuLIPTB(:Si, nkpoints = nkpoints)

# Atoms Matrices
Ha, Ma = hamiltonian(tba, at)
Ha = full(Ha)
Ma = full(Ma)

@show vecnorm(Hj - Ha, Inf)
@show vecnorm(Mj - Ma, Inf)
ae = full(Ha) |> eigvals |> real |> sort
je = full(Hj) |> eigvals |> real |> sort
@show vecnorm(ae - je, Inf)







# ==================================================================

# # OLD VERSION
# # ===========
#
# # save to a file
# X = positions(at) |> mat
# bc = pbc(at)
# C = cell(at)
# save("/Users/ortner/temp1.jld", "X", X, "pbc", bc, "C", C)
#
# # now ask Atoms.jl to compute H and M as well
# run(`/Applications/Julia-0.4.7.app/Contents/Resources/julia/bin/julia /Users/ortner/gits/Atoms.jl/tbH.jl /Users/ortner/temp1.jld /Users/ortner/temp2.jld`)
#
# # load the stuff
# Ha, Ma = load("/Users/ortner/temp2.jld", "H", "M")
#
# vecnorm(H - Ha, Inf)
# vecnorm(M - Ma, Inf)
#
# display(H)
# display(Ha)
