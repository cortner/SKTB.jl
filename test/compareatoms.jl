using JuLIP, TightBinding
using JLD

TB = TightBinding
at = bulk("Si", cubic = true)
rattle!(at, 0.02)
tbm = TB.NRLTB.NRLTBModel(elem = TB.NRLTB.Si_sp, nkpoints = (0,0,0))
H, M = hamiltonian(tbm, at)
H = full(H)
M = full(M)

# save to a file
X = positions(at) |> mat
bc = pbc(at)
C = cell(at)
save("/Users/ortner/temp1.jld", "X", X, "pbc", bc, "C", C)

# now ask Atoms.jl to compute H and M as well
run(`/Applications/Julia-0.4.7.app/Contents/Resources/julia/bin/julia /Users/ortner/gits/Atoms.jl/tbH.jl /Users/ortner/temp1.jld /Users/ortner/temp2.jld`)

# load the stuff
Ha, Ma = load("/Users/ortner/temp2.jld", "H", "M")

vecnorm(H - Ha, Inf)
vecnorm(M - Ma, Inf)

display(H)
display(Ha)
