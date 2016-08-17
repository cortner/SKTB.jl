using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

at = Atoms("Al", repeatcell=(1,1,1), pbc=(false,false,false), cubic=true)
tbm = TightBinding.ToyModels.ToyTBModel(r0=2.5, rcut=8.0)

X = copy(positions(at)) |> mat
X[:, 2] += [0.123; 0.234; 0.01]
set_positions!(at, pts(X))


H, M = hamiltonian(tbm, at)
display( round(real(full(H)), 6) )
println()
display( round(real(full(M)), 6) )
println()



@show length(at)
println("============================================")
println("    TightBinding Tests  ")
println("============================================")
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
println("ok.")
print("check that `energy` evaluates ... ")
E = energy(tbm, at)
println("ok : E = ", E, ".")
print("check that `forces` evaluates ... ")
frc = forces(tbm, at) |> mat
# frc = real( TightBinding.forces_k(positions(at) |> mat, tbm, neighbourlist(at, TightBinding.cutoff(tbm)), [0.0;0.0;0.0]) )
println("ok : |f|∞ = ", vecnorm(frc, Inf), ".")

display(round(frc, 6))

# quit()


println("-------------------------------------------")
println("  Finite-difference test with ToyTBModel:  ")
println("-------------------------------------------")
fdtest(tbm, at, verbose=true)
println("-------------------------------------------")
println(" and now by hand: ")

X = positions(at) |> mat |> copy
X += 0.02 * rand(size(X))
set_positions!(at, X)
X = positions(at) |> mat |> copy
f = energy(tbm, at)
df = forces(tbm, at) |> mat

df_old = [0.14964512941122293 0.1298107651332044 -0.1434983825454178 -0.13595751199900952;
 0.15663708962680695 -0.17619277767202862 0.156645333211539 -0.13708964516631728;
 0.14840023680352574 -0.14410681488892718 -0.14463504760589468 0.14034162569129616]

@show vecnorm(df - df_old)

# df = df[:]
println("-----------------------------")
println("  p | error ")
println("----|------------------------")
for p = 2:12
   h = 0.1^p
   dfh = zeros(size(df))
   for n = 1:length(df)
      X[n] += h
      set_positions!(at, pts(X))
      dfh[n] = - (energy(tbm, at) - f) / h
      X[n] -= h
   end
   @printf(" %2d | %1.7e \n", p, vecnorm(dfh - df, Inf))
end
println("-----------------------------")
