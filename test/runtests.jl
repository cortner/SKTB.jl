using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

println("============================================")
println("    TightBinding Tests  ")
println("============================================")

# include("compareatoms.jl")

# include("testtoymodel.jl")
include("testnrltb.jl")
# include("testsiteE.jl")
# include("perfsiteE.jl")
# include("comparequip.jl")

# include("compareatoms.jl")


# using BenchmarkTools
#
# TB=TightBinding
# at = (1,2,2) * bulk("Si", pbc=false, cubic=true)
# β, eF, fixed_eF = 1.0, 0.0, true
# tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF), bzquad = TB.GammaPoint() )
# X = copy(positions(at)) |> mat
# X[:, 2] += [0.123; 0.234; 0.01]
# set_positions!(at, vecs(X))
# X = positions(at)
#
# R = X[2] - X[1]
# r = norm(R)
# bonds = zeros(4)
# dbonds = zeros(4)
# dH_nm = zeros(3, 4, 4)
# H_nm = zeros(4,4)
# A = zeros(ForwardDiff.Dual{3, Float64}, 4, 4)
# dA = zeros(16, 3)
# bondsad = zeros(ForwardDiff.Dual{3, Float64}, 4)
#
#
#
# b = rand(4)
# db = zeros(4)
#
# fb(r, i) = sin(r * i)
# dfb(r, i) = i * cos(r * i)
# Fb(r) = [fb(r, i) for i = 1:4]
# dFb(r) = [dfb(r, i) for i = 1:4]
#
# f(R) = TB._sk4!(R/norm(R), Fb(norm(R)), H_nm)[3,3]
#
# df(R) = TB._sk4_d!(R/norm(R), norm(R), Fb(norm(R)), dFb(norm(R)), dH_nm)[:,3,3]
#
# # adf(R) = ForwardDiff.jacobian(f, R)
#
# # function df(R)
# #    r = norm(R)
# #    U = R / r
# #    l, m, n = U[1], U[2], U[3]
# #    dl = [(1.0-l*l)/r ,     - l*m/r ,     - l*n/r]
# #    return 2*l * b[3] * dl - 2*l * b[4] * dl
# # end
#
# @show df(R)
#
# df0 = df(R)
# f0 = f(R)
# R = Vector(R)
#
# for p = 3:10
#    h = .1^p
#    dfh = zeros(3)
#    for i = 1:3
#       R[i] += h
#       dfh[i] = (f(R) - f0) / h
#       R[i] -= h
#    end
#    println(p, ": ", vecnorm(df0 - dfh, Inf))
# end
#
#
#
#
# # function test_hop_d(H, R, dH_nm, bonds, dbonds)
# #    r = norm(R)
# #    TB.hop_d!(H, r, bonds, dbonds)
# #    TB.sk_d!(H, r, R, bonds, dbonds, dH_nm)
# # end
# #
# # function test_hop_ad(H, R, dH_nm)   # , A, dA, bondsad
# #    TB.sk_ad!(H, R, TB.hop, dH_nm)
# # end
#
# # BENCHMARKING
# # test_hop_d(tbm.H, R, dH_nm, bonds, dbonds)
# # test_hop_ad(tbm.H, R, dH_nm, A, dA, bondsad)
# # println("Benchmark: manual derivatives")
# # display(@benchmark test_hop_d(tbm.H, R, dH_nm, bonds, dbonds))
# # println("Benchmark: ForwardDiff derivatives")
# # display(@benchmark test_hop_ad(tbm.H, R, dH_nm, A, dA, bondsad))
#
# # TYPE INSTABILITIES
# # r = norm(R)
# # @code_warntype TB.hop_d!(tbm.H, r, bonds, dbonds)
# # TB.hop_d!(tbm.H, r, bonds, dbonds)
# # @code_warntype TB._sk4_d!(R/r, r, bonds, dbonds, dH_nm)
#
# # using ForwardDiff
# # @code_warntype TB.sk_ad_test!(tbm.H, R, TB.hop, dH_nm, A, dA)
#
# # COMPARE THE TWO ARRAYS
# # test_hop_d(tbm.H, R, dH_nm, bonds, dbonds)
# # dH = copy(dH_nm)
# # test_hop_ad(tbm.H, R, dH_nm, A, dA, bondsad)
# # adH = copy(dH_nm)
# #
# # println("manual:")
# # display(dH)
# # println("auto:")
# # display(adH)
#
#
#
# # # COMPARE THE BOND CALCULATIONS: seems correct
# # TB.hop_d!(tbm.H, r, bonds, dbonds)
# # TB.hop!(tbm.H, r, bonds)
# # b0 = copy(bonds)
# # for p = 3:8
# #    h = .1^p
# #    TB.hop!(tbm.H, r+h, bonds)
# #    dbh = (bonds - b0) / h
# #    println(p, ": ", vecnorm(dbh - dbonds, Inf))
# # end
