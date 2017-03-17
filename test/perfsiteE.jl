addprocs(3)

@everywhere using JuLIP, JuLIP.ASE
@everywhere using TightBinding
# using ProfileView

# test parameters
beta = 20.0        # temperature / smearing paramter
                  # 10 to 50 for room temperature
n0 = 1            # site index where we compute the site energy
NQUAD = 8         # number of contour points
DIM = (7,7,7)


TB=TightBinding

# define the model
tbm = TB.NRLTB.NRLTBModel(elem = TB.NRLTB.Si_sp, nkpoints = (0,0,0))
calc = TB.Contour.ContourCalculator(tbm, NQUAD)
# now the real system to test on
at = DIM * bulk("Si", pbc=true, cubic=true)
JuLIP.rattle!(at, 0.02)
TB.Contour.calibrate2!(calc, at, beta, nkpoints=(6,6,6))
@show length(at)
Nat = length(at)
Isub = 1:(length(at) ÷ 3)


@time Es, dEs = TB.Contour.site_energy(calc, at, n0, true)
@time Es, dEs = TB.Contour.site_energy(calc, at, n0, true)
@time Es, dEs = TB.Contour.partial_energy(calc, at, Isub, true)
@time Es, dEs = TB.Contour.partial_energy(calc, at, Isub, true);


# MEMORY:
#   216 >>>   103,514,112   ~ 100MB
#  1000 >>>   479,232,000   ~ 500MB
#  2744 >>> 1,315,012,608   ~ 1.3GBP
#  5000 >>> 2,500,000,000   ~ 2.5 GB  (guessed)
# 10000 >>>  ~ 5GB   (ca 21^3 cell)
#    i.e. a 41^3 ~ 100k atoms cell will lead to ~ 17 GB, which is still OK!
#    on Apple Laopto: 16 GB  i.e. up to 10k atoms
#    on Kaiju we have 32,897,748k = 32GB   i.e. up to 100k atoms
#
# storage for neighbourlist:
#     4_096_000_000 atoms  >>> not a chance


# @code_warntype TB.Contour.site_energy(calc, at, n0, true);
# Es, dEs = TB.Contour.site_energy(calc, at, n0, true)
# ;


# TESTING HOW MUCH STORAGE WOULD BE REQUIRED TO KEEP THE RESIDUALS!
# PROBABLY TOO MUCH!
# for d in (3, 5, 7)
#    at = (d,d,d) * bulk("Si", pbc=(false,false,false), cubic=true)
#    Isub = 1:(length(at) ÷ 3)
#    @show length(at)
#    Es, dEs = TB.Contour.partial_energy(calc, at, Isub, true)
# end
# quit()




# cross-over between full and sparse seems to be
# around (1,10,10) in 2D and  smaller in 3D


# for n in (4,8,16)
#    @show n
#
#    # now go to the real system
#    at = (1,n,n) * bulk("Si", pbc=(true,true,true), cubic=false)
#    JuLIP.rattle!(at, 0.02)
#
#    H, M = TB.hamiltonian(tbm, at)
#    Hf = full(H); Mf = full(M)
#
#    rhs = rand(size(H,1), size(H,1) ÷ 3)
#
#    println("sparse")
#    @time LU=lufact(H - 0.1im * M)
#    @time LU=lufact(H - 0.1im * M)
#    @time LU \ rhs
#    @time LU \ rhs
#    println("full")
#    @time LU=lufact(Hf - 0.1im * Mf)
#    @time LU=lufact(Hf - 0.1im * Mf)
#    @time LU \ rhs
#    @time LU \ rhs
# end


# for n in (4,6,8)
#    @show n
#
#    # now go to the real system
#    at = (n,n,n) * bulk("Si", pbc=(true,true,true), cubic=false)
#    JuLIP.rattle!(at, 0.02)
#
#    H, M = TB.hamiltonian(tbm, at)
#    Hf = full(H); Mf = full(M)
#
#    rhs = rand(size(H,1))
#
#    println("sparse")
#    @time LU=lufact(H - 0.1im * M)
#    @time LU=lufact(H - 0.1im * M)
#    @time LU \ rhs
#    @time LU \ rhs
#    println("full")
#    @time LU=lufact(Hf - 0.1im * Mf)
#    @time LU=lufact(Hf - 0.1im * Mf)
#    @time LU \ rhs
#    @time LU \ rhs
# end





# # timing test
# println("timing with nquad = ", calc.nquad, "  (ca. 6 digits)")
# @time TB.Contour.site_energy(calc, at, n0)
# @time TB.Contour.site_energy(calc, at, n0)
# @show length(at) * tbm.norbitals
