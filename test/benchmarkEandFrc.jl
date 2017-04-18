import TightBinding
using JuLIP, JuLIP.ASE

TB=TightBinding
β, eF, fixed_eF = 1.0, 0.0, true

tests = Any[]

# precompile
at = bulk("Si", pbc = false, cubic=true)
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           bzquad = TB.GammaPoint() )
energy(tbm, at);
forces(tbm, at);
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           bzquad = TB.MPGrid(at, (2, 2, 2)) )
rattle!(at, 0.01)
energy(tbm, at);
forces(tbm, at);


test1 = "Large Cluster"
at1 = (3,3,3) * bulk("Si", pbc=false, cubic=true)
tbm1 = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           bzquad = TB.GammaPoint() )
push!(tests, (test1, at1, tbm1))

test2 = "2D Cluster + k in z direction"
at2 = (4,4,1) * bulk("Si", pbc=false, cubic=true)
tbm2 = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           bzquad=TB.MPGrid(at2, (0, 0, 8)) )
push!(tests, (test2, at2, tbm2))



# using ProfileView
# @profile energy(tbm2, at2)
# ProfileView.view()


for (test, at, tbm) in tests
   println("====================================================")
   println(test)
   println("Energy: " )
   for n = 1:4
      n <= 2 && rattle!(at, 0.001)
      @time energy(tbm, at)
   end
   println("Forces: " )
   for n = 1:4
      n <= 4 && rattle!(at, 0.001)
      @time forces(tbm, at)
   end
end
println("====================================================")
