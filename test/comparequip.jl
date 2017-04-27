
using JuLIP
using TightBinding

# @testset "Compare TightBinding.jl against QUIP" begin

beta = 10.0

TB = TightBinding
at = bulk("Si", cubic = true) * 2
rattle!(at, 0.02)
tbm = TB.NRLTB.NRLTBModel(:Si, FermiDiracSmearing(beta), orbitals = :sp, cutoff=:original)

H, M = hamiltonian(tbm, at)
H = full(H)
M = full(M)

using PyCall
@pyimport quippy

function make_quip_tb(at, Nk)
    k, weight = TB.monkhorstpackgrid(cell(at), (0, 0, Nk))
    kpoint_xml =  "<KPoints N=\"$(length(weight))\">\n" *
        join(["<point weight=\"$w\">$kx $ky $kz</point>\n" for (w, (kx, ky, kz)) in zip(weight, k)]) *
        "</KPoints>"
    println(kpoint_xml)

    xml_str = "<params>" *
        readstring(open("/Users/ortner/gits/QUIP/share/Parameters/tightbind.parms.NRL_TB.Si.xml", "r")) *
        kpoint_xml * "</params>"

    pot = quippy.Potential("TB NRL-TB", param_str=xml_str)
    quip_tb = ASECalculator(pot)
    return quip_tb
end

function tb_matrices(calc, at, tbm)
   norb = 4
   H = zeros(norb * length(at), norb  * length(at))
   M = zeros(norb * length(at), norb  * length(at))
   qsi = quippy.Atoms(at.po)
   calc.po[:calc_tb_matrices](qsi, hd=H, sd=M)

   # permute
   for n = 1:length(at)
      In = TB.indexblock(n, tbm.H)[2:4]
      Inp = [In[3], In[1], In[2]]
      H[:, In] = H[:, Inp]
      M[:, In] = M[:, Inp]
      H[In, :] = H[Inp, :]
      M[In, :] = M[Inp, :]
   end

   return H, M
end

quip_tb = make_quip_tb(at, 0)
Hq, Mq = tb_matrices(quip_tb, at, tbm)

M, H, Mq, Hq = real(M), real(H), real(Mq), real(Hq)

@show vecnorm(H - Hq, Inf)
@test vecnorm(H - Hq, Inf) < 1e-5
@show vecnorm(M - Mq, Inf)
@test vecnorm(M - Mq, Inf) < 1e-5

E = sort(real(eigvals(H, M)))
Eq = sort(real(eigvals(Hq, Mq)))
@show norm(E - Eq, Inf)
@test norm(E - Eq, Inf) < 1e-5

# println("JuLIP H: " )
# display(round(H[1:12, 1:16], 3)); println()
# println("QUIP H: ")
# display(round(Hq[1:12,1:16], 3)); println()
#
# println("JuLIP M: " )
# display(round(M[1:12, 1:16], 3)); println()
# println("QUIP M: ")
# display(round(Mq[1:12,1:16], 3)); println()

# end
