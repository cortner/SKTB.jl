
# # TODO: where does this go?????
# # TODO: default evaluate!; should this potentially go into JuLIP.Potentials?
# evaluate!(pot, r, R, target)  = copy!(target, evaluate(pot, r, R))
# evaluate_d!(pot, r, R, target)  = copy!(target, evaluate_d(pot, r, R))
# grad(pot, r, R) = R .* (evaluate_d(pot, r, R) ./ r)'
# grad!(p, r, R, G) = copy!(G, grad(p, r, R))




############################################################
### Standard Calculator Functions





function band_structure_all(at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   na = length(at) * tbm.norbitals
   K, weight = monkhorstpackgrid(at, tbm)
   E = zeros(na, length(K))
   Ne = tbm.norbitals * length(at)
   nf = round(Int, ceil(Ne/2))
   for n = 1:length(K)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      for j = 1:na
         E[j,n] = epsn_k[j]
      end
   end
   return K, E
end


# get 2*Nb+1 bands around the fermi level
function band_structure_near_eF(Nb, at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(at, tbm)
   E = zeros(2*Nb+1, length(K))
   Ne = tbm.norbitals * length(at)
   nf = round(Int, ceil(Ne/2))
   for n = 1:length(K)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      E[Nb+1,n] = epsn_k[nf]
      for j = 1:Nb
         E[Nb+1-j,n] = epsn_k[nf-j]
         E[Nb+1+j,n] = epsn_k[nf+j]
      end
   end
   return K, E
end
