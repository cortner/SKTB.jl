


function site_energy(tbm::TBModel, at::AbstractAtoms, n0::Integer)
   update!(at, tbm)
   In0 = (indexblock(n0, tbm) |> Vector)[1:1]   # (DEBUG)
   # In0 = indexblock(n0, tbm) |> Vector
   K, weight = monkhorstpackgrid(at, tbm)
   Esite = 0.0
   for (k, w) in zip(K, weight)
      epsn_k = get_k_array(tbm, :epsn, k)
      M_k = get_k_array(tbm, :M, k)
      C_k = get_k_array(tbm, :C, k)
      MC_k = M_k[In0, :] * C_k
      ψ² = sum( conj(C_k[In0, :] .* MC_k), 1 )[:]
      Esite += w * sum(tbm.smearing(epsn_k, tbm.eF) .* epsn_k  .* ψ²)
   end
   return real(Esite)
end
