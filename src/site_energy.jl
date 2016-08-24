


function site_energy(tbm::TBModel, at::AbstractAtoms, n0::Integer)
   update!(at, tbm)
   In0 = indexblock(n0, tbm) |> Vector
   K, weight = monkhorstpackgrid(at, tbm)
   Esite = 0.0
   for (k, w) in zip(K, weight)
      epsn_k = get_k_array(tbm, :epsn, k)
      C_k = get_k_array(tbm, :C, k)
      loc = sum( C_k[In0, :].^2, 1 )[:]
      # TODO: this is wrong
      Esite += w * sum(tbm.smearing(epsn_k, tbm.eF) .* epsn_k  .* loc)
   end
   return real(Esite)
end
