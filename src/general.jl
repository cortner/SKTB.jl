










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

# TODO: check the S thing and probably don't pass X !!!!!
function forces_k(X::JVecsF, tbm::TBModel, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(tbm, :epsn, k)
   C = get_k_array(tbm, :C, k)
   df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))

   # precompute some products
   const C_df_Ct = (C * (df' .* C)')::Matrix{Complex{Float64}}
   const C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')::Matrix{Complex{Float64}}

   # allocate forces
   const frc = zeros(Complex{Float64}, 3, length(X))

   # pre-allocate dH, with a (dumb) initial guess for the size
   # TODO: re-interpret these as arrays of JVecs, where the first argument is the JVec
   const dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, 6)
   const dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   const dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)

   # loop through all atoms, to compute the force on atm[n]
   for (n, neigs, r, R, _) in sites(nlist)
      # neigs::Vector{Int}   # TODO: put this back in?!?  > PROFILE IT AGAIN
      # R::Matrix{Float64}  # TODO: put this back in?!?
      #   IT LOOKS LIKE the type of R is not inferred!
      # compute the block of indices for the orbitals belonging to n
      In = indexblock(n, tbm)

      # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
      # dH_nn should be 3 x norbitals x norbitals x nneigs
      if length(neigs) > size(dH_nn, 4)
         dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, ceil(Int, 1.5*length(neigs)))
      end
      evaluate_d!(tbm.onsite, r, R, dH_nn)

      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         tmp = R[i_n] - (X[neigs[i_n]] - X[n])
         kR = dot(tmp, k)
         eikr = exp(im * kR)::Complex{Float64}
         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         grad!(tbm.hop, r[i_n], - R[i_n], dH_nm)
         grad!(tbm.overlap, r[i_n], - R[i_n], dM_nm)

         # the following is a hack to put the on-site assembly into the
         # innermost loop
         # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
         for a = 1:tbm.norbitals, b = 1:tbm.norbitals
            t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
            t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
            t3 = C_df_Ct[In[a],In[b]]
            # add contributions to the force
            # TODO: can re-write this as sum over JVecs
            for j = 1:3
               frc[j,n] += - dH_nm[j,a,b] * t1 + dM_nm[j,a,b] * t2 + dH_nn[j,a,b,i_n] * t3
               frc[j,m] += - t3 * dH_nn[j,a,b,i_n]
            end
         end

      end  # m in neigs-loop
   end  #  sites-loop

   # TODO: in the future assemble the forces already in JVecsF format
   return real(frc) |> vecs
end



function forces(tbm::TBModel, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm))
   K, weight = monkhorstpackgrid(atm, tbm)
   X = positions(atm)
   frc = zerovecs(length(atm))
   for iK = 1:length(K)
      frc +=  weight[iK] * forces_k(X, tbm, nlist, K[iK])
   end
   return frc
end
