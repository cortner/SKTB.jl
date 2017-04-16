




############################################################
##### update functions






"""
`update_eF!(tbm::TBModel)`: recompute the correct
fermi-level; using the precomputed data in `tbm.arrays`
"""
function update_eF!(atm::AbstractAtoms, tbm::TBModel)
   if tbm.fixed_eF
      set_eF!(tbm.smearing, tbm.eF)
      return
   end
   # the following algorithm works for Fermi-Dirac, not general Smearing
   K, weight = monkhorstpackgrid(atm, tbm)
   Ne = tbm.norbitals * length(atm)
   nf = round(Int, ceil(Ne/2))
   # update_eig!(atm, tbm)
   # set an initial eF
   μ = 0.0
   for n = 1:length(K)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      μ += weight[n] * (epsn_k[nf] + epsn_k[nf+1]) /2
   end
   # iteration by Newton algorithm
   err = 1.0
   while abs(err) > 1.0e-8
      Ni = 0.0
      gi = 0.0
      for n = 1:length(K)
         k = K[n]
         epsn_k = get_k_array(tbm, :epsn, k)
         Ni += weight[n] * sum_kbn( tbm.smearing(epsn_k, μ) )
         gi += weight[n] * sum_kbn( @D tbm.smearing(epsn_k, μ) )
      end
      err = Ne - Ni
      #println("\n err=");  print(err)
      μ = μ - err / gi
   end
   tbm.eF = μ
   set_eF!(tbm.smearing, tbm.eF)
end



############################################################
### Hamiltonian Evaluation


"""
`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
binding model.

#### Parameters:

* `atm::AbstractAtoms`
* `tbm::TBModel`
* `k = k=[0.;0.;0.]` : k-point at which the hamiltonian is evaluated

### Output: H, M

* `H` : hamiltoian in CSC format
* `M` : overlap matrix in CSC format
"""
function hamiltonian(atm::AbstractAtoms, tbm::TBModel, k)
   nlist = neighbourlist(atm, cutoff(tbm))
   nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
   It = zeros(Int32, nnz_est)
   Jt = zeros(Int32, nnz_est)
   Ht = zeros(Complex{Float64}, nnz_est)
   Mt = zeros(Complex{Float64}, nnz_est)
   X = positions(atm)
   return hamiltonian!( tbm, k, It, Jt, Ht, Mt, nlist, X)
end

hamiltonian(tbm::TBModel, atm::AbstractAtoms) =
   hamiltonian(atm::AbstractAtoms, tbm::TBModel, JVecF(0.0,0.0,0.0))



# TODO: the following method overload are a bit of a hack
#       it would be better to implement broadcasting in FixedSizeArrays

import Base.-
-(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]

dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]

function append!(It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, norbitals, idx)
   @inbounds for i = 1:norbitals, j = 1:norbitals
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
      Mt[idx] = M_nm[i,j] * exp_i_kR
   end
   return idx
end

function hamiltonian!(tbm::TBModel, k, It, Jt, Ht, Mt, nlist, X)

   idx = 0                     # index into triplet format
   H_nm = zeros(tbm.norbitals, tbm.norbitals)    # temporary arrays
   M_nm = zeros(tbm.norbitals, tbm.norbitals)
   temp = zeros(10)

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, tbm)   # index-block for atom index n
      # loop through the neighbours of the current atom
      for m = 1:length(neigs)
         # note: we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
         #       but this would actually be less efficient, and less clear
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )

         Im = indexblock(neigs[m], tbm)
         # compute hamiltonian block
         H_nm = evaluate!(tbm.hop, r[m], R[m], H_nm)
         # compute overlap block
         M_nm = evaluate!(tbm.overlap, r[m], R[m], M_nm)
         # add new indices into the sparse matrix
         idx = append!(It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, tbm.norbitals, idx)
      end
      # now compute the on-site terms
      # TODO: we could move these to be done in-place???
      # (first test: with small vectors and matrices in-place operations
      #              may become unnecessary)
      H_nn = tbm.onsite(r, R)
      M_nn = tbm.overlap(0.0)
      # H_nn = zeros(M_nn)    # for debugging
      # add into sparse matrix
      idx = append!(It, Jt, Ht, Mt, In, In, H_nn, M_nn, 1.0, tbm.norbitals, idx)
   end

   # convert M, H into Sparse CCS and return
   #   NOTE: The conversion to sparse format accounts for about 1/2 of the
   #         total cost. Since It, Jt are in an ordered format, it should be
   #         possible to write a specialised code that converts it to
   #         CCS much faster, possibly with less additional allocation?
   #         another option would be to store a single It, Jt somewhere
   #         for ALL the hamiltonians, and store multiple Ht, Mt and convert
   #         these "on-the-fly", depending on whether full or sparse is needed.
   #         but at the moment, eigfact cost MUCH more than the assembly,
   #         so we could choose to stop here.
   return sparse(It, Jt, Ht), sparse(It, Jt, Mt)
end



"""`densitymatrix(at::AbstractAtoms, tbm::TBModel) -> rho`:

### Input
* `at::AbstractAtoms` : configuration
* `tbm::TBModel` : calculator

### Output
* `rho::Matrix{Float64}`: density matrix,
    ρ = ∑_s f(ϵ_s) ψ_s ⊗ ψ_s
where `f` is given by `tbm.SmearingFunction`. With BZ integration, it becomes
    ρ = ∑_k w^k ∑_s f(ϵ_s^k) ψ_s^k ⊗ ψ_s^k
"""
function densitymatrix(at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(atm, tbm)
   rho = 0.0
   for n = 1:length(K)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      C_k = get_k_array(tbm, :C, k)
      f = tbm.smearing(epsn_k, tbm.eF)
      for m = 1:length(epsn_k)
         rho += weight[n] * f[m] * C_k[:,m] * C_k[:,m]'
      end
   end
   return rho
end



############################################################
### Standard Calculator Functions


function energy(tbm::TBModel, at::AbstractAtoms)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(at, tbm)
   E = 0.0
   for n = 1:length(K)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      E += weight[n] * sum_kbn(tbm.smearing(epsn_k, tbm.eF) .* epsn_k)
   end
   return E
end



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
