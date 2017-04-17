
using JuLIP: set_transient!, get_transient


# this file implements the standard spectral decomposition
# calculator for energy, forces, etc.



# ================= diagonalisation ====================

# TODO: should we distinguish k = 0 (Real, Symmetric), k ≠ 0 (Complex, Hermitian)
#       for large systems with only Gamma-point, this might be useful?
#       but it actually seems that the timings are not so different, e.g., for
#       a 1000 x 1000 system, Float64 ~ 400ms, Complex128 ~ 500ms per `eig`.
#       this indicates that we probably shouldn't bother


# for the standard calculator we need to convert
# the hamiltonian to full matrices
function full_hermitian{T <: Complex}(A::AbstractMatrix{T})
   A = 0.5 * (A + A')
   A[diagind(A)] = real(A[diagind(A)])
   return Hermitian(full(A))
end

function full_hermitian{T <: Real}(A::AbstractMatrix{T})
   return Symmetric(full(0.5 * (A + A')))
end

"""
`sorted_eig`:  helper function to compute eigenvalues, then sort them
in ascending order and sort the eig-vectors as well.
"""
function sorted_eig(H, M::AbstractMatrix)
   epsn, C = eig(full_hermitian(H), full_hermitian(M))
   Isort = sortperm(epsn)
   return epsn[Isort], C[:, Isort]
end

function sorted_eig(H, ::UniformScaling)
   epsn, C = eig(full_hermitian(H))
   Isort = sortperm(epsn)
   return epsn[Isort], C[:, Isort]
end


"""
`update_eig!(atm::AbstractAtoms, tbm::TBModel)` : updates the hamiltonians
and spectral decompositions on the MP grid.
"""
function update_eig!(atm::AbstractAtoms, tbm::TBModel)
   nlist = neighbourlist(atm, cutoff(tbm.H))
   nnz_est = estimate_nnz(tbm.H, atm)
   It = zeros(Int32, nnz_est)
   Jt = zeros(Int32, nnz_est)
   Ht = zeros(Complex{Float64}, nnz_est)
   Mt = zeros(Complex{Float64}, nnz_est)
   X = positions(atm)
   for (w, k) in tbm.bzquad
      H, M = assemble!(tbm.H, k, It, Jt, Ht, Mt, nlist, X)
      epsn, C = sorted_eig(H, M)
      set_k_array!(atm, M, :M, k)
      set_k_array!(atm, H, :H, k)
      set_k_array!(atm, epsn, :epsn, k)
      set_k_array!(atm, C, :C, k)
   end
end



# ==================== update all arrays =================
#    e.g., prior to an energy or force calculation


"""
`update!(atm::AbstractAtoms, tbm:TBModel)`: checks whether the precomputed
data stored in `at` is still there - JuLIP deletes all arrays when
atom positions are updated.

* spectral decompositions (`update_eig!`)
* the fermi-level (`update_eF!`)
"""
function update!(at::AbstractAtoms, tbm::TBModel)
   # check whether the :tbupdateflag exists;  if it does then atom positions
   # have not changed since last time that update! was called
   if has_transient(at, :tbupdateflag)
      return nothing
   end
   # if the flag does not exist, then we update everything
   update_eig!(at, tbm)
   update!(at, tbm.smearing)
   # set the update flag (will be deleted as soon as atom positions change)
   set_transient!(at, :tbupdateflag, 0)
   return nothing
end



# ================ Density Matrix and Energy ================


"""
`densitymatrix(tbm, at) -> Γ`:

### Input

* `tbm::TBModel` : calculator
* `at::AbstractAtoms` : configuration

### Output

`Γ::Matrix{Float64}`: density matrix Γ = Σ_k w^k Σ_s f(ϵ_s^k) ψ_s^k ⊗ ψ_s^k
and f(ϵ) is the occupancy
"""
function densitymatrix(tbm::TBModel, at::AbstractAtoms)
   N = ndofs(tbm.H, at)
   Γ = zeros(N, N)
   for (w, _, ϵ, ψ) in BZiter(tbm, at)
      fs = occupancy(tbm.smearing, ϵ)
      for a = 1:N, b = 1:N
         Γ[a,b] += w * fs * real(ψ[a] * ψ[b]')
      end
   end
   return Γ
end


# this is imported from JuLIP
energy(tbm::TBModel, at::AbstractAtoms) =
   sum( w * energy(tbm.smearing, ϵ) for (w, _1, ϵ, _2) in BZiter(tbm, at) )




# ========================== Forces ==========================

#
# this is an old force computation that *requires* a SKHamiltonian structure
#  therefore, I added H as an argument so _forces_k dispatches on its type
#
# TODO: rewrite this in a type-agnostic way!
#      (but probably keep this version for performance? or at least comparison?)
#
function _forces_k{ISORTH, NORB}(X::JVecsF, tbm::TBModel{ISORTH},
                  H::SKHamiltonian{ISORTH,NORB}, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(tbm, :epsn, k)::Vector{Float64}
   C = get_k_array(tbm, :C, k)::Matrix{Complex128}
   df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))

   # precompute some products
   const C_df_Ct = (C * (df' .* C)')               #::Matrix{Complex{Float64}}
   const C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')   #::Matrix{Complex{Float64}}

   # allocate array for forces
   const frc = zeros(Complex{Float64}, 3, length(X))

   # pre-allocate dH, with a (dumb) initial guess for the size
   # TODO:
   #   * re-interpret these as arrays of JVecs, where the first argument is the JVec
   #   * move them outside to prevent multiple allocations
   const dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, 6)
   const dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   const dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)

   # loop through all atoms, to compute the force on atm[n]
   for (n, neigs, r, R, _) in sites(nlist)
      # neigs::Vector{Int}   # TODO: put this back in?!?  > PROFILE IT AGAIN
      # R::Matrix{Float64}  # TODO: put this back in?!?
      #   IT LOOKS LIKE the type of R might not be inferred!
      
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


# imported from JuLIP
function forces(tbm::TBModel, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm.H))
   X = positions(atm)
   frc = zerovecs(length(atm))
   for (w, k) in tbm.bzquad
      frc +=  w * _forces_k(X, tbm, nlist, k)
   end
   return frc
end
