
using JuLIP: set_transient!, get_transient


# this file implements the standard spectral decomposition
# calculator for energy, forces, etc.


# ================ storage of k-point dependent arrays ================

"""
store k-point dependent arrays
"""
set_k_array!(at::AbstractAtoms, q, symbol, k) = set_transient!(at, (symbol, k), q)

"""
retrieve k-point dependent arrays
"""
get_k_array(at::AbstractAtoms, symbol, k) = get_transient(at, (symbol, k))

"""
check that a k-array exists
"""
has_k_array(at::AbstractAtoms, symbol, k) = has_transient(at, (symbol, k))


# TODO: we could write a pretty little iterator over
#   for (w, k, epsn_k, C_k) in BZ(tbm)
#       ...


# ================= diagonalisation ====================

# TODO: should we distinguish k = 0 (Real, Symmetric), k ≠ 0 (Complex, Hermitian)
#       for large systems with only Gamma-point, this might be useful?
#       but it actually seems that the timings are not so different, e.g., for
#       a 1000 x 1000 system, Float64 ~ 400ms, Complex128 ~ 500ms per `eig`.


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



# ================ Density Matrix ================


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
   update!(at, tbm)
   Γ = 0.0
   for (w, k) in tbm.bzquad
      epsn_k = get_k_array(at, :epsn, k)
      C_k = get_k_array(at, :C, k)
      for s = 1:length(epsn_k)
         fs = occupancy(tbm.smearing, epsn_k[s])
         Γ += (w * fs * C_k[:,s]) * C_k[:, s]'
      end
   end
   return Γ
end
