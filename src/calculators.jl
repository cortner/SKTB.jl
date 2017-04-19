
using JuLIP: set_transient!, get_transient


# this file implements the standard spectral decomposition
# calculator for energy, forces, etc.



# ================= diagonalisation ====================

# TODO: should we distinguish k = 0 (Real, Symmetric), k ≠ 0 (Complex, Hermitian)
#       for large systems with only Gamma-point, this might be useful?
#       but it actually seems that the timings are not so different, e.g., for
#       a 1000 x 1000 system, Float64 ~ 400ms, Complex128 ~ 500ms per `eig`.
#       this indicates that we probably shouldn't bother


"""
`sorted_eig`:  helper function to compute eigenvalues, then sort them
in ascending order and sort the eig-vectors as well.
"""
function sorted_eig(H::Hermitian, M::Hermitian)
   epsn, C = eig(H, M)
   Isort = sortperm(epsn)
   return epsn[Isort], C[:, Isort]
end

function sorted_eig(H::Hermitian, ::UniformScaling)
   epsn, C = eig(H)
   Isort = sortperm(epsn)
   return epsn[Isort], C[:, Isort]
end

# compute and store the SparseSKH thing from the hamiltonian
function update!(at::AbstractAtoms, H::SKHamiltonian)
   if has_transient(at, :skh)
      return get_transient(at, :skh)
   else
      skh = SparseSKH(H, at)
      set_transient!(at, skh, :skh)
      return skh
   end
end

"""
`update_eig!(atm::AbstractAtoms, tbm::TBModel)` : updates the hamiltonians
and spectral decompositions on the MP grid.
"""
function update_eig!{ISORTH}(atm::AbstractAtoms, H::SparseSKH{ISORTH}, tbm::TBModel)
   wrk = _alloc_full(H)
   for (w, k) in tbm.bzquad
      Hf, Mf = full!(wrk, H, k)
      epsn, C = sorted_eig(Hf, Mf)
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
   skh = update!(at, tbm.H)
   update_eig!(at, skh, tbm)
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

# TODO: implement a generic force calculator
