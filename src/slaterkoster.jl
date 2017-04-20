
import JuLIP.Potentials: evaluate, evaluate_d

using ForwardDiff

# slaterkoster.jl
#
# Collect all generic stuff for Slater-Koster-type Tight-binding
# models (which is 99.99% of non-selfconsistent TB models)
#

abstract SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH}

norbitals{ISORTH,NORB}(::SKHamiltonian{ISORTH, NORB}) = NORB

nbonds{ISORTH}(::SKHamiltonian{ISORTH, 1}) = 1
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 4}) = 4
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 9}) = 10

ndofs(H::SKHamiltonian, at::AbstractAtoms) = norbitals(H) * length(at)


############################################################
### indexing for SKHamiltonians

"""
`indexblock`:
a little auxiliary function to compute indices of Slater Koster orbitals,
this is returned as an SVector, i.e. it is generated on the stack so that no
heap memory is allocated.
"""
indexblock{IO,NORB}(n::Integer, H::SKHamiltonian{IO,NORB}) =
   SVector{NORB, Int}( ((n-1)*NORB+1):(n*NORB) )


# The following code is uncommented because we don't need it right now
# it will probably used for the ContourCalculator
#
# function skindexblock{T <: Integer}(Iat::AbstractVector{T}, norb::Integer)
#    out = T[]
#    for n in Iat
#       append!(out, indexblock(n, norb))
#    end
#    return out
# end
#
# indexblock{T <: Integer}(Iat::AbstractVector{T}, H::SKHamiltonian) =
#    skindexblock(Iat, norbitals(H))





############################################################
### Hamiltonian entries


######## s-orbital model

function sk!{IO}(H::SKHamiltonian{IO, 1}, U, bonds, out)
   setindex!(out, bonds[1], 1)
   return out
end

function sk_d!{IO}(H::SKHamiltonian{IO, 1}, r, R, b, db, dH_nm)
   # dH_nm is 3 x 1 x 1 so we can just index it linearly    (NORB = 1)
   for a = 1:3
      dH_nm[a] = db[1] * R[a] / r
   end
   return dH_nm
end

######## sp-orbital model

sk!{IO}(H::SKHamiltonian{IO, 4}, U, bonds, out) = _sk4!(U, bonds, out)
sk_d!{IO}(H::SKHamiltonian{IO, 4}, r, R, b, db, dout) = _sk4_d!(R/r, r, b, db, dout)

######## spd-orbital model

sk!{IO}(H::SKHamiltonian{IO, 9}, U, bonds, out) = _sk9!(U, bonds, out)
sk_d!{IO}(H::SKHamiltonian{IO, 9}, r, R, b, db, dout) = _sk9_d!(R/r, r, b, db, dout)


# ======================================================================
# prototypes for the functions needed to assemble the hamiltonian
# and its derivatives

# TODO: documentation, what is this called?

"""
`hop(H::SKHamiltonian, r, i)`: where `r` is real, `i` integer, this
should return the
"""
@protofun hop(::SKHamiltonian, ::Any, ::Any)


# TODO: this is probably type-unstable!
hop_d(H::SKHamiltonian, r, i) = ForwardDiff.derivative(s -> hop(H,s,i), r)


function hop!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = hop(H, r, i)
   end
   return bonds
end

# TODO: hop_d should return hop and hop_d, since we need both!!!!
function hop_d!(H::SKHamiltonian, r, b, db)
   for i = 1:nbonds(H)
      b[i] = hop(H, r, i)
      db[i] = hop_d(H, r, i)
   end
   return b, db
end


@protofun overlap(::SKHamiltonian, ::Real, ::Integer)

overlap_d(H::SKHamiltonian, r::Real, i) = ForwardDiff.derivative(s->overlap(H,s,i), r)

function overlap!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = overlap(H, r, i)
   end
   return bonds
end

function overlap_d!(H::SKHamiltonian, r, b, db)
   for i = 1:nbonds(H)
      b[i] = overlap(H, r, i)
      db[i] = overlap_d(H, r, i)
   end
   return b, db
end

# rewrite onsite! as diagonals???
# but we don't provide AD for these since they depend on too many variables,
# so AD will necessarily be inefficient.

function onsite! end
function onsite_grad! end


# # prototypes for the defaul diagonal on-site terms
# function diagonsite! end
# function diagonsite_d! end
#
# function onsite!(H::SKHamiltonian, r, R, H_nn)
#    diagonsite!(H, r, H_nn)   # this treats H_nn as linear memory
#    for i = 1:size(H_nn, 1)
#       H_nn[i,i] = H_nn[i]    # hence H_nn[i] must be copied to the diagonal
#       H_nn[i] = 0.0
#    end
#    return H_nn
# end

# function onsite_grad!(H::SKHamiltonian, r, dH_nn)
#    temp = zeros(
#
# end




############################################################
### Hamiltonian Evaluation

function evaluate(H::SKHamiltonian, at::AbstractAtoms, k::AbstractVector,
                  T=Matrix)
   # this either retrieves or computes the SparseSKH thing
   skh = update!(at, H)
   return T(skh)
end



# ============ SPECIALISED SK FORCES IMPLEMENTATION =============
#
# this is an old force computation that *requires* a SKHamiltonian structure
#    F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
# but instead of computing H,n, M,n as matrices, this assembly loops over
# the non-zero blocks first and the inner loop is over the derivative ,n.
#
# TODO: after rewriting the generic force assembly, compare against this
#       implementation for performance
#
function _forces_k{ISORTH, NORB}(X::JVecsF, at::AbstractAtoms, tbm::TBModel,
                                 H::SKHamiltonian{ISORTH,NORB}, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(at, :epsn, k)::Vector{Float64}
   C = get_k_array(at, :C, k)::Matrix{Complex128}
   # df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
   df = grad(tbm.smearing, epsn)::Vector{Float64}

   # precompute some products
   C_df_Ct = (C * (df' .* C)')
   C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')

   # allocate array for forces
   const frc = zeros(Complex{Float64}, 3, length(X))

   # pre-allocate dH, with a (dumb) initial guess for the size
   # TODO:
   #   * re-interpret these as arrays of JVecs, where the first argument is the JVec
   #   * move them outside to prevent multiple allocations
   const dH_nn = zeros(3, NORB, NORB, 6)
   const dH_nm = zeros(3, NORB, NORB)
   const dM_nm = zeros(3, NORB, NORB)

   const bonds = zeros(nbonds(H))
   const dbonds = zeros(nbonds(H))

   # loop through all atoms, to compute the force on atm[n]
   for (n, neigs, r, R, _) in sites(nlist)
      # compute the block of indices for the orbitals belonging to n
      In = indexblock(n, H)

      # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
      # dH_nn should be 3 x norbitals x norbitals x nneigs
      if length(neigs) > size(dH_nn, 4)
         dH_nn = zeros(3, NORB, NORB, ceil(Int, 1.5*length(neigs)))
      end
      onsite_grad!(H, r, R, dH_nn)

      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, H)
         # kR is actually independent of X, so need not be differentiated
         kR = dot(R[i_n] - (X[m] - X[n]), k)
         eikr = exp(im * kR)::Complex{Float64}

         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         hop_d!(H, r[i_n], bonds, dbonds)
         sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dH_nm)
         if !ISORTH
            overlap_d!(H, r[i_n], bonds, dbonds)
            sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dM_nm)
         end

         # the following is a hack to put the on-site assembly into the innermost loop
         # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
         for a = 1:NORB, b = 1:NORB
            t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
            t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
            t3 = C_df_Ct[In[a],In[b]]
            # add contributions to the force
            # TODO: can re-write this as sum over JVecs
            for j = 1:3   # -dH_nm[j,a,b] * t1 + dM_nm[j,a,b] * t2 +
               frc[j,n] +=  -dH_nm[j,a,b] * t1 + dM_nm[j,a,b] * t2 + dH_nn[j,a,b,i_n] * t3
               frc[j,m] -= t3 * dH_nn[j,a,b,i_n]
            end
         end

      end  # m in neigs-loop
   end  #  sites-loop

   # TODO: in the future assemble the forces already in JVecsF format
   return real(frc) |> vecs
end


# * `forces` is imported from JuLIP
# * this implementation is the old version from Atoms.jl, which makes
#   specific assumptions about the structure of the hamiltonian, hence is
#   only valid for SK-type hamiltonians.
#
# TODO: after implementing the generic force assembly, we need to benchmark
#       them against each other!
function forces{HT <: SKHamiltonian}(tbm::TBModel{HT}, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm.H))
   frc = zerovecs(length(atm))
   X = positions(atm)
   for (w, k) in tbm.bzquad
      frc +=  w * _forces_k(X, atm, tbm, tbm.H, nlist, k)
   end
   return frc
end



# TODO: in the future assemble the forces already in JVecsF format
function _forcesnew_k{ISORTH, NORB}(at::AbstractAtoms, tbm::TBModel,
                                 H::SKHamiltonian{ISORTH,NORB}, k::JVecF,
                                 skhg)
   # obtain the precomputed arrays
   epsn = get_k_array(at, :epsn, k)::Vector{Float64}
   C = get_k_array(at, :C, k)::Matrix{Complex128}
   df = grad(tbm.smearing, epsn)::Vector{Float64}

   # precompute some products
   # TODO: optimise these two lines?
   #       note also these are O(N^3) scaling, while overall
   #       force assembly should be just O(N^2)
   C_df_Ct = (C * (df' .* C)')
   C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')
   # tmp = C'
   # scale!(df, tmp)
   # C_df_Ct = C * tmp
   # scale!(epsn, tmp)
   # C_dfepsn_Ct = C * tmp

   X = positions(at)
   const dH_nn = zeros(3, NORB, NORB, 6)
   const dH_nm = zeros(3, NORB, NORB)
   const dM_nm = zeros(3, NORB, NORB)

   const bonds = zeros(nbonds(H))
   const dbonds = zeros(nbonds(H))


   # allocate array for forces
   const frc = zeros(Float64, 3, length(at))

   for n = 1:length(skhg.i)
      i, j, dH_ij, dH_ii, S = skhg.i[n], skhg.j[n], skhg.dH[n], skhg.dOS[n], skhg.Rcell[n]
      if !ISORTH; dM_ij = skhg.dM[n]; end
      Ii, Ij = indexblock(i, H), indexblock(j, H)
      eikr = exp(im * dot(S, k))::Complex{Float64}

      R = X[j] - X[i] + S
      r = norm(R)
      hop_d!(H, r, bonds, dbonds)
      sk_d!(H, r, -R, bonds, dbonds, dH_nm)
      # @show vecnorm(dH_nm - dH_ij,Inf), vecnorm(dH_nm,Inf)
      # @show vecnorm(dH_nm + dH_ij)

      @inbounds for a = 1:NORB, b = 1:NORB
         t1 = 2.0 * real(C_df_Ct[Ij[a], Ii[b]] * eikr)
         t2 = 2.0 * real(C_dfepsn_Ct[Ij[a], Ii[b]] * eikr)
         t3 = real(C_df_Ct[Ii[a],Ii[b]])

         for c = 1:3  # -dH_ij[c,a,b] * t1 + dM_ij[c,a,b] * t2 +
            frc[c,i] += -dH_ij[c,a,b] * t1 + dM_ij[c,a,b] * t2 + dH_ii[c,a,b] * t3
            frc[c,j] -= t3 * dH_ii[c,a,b]
         end
      end
   end

   return real(frc) |> vecs
end




# * `forces` is imported from JuLIP
# * this implementation is the old version from Atoms.jl, which makes
#   specific assumptions about the structure of the hamiltonian, hence is
#   only valid for SK-type hamiltonians.
#
# TODO: after implementing the generic force assembly, we need to benchmark
#       them against each other!
function forcesnew{HT <: SKHamiltonian}(tbm::TBModel{HT}, atm::AbstractAtoms)
   update!(atm, tbm)
   skhg = SparseSKHgrad(tbm.H, atm)
   frc = zerovecs(length(atm))
   for (w, k) in tbm.bzquad
      frc +=  w * _forcesnew_k(atm, tbm, tbm.H, k, skhg)
   end
   return frc
end
