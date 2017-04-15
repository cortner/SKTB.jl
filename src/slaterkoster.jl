
import JuLIP.Potentials: evaluate, evaluate_d

# slaterkoster.jl
#
# Collect all generic stuff for Slater-Koster-type Tight-binding
# models (which is 99% of non-selfconsistent TB models)
#

abstract SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH}

norbitals{ISORTH,NORB}(::SKHamiltonian{ISORTH, NORB}) = NORB

nbonds{ISORTH}(::SKHamiltonian{ISORTH, 1}) = 1
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 4}) = 4
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 9}) = 10



############################################################
### indexing for SKHamiltonians

skindexblock(n::Integer, norb::Integer) = Int[(n-1) * norb + j for j = 1:norb]

"""
`indexblock`:
a little auxiliary function to compute indices of Slater Koster orbitals
"""
indexblock(n::Integer, H::SKHamiltonian) = skindexblock(n, norbitals(H))


function skindexblock{T <: Integer}(Iat::AbstractVector{T}, norb::Integer)
   out = T[]
   for n in Iat
      append!(out, indexblock(n, norb))
   end
   return out
end

indexblock{T <: Integer}(Iat::AbstractVector{T}, H::SKHamiltonian) =
   skindexblock(Iat, norbitals(H))


############################################################
### Slater-Koster Paramters


"""
Assemble a hamiltonian block for a Slater-Koster type
hamiltonian, with 4 orbitals (s, p).

### Parameters:
* `U` : R / |R|, orientation of the bond, must be a 3-vector
* `hop` : vector of hopping function values, must be a 4-vector
* `mat` : output matrix, must be at least 4 x 4
"""
function sk4!(U, hop, mat)
   l, m, n = U[1], U[2], U[3]
   # 4 orbitals are s, px, py, pz, these are the mat-indices
   # 4 bond types are : ssσ,spσ,ppσ,ppπ, these are the hop-indices
   mat[1,1] = hop[1]                            # E_ss = V_ssσ
   mat[2,2] = l*l * hop[3] + (1-l*l) * hop[4]   # E_xx = l² V_ppσ + (1-l²) V_ppπ
   mat[3,3] = m*m * hop[3] + (1-m*m) * hop[4]   # E_yy = m² V_ppσ + (1-m²) V_ppπ
   mat[4,4] = n*n * hop[3] + (1-n*n) * hop[4]   # E_zz = n² V_ppσ + (1-n²) V_ppπ
   mat[1,2] = l * hop[2]                        # E_sx = l V_spσ
   mat[1,3] = m * hop[2]                        # E_sy = m V_spσ
   mat[1,4] = n * hop[2]                        # E_sz = n V_spσ
   mat[2,1] = - mat[1,2]                        # E_xs = - E_sx
   mat[3,1] = - mat[1,3]                        # E_ys = - E_sy
   mat[4,1] = - mat[1,4]                        # E_zs = - E_sz
   mat[2,3] = l * m * (hop[3] - hop[4])         # E_xy = l m (V_ppσ - V_ppπ)
   mat[2,4] = l * n * (hop[3] - hop[4])         # E_xz = l n (V_ppσ - V_ppπ)
   mat[3,4] = m * n * (hop[3] - hop[4])         # E_yz = n m (V_ppσ - V_ppπ)
   mat[3,2] =  mat[2,3]                         # E_yx = E_xy
   mat[4,2] =  mat[2,4]                         # E_zx = E_xz
   mat[4,3] =  mat[3,4]                         # E_zy = E_yz
   return mat
end


"""
Assemble a hamiltonian block for a slater-koster type
hamiltonian, with 9 orbitals (s, p, d).

* `U` : orientation of the bond
* `hop` : hopping functions for bond type, 10-vector
* `mat` : output matrix 9 x 9
"""
function sk9!(U, hop, mat)
   # fill the [1:4, 1:4] block
   sk4!(U, hop, mat)
   # and then all the rest
   l, m, n = U[1], U[2], U[3]
   # l2, m2, n2 = l*l, m*m, n*n    # TODO: for faster evaluation
   # lm, ln, mn = l*m, l*n, m*n
   # rt3 = √3

   # sd
   mat[1,5] = √3 * l * m * hop[5]
   mat[1,6] = √3 * m * n * hop[5]
   mat[1,7] = √3 * l * n * hop[5]
   mat[1,8] = √3/2 * (l^2 - m^2) * hop[5]
   mat[1,9] = ( n^2 - (l^2 + m^2)/2 ) * hop[5]
   mat[5,1] = mat[1,5]
   mat[6,1] = mat[1,6]
   mat[7,1] = mat[1,7]
   mat[8,1] = mat[1,8]
   mat[9,1] = mat[1,9]

   # pd
   mat[2,5] = √3 * l * l * m * hop[6] + m * (1.0 - 2.0 * l^2) * hop[7]
   mat[2,6] = √3 * l * m * n * hop[6] - 2.0 * l * m * n * hop[7]
   mat[2,7] = √3 * l * l * n * hop[6] + n * (1.0 - 2.0 * l^2) * hop[7]
   mat[2,8] = √3/2 * l * (l^2 - m^2) * hop[6] + l * (1.0 - l^2 + m^2) * hop[7]
   mat[2,9] = l * (n^2 - (l^2 + m^2)/2) * hop[6] - √3 * l * n^2 * hop[7]
   mat[5,2] = - mat[2,5]
   mat[6,2] = - mat[2,6]
   mat[7,2] = - mat[2,7]
   mat[8,2] = - mat[2,8]
   mat[9,2] = - mat[2,9]
   mat[3,5] = √3 * l * m * m * hop[6] + l * (1.0 - 2.0 * m^2) * hop[7]
   mat[3,6] = √3 * m * m * n * hop[6] + n * (1.0 - 2.0 * m^2) * hop[7]
   mat[3,7] = √3 * l * m * n * hop[6] - 2.0 * l * m * n * hop[7]
   mat[3,8] = √3/2 * m * (l^2 - m^2) * hop[6] - m * (1.0 + l^2 - m^2) * hop[7]
   mat[3,9] = m * (n^2 - (l^2 + m^2)/2) * hop[6] - √3 * m * n^2 * hop[7]
   mat[5,3] = - mat[3,5]
   mat[6,3] = - mat[3,6]
   mat[7,3] = - mat[3,7]
   mat[8,3] = - mat[3,8]
   mat[9,3] = - mat[3,9]
   mat[4,5] = √3 * l * m * n * hop[6] - 2.0 * l * m * n * hop[7]
   mat[4,6] = √3 * m * n * n * hop[6] + m * (1.0 - 2.0 * n^2) * hop[7]
   mat[4,7] = √3 * l * n * n * hop[6] + l * (1.0 - 2.0 * n^2) * hop[7]
   mat[4,8] = √3/2 * n * (l^2 - m^2) * hop[6] - n * (l^2 - m^2) * hop[7]
   mat[4,9] = n * (n^2 - (l^2 + m^2)/2) * hop[6] + √3 * n * (l^2 + m^2) * hop[7]
   mat[5,4] = - mat[4,5]
   mat[6,4] = - mat[4,6]
   mat[7,4] = - mat[4,7]
   mat[8,4] = - mat[4,8]
   mat[9,4] = - mat[4,9]

   # dd
   mat[5,5] = 3.0 * l^2 * m^2 * hop[8] + (l^2 + m^2 - 4.0 * l^2 * m^2) * hop[9] +
               (n^2 + l^2 * m^2) * hop[10]
   mat[6,6] = 3.0 * m^2 * n^2 * hop[8] + (m^2 + n^2 - 4.0 * m^2 * n^2) * hop[9] +
               (l^2 + m^2 * n^2) * hop[10]
   mat[7,7] = 3.0 * l^2 * n^2 * hop[8] + (l^2 + n^2 - 4.0 * l^2 * n^2) * hop[9] +
               (m^2 + l^2 * n^2) * hop[10]
   mat[8,8] = 3.0/4 * (l^2 - m^2)^2 * hop[8] + (l^2 + m^2 - (l^2 - m^2)^2) * hop[9] +
               (n^2 + (l^2 - m^2)^2 /4 ) * hop[10]
   mat[9,9] = (n^2 - (l^2 + m^2) /2)^2 * hop[8] + 3.0 * n^2 * (l^2 + m^2) * hop[9] +
               3.0/4 * (l^2 + m^2)^2 * hop[10]
   mat[5,6] = 3.0 * l * m^2 * n * hop[8] + l * n * (1.0 - 4.0 * m^2) * hop[9] +
               l * n * (m^2 - 1.0) * hop[10]
   mat[5,7] = 3.0 * l^2 * m * n * hop[8] + m * n * (1.0 - 4.0 * l^2) * hop[9] +
               m * n * (l^2 - 1.0) * hop[10]
   mat[5,8] = 3.0/2 * l * m * (l^2 - m^2) * hop[8] + 2.0 * l * m * (m^2 - l^2) * hop[9] +
               1.0/2 * l * m * (l^2 - m^2) * hop[10]
   mat[5,9] = √3 * l * m * (n^2 - (l^2 + m^2)/2) * hop[8] - 2.0*√3 * l * m * n^2 * hop[9] +
               √3/2 * l * m * (1.0 + n^2) * hop[10]
   mat[6,7] = 3.0 * l * m * n^2 * hop[8] + l * m * (1.0 - 4.0 * n^2) * hop[9] +
               l * m * (n^2 - 1.0) * hop[10]
   mat[6,8] = 3.0/2 * m * n * (l^2 - m^2) * hop[8] -
               m * n * (1.0 + 2.0 * (l^2 - m^2)) * hop[9] +
               m * n * (1.0 + (l^2 - m^2) /2) * hop[10]
   mat[6,9] = √3 * m * n * (n^2 - (l^2 + m^2)/2) * hop[8] +
               √3 * m * n * (l^2 + m^2 - n^2) * hop[9] -
               √3/2 * m * n * (l^2 + m^2) * hop[10]
   mat[7,8] = 3.0/2 * l * n * (l^2 - m^2) * hop[8] +
               l * n * (1.0 - 2.0 * (l^2 - m^2)) * hop[9] -
               l * n * (1.0 - (l^2 - m^2) /2) * hop[10]
   mat[7,9] = √3 * l * n * (n^2 - (l^2 + m^2) /2) * hop[8] +
               √3 * l * n * (l^2 + m^2 - n^2) * hop[9] -
               √3/2 * l * n * (l^2 + m^2) * hop[10]
   mat[8,9] = √3/2 * (l^2 - m^2) * (n^2 - (l^2 + m^2) /2) * hop[8] +
               √3 * n^2 * (m^2 - l^2) * hop[9] +
               √3/4 * (1.0 + n^2) * (l^2 - m^2) * hop[10]
   mat[6,5] = mat[5,6]
   mat[7,5] = mat[5,7]
   mat[8,5] = mat[5,8]
   mat[9,5] = mat[5,9]
   mat[7,6] = mat[6,7]
   mat[8,6] = mat[6,8]
   mat[9,6] = mat[6,9]
   mat[8,7] = mat[7,8]
   mat[9,7] = mat[7,9]
   mat[9,8] = mat[8,9]
   return mat
end



############################################################
### Hamiltonian entries

sk!{IO}(H::SKHamiltonian{IO, 1}, U, bonds, out) = bonds[1]

sk!{IO}(H::SKHamiltonian{IO, 4}, U, bonds, out) = sk4!(U, bonds, out)

sk!{IO}(H::SKHamiltonian{IO, 9}, U, bonds, out) = sk9!(U, bonds, out)


############################################################
### Hamiltonian Evaluation
#   most of this could become a *generic* code!
#   it just needs generalising the atom-orbital-dof map.


function evaluate(H::SKHamiltonian, at::AbstractAtoms, k::AbstractVector)
   nlist = neighbourlist(at, cutoff(H))
   # pre-allocate memory for the triplet format
   norb = norbitals(H)
   nnz_est = length(nlist) * norb^2 + length(at) * norb^2
   It = zeros(Int32, nnz_est)
   Jt = zeros(Int32, nnz_est)
   Ht = zeros(Complex{Float64}, nnz_est)
   if isorthogonal(H)
      return assemble!( H, k, It, Jt, Ht, nlist, positions(at))
   else
      Mt = zeros(Complex{Float64}, nnz_est)
      return assemble!( H, k, It, Jt, Ht, Mt, nlist, positions(at))
   end
   error("nomansland")
end


# TODO: the following method overload are a bit of a hack
#       it would be better to implement broadcasting in FixedSizeArrays

import Base.-
-(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]

dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]

import Base.append!


# TODO: hide all this sparse matrix crap in a nice type

# append to triplet format: version 1 for H and M (non-orth TB)
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

# append to triplet format: version 2 for H only (orthogonal TB)
function append!(It, Jt, Ht, In, Im, H_nm, exp_i_kR, norbitals, idx)
   @inbounds for i = 1:norbitals, j = 1:norbitals
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
   end
   return idx
end

# prototypes for the functions needed in `assemble!`
function hop! end
function overlap!  end
function onsite! end

# inner Hamiltonian assembly:  non-orthogonal SK tight-binding
#
# exp_i_kR = complex multiplier needed for BZ integration
# note: we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
#       but this would actually be less efficient, and less clear
#
function assemble!{NORB}(H::SKHamiltonian{NONORTHOGONAL, NORB},
                           k, It, Jt, Ht, Mt, nlist, X)

   idx = 0                     # initialise index into triplet format
   H_nm = zeros(NORB, NORB)    # temporary arrays for computing H and M entries
   M_nm = zeros(NORB, NORB)
   bonds = zeros(nbonds(H))     # temporary array for storing the potentials

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, H)     # index-block for atom index n
      # loop through the neighbours of the current atom (i.e. the bonds)
      for m = 1:length(neigs)
         U = R[m]/r[m]
         # compute hamiltonian block
         sk!(H, U, hop!(H, r[m], bonds), H_nm)
         # compute overlap block
         sk!(H, U, overlap!(H, r[m], bonds), M_nm)
         # add new indices into the sparse matrix
         Im = indexblock(neigs[m], H)
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
         idx = append!(It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, NORB, idx)
      end

      # now compute the on-site blocks;
      # TODO: revisit this (can one do the scalar temp trick again?)
      onsite!(H, r, R, H_nm)
      overlap!(H, M_nm)
      # add into sparse matrix
      idx = append!(It, Jt, Ht, Mt, In, In, H_nm, M_nm, 1.0, NORB, idx)
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


# inner Hamiltonian assembly:  orthogonal tight-binding
function assemble!{NORB}(H::SKHamiltonian{ORTHOGONAL, NORB},
                         k, It, Jt, Ht, nlist, X)

   idx = 0                     # initialise index into triplet format
   H_nm = zeros(NORB, NORB)    # temporary arrays for computing H and M entries
   bonds = zeros(nbonds(H))     # temporary array for storing the potentials

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, H)   # index-block for atom index n
      # loop through the neighbours of the current atom
      for m = 1:length(neigs)
         U = R[m]/r[m]
         # compute hamiltonian block
         sk!(H, U, hop!(H, r[m], bonds), H_nm)
         # add new indices into the sparse matrix
         Im = indexblock(neigs[m], H)
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
         idx = append!(It, Jt, Ht, In, Im, H_nm, exp_i_kR, NORB, idx)
      end
      # now compute the on-site terms
      onsite!(H, r, R, H_nm)
      # add into sparse matrix
      idx = append!(It, Jt, Ht, In, In, H_nm, 1.0, NORB, idx)
   end
   return sparse(It, Jt, Ht), I
end
