#
# This File contains the core routines for Slater-Koster type TB models
#
############################################################


"""
Assemble a hamiltonian block for a Slater-Koster type
hamiltonian, with 4 orbitals (s, p).

### Parameters:
* `U` : R / |R|, orientation of the bond, must be a 3-vector
* `hop` : vector of hopping function values, must be a 4-vector
* `mat` : output matrix, must be at least 4 x 4
"""
function _sk4!(U, hop, mat)
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
function _sk9!(U, hop, mat)
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



# =============== Derivatives of the sk*! functions ================

#
# hh[i] = V(r)
# dhh[i] = V'(r)
#
# in other parts of the code hh ≡ bonds ≡ b  and dhh ≡ dbonds ≡ db
#
function _sk4_d!(U, r, hh, dhh, dmat)
   # U = ∇r, so it appears here both as the orientation and as ∇r
   l, m, n = U[1], U[2], U[3]
   ll, lm, ln, mm, mn, nn = l*l, l*m, l*n, m*m, m*n, n*n
   dl = ( (1.0-ll)/r ,     - lm/r ,     - ln/r )
   dm = (     - lm/r , (1.0-mm)/r ,     - mn/r )
   dn = (     - ln/r ,     - mn/r , (1.0-nn)/r )

   for d = 1:3
      dmat[d,1,1] = dhh[1] * U[d]
      dmat[d,2,2] = (ll * dhh[3] + (1.-ll) * dhh[4]) * U[d] + 2*l * hh[3] * dl[d] - 2*l * hh[4] * dl[d]
      dmat[d,3,3] = (mm * dhh[3] + (1.-mm) * dhh[4]) * U[d] + 2*m * hh[3] * dm[d] - 2*m * hh[4] * dm[d]
      dmat[d,4,4] = (nn * dhh[3] + (1.-nn) * dhh[4]) * U[d] + 2*n * hh[3] * dn[d] - 2*n * hh[4] * dn[d]
      dmat[d,1,2] = l * dhh[2] * U[d] + hh[2] * dl[d]
      dmat[d,1,3] = m * dhh[2] * U[d] + hh[2] * dm[d]
      dmat[d,1,4] = n * dhh[2] * U[d] + hh[2] * dn[d]
      dmat[d,2,1] = - dmat[d,1,2]
      dmat[d,3,1] = - dmat[d,1,3]
      dmat[d,4,1] = - dmat[d,1,4]
      dmat[d,2,3] = lm * (dhh[3] - dhh[4]) * U[d] + (dl[d] * m + l * dm[d]) * (hh[3] - hh[4])
      dmat[d,2,4] = ln * (dhh[3] - dhh[4]) * U[d] + (dl[d] * n + l * dn[d]) * (hh[3] - hh[4])
      dmat[d,3,4] = mn * (dhh[3] - dhh[4]) * U[d] + (dm[d] * n + m * dn[d]) * (hh[3] - hh[4])
      dmat[d,3,2] =  dmat[d,2,3]
      dmat[d,4,2] =  dmat[d,2,4]
      dmat[d,4,3] =  dmat[d,3,4]
   end
   return dmat
end



function _sk9_d!(U, r, hh, dhh, dmat)
   l, m, n = U[1], U[2], U[3]
   dl = ( (1.0-ll)/r ,     - lm/r ,     - ln/r )
   dm = (     - lm/r , (1.0-mm)/r ,     - mn/r )
   dn = (     - ln/r ,     - mn/r , (1.0-nn)/r )

   for d = 1 : dim
       # ss
       dmat[d,1,1] = dhh[1] * U[d]
       # sp
       dmat[d,1,2] = l * dhh[2] * U[d] + hh[2] * dl[d]
       dmat[d,1,3] = m * dhh[2] * U[d] + hh[2] * dm[d]
       dmat[d,1,4] = n * dhh[2] * U[d] + hh[2] * dn[d]
       dmat[d,2,1] = - dmat[d,1,2]
       dmat[d,3,1] = - dmat[d,1,3]
       dmat[d,4,1] = - dmat[d,1,4]
       # pp
       dmat[d,2,2] = l*l * dhh[3] * U[d] + (1.-l*l) * dhh[4] * U[d] + 2*l * hh[3] * dl[d] - 2*l * hh[4] * dl[d]
       dmat[d,3,3] = m*m * dhh[3] * U[d] + (1.-m*m) * dhh[4] * U[d] + 2*m * hh[3] * dm[d] - 2*m * hh[4] * dm[d]
       dmat[d,4,4] = n*n * dhh[3] * U[d] + (1.-n*n) * dhh[4] * U[d] + 2*n * hh[3] * dn[d] - 2*n * hh[4] * dn[d]
       dmat[d,2,3] = l * m * (dhh[3] - dhh[4]) * U[d] + (dl[d] * m + l * dm[d]) * (hh[3] - hh[4])
       dmat[d,2,4] = l * n * (dhh[3] - dhh[4]) * U[d] + (dl[d] * n + l * dn[d]) * (hh[3] - hh[4])
       dmat[d,3,4] = m * n * (dhh[3] - dhh[4]) * U[d] + (dm[d] * n + m * dn[d]) * (hh[3] - hh[4])
       dmat[d,3,2] =  dmat[d,2,3]
       dmat[d,4,2] =  dmat[d,2,4]
       dmat[d,4,3] =  dmat[d,3,4]
       # sd
       dmat[d,1,5] = √3 * l * m * dhh[5] * U[d] + √3 * ( dl[d] * m + l * dm[d] ) * hh[5]
       dmat[d,1,6] = √3 * m * n * dhh[5] * U[d] + √3 * ( dm[d] * n + m * dn[d] ) * hh[5]
       dmat[d,1,7] = √3 * l * n * dhh[5] * U[d] + √3 * ( dl[d] * n + l * dn[d] ) * hh[5]
       dmat[d,1,8] = √3/2 * (l^2 - m^2) * dhh[5] * U[d] + √3/2 * ( 2*l * dl[d] - 2*m * dm[d]) * hh[5]
       dmat[d,1,9] = ( n^2 - (l^2 + m^2)/2 ) * dhh[5] * U[d] + ( 2*n * dn[d] - (l * dl[d] + m * dm[d]) ) * hh[5]
       dmat[d,5,1] = dmat[d,1,5]
       dmat[d,6,1] = dmat[d,1,6]
       dmat[d,7,1] = dmat[d,1,7]
       dmat[d,8,1] = dmat[d,1,8]
       dmat[d,9,1] = dmat[d,1,9]
       # pd
       dmat[d,2,5] = √3 * l * l * m * dhh[6] * U[d] + m * (1.0 - 2.0 * l^2) * dhh[7] * U[d] +
                   √3 * (2*l * m * dl[d] + l^2 * dm[d]) * hh[6] +
                   ( dm[d] * (1.0 - 2.0 * l^2) - 4*m*l * dl[d] ) * hh[7]
       dmat[d,2,6] = √3 * l * m * n * dhh[6] * U[d] - 2.0 * l * m * n * dhh[7] * U[d] +
                   √3 * (dl[d] * m * n + l * dm[d] * n + l * m * dn[d]) * hh[6] -
                   2.0 * (dl[d] * m * n + l * dm[d] * n + l * m * dn[d]) * hh[7]
       dmat[d,2,7] = √3 * l * l * n * dhh[6] * U[d] + n * (1.0 - 2.0 * l^2) * dhh[7] * U[d] +
                   √3 * (2*l * dl[d] * n + l*l * dn[d]) * hh[6] +
                   ( dn[d] * (1.0 - 2.0 * l^2) - 4*l*n * dl[d] ) * hh[7]
       dmat[d,2,8] = √3/2 * l * (l^2 - m^2) * dhh[6] * U[d] + l * (1.0 - l^2 + m^2) * dhh[7] * U[d] +
                   √3/2 * (dl[d] * (l^2 - m^2) + l * (2*l * dl[d] - 2*m * dm[d])) * hh[6] +
                   (dl[d] * (1.0 - l^2 + m^2) + l * (-2*l * dl[d] + 2*m * dm[d]) ) * hh[7]
       dmat[d,2,9] = l * (n^2 - (l^2 + m^2)/2) * dhh[6] * U[d] - √3 * l * n^2 * dhh[7] * U[d] +
                   ( dl[d] * (n^2 - (l^2 + m^2)/2) + l * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] -
                   √3 * (dl[d] * n^2 + 2*l*n * dn[d])* hh[7]
       dmat[d,5,2] = - dmat[d,2,5]
       dmat[d,6,2] = - dmat[d,2,6]
       dmat[d,7,2] = - dmat[d,2,7]
       dmat[d,8,2] = - dmat[d,2,8]
       dmat[d,9,2] = - dmat[d,2,9]
       dmat[d,3,5] = √3 * l * m * m * dhh[6] * U[d] + l * (1.0 - 2.0 * m^2) * dhh[7] * U[d] +
                   √3 * (dl[d] * m^2 + 2*l*m * dm[d]) * hh[6] +
                   ( dl[d] * (1.0 - 2.0 * m^2) - 4*l*m * dm[d] ) * hh[7]
       dmat[d,3,6] = √3 * m * m * n * dhh[6] * U[d] + n * (1.0 - 2.0 * m^2) * dhh[7] * U[d] +
                   √3 * (2*m*n * dm[d] + m^2 * dn[d]) * hh[6] +
                   ( dn[d] * (1.0 - 2.0 * m^2) - 4*m*n * dm[d] ) * hh[7]
       dmat[d,3,7] = √3 * l * m * n * dhh[6] * U[d] - 2.0 * l * m * n * dhh[7] * U[d] +
                   ( dl[d] * m * n + l * dm[d] * n + l * m * dn[d] ) * ( √3 * hh[6] - 2.0 * hh[7] )
       dmat[d,3,8] = √3/2 * m * (l^2 - m^2) * dhh[6] * U[d] - m * (1.0 + l^2 - m^2) * dhh[7] * U[d] +
                   √3/2 * ( dm[d] * (l^2 - m^2) + m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[6] -
                   ( dm[d] * (1.0 + l^2 - m^2) + m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[7]
       dmat[d,3,9] = m * (n^2 - (l^2 + m^2)/2) * dhh[6] * U[d] - √3 * m * n^2 * dhh[7] * U[d] +
                   ( dm[d] * (n^2 - (l^2 + m^2)/2) + m * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] -
                   √3 * ( dm[d] * n^2 + 2*m*n * dn[d] ) * hh[7]
       dmat[d,5,3] = - dmat[d,3,5]
       dmat[d,6,3] = - dmat[d,3,6]
       dmat[d,7,3] = - dmat[d,3,7]
       dmat[d,8,3] = - dmat[d,3,8]
       dmat[d,9,3] = - dmat[d,3,9]
       dmat[d,4,5] = √3 * l * m * n * dhh[6] * U[d] - 2.0 * l * m * n * dhh[7] * U[d] +
                   ( dl[d] * m * n + l * dm[d] * n + l * m * dn[d] ) * ( √3 * hh[6] - 2.0 * hh[7] )
       dmat[d,4,6] = √3 * m * n * n * dhh[6] * U[d] + m * (1.0 - 2.0 * n^2) * dhh[7] * U[d] +
                   √3 * ( dm[d] * n^2 + 2*m*n * dn[d] ) * hh[6] +
                   ( dm[d] * (1.0 - 2.0 * n^2) - 4*m*n * dn[d] ) * hh[7]
       dmat[d,4,7] = √3 * l * n * n * dhh[6] * U[d] + l * (1.0 - 2.0 * n^2) * dhh[7] * U[d] +
                   √3 * ( dl[d] * n^2 + 2*l*n * dn[d] ) * hh[6] +
                   ( dl[d] * (1.0 - 2.0 * n^2) - 4*l*n * dn[d] ) * hh[7]
       dmat[d,4,8] = √3/2 * n * (l^2 - m^2) * dhh[6] * U[d] - n * (l^2 - m^2) * dhh[7] * U[d] +
                   √3/2 * ( dn[d] * (l^2 - m^2) + n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[6] -
                   ( dn[d] * (l^2 - m^2) + n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[7]
       dmat[d,4,9] = n * (n^2 - (l^2 + m^2)/2) * dhh[6] * U[d] + √3 * n * (l^2 + m^2) * dhh[7] * U[d] +
                   ( dn[d] * (n^2 - (l^2 + m^2)/2) +  n * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] +
                   √3 * ( dn[d] * (l^2 + m^2) + n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[7]
       dmat[d,5,4] = - dmat[d,4,5]
       dmat[d,6,4] = - dmat[d,4,6]
       dmat[d,7,4] = - dmat[d,4,7]
       dmat[d,8,4] = - dmat[d,4,8]
       dmat[d,9,4] = - dmat[d,4,9]
       # dd
       dmat[d,5,5] = 3.0 * l^2 * m^2 * dhh[8] * U[d] + (l^2 + m^2 - 4.0 * l^2 * m^2) * dhh[9] * U[d] +
             (n^2 + l^2 * m^2) * dhh[10] * U[d] +
             3.0 * ( 2*l * dl[d] * m^2 + l^2 * 2*m * dm[d] ) * hh[8] +
                   (2*l * dl[d] + 2*m * dm[d] - 8*l * dl[d] * m^2 - 8*m * l^2 * dm[d]) * hh[9] +
                   (2*n * dn[d] + 2*l * dl[d] * m^2 + 2*m * l^2 * dm[d]) * hh[10]
       dmat[d,6,6] = 3.0 * m^2 * n^2 * dhh[8] * U[d] + (m^2 + n^2 - 4.0 * m^2 * n^2) * dhh[9] * U[d] +
             (l^2 + m^2 * n^2) * dhh[10] * U[d] +
                   3.0 * ( 2*m * dm[d] * n^2 + m^2 * 2*n * dn[d] ) * hh[8] +
                   (2*m * dm[d] + 2*n * dn[d] - 8*m * dm[d] * n^2 - 8*n * m^2 * dn[d]) * hh[9] +
                   (2*l * dl[d] + 2*m * dm[d] * n^2 + 2*n * m^2 * dn[d]) * hh[10]
       dmat[d,7,7] = 3.0 * l^2 * n^2 * dhh[8] * U[d] + (l^2 + n^2 - 4.0 * l^2 * n^2) * dhh[9] * U[d] +
             (m^2 + l^2 * n^2) * dhh[10] * U[d] +
                   3.0 * ( 2*l * dl[d] * n^2 + l^2 * 2*n * dn[d] ) * hh[8] +
                   (2*l * dl[d] + 2*n * dn[d] - 8*l * dl[d] * n^2 - 8*n * l^2 * dn[d]) * hh[9] +
                   (2*m * dm[d] + 2*l * dl[d] * n^2 + 2*n * l^2 * dn[d]) * hh[10]
       dmat[d,8,8] = 3.0/4 * (l^2 - m^2)^2 * dhh[8] * U[d] + (l^2 + m^2 - (l^2 - m^2)^2) * dhh[9] * U[d] +
             (n^2 + (l^2 - m^2)^2 /4) * dhh[10] * U[d] +
                   3.0 * (l^2 - m^2) * (l * dl[d] - m * dm[d]) * hh[8] +
                   ( 2*l * dl[d] + 2*m * dm[d] - 4.0 * (l^2 - m^2) * (l * dl[d] - m * dm[d]) ) * hh[9] +
                   ( 2*n * dn[d] + (l^2 - m^2) * (l * dl[d] - m * dm[d]) ) * hh[10]
       dmat[d,9,9] = (n^2 - (l^2 + m^2) /2)^2 * dhh[8] * U[d] + 3.0 * n^2 * (l^2 + m^2) * dhh[9] * U[d] +
             3.0/4 * (l^2 + m^2)^2 * dhh[10] * U[d] +
                   2.0 * (n^2 - (l^2 + m^2) /2) * (2*n * dn[d] - (l * dl[d] + m * dm[d])) * hh[8] +
                   3.0 * ( 2*n * dn[d] * (l^2 + m^2) + n^2 * (2*l * dl[d] + 2*m * dm[d]) ) * hh[9] +
                   3.0 * (l^2 + m^2) * (l * dl[d] + m * dm[d]) * hh[10]
       dmat[d,5,6] = 3.0 * l * m^2 * n * dhh[8] * U[d] + l * n * (1.0 - 4.0 * m^2) * dhh[9] * U[d] +
             l * n * (m^2 - 1.0) * dhh[10] * U[d] +
                   3.0 * (dl[d] * m^2 * n + l * 2*m * dm[d] * n + l * m^2 * dn[d]) * hh[8] +
                   ( dl[d] * n * (1.0 - 4.0 * m^2) + l * dn[d] * (1.0 - 4.0 * m^2) - l * n * 8*m * dm[d] ) * hh[9] +
                   ( dl[d] * n * (m^2 - 1.0) + l * dn[d] * (m^2 - 1.0) + l * n * 2*m * dm[d] ) * hh[10]
       dmat[d,5,7] = 3.0 * l^2 * m * n * dhh[8] * U[d] + m * n * (1.0 - 4.0 * l^2) * dhh[9] * U[d] +
             m * n * (l^2 - 1.0) * dhh[10] * U[d] +
                   3.0 * (2*l * dl[d] * m * n + l^2 * dm[d] * n + l^2 * m * dn[d]) * hh[8] +
                   ( dm[d] * n * (1.0 - 4.0 * l^2) + m * dn[d] * (1.0 - 4.0 * l^2) - m * n * 8*l * dl[d] ) * hh[9] +
                   ( dm[d] * n * (l^2 - 1.0) + m * dn[d] * (l^2 - 1.0) + m * n * 2*l * dl[d] ) * hh[10]
       dmat[d,5,8] = 3.0/2 * l * m * (l^2 - m^2) * dhh[8] * U[d] + 2.0 * l * m * (m^2 - l^2) * dhh[9] * U[d] +
             1.0/2 * l * m * (l^2 - m^2) * dhh[10] * U[d] +
                   3.0/2 * ( dl[d] * m * (l^2 - m^2) + l * dm[d] * (l^2 - m^2) + l * m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] +
                   2.0 * ( dl[d] * m * (m^2 - l^2) + l * dm[d] * (m^2 - l^2) + l * m * (2*m * dm[d] - 2*l * dl[d]) ) * hh[9] +
                   ( dl[d] * m * (l^2 - m^2)/2 + l * dm[d] * (l^2 - m^2)/2 + l * m * (l*dl[d] - m*dm[d]) ) * hh[10]
       dmat[d,5,9] = √3 * l * m * (n^2 - (l^2 + m^2)/2) * dhh[8] * U[d] - 2.0*√3 * l * m * n^2 * dhh[9] * U[d] +
             √3/2 * l * m * (1.0 + n^2) * dhh[10] * U[d] +
                   √3 * ( dl[d] * m * (n^2 - (l^2 + m^2)/2) + l * dm[d] * (n^2 - (l^2 + m^2)/2) + l * m * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] -
                   2.0*√3 * ( dl[d] * m * n^2 + l * dm[d] * n^2 + l * m * 2*n * dn[d] ) * hh[9] +
                   √3/2 * ( dl[d] * m * (1.0 + n^2) + l * dm[d] * (1.0 + n^2) + l * m * 2*n * dn[d] ) * hh[10]
       dmat[d,6,7] = 3.0 * l * m * n^2 * dhh[8] * U[d] + l * m * (1.0 - 4.0 * n^2) * dhh[9] * U[d] +
             l * m * (n^2 - 1.0) * dhh[10] * U[d] +
                   3.0 * ( dl[d] * m * n^2 + l * dm[d] * n^2 + l * m * 2*n * dn[d] ) * hh[8] +
                   ( dl[d] * m * (1.0 - 4.0 * n^2) + l * dm[d] * (1.0 - 4.0 * n^2) - l * m * 8*n * dn[d] ) * hh[9] +
                   ( dl[d] * m * (n^2 - 1.0) + l * dm[d] * (n^2 - 1.0) + l * m * 2*n * dn[d] ) * hh[10]
       dmat[d,6,8] = 3.0/2 * m * n * (l^2 - m^2) * dhh[8] * U[d] - m * n * (1.0 + 2.0 * (l^2 - m^2)) * dhh[9] * U[d] +
             m * n * (1.0 + (l^2 - m^2) /2) * dhh[10] * U[d] +
                   3.0/2 * ( dm[d] * n * (l^2 - m^2) + m * dn[d] * (l^2 - m^2) + m * n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] -
                   ( dm[d] * n * (1.0 + 2.0 * (l^2 - m^2)) + m * dn[d] * (1.0 + 2.0 * (l^2 - m^2)) + m * n * 4.0 * (l * dl[d] - m * dm[d]) ) * hh[9] +
                   ( dm[d] * n * (1.0 + (l^2 - m^2) /2) + m * dn[d] * (1.0 + (l^2 - m^2) /2) + m * n * (l * dl[d] - m * dm[d]) ) * hh[10]
       dmat[d,6,9] = √3 * m * n * (n^2 - (l^2 + m^2) /2) * dhh[8] * U[d] + √3 * m * n * (l^2 + m^2 - n^2) * dhh[9] * U[d] -
             √3/2 * m * n * (l^2 + m^2) * dhh[10] * U[d] +
                   √3 * ( dm[d] * n * (n^2 - (l^2 + m^2) /2) + m * dn[d] * (n^2 - (l^2 + m^2) /2) +  m * n * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] +
                   √3 * ( dm[d] * n * (l^2 + m^2 - n^2) + m * dn[d] * (l^2 + m^2 - n^2) + m * n * (2*l * dl[d] + 2*m * dm[d] - 2*n * dn[d]) ) * hh[9] -
                   √3/2 * ( dm[d] * n * (l^2 + m^2) + m * dn[d] * (l^2 + m^2) + m * n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[10]
       dmat[d,7,8] = 3.0/2 * l * n * (l^2 - m^2) * dhh[8] * U[d] + l * n * (1.0 - 2.0 * (l^2 - m^2)) * dhh[9] * U[d] -
             l * n * (1.0 - (l^2 - m^2) /2) * dhh[10] * U[d] +
                   3.0/2 * ( dl[d] * n * (l^2 - m^2) + l * dn[d] * (l^2 - m^2) + l * n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] +
                   ( dl[d] * n * (1.0 - 2.0 * (l^2 - m^2)) + l * dn[d] * (1.0 - 2.0 * (l^2 - m^2)) - l * n * 4.0 * (l * dl[d] - m * dm[d]) ) * hh[9] -
                   ( dl[d] * n * (1.0 - (l^2 - m^2) /2) + l * dn[d] * (1.0 - (l^2 - m^2) /2) - l * n * (l * dl[d] - m * dm[d]) ) * hh[10]
       dmat[d,7,9] = √3 * l * n * (n^2 - (l^2 + m^2) /2) * dhh[8] * U[d] + √3 * l * n * (l^2 + m^2 - n^2) * dhh[9] * U[d] -
             √3/2 * l * n * (l^2 + m^2) * dhh[10] * U[d] +
                   √3 * ( dl[d] * n * (n^2 - (l^2 + m^2) /2) + l * dn[d] * (n^2 - (l^2 + m^2) /2) + l * n * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] +
                   √3 * ( dl[d] * n * (l^2 + m^2 - n^2) + l * dn[d] * (l^2 + m^2 - n^2) + l * n * (2*l * dl[d] + 2*m * dm[d] - 2*n * dn[d]) ) * hh[9] -
                   √3/2 * ( dl[d] * n * (l^2 + m^2) + l * dn[d] * (l^2 + m^2) + l * n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[10]
       dmat[d,8,9] = √3/2 * (l^2 - m^2) * (n^2 - (l^2 + m^2) /2) * dhh[8] * U[d] + √3 * n^2 * (m^2 - l^2) * dhh[9] * U[d] +
             √3/4 * (1.0 + n^2) * (l^2 - m^2) * dhh[10] * U[d] +
                   √3/2 * ( (2*l * dl[d] - 2*m * dm[d]) * (n^2 - (l^2 + m^2) /2) + (l^2 - m^2) * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] +
                   √3 * ( 2*n * dn[d] * (m^2 - l^2) + n^2 * (2*m * dm[d] - 2*l * dl[d]) ) * hh[9] +
                   √3/4 * ( 2*n * dn[d] * (l^2 - m^2) + (1.0 + n^2) * (2*l * dl[d] - 2*m * dm[d]) ) * hh[10]
       dmat[d,6,5] = dmat[d,5,6]
       dmat[d,7,5] = dmat[d,5,7]
       dmat[d,8,5] = dmat[d,5,8]
       dmat[d,9,5] = dmat[d,5,9]
       dmat[d,7,6] = dmat[d,6,7]
       dmat[d,8,6] = dmat[d,6,8]
       dmat[d,9,6] = dmat[d,6,9]
       dmat[d,8,7] = dmat[d,7,8]
       dmat[d,9,7] = dmat[d,7,9]
       dmat[d,9,8] = dmat[d,8,9]
   end

   return dmat
end
