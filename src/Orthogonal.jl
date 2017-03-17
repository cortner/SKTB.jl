
module Orthogonal

using Parameters, StaticArrays
using TightBinding: TightBindingModel
using ReverseDiffSource

abstract OrthogonalTightBinding # <: TightBindingModel

# α   1     2    3     4
#    ssσ   spσ  ppσ   ppπ

@with_kw type KwonParams
   r0::Float64 = 2.360352   # Å
   Es::Float64 = 5.25       # eV
   Ep::Float64 = 1.2        # eV
   E0::Float64 = 8.7393204  # eV
   # ------------------------------- Electronic Parameters
   hr0::NTuple{4, Float64} = (-2.038, 1.745, 2.75, -1.075)
   nc::NTuple{4, Float64} = (9.5, 8.5, 7.5, 7.5)
   rc::NTuple{4, Float64} = (3.4, 3.55, 3.7, 3.7)
   # ------------------------------- 2-body parameters
   m::Float64 = 6.8755
   mc::Float64 = 13.017
   dc::Float64 = 3.66995     # Å
   C::NTuple{4, Float64} = (2.1604385, -0.1384393, 5.8398423e-3, -8.0263577e-5)
   # ----------------------------- storage
   mat::MMatrix{4, 4, Float64, 16} = MMatrix{4, 4}(zeros(16))
end

# hopping function
h(tbm::KwonParams, r, α) = ( tbm.hr0[α] * (tbm.r0 / r)^2 *
   exp( - 2 * (r/tbm.rc[α])^tbm.nc[α] + 2 * (tbm.r0/tbm.rc[α])^tbm.nc[α] ) )
dh_rdiff = rdiff(h, (KwonParams, Float64, Int); order = 1, allorders = false, ignore = (:tbm, :α))
dh(tbm::KwonParams, r, α) = dh_rdiff(tbm, r, α)[1]

# embedding function
f(tbm::KwonParams, x) =
   x * (tbm.C[1] + x * (tbm.C[2] + x * (tbm.C[3] + x * tbm.C[4])))
df_rdiff = rdiff(f, (KwonParams, Float64); order = 1, allorders = false, ignore = (:tbm,))
df(tbm::KwonParams, x) = df_rdiff(tbm, x)[1]

# pair potential
phi(tbm::KwonParams, r) = ( (tbm.r0/r)^tbm.m *
   exp( - tbm.m * (r/tbm.dc)^tbm.mc + tbm.m * (tbm.r0/tbm.dc)^tbm.mc ) )



function sk4!(U, hop, mat)
   l, m, n = U[1], U[2], U[3]
   # 4 orbitals are s, px, py, pz, these are the mat-indices
   # 4 bond types are : ssσ,spσ,ppσ,ppπ, these are the hop-indices
   mat[1,1] = hop[1]                            # E_ss = V_ssσ
   mat[2,2] = l*l * hop[3] + (1.-l*l) * hop[4]  # E_xx = l² V_ppσ + (1-l²) V_ppπ
   mat[3,3] = m*m * hop[3] + (1.-m*m) * hop[4]  # E_yy = m² V_ppσ + (1-m²) V_ppπ
   mat[4,4] = n*n * hop[3] + (1.-n*n) * hop[4]  # E_zz = n² V_ppσ + (1-n²) V_ppπ
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
end


function d_sk4!(U, task::Symbol, dh::Array{Float64, 3})
   l, m, n = U[1], U[2], U[3]
    R = r
    dR = (l, m, n)
    dl = (1./R - l*l/R , - l*m/R , - l*n/R)
    dm = (- l*m/R , 1./R - m*m/R , - m*n/R)
    dn = (- l*n/R , - m*n/R , 1./R - n*n/R)

    # use different functions for different tasks
    if task == :dH
        hh = Float64[ h_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
        dhh = Float64[ dR_h_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    elseif task == :dM
      hh = zeros(Nb)
      dhh = zeros(Nb)
      for bond_type = 1:Nb
         hh[bond_type] = m_hop(r, bond_type, elem)
         dhh[bond_type] = dR_m_hop(r, bond_type, elem)
      end
      #   hh = Float64[ m_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
      #   dhh = Float64[ dR_m_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    else
        throw(ArgumentError("this task has not been implemented yet"))
    end

       for d = 1 : dim
            dh[d,1,1] = dhh[1] * dR[d]
            dh[d,2,2] = l*l * dhh[3] * dR[d] + (1.-l*l) * dhh[4] * dR[d] + 2*l * hh[3] * dl[d] - 2*l * hh[4] * dl[d]
            dh[d,3,3] = m*m * dhh[3] * dR[d] + (1.-m*m) * dhh[4] * dR[d] + 2*m * hh[3] * dm[d] - 2*m * hh[4] * dm[d]
            dh[d,4,4] = n*n * dhh[3] * dR[d] + (1.-n*n) * dhh[4] * dR[d] + 2*n * hh[3] * dn[d] - 2*n * hh[4] * dn[d]
            dh[d,1,2] = l * dhh[2] * dR[d] + hh[2] * dl[d]
            dh[d,1,3] = m * dhh[2] * dR[d] + hh[2] * dm[d]
            dh[d,1,4] = n * dhh[2] * dR[d] + hh[2] * dn[d]
            dh[d,2,1] = - dh[d,1,2]
            dh[d,3,1] = - dh[d,1,3]
            dh[d,4,1] = - dh[d,1,4]
            dh[d,2,3] = l * m * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * m + l * dm[d]) * (hh[3] - hh[4])
            dh[d,2,4] = l * n * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * n + l * dn[d]) * (hh[3] - hh[4])
            dh[d,3,4] = m * n * (dhh[3] - dhh[4]) * dR[d] + (dm[d] * n + m * dn[d]) * (hh[3] - hh[4])
            dh[d,3,2] =  dh[d,2,3]
            dh[d,4,2] =  dh[d,2,4]
            dh[d,4,3] =  dh[d,3,4]
        end
end




@pot type KwonHop <: PairPotential
    params::KwonParams
end

function evaluate!(p::KwonHop, r, R, H, temp)
   U = R / r
   for n = 1:4
      temp[n] = h(p.params, r, n)
   end
   sk4!(U, temp, H)
   return H
end

function grad!(p::KwonHop, r, R, dH)
   temp = [dh(p.params, r, n) for n = 1:4]
   d_mat_local!(r/BOHR, R/BOHR, p.elem, :dH, dH)
   scale!(dH, 1.0/BOHR)
end



# Kwon TB on-site term

@pot type KwonOnsite <: Potential
   elem::KwonParams
end


# Kwon TB Overlap matrix: this is orthogonal TB i.e. just an identity

@pot type KwonOverlap <: PairPotential
    elem::KwonParams
end

function evaluate!(p::KwonOverlap, r, R, M, temp)
   fill!(M, 0.0)
   for n = 1:3; M[n,n] = 1.0; end
   return M
end

grad!(p::KwonOverlap, r, R, dM::Array{Float64, 3}) = fill!(dM, 0.0)




end
