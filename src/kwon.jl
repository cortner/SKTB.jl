


# α   1     2    3     4
#    ssσ   spσ  ppσ   ppπ

@with_kw type KwonHamiltonian
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
