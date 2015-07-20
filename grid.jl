
module grid

export ndgrid, ndgrd_fill, meshgrid, dgrid, dgrid_list

#
#
# Grids and meshes
#
#

function ndgrid{T}(v1::AbstractVector{T}, v2::AbstractVector{T})
    m, n = length(v1), length(v2)
    v1 = reshape(v1, m, 1)
    v2 = reshape(v2, 1, n)
    (repmat(v1, 1, n), repmat(v2, m, 1))
end

function ndgrid_fill(a, v, s, snext)
    for j = 1:length(a)
        a[j] = v[div(rem(j-1, snext), s)+1]
    end
end

function ndgrid{T}(vs::AbstractVector{T}...)
    n = length(vs)
    sz = map(length, vs)
    out = ntuple(n, i->Array(T, sz))
    s = 1
    for i=1:n
        a = out[i]::Array
        v = vs[i]
        snext = s*size(a,i)
        ndgrid_fill(a, v, s, snext)
        s = snext
    end
    out
end


function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T},
                     vz::AbstractVector{T})
    m, n, o = length(vy), length(vx), length(vz)
    vx = reshape(vx, 1, n, 1)
    vy = reshape(vy, m, 1, 1)
    vz = reshape(vz, 1, 1, o)
    om = ones(Int, m)
    on = ones(Int, n)
    oo = ones(Int, o)
    (vx[om, :, oo], vy[:, on, oo], vz[om, on, :])
end


## function dgrid(vxyz, d)
# generate d-dimensional grid, return is a meshgrid-like d-tuple
function dgrid(vxyz, d)
    if d == 1
        return vxyz
    elseif d == 2
        return meshgrid(vxyz, vxyz)
    elseif d == 3
        return meshgrid(vxyz, vxyz, vxyz)
    else 
        throw(ArgumentError("dgrid: d must be 1, 2, or 3."))
    end
end

## function dgrid_list
# like dgrid, but returns a (d x npoints) array
function dgrid_list(vxyz, d)
    if d == 2
        order = [2, 1]
    elseif d == 3
        order = [2,1,3]
    end
    grids = dgrid(vxyz, d)
    if d == 1; grids = (grids,); end
    nX = length(grids[1])
    x = zeros(d, nX)
    for α = 1:d
        x[α, :] = reshape(grids[order[α]], (1, nX))
    end
    return x
end

end
