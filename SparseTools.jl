
module SparseTools

using Base.Markdown

export sparse_flexible,
       sparse_static
       # bilinear_form,
       # N_quad_forms


#########################################
###    Some Sparse Matrix Tools       ###
#########################################


"""
A Sparse matrix that we can add entries to using +=
and which is grown automatically (if necessary). Usage:
```{.julia}
At = SparseTriplet(N)        # Initialise the triplet format matrix
# . . . some code to compute
#        m, n : (arrays of) integers, row and col indices
#         a  : length(m) x length(n) full matrix
At[m, n] += a                # add to the entries of A
# . . . more adding to A . . .
A = sparse(At)               # convert to CCS format
```
"""
type SparseTriplet{T}
    I::Array{Int32,1}
    J::Array{Int32,1}
    #V::Array{Float64,1}
    V::Array{T,1}
    idx::Int32
end

#  default constructor for the SparseTriplet type
#  returns an empty SparseTriplet matrix with
#  N possible triplet entries
# (doc is included in doc for SparseTriplet type)
SparseTriplet(N) = SparseTriplet(zeros(Int32,N), zeros(Int32,N), zeros(Float64,N), 0)
# Further add a constructor
SparseTriplet(N, T::DataType) =
    SparseTriplet(zeros(Int32,N), zeros(Int32,N), zeros(T,N), 0)


"""
function to automatically grow the storage for the triplet format,
called from setindex!, if not enough storage is left
"""
function grow!(A::SparseTriplet)
    N = length(A.I)
    Nnew = 2 * N
    Iold = A.I; Jold = A.J; Vold = A.V
    A.I = zeros(Int32,Nnew)
    A.I[1:N] = Iold
    A.J = zeros(Int32,Nnew)
    A.J[1:N] = Jold
    A.V = zeros(Float64,Nnew)
    A.V[1:N] = Vold
end
function grow!(A::SparseTriplet, T::DataType)
    N = length(A.I)
    Nnew = 2 * N
    Iold = A.I; Jold = A.J; Vold = A.V
    A.I = zeros(Int32,Nnew)
    A.I[1:N] = Iold
    A.J = zeros(Int32,Nnew)
    A.J[1:N] = Jold
    A.V = zeros(T,Nnew)
    A.V[1:N] = Vold
end




# This setindex is designed so that
#   A[iRow, iCol] += v
# just adds A[iRow[j], iCol[k]] += v[j,k]
# relies also on the getindex method below
# (hidden so no doc needed)
import Base.setindex!
"`setindex!(A::SparseTriplet, v, iRow, iCol)`: see documentation for `SparseTriplet`"
function setindex!(A::SparseTriplet, v, iRow, iCol)
    if A.idx == length(A.I)
        grow!(A)
    end
    for i=1:length(iRow), j = 1:length(iCol)
        A.idx += 1
        A.I[A.idx] = iRow[i]
        A.J[A.idx] = iCol[j]
        A.V[A.idx] = v[i,j]
    end
end

# this works together with setindex! above to make += work
import Base.getindex
"`getindex(A::SparseTriplet, v, iRow, iCol)`: see documentation for `SparseTriplet`"
getindex(A::SparseTriplet, iRow, iCol) = zeros(length(iRow), length(iCol))

# we need a conversion to sparse matrix format
# to do this we first need to explicityly import Base.sparse
# (probably because we are overloading it?)
import Base.sparse
"Convert `SparseTriplet` to `SparseMatrixCSC`"
sparse(A::SparseTriplet) =
    sparse( A.I[1:A.idx], A.J[1:A.idx], A.V[1:A.idx] );


# to be able to write code that does not know about what format is used
# for assembly, we provide a generic constructor
"""
Creates an empty sparse matrix, suitable for assembly.
The current version uses the SparseTriplet type.
"""
sparse_flexible(N) = SparseTriplet(N)
sparse_flexible(N, T) = SparseTriplet(N, T)

# . . . and a generic function to convert it to CCS
"""
Converts a flexible sparse format to a static sparse format (CCS)
used for fast algebra.
"""
sparse_static(A::SparseTriplet) = sparse(A)


# some multigrid functionality
# include("multigrid.jl")

# conjugate gradients
#  TODO: have Julia wrappers for a PCG solver




####### The next few functions were used to optimise the NRLTB module.
####### They could probably be removed soon since we are not using
####### these datastructures anymore

# it turns out that it is useful to have a bilinear form implemented in
# SparseTripletFormat:
"""`bilinear_form(A::SparseTriplet, x, y)` :
# returns x' * A * y
# """
function bilinear_form(A::SparseTriplet, x::Vector{Float64}, y::Vector{Float64})
    ljerror("""the function `LjSparse.bilinear_form` has been removed; add 
            from a previous commit""")
end



function N_quad_forms(A::SparseTriplet, C::Array{Float64, 2}, f::Vector{Float64})
    ljerror("""the function `LjSparse.N_quad_forms` has been removed; add
            from a previous commit""" )
end


end
