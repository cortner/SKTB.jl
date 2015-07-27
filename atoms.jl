module atoms

export Atoms, convert, get_array, set_array!, get_positions, set_positions!
export get_cell, set_cell!
export set_calculator, get_forces, get_potential_energy, get_stress
export repeat, bulk, length
export neighbours

using PyCall
@pyimport ase
@pyimport ase.lattice as lattice

type Atoms
    po::PyObject # ase.Atoms instance
    
    Atoms(po) = new(po)
end

import PyCall.PyObject
PyObject(a::Atoms) = a.po

import Base.convert
convert{T<:Atoms}(::Type{T}, po::PyObject) = Atoms(po)

get_array(a::Atoms, name) = a.po[:get_array(name)]
set_array!(a::Atoms, name, value) = a.po[:set_array(name, value)]

get_positions(a::Atoms) = a.po[:get_positions]()
set_positions!(a::Atoms, p::Array{Float64,2}) = a.po[:set_positions](p)

get_cell(a::Atoms) = a.po[:get_cell]()
set_cell!(a::Atoms, p::Array{Float64,2}) = a.po[:set_cell](p)

set_calculator(a::Atoms, calculator::PyObject) = a.po[:set_calculator](calculator)

get_forces(a::Atoms) = a.po[:get_forces]()
get_potential_energy(a::Atoms) = a.po[:get_potential_energy]()
get_stress(a::Atoms) = a.po[:get_stress]()

import Base.repeat
repeat(a::Atoms, n::(Int64,Int64,Int64)) = convert(Atoms, a.po[:repeat](n))

bulk(name; kwargs...) = convert(Atoms, lattice.bulk(name; kwargs...))

import Base.length
length(a::Atoms) = a.po[:__len__]()

@pyimport matscipy.neighbours as matscipy_neighbours

function neighbours(atoms::Atoms, quantities, cutoff)
    results = matscipy_neighbours.neighbour_list(quantities,
                                                 PyObject(atoms),
                                                 cutoff)    
    results = collect(results) # tuple -> array so we can change in place
    # translate from 0- to 1-based indices
    for (idx, quantity) in enumerate(quantities)
        if quantity == 'i' || quantity == 'j'
            results[idx] += 1
        end
    end
   return results
end

end
