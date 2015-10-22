
module TestAtoms

using Potentials, AtomsInterface, ASE, MatSciPy

export test_ScalarFunction, test_potentialenergy

"""`test_ScalarFunction(p, r0)`

finite difference test for `ScalarFunction` implementations
"""
function test_ScalarFunction(p::SimpleFunction, r0)
    r0 = r0[:]
    f = p(r0)
    df = @D p(r0)   # evaluate_d(p, r0)
    println("-----------------------------")
    println("  p | error ")
    println("----|------------------------")
    for q = 2:10
        h = 0.1.^q
        dfh = (p(r0 + h) - f) / h
        @printf(" %2d | %1.7e \n", q, norm(df - dfh, Inf))
    end
    println("-----------------------------")
end


"""`test_potentialenergy`

finite difference test for consistency of `potentialenergy` with 
`potential_energy_d`.
"""
function test_potentialenergy(calc::AbstractCalculator, at::AbstractAtoms)
    X = positions(at)
    f = potential_energy(at, calc)
    df = potential_energy_d(at, calc)[:]
    println("-----------------------------")
    println("  p | error ")
    println("----|------------------------")
    for p = 2:10
        h = 0.1^p
        dfh = zeros(length(df))
        for n = 1:length(df)
            X[n] += h
            set_positions!(at, X)
            dfh[n] = (potential_energy(at, calc) - f) / h
            X[n] -= h
        end
        @printf(" %2d | %1.7e \n", p, norm(df - dfh, Inf))
    end
    println("-----------------------------")    
end




end
