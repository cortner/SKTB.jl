using SKTB.FermiContour
using Test

# Compare the rate of convergence with the predicted rate.
# Not a proper unit test, but good enough for our purposes.


E1 = 0
E2 = 1
β = 10.0
n = 2:2:18
x = range(E1,stop=E2,length=100)

# Compute empirical convergence rate
error = zeros(length(n))
for i = 1:length(n)
    w,z = fermicontour(E1,E2,β,n[i])
    fx = sum(real(w*fermidirac.(z,β) ./ (z  .- x)) for (w,z) in zip(w,z))
    error[i] = maximum(abs.(fermidirac.(x,β) .- fx))
end

# Compute theoretical convergence rate
m = E1^2 + π^2/β^2
M = E2^2 + π^2/β^2
k = (sqrt(M/m)-1)/(sqrt(M/m)+1)
K = SKTB.FermiContour.JacobiFunc.K(k^2)
iK = SKTB.FermiContour.JacobiFunc.iK(k^2)
predicted = 2 * exp.(-π*iK/(2K)*n)

println("n   error    predicted    err/pred")
display([n  error predicted error./predicted]); println()
@test maximum(abs.(error./predicted .- 1.0)) < 0.1

# TODO: implement analogous test for 0T contour!

