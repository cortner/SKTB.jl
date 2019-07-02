# SKTB.jl

[![Build Status](https://travis-ci.org/cortner/SKTB.jl.svg?branch=master)](https://travis-ci.org/cortner/SKTB.jl)

[![Coverage Status](https://coveralls.io/repos/cortner/SKTB.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/cortner/SKTB.jl?branch=master)

[![codecov.io](http://codecov.io/github/cortner/SKTB.jl/coverage.svg?branch=master)](http://codecov.io/github/cortner/SKTB.jl?branch=master)

This is an implementation of Slater-Koster type tight-binding models.
The intention is to have a flexible but fairly performant tight-binding
code to experiment with new models, and non-standard multi-scale schemes.

## Installation

`SKTB.jl` is not registered, install by cloning:
```julia
Pkg.clone("https://github.com/cortner/SKTB.jl.git")
```

The `master` branch requires `Julia v0.6`. To use `SKTB.jl` with
`Julia v0.5` checkout the `v0.5` branch: from the Julia REPL this can be
achieved via
```julia
cd(Pkg.dir("SKTB"))
run(`git checkout v0.5`)
```

Please run
```julia
Pkg.test("SKTB")
```
and file an issue if there are any failed tests.


## Authors

This module was written by [Huajie Chen](https://github.com/hjchen1983) and [Christoph Ortner](http://homepages.warwick.ac.uk/staff/C.Ortner/), the FermiContour
submodule was merged from [FermiContour.jl](https://github.com/ettersi/FermiContour.jl)
by [ettersi](https://github.com/ettersi).
