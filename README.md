# TightBinding.jl

[![Build Status](https://travis-ci.org/cortner/TightBinding.jl.svg?branch=master)](https://travis-ci.org/cortner/TightBinding.jl)

[![Coverage Status](https://coveralls.io/repos/cortner/TightBinding.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/cortner/TightBinding.jl?branch=master)

[![codecov.io](http://codecov.io/github/cortner/TightBinding.jl/coverage.svg?branch=master)](http://codecov.io/github/cortner/TightBinding.jl?branch=master)

This is an implementation of Slater-Koster type tight-binding models.
The intention is to have a flexible but fairly performant tight-binding
code to experiment with new models, and non-standard multi-scale schemes.

## Installation

The module depends on the latest versions of [JuLIP.jl](https://github.com/libAtoms/JuLIP.jl) and
[FermiContour.jl](https://github.com/ettersi/FermiContour.jl). Install
these via
```
Pkg.add("JuLIP")
Pkg.checkout("JuLIP")
Pkg.clone("https://github.com/ettersi/FermiContour.jl")
```


## Authors

This module was written by [Huajie Chen](https://github.com/hjchen1983) and [Christoph Ortner](http://homepages.warwick.ac.uk/staff/C.Ortner/).
