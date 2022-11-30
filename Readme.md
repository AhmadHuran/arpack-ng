# arpack-ng

This package provides Rust bindings for [https://github.com/opencollab/arpack-ng](arpack-ng), a library solve large eigenvalue problems.

Currently only the functionality of `zneupd` is exposed.
This means that this library can be used to calculate eigenvalues and vectors using **I**mplicitely **R**estarted **A**rnoldi **M**ethod.

Supported are `ndarray` matrices (or alternative dynamically sized `nalgebra` matrices with the `nalgebra` flag) or closures that calculate a matrix vector multiplication.

Refer to the `simple` example for the matrix case and the `closure` example otherwise.