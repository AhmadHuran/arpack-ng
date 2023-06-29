use arpack_ng_sys::*;
use lazy_static::lazy_static;
use std::{fmt, sync::Mutex};

#[cfg(feature = "ndarray")]
mod ndarray;
#[cfg(feature = "ndarray")]
pub use crate::ndarray::*;

#[cfg(feature = "nalgebra")]
mod nalgebra;
#[cfg(feature = "nalgebra")]
pub use crate::nalgebra::*;

lazy_static! {
    static ref MUTEX: Mutex<()> = Mutex::new(());
}

#[derive(Debug)]
pub enum Error {
    NonSquare,
    IllegalParameters(String),
    Other(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::NonSquare => f.write_str("Non square matrix."),
            Error::IllegalParameters(s) => write!(f, "Invalid parameters: {}", s),
            Error::Other(i) => write!(f, "Arpack error (code {}", i),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Which {
    LargestAlgebraic,
    SmallestAlgebraic,
    LargestMagnitude,
    SmallestMagnitude,
    LargestRealPart,
    SmallestRealPart,
    LargestImaginaryPart,
    SmallestImaginaryPart,
}

impl Which {
    fn as_str(&self) -> &'static str {
        match self {
            Which::LargestAlgebraic => "LA",
            Which::SmallestAlgebraic => "SA",
            Which::LargestMagnitude => "LM",
            Which::SmallestMagnitude => "SM",
            Which::LargestRealPart => "LR",
            Which::SmallestRealPart => "SR",
            Which::LargestImaginaryPart => "LI",
            Which::SmallestImaginaryPart => "SI",
        }
    }
}

impl std::error::Error for Error {}

pub trait Arpack {
    type Result;
    type ResultVec;

    fn eigenvalues(
        &self,
        which: &Which,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::Result, Error>;
    fn eigenvectors(
        &self,
        which: &Which,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::ResultVec, Error>;
}

const ZERO: __BindgenComplex<f64> = __BindgenComplex { re: 0., im: 0. };
