use arpack_ng_sys::*;
use lazy_static::lazy_static;
use ndarray::prelude::*;
use num_complex::Complex64;
use std::{f64::EPSILON, sync::Mutex, fmt};

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
            Error::Other(i) => write!(f, "Arpack error (code {}", i)
        }
    }
}

impl std::error::Error for Error {}

pub trait Arpack {
    type Result;
    type ResultVec;

    fn eigenvalues(&self, nev: usize, ncv: usize, maxiter: usize) -> Result<Self::Result, Error>;
    fn eigenvectors(
        &self,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::ResultVec, Error>;
}

const ZERO: __BindgenComplex<f64> = __BindgenComplex { re: 0., im: 0. };

impl Arpack for Array2<Complex64> {
    type Result = Array1<Complex64>;
    type ResultVec = (Array1<Complex64>, Array2<Complex64>);

    fn eigenvalues(&self, nev: usize, ncv: usize, maxiter: usize) -> Result<Self::Result, Error> {
        if !self.is_square() {
            return Err(Error::NonSquare);
        }
        let n = self.dim().0;
        let (val, _) = arpack_c64(
            |v1, mut v2| v2.assign(&self.dot(&v1)),
            n,
            nev,
            ncv,
            maxiter,
            true,
        )?;
        Ok(val)
    }

    fn eigenvectors(
        &self,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::ResultVec, Error> {
        if !self.is_square() {
            return Err(Error::NonSquare);
        }
        let n = self.dim().0;
        arpack_c64(
            |v1, mut v2| v2.assign(&self.dot(&v1)),
            n,
            nev,
            ncv,
            maxiter,
            true,
        )
    }
}

pub fn eigenvalues<F>(
    av: F,
    n: usize,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<Array1<Complex64>, Error>
where
    F: FnMut(ArrayView1<Complex64>, ArrayViewMut1<Complex64>),
{
    let (res, _) = arpack_c64(av, n, nev, ncv, maxiter, true)?;
    Ok(res)
}

pub fn eigenvectors<F>(
    av: F,
    n: usize,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<(Array1<Complex64>, Array2<Complex64>), Error>
where
    F: FnMut(ArrayView1<Complex64>, ArrayViewMut1<Complex64>),
{
    arpack_c64(av, n, nev, ncv, maxiter, true)
}

fn arpack_c64<F>(
    mut av: F,
    n: usize,
    nev: usize,
    ncv: usize,
    maxiter: usize,
    vectors: bool,
) -> Result<(Array1<Complex64>, Array2<Complex64>), Error>
where
    F: FnMut(ArrayView1<Complex64>, ArrayViewMut1<Complex64>),
{
    let g = MUTEX.lock().unwrap();
    let mut ido = 0;
    let mut resid: Array1<Complex64> = Array1::zeros(n);
    let mut v: Array2<Complex64> = Array2::zeros((n, ncv));
    let mut iparam = [0; 11];
    iparam[0] = 1;
    iparam[2] = maxiter as i32;
    iparam[6] = 1;
    let mut ipntr = [0; 14];
    let mut workd = Array1::zeros(3 * n);
    let lworkl = 3 * ncv.pow(2) + 6 * ncv;
    let mut workl: Array1<Complex64> = Array1::zeros(lworkl);
    let mut rwork = vec![0.; ncv];
    let mut info = 0;
    while ido != 99 {
        unsafe {
            znaupd_c(
                &mut ido,
                "I".as_ptr() as *const i8,
                n as i32,
                "LR".as_ptr() as *const i8,
                nev as i32,
                EPSILON,
                resid.as_mut_ptr() as *mut __BindgenComplex<f64>,
                ncv as i32,
                v.as_mut_ptr() as *mut __BindgenComplex<f64>,
                n as i32,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr() as *mut __BindgenComplex<f64>,
                workl.as_mut_ptr() as *mut __BindgenComplex<f64>,
                lworkl as i32,
                rwork.as_mut_ptr(),
                &mut info,
            );
        }
        if (ido == -1) || (ido == 1) {
            let v = workd
                .slice(s![ipntr[0] as usize - 1..ipntr[0] as usize + n - 1])
                .to_owned();
            av(
                v.view(),
                workd.slice_mut(s![ipntr[1] as usize - 1..ipntr[1] as usize + n - 1]),
            );
        }
    }
    match info {
        0 | 1 | 2 => {}
        -1 => return Err(Error::IllegalParameters("N must be positive.".to_string())),
        -2 => {
            return Err(Error::IllegalParameters(
                "NEV must be positive.".to_string(),
            ))
        }
        -3 => {
            return Err(Error::IllegalParameters(
                "NCV-NEV >= 2 and less than or equal to N.".to_string(),
            ))
        }
        -4 => {
            return Err(Error::IllegalParameters(
                "Maximum iterations must be greater than 0.".to_string(),
            ))
        }
        -5 => {
            return Err(Error::IllegalParameters(
                "Maximum iterations must be greater than 0.".to_string(),
            ))
        }
        i => return Err(Error::Other(i)),
    }

    let select = vec![false as i32; ncv];
    let mut d: Array1<Complex64> = Array1::zeros(nev + 1);
    let mut z: Array2<Complex64> = Array2::zeros((n, nev));
    let mut workev: Array1<Complex64> = Array1::zeros(2 * ncv);
    unsafe {
        zneupd_c(
            vectors as i32,
            "A".as_ptr() as *const i8,
            select.as_ptr(),
            d.as_mut_ptr() as *mut __BindgenComplex<f64>,
            z.as_mut_ptr() as *mut __BindgenComplex<f64>,
            n as i32,
            ZERO,
            workev.as_mut_ptr() as *mut __BindgenComplex<f64>,
            "I".as_ptr() as *const i8,
            n as i32,
            "LR".as_ptr() as *const i8,
            nev as i32,
            EPSILON,
            resid.as_mut_ptr() as *mut __BindgenComplex<f64>,
            ncv as i32,
            v.as_mut_ptr() as *mut __BindgenComplex<f64>,
            n as i32,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr() as *mut __BindgenComplex<f64>,
            workl.as_mut_ptr() as *mut __BindgenComplex<f64>,
            lworkl as i32,
            rwork.as_mut_ptr(),
            &mut info,
        );
    }
    drop(g);
    Ok((d.slice(s![0..nev]).to_owned(), z))
}
