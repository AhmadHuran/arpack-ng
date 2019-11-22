use arpack_ng_sys::*;
use ndarray::prelude::*;
use num_complex::Complex64;
use std::f64::EPSILON;

#[derive(Debug)]
pub enum Error {
    NonSquare,
    Other,
}

pub trait Arpack {
    type Result;

    fn eigenvalues(&self, nev: usize, ncv: usize, maxiter: usize) -> Result<Self::Result, Error>;
}

const ZERO: __BindgenComplex<f64> = __BindgenComplex { re: 0., im: 0. };

impl Arpack for Array2<Complex64> {
    type Result = Array1<Complex64>;

    fn eigenvalues(&self, nev: usize, ncv: usize, maxiter: usize) -> Result<Self::Result, Error> {
        if !self.is_square() {
            return Err(Error::NonSquare);
        }
        let n = self.dim().0;
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
                workd
                    .slice_mut(s![ipntr[1] as usize - 1..ipntr[1] as usize + n - 1])
                    .assign(&self.dot(&v));
            }
        }
        if info < 0 {
            return Err(Error::Other);
        }

        let select = vec![false; ncv];
        let mut d: Array1<Complex64> = Array1::zeros(nev + 1);
        let mut z: Array2<Complex64> = Array2::zeros((n, nev));
        let mut workev: Array1<Complex64> = Array1::zeros(2 * ncv);
        unsafe {
            zneupd_c(
                false as i32,
                "A".as_ptr() as *const i8,
                select.as_ptr() as *const i32,
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
        Ok(d.slice(s![0..nev]).to_owned())
    }
}
