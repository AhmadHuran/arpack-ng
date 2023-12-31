use std::f64::EPSILON;

use arpack_ng_sys::*;
use nalgebra::{DMatrix, DVector, DVectorSliceMut};
use num_complex::Complex64;

use crate::{Arpack, Error, Which, MUTEX, ZERO};

impl Arpack for DMatrix<Complex64> {
    type Result = DVector<Complex64>;
    type ResultVec = (DVector<Complex64>, DMatrix<Complex64>);

    fn eigenvalues(
        &self,
        which: &Which,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::Result, Error> {
        if !self.is_square() {
            return Err(Error::NonSquare);
        }
        let n = self.ncols();
        let (val, _) = arpack_c64(
            |v1, mut v2| v2.copy_from(&(self * v1)),
            n,
            which.as_str(),
            nev,
            ncv,
            maxiter,
            true,
        )?;
        Ok(val)
    }

    fn eigenvectors(
        &self,
        which: &Which,
        nev: usize,
        ncv: usize,
        maxiter: usize,
    ) -> Result<Self::ResultVec, Error> {
        if !self.is_square() {
            return Err(Error::NonSquare);
        }
        let n = self.ncols();
        arpack_c64(
            |v1, mut v2| v2.copy_from(&(self * v1)),
            n,
            which.as_str(),
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
    which: &Which,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<DVector<Complex64>, Error>
where
    F: FnMut(DVector<Complex64>, DVectorSliceMut<Complex64>),
{
    let (res, _) = arpack_c64(av, n, which.as_str(), nev, ncv, maxiter, true)?;
    Ok(res)
}

pub fn eigenvectors<F>(
    av: F,
    n: usize,
    which: &Which,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<(DVector<Complex64>, DMatrix<Complex64>), Error>
where
    F: FnMut(DVector<Complex64>, DVectorSliceMut<Complex64>),
{
    arpack_c64(av, n, which.as_str(), nev, ncv, maxiter, true)
}

fn arpack_c64<F>(
    mut av: F,
    n: usize,
    which: &str,
    nev: usize,
    ncv: usize,
    maxiter: usize,
    vectors: bool,
) -> Result<(DVector<Complex64>, DMatrix<Complex64>), Error>
where
    F: FnMut(DVector<Complex64>, DVectorSliceMut<Complex64>),
{
    let g = MUTEX.lock().unwrap();
    let mut ido = 0;
    let mut resid: DVector<Complex64> = DVector::zeros(n);
    let mut v: DMatrix<Complex64> = DMatrix::zeros(n, ncv);
    let mut iparam = [0; 11];
    iparam[0] = 1;
    iparam[2] = maxiter as i32;
    iparam[6] = 1;
    let mut ipntr = [0; 14];
    let mut workd = DVector::zeros(3 * n);
    let lworkl = 3 * ncv.pow(2) + 6 * ncv;
    let mut workl: DVector<Complex64> = DVector::zeros(lworkl);
    let mut rwork = vec![0.; ncv];
    let mut info = 0;
    while ido != 99 {
        unsafe {
            znaupd_c(
                &mut ido,
                "I".as_ptr() as *const i8,
                n as i32,
                which.as_ptr() as *const i8,
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
            let v = DVector::from(workd.rows(ipntr[0] as usize - 1, n));
            av(v, workd.rows_mut(ipntr[1] as usize - 1, n))
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
    let mut d: DVector<Complex64> = DVector::zeros(nev + 1);
    let mut z: DMatrix<Complex64> = DMatrix::zeros(n, nev);
    let mut workev: DVector<Complex64> = DVector::zeros(2 * ncv);
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
    Ok((d.rows(0, nev).into(), z))
}
