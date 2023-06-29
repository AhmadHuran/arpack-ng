
use std::f64::EPSILON;
use std::os::raw::c_double;

use arpack_ng_sys::*;
use ndarray::prelude::*;

use crate::{Error, Which, MUTEX};

pub fn eigenvalues<F>(
    av: F,
    n: usize,
    which: &Which,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<Array1<f64>, Error>
where
    F: FnMut(ArrayView1<f64>, ArrayViewMut1<f64>),
{
    let (res, _) = arpack_f64(av, n, which.as_str(), nev, ncv, maxiter, true)?;
    Ok(res)
}

pub fn eigenvectors<F>(
    av: F,
    n: usize,
    which: &Which,
    nev: usize,
    ncv: usize,
    maxiter: usize,
) -> Result<(Array1<f64>, Array2<f64>), Error>
where
    F: FnMut(ArrayView1<f64>, ArrayViewMut1<f64>),
{
    arpack_f64(av, n, which.as_str(), nev, ncv, maxiter, true)
}

fn arpack_f64<F>(
    mut av: F,
    n: usize,
    which: &str,
    nev: usize,
    ncv: usize,
    maxiter: usize,
    vectors: bool,
) -> Result<(Array1<f64>, Array2<f64>), Error>
where
    F: FnMut(ArrayView1<f64>, ArrayViewMut1<f64>),
{
    let g = MUTEX.lock().unwrap();
    let mut ido = 0;
    let mut resid: Array1<f64> = Array1::zeros(n);
    let mut v: Array2<f64> = Array2::zeros((n, ncv));
    let mut iparam = [0; 11];
    iparam[0] = 1;
    iparam[2] = maxiter as i32;
    iparam[6] = 1;
    let mut ipntr = [0; 14];
    let mut workd = Array1::zeros(3 * n);
    let lworkl = 3 * ncv.pow(2) + 6 * ncv;
    let mut workl: Array1<f64> = Array1::zeros(lworkl);
    let mut info = 0;
    while ido != 99 {
        unsafe {
            dsaupd_c(
                &mut ido,
                "I".as_ptr() as *const i8,
                n as i32,
                which.as_ptr() as *const i8,
                nev as i32,
                EPSILON,
                resid.as_mut_ptr() as *mut c_double,
                ncv as i32,
                v.as_mut_ptr() as *mut c_double,
                n as i32,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr() as *mut c_double,
                workl.as_mut_ptr() as *mut c_double,
                lworkl as i32,
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
    let mut d: Array1<f64> = Array1::zeros(nev + 1);
    let mut z: Array2<f64> = Array2::zeros((n, nev));
    unsafe {
        dseupd_c(
            vectors as i32,
            "A".as_ptr() as *const i8,
            select.as_ptr(),
            d.as_mut_ptr() as *mut c_double,
            z.as_mut_ptr() as *mut c_double,
            n as i32,
            0.0,
            "I".as_ptr() as *const i8,
            n as i32,
            which.as_ptr() as *const i8,
            nev as i32,
            EPSILON,
            resid.as_mut_ptr() as *mut c_double,
            ncv as i32,
            v.as_mut_ptr() as *mut c_double,
            n as i32,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr() as *mut c_double,
            workl.as_mut_ptr() as *mut c_double,
            lworkl as i32,
            &mut info,
        );
    }
    drop(g);
    Ok((d.slice(s![0..nev]).to_owned(), z))
}

