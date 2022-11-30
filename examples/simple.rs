use arpack_ng::*;
#[cfg(feature = "nalgebra")]
use nalgebra::DMatrix;
#[cfg(feature = "ndarray")]
use ndarray::prelude::*;
#[cfg(feature = "nalgebra")]
use num_complex::Complex64;

#[cfg(feature = "ndarray")]
fn main() -> Result<(), Error> {
    let m = Array2::ones((200, 200));
    let (val, vec) = m.eigenvectors(&Which::LargestRealPart, 2, 50, 100)?;
    println!("{:?} {:?}", val.shape(), vec.shape());
    for i in 0..val.len() {
        println!("{} => {:?}", val[i], vec.slice(s![.., i]));
    }
    Ok(())
}

#[cfg(feature = "nalgebra")]
fn main() -> Result<(), Error> {
    let m = DMatrix::from_element(200, 200, Complex64::new(1., 0.));
    let (val, vec) = m.eigenvectors(&Which::LargestRealPart, 2, 50, 100)?;
    println!("{:?} {:?}", val.shape(), vec.shape());
    for i in 0..val.len() {
        println!("{} => {:.1}", val[i], vec.column(i));
    }
    Ok(())
}
