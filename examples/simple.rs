use arpack_ng::*;
use ndarray::prelude::*;
use num_complex::Complex64;

fn main() -> Result<(), Error> {
    let m = array![
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [-1., -3., 4., 2.],
        [-1., -3., 4., 2.],
    ]
    .map(|&x| Complex64::new(x, 0.));
    println!("{}", m.eigenvalues(2, 4, 100)?);
    Ok(())
}
