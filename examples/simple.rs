use arpack_ng::*;
use ndarray::{s, prelude::*};

fn main() -> Result<(), Error> {
    let m = Array2::ones((200, 200));
    let (val, vec) = m.eigenvectors(2, 50, 100)?;
    println!("{:?} {:?}", val.shape(), vec.shape());
    for i in 0..val.len() {
        println!("{} => {:?}", val[i], vec.slice(s![.., i]));
    }
    Ok(())
}
