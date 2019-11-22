use arpack_ng::*;
use ndarray::prelude::*;

fn main() -> Result<(), Error> {
    let m = Array2::ones((200, 200));
    println!("{}", m.eigenvalues(1, 200, 10000)?);
    Ok(())
}
