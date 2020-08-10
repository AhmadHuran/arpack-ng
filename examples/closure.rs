use arpack_ng::{self, Error, Which};

const N: usize = 100;

fn main() -> Result<(), Error> {
    let (val, _vec) = arpack_ng::eigenvectors(
        |v1, mut v2| {
            for i in 0..N {
                v2[i] = v1[(i + 1) % N] + v1[(i + N - 1) % N];
            }
        },
        N,
        &Which::LargestRealPart,
        2,
        10,
        100,
    )?;
    println!("{}", val);

    Ok(())
}
