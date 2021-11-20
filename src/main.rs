use mace_rs::*;

fn main() {
    let data = read_csv("tests/example.csv");
    let unique_values = compute_unique_values(&data);
    let mace = MACE::new("tests/example.csv");

    e = e_step(mace)
    m = m_step(e)
    
    println!("{:?}", data);
    println!("{:?}", unique_values);
    println!("{:?}", mace);
}
