use std::collections::HashSet;
use std::fs::read_to_string;

const MISSING_VALUE: &str = "";


trait EM {
    fn new()
    fn e_step();
    fn m_step();
    fn fit(&self) {
        self.e_step();
        self.m_step();
    }
}

struct BayesMACE {
} 

impl EM for BayesMACE {
    fn e_step() {
        //bayes e-step
    }
    fn m_step() {

    }
}

struct VarMACE {

}

impl EM for VarMACE {
    fn e_step() {
        //variational e-step
    }
    fn m_step() {

    }
}

fn main() {
    println!("Hello, world!");
}


#[derive(Debug)]
pub struct MACE {
    data: Vec<Vec<String>>,
    labels: HashSet<String>,
    num_labels: usize,
    num_instances: usize,
}

impl MACE {
    pub fn new(file_path: &str) -> MACE {
        let data = read_csv(file_path);
        let labels = compute_unique_values(&data);
        let num_labels = labels.len();
        let num_instances = data.len();

        MACE {
            data,
            labels,
            num_labels,
            num_instances,
        }
    }
}

impl MACE {
    fn e_step() {

    }

    fn m_step() {

    } 

    fn fit(check_end) {
        e_step()
        m_step()
    }
    
    fn fit_tol(tol) {
        check_end()
    }
}

pub fn read_csv(file_path: &str) -> Vec<Vec<String>> {
    read_to_string(file_path)
        .unwrap()
        .lines()
        .map(|x| x.split(',').map(String::from).collect())
        .collect()
}

pub fn compute_unique_values(inputs: &[Vec<String>]) -> HashSet<String> {
    let mut unique = HashSet::new();
    for line in inputs {
        for annotation in line {
            unique.insert(annotation.clone());
        }
    }
    // remove the MISSING_VALUE from the result
    unique.remove(MISSING_VALUE);

    unique
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
