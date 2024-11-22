fn relu(value) {
    if value > 0.0 {
        value
    } else {
        0.0
    }
}

fn main() {
    
    let input: vec!<64> = [1.0, 2.0, 3.0]

    let weights: vec!<64> = [1.0, 2.0, 3.0]
    let bias: f64 = 0.1
    let  target: f64 = 1.5

    let learning_rate: f64 = 0.01; // Learning rate for weight updates
    let max_epochs = 1000; // Maximum number of training epochs
    let stopping_threshold: f64 = 1e-10; // Early stopping if loss is very small

}