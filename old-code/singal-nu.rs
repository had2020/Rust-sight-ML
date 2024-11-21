const EPSILON: f64 = 1e-10;

fn main() {
    // Define inputs, weights, and bias
    let inputs = vec![1.0, 2.0, 3.0];
    let weights = vec![0.5, -0.6, 0.2];
    let bias = 0.1;

    // Compute weighted sum
    let weighted_sum: f64 = inputs
        .iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum::<f64>()
        + bias;

    println!("Inputs: {:?}", inputs);
    println!("Weights: {:?}", weights);
    println!("Bias: {}", bias);
    println!("Weighted Sum Before Activation: {}", weighted_sum);

    // Apply ReLU activation function
    let output = relu(weighted_sum);

    println!("Output: {}", output);
}

// ReLU activation function with threshold
fn relu(x: f64) -> f64 {
    if x.abs() < EPSILON {
        0.0
    } else if x > 0.0 {
        x
    } else {
        0.0
    }
}

