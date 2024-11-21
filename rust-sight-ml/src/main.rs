fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn main() {
    let inputs = vec![1.0, 2.0, 3.0];
    let mut weights = vec![0.5, -0.6, 0.2];
    let mut bias = 0.1;
    let target = 1.5; // Example target output

    let learning_rate = 0.01;

    for epoch in 0..1000 { // Train for 1000 epochs
        // Forward pass
        let weighted_sum: f64 = inputs
            .iter()
            .zip(weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + bias;

        let output = relu(weighted_sum);

        // Compute loss (mean squared error)
        let loss = (output - target).powi(2);

        // Backpropagation
        let error_gradient = 2.0 * (output - target);
        let relu_gradient = if weighted_sum > 0.0 { 1.0 } else { 0.0 };
        let total_gradient = error_gradient * relu_gradient;

        // Update weights and bias
        for i in 0..weights.len() {
            weights[i] -= learning_rate * total_gradient * inputs[i];
        }
        bias -= learning_rate * total_gradient;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss);
        }
    }

    println!("Trained Weights: {:?}", weights);
    println!("Trained Bias: {}", bias);
}
