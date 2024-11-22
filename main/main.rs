fn relu(x: f64) -> f64 {
    // Activation function: ReLU (Rectified Linear Unit)
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn main() {
    // Input signals, weights, bias, and target
    let inputs: Vec<f64> = vec![1.0, 2.0, 3.0]; // Incoming signals, could come from other neurons
    let mut weights: Vec<f64> = vec![0.5, -0.6, 0.2]; // Signal importance: determines whether to strengthen or weaken the signal
    let mut bias: f64 = 0.1; // Adjusts weight: determines whether the neuron should fire
    let target: f64 = 1.5; // Desired output value

    let learning_rate: f64 = 0.01; // Learning rate for weight updates
    let max_epochs = 1000; // Maximum number of training epochs
    let stopping_threshold: f64 = 1e-10; // Early stopping if loss is very small

    for epoch in 0..max_epochs {
        // Forward pass: Compute the weighted sum
        let weighted_sum: f64 = inputs
            .iter()
            .zip(weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>()
            + bias;

        // Apply activation function (ReLU)
        let output = relu(weighted_sum);

        // Compute loss (mean squared error)
        let loss = (output - target).powi(2);

        // Backpropagation: Compute gradients
        let error_gradient = 2.0 * (output - target); // Derivative of MSE loss
        let relu_gradient = if weighted_sum > 0.0 { 1.0 } else { 0.0 }; // Derivative of ReLU
        let total_gradient = error_gradient * relu_gradient;

        // Update weights and bias
        for i in 0..weights.len() {
            weights[i] -= learning_rate * total_gradient * inputs[i];
        }
        bias -= learning_rate * total_gradient;

        // Display loss and status every 100 epochs
        if epoch % 100 == 0 || loss < stopping_threshold {
            println!(
                "Epoch {}: Loss = {:.15e}, Output = {:.5}, Target = {:.5}",
                epoch, loss, output, target
            );
        }

        // Early stopping if the loss is sufficiently small
        if loss < stopping_threshold {
            println!("Early stopping at epoch {} due to low loss.", epoch);
            break;
        }
    }

    // Final weights and bias
    println!("\nTrained Weights: {:?}", weights);
    println!("Trained Bias: {}", bias);

    // Verify final output
    let final_weighted_sum: f64 = inputs
        .iter()
        .zip(weights.iter())
        .map(|(input, weight)| input * weight)
        .sum::<f64>()
        + bias;
    let final_output = relu(final_weighted_sum);

    println!(
        "Final Output = {:.5}, Target = {:.5}, Final Loss = {:.15e}",
        final_output,
        target,
        (final_output - target).powi(2)
    );
}
