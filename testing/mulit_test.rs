fn relu(x: f64) -> f64 { 
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn main() {
    let mut final_output: f64 = 0.0;
    println!("loading test");

    let inputs = vec![1.0, 2.0, 3.0];

    let mut weights = vec![0.5, 1.5, 2.0];

    let mut bias: f64 = 0.1;

    let target: f64 = 1.5; // possible programable award

    let max_epochs = 1000;

    let stopping_threshold: f64 = 1e-10; 

    let learning_rate: f64 = 0.01; 

    for epoch in 0..max_epochs {
        let mut weighted_sum: f64 = 0.0;

        for (index, weight) in weights.iter().enumerate() {
            //println!("Index: {}, Weight: {}", index, weight); //DEBUG
            let input = inputs[index];
            weighted_sum += input * weight;
        }

        weighted_sum += bias;

        let output = relu(weighted_sum);

        // learning belown
        let loss = (output - target).powi(2); //compute error // TODO focus on when this place on training a nural network process 

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
            final_output = output;
            println!("Early stopping at epoch {} due to low loss.", epoch);
            break;
        }
        
        if epoch == 100 {
            final_output = output
        }
    }

    println!("\nTrained Weights: {:?}", weights);
    println!("Trained Bias: {}", bias);
    println!("Output: {final_output}");

}
