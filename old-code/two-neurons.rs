fn main() {
    let inputs = vec![1.0, 2.0, 3.0]; // Inputs to the layer
    let weights = vec![
        vec![0.5, -0.6, 0.2], // Weights for Neuron 1
        vec![0.3, 0.8, -0.5], // Weights for Neuron 2
    ];
    let biases = vec![0.1, -0.2]; // Biases for each neuron

    // Outputs from both neurons
    let outputs: Vec<f64> = weights.iter().zip(biases.iter())
        .map(|(neuron_weights, bias)| {
            let weighted_sum: f64 = inputs.iter()
                .zip(neuron_weights.iter())
                .map(|(input, weight)| input * weight)
                .sum::<f64>() + bias;

            relu(weighted_sum) // Activation function
        })
        .collect();

    println!("Outputs: {:?}", outputs);
}
