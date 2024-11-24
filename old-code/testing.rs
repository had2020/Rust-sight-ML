fn main() {
    // Input signals, weights, bias, and target
    let inputs: Vec<f64> = vec![1.0, 2.0, 3.0]; // Incoming signals, could come from other neurons

    // Signal importance: determines whether to strengthen or weaken the signal
    let weights = vec![
        vec![ // Layer 1
            vec![0.5, -0.6, 0.2], // Neuron 1
            vec![0.3, 0.8, -0.5], // Neuron 2
        ],
        vec![ // Layer 2
            vec![0.1, 0.4, -0.3], // Neuron 1
            vec![0.6, -0.2, 0.1], // Neuron 2
        ],
    ];

    let mut bias: f64 = 0.1; // Adjusts weight: determines whether the neuron should fire
    let target: f64 = 1.5; // Desired output value

    let learning_rate: f64 = 0.01; // Learning rate for weight updates
    let max_epochs = 1000; // Maximum number of training epochs
    let stopping_threshold: f64 = 1e-10; // Early stopping if loss is very smal

    // Iterating over weights in depth
    for (layer_index, layer) in weights.iter().enumerate() {
        println!("Layer {}", layer_index + 1);
        for (neuron_index, neuron) in layer.iter().enumerate() {
            println!("  Neuron {}", neuron_index + 1);
            for (input_index, weight) in neuron.iter().enumerate() {
                println!("    Input {}: {}", input_index + 1, weight);
            }
        }
    }

    // TODO relu function 
}