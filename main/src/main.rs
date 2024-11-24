use tch::{nn, nn::OptimizerConfig, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple tensor
    let tensor = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Original tensor: {:?}", tensor);

    // Perform basic operations
    let squared = tensor * tensor;
    println!("Squared tensor: {:?}", squared);

    // Define a simple linear regression model
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let linear = nn::linear(&vs.root(), 1, 1, Default::default());

    // Training data
    let xs = Tensor::of_slice(&[1.0, 2.0, 3.0, 4.0]).unsqueeze(1);
    let ys = Tensor::of_slice(&[2.0, 4.0, 6.0, 8.0]).unsqueeze(1);

    // Optimizer
    let mut opt = nn::Sgd::default().build(&vs, 0.01)?;

    // Training loop
    for epoch in 1..101 {
        let predictions = xs.apply(&linear);
        let loss = predictions.mse_loss(&ys, tch::Reduction::Mean);
        opt.backward_step(&loss);

        if epoch % 10 == 0 {
            println!("Epoch: {}, Loss: {:?}", epoch, f64::from(&loss));
        }
    }

    // Test the trained model
    let test_xs = Tensor::of_slice(&[5.0, 6.0]).unsqueeze(1);
    let test_preds = test_xs.apply(&linear);
    println!("Predictions for input {:?}: {:?}", test_xs, test_preds);

    Ok(())
}
