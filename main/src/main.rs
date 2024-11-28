use candle_core::backend::BackendDevice;
use candle_core::{MetalDevice, WithDType};
use candle_core::{Device, Result, Tensor};

fn cross_entropy_loss(predictions: &[f32], target: usize) -> f32 {
    // makes predictions that are valid probabilities
    let eps = 1e-10; // prevents log(0)
    -predictions[target].max(eps).ln()
}


fn error_gradient(predictions: &[f32], target: usize) -> f32 {
    let eps = 1e-10; 
    let x = -predictions[target].max(eps).ln();
    return x * 2.0_f32
}

fn relu_gradient(weighted_sum: f32) -> f32 {
    if weighted_sum > 0.0_f32 {
        weighted_sum
    } else {
        0.0_f32
    }
}

struct Model {
    first_weights: Tensor,
    second_weights: Tensor,
    third_weights: Tensor,

    first_bias: Tensor,
    second_bias: Tensor,
    third_bias: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first_weights)?;
        let x = (x + &self.first_bias)?;
        let x = x.relu()?;
        let x = x.matmul(&self.second_weights)?;
        let x = (x + &self.second_bias)?;
        let x = x.relu()?;
        let x = x.matmul(&self.third_weights)?;
        let x = (x + &self.third_bias)?;
        Ok(x)
    }
}

fn main() -> Result<()> {
    // setup device
    let device = match MetalDevice::new(0) {
        Ok(metal_device) => Device::Metal(metal_device),
        Err(_) => {
            eprintln!("Failed to create MetalDevice. Falling back to CPU.");
            Device::Cpu
        }
    };

    // setup model
    let first_weights = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second_weights = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let third_weights = Tensor::randn(0f32, 1.0, (10, 10), &device)?;

    let first_bias = Tensor::randn(0f32, 1.0, (1, 100), &device)?;
    let second_bias = Tensor::randn(0f32, 1.0, (1, 10), &device)?;
    let third_bias = Tensor::randn(0f32, 1.0, (1, 10), &device)?;

    let mut model = Model {
         first_weights: first_weights.clone(),
         second_weights: second_weights.clone(), 
         third_weights: third_weights.clone(), 
         first_bias: first_bias.clone(), 
         second_bias: second_bias.clone(), 
         third_bias: third_bias.clone(),
        };

    // random example input image
    let input_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;
    //let target_label = Tensor::new(&[1], &device)?; // Example: label 1

    // the learning rate
    let learning_rate = 0.01;

    // training loops
    for epoch in 0..500 { 
        // a forward pass
        let logits = model.forward(&input_image)?;

        // softmax for output probabilities
        let probabilities = candle_nn::ops::softmax(&logits, 1)?;

        // convert the flattened tensor to a Vec<f32>
        //let probabilities_vec: Vec<f64> = probabilities.to_vec1()?; // to_vec1 for 1D tensors
        let probabilities_vec: Vec<f32> = probabilities.squeeze(0)?.to_vec1()?;

        let target_label = 1; // example replace, it is the class it should be

        // next two lines debug variables
        let loss = cross_entropy_loss(&probabilities_vec, target_label);
        //let loss = candle_nn::loss::cross_entropy(inp, target) TODO
        //let predicted_class = probabilities.argmax(1)?;

        let predicted_class_d = probabilities.argmax(1)?.to_dtype(candle_core::DType::F32)?;
        //let f32_predicted_class = predicted_class.to_scalar::<f32>()?; // tensor to f32
        //let f32_predicted_class = predicted_class.get(0)?.to_scalar::<f32>()?;
        let f32_predicted_class = predicted_class_d.squeeze(0)?.to_scalar::<f32>()?;

        // backward pass (computing gradients)
        let error_gradient:f32 = error_gradient(&probabilities_vec, target_label);
        let relu_gradient:f32 = relu_gradient(f32_predicted_class);
        let total_gradient:f32 = error_gradient + relu_gradient;

        // display
        if epoch % 100 == 0 {
            println!("\n Epoch: {epoch} \n ----------- \n"); // just for readablity
            println!("Predicted class: {}", f32_predicted_class);
            println!("Probabilities: {probabilities_vec:?}");
            println!("Loss: {}", loss);
            println!("Relu gradient: {}", relu_gradient);
            println!("Error gradient: {}", error_gradient);
        }

        // update weights 
        let scalar = learning_rate * total_gradient.to_f64();
        
        model.first_weights = first_weights.clone().affine(1.0, scalar)?;
        model.second_weights = second_weights.clone().affine(1.0, scalar)?;
        model.third_weights = third_weights.clone().affine(1.0, scalar)?;
    }

    Ok(())
}
