use candle_core::backend::BackendDevice;
use candle_core::{scalar, MetalDevice, WithDType};
use candle_core::{Device, Result, Tensor};
use candle_nn::{ops, AdamW};
use candle_nn::loss::{binary_cross_entropy_with_logit, cross_entropy};
use candle_nn::{Module, Optimizer};
use candle_core::Var;
use candle_nn::ParamsAdamW;

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
    
    //let device = Device::Cpu;

    // setup model
    let first_weights = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second_weights = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let third_weights = Tensor::randn(0f32, 1.0, (10, 10), &device)?;

    // TODO zeros
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

    let config = ParamsAdamW {
        lr: 0.001,          // learning rate
        beta1: 0.9,         // beta1 coefficient
        beta2: 0.999,       // beta2 coefficient
        eps: 1e-8,          // epsilon for numerical stability
        weight_decay: 0.01, // weight decay (L2 regularization)
    };  

    let first_var = Var::from_tensor(&first_weights)?;
    let second_var = Var::from_tensor(&second_weights)?;
    let third_var = Var::from_tensor(&third_weights)?;
    let first_bias_var = Var::from_tensor(&first_bias)?;
    let second_bias_var = Var::from_tensor(&second_bias)?;
    let third_bias_var = Var::from_tensor(&third_bias)?;

    let layers = vec![
        first_var,
        second_var,
        third_var,
        first_bias_var,
        second_bias_var,
        third_bias_var,
    ];

    let mut optimizer = AdamW::new(layers, config)?;


    let learning_rate = 0.01;

    let target= Tensor::new(&[1.0_f32], &device)?;
    
    for epoch in 0..10 { 

        let logits = model.forward(&input_image)?;

        //let loss = (cross_entropy(&logits, &target)?);
        let loss= binary_cross_entropy_with_logit(&logits, &target)?;
        //let loss= candle_nn::loss::cross_entropy(&logits, &target)?;

        // Backward pass
        optimizer.backward_step(&loss)?;

        //optimizer.step()?;

        //optimizer.zero_grad()?;

        /* 
        let scalar = 1_f64; //loss.to_f64();

        model.first_weights = (&first_weights - scalar)?;
        model.second_weights = (&second_weights - scalar)?;
        model.third_weights = (&third_weights - scalar)?;
        */
        println!("loss {}", loss)
    }

    Ok(())
}
