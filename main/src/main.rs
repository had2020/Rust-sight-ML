use candle_core::backend::BackendDevice;
use candle_core::MetalDevice;
use candle_core::{Device, Result, Tensor};

struct Model {
    first: Tensor,
    second: Tensor,
    third: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        let x = x.matmul(&self.second)?;
        let x = x.relu()?;
        let x = x.matmul(&self.third)?;
        Ok(x)
    }
}

fn main() -> Result<()> {
    //let device = Device::Cpu;
    let metal_device = match MetalDevice::new(0) {
        Ok(device) => device,
        Err(e) => {
            eprintln!("Failed to create MetalDevice: {:?}", e);
            return Err(e); // Or handle the error appropriately.
        }
    };
    let device = Device::Metal(metal_device);

    // 3D tensor let first = Tensor::randn(0f32, 1.0, (10, 10, 10), &device)?;
    let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?; // frist two random, tensor with shape 784, 100
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?; // the second number must match the first number in the under row
    let third = Tensor::randn(0f32, 1.0, (10, 10), &device)?;
    let model = Model { first, second, third };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;


    println!("Digit {digit:?} digit");
    Ok(())
}
