use wgpu::util::DeviceExt;

async fn run_cross_entropy_loss() {
    // Initialize GPU
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let device = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();
    let queue = device.create_queue();

    // Data
    let predictions = vec![0.8, 0.1, 0.1];
    let labels = vec![1.0, 0.0, 0.0];
    let mut results = vec![0.0; predictions.len()];

    // Buffers
    let predictions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Predictions Buffer"),
        contents: bytemuck::cast_slice(&predictions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let labels_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Labels Buffer"),
        contents: bytemuck::cast_slice(&labels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Result Buffer"),
        contents: bytemuck::cast_slice(&results),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Shader
    let shader = device.create_shader_module(wgpu::include_spirv!("cross_entropy_loss.spv"));

    // Pipeline
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &shader,
        entry_point: "cross_entropy_loss",
    });

    // Execute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Command Encoder") });
    encoder.copy_buffer_to_buffer(&result_buf, 0, &result_buf, 0, (results.len() * std::mem::size_of::<f32>()) as u64);
    queue.submit(Some(encoder.finish()));

    // Read back results
    device.poll(wgpu::Maintain::Wait);
    let buffer_slice = result_buf.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    receiver.await.unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    result_buf.unmap();

    println!("Results: {:?}", result);
}
