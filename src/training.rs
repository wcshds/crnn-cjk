use std::{fs, time::Instant};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    nn::loss::Reduction,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, Int, Tensor},
};

use crate::{
    burn_ext::ctc::CTCLoss,
    converter::Converter,
    dataset::{TextImgBatcher, TextImgDataset},
    img_gen::generator::GeneratorConfig,
    model::CRNNConfig,
};

#[derive(Config)]
pub struct CRNNTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 30)]
    pub batch_size: usize,
    #[config(default = 10)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 0.001)]
    pub lr: f64,
    pub model: CRNNConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let start = Instant::now();
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    // Create the configuration.
    let config_model = CRNNConfig::new(1, 110001, 256);
    let config_optimizer = AdamConfig::new();
    let config = CRNNTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    // Create the model and optimizer.
    let mut model = config.model.init(&device);
    let mut optim = config.optimizer.init();

    // let mut model = model
    //     .load_file("./iter-00000", &bfr)
    //     .unwrap();

    // Create the batcher.
    let batcher_train = TextImgBatcher::<B>::new(device.clone());
    let batcher_valid = TextImgBatcher::<B::InnerBackend>::new(device.clone());

    let generator_config = GeneratorConfig {
        font_size: 50,
        line_height: 64,
        image_width: 2000,
        image_height: 64,
    };
    let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
    let converter = Converter::new(&lexicon);

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            10..15,
            1000_0000,
            "./font",
            "./ch.txt",
            generator_config,
            converter.clone(),
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            10..15,
            5_0000,
            "./font",
            "./ch.txt",
            generator_config,
            converter.clone(),
        ));

    println!(
        "Loading completed, took {} seconds.",
        start.elapsed().as_secs_f64()
    );

    let mut count = 0;
    // Iterate over our training and validation loop for X epochs.
    // Implement our training loop.
    for (iteration, batch) in dataloader_train.iter().enumerate() {
        let output = model.forward(batch.images);
        let device = output.clone().device();
        let [batch_size, seq_length, _] = output.clone().dims();
        let input_lengths = Tensor::<B, 1, Int>::full([batch_size], seq_length as i32, &device);
        let loss = CTCLoss::new(0).forward(
            output.clone(),
            batch.targets.clone(),
            input_lengths.clone(),
            batch.target_lengths.clone(),
            Some(Reduction::Mean),
        );

        println!(
            "[Train - Iteration {}] Loss {:.5}",
            iteration,
            loss.clone().into_scalar()
        );

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);
        // Update the model using the optimizer.
        model = optim.step(config.lr, model, grads);

        if iteration % 1000 == 0 {
            model
                .clone()
                .save_file(format!("./iter-{count:05}"), &bfr)
                .unwrap();
            count += 1;
        }
    }

    // Get the model without autodiff.
    let model_valid = model.valid();

    // Implement our validation loop.
    for (iteration, batch) in dataloader_test.iter().enumerate() {
        let output = model_valid.forward(batch.images);
        let [batch_size, seq_length, _] = output.dims();
        let input_lengths =
            Tensor::<B::InnerBackend, 1, Int>::full([batch_size], seq_length as i32, &device);
        let loss = CTCLoss::new(0).forward(
            output.clone(),
            batch.targets.clone(),
            input_lengths.clone(),
            batch.target_lengths.clone(),
            Some(Reduction::Mean),
        );
        // let accuracy = accuracy(output, batch.targets);

        println!(
            "[Valid - Iteration {}] Loss {}",
            iteration,
            loss.clone().into_scalar(),
            // accuracy,
        );
    }
}
