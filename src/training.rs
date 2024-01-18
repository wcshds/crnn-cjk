use std::{fs, path::Path, time::Instant};

use burn::{
    data::dataloader::DataLoaderBuilder,
    module::{AutodiffModule, Module},
    nn::loss::Reduction,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::AutodiffBackend, Int, Tensor},
};

use crate::{
    burn_ext::ctc::CTCLoss,
    converter::Converter,
    dataset::{TextImgBatcher, TextImgDataset},
    img_gen::parse_config::GeneratorConfig,
    model::CRNNConfig,
    parse_config::CrnnTrainingConfig,
};

pub fn run<B: AutodiffBackend>(device: B::Device, config: &CrnnTrainingConfig) {
    let start = Instant::now();
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    // Create the configuration.
    let config_model = CRNNConfig::new(config.crnn_num_classes, config.crnn_rnn_hidden_size);
    let config_optimizer = AdamConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(0.0)));

    if config.random_seed != 0 {
        B::seed(config.random_seed);
    }

    // Create the model and optimizer.
    let mut model = config_model.init(&config.cnn_structure, &device);
    let mut optim = config_optimizer.init();

    if config.pretrained_model_path.len() > 0 {
        model = model
            .load_file(&config.pretrained_model_path, &bfr)
            .expect("fail to read pretrained model");
        println!("{} loaded.", config.pretrained_model_path);
    }

    // Create the batcher.
    let batcher_train = TextImgBatcher::<B>::new(device.clone());
    let batcher_valid = TextImgBatcher::<B::InnerBackend>::new(device.clone());

    let generator_config = GeneratorConfig::from_yaml(&config.generator_config_path);
    let lexicon = fs::read_to_string(&config.lexicon_path).unwrap();
    let converter = Converter::new(&lexicon);

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.random_seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            10..15,
            1000_0000,
            generator_config.clone(),
            converter.clone(),
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.random_seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            10..15,
            5_0000,
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
        let output = model.forward(batch.images, &config.cnn_structure);
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
        model = optim.step(config.learning_rate, model, grads);

        if iteration % config.save_interval == 0 {
            model
                .clone()
                .save_file(
                    Path::new(&config.save_dir).join(format!("iter-{count:05}")),
                    &bfr,
                )
                .unwrap();
            count += 1;
        }
    }

    // Get the model without autodiff.
    let model_valid = model.valid();

    // Implement our validation loop.
    for (iteration, batch) in dataloader_test.iter().enumerate() {
        let output = model_valid.forward(batch.images, &config.cnn_structure);
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
