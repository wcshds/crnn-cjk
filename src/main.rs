use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use crnn::{parse_config::CrnnTrainingConfig, training::run};

fn main() {
    let config = CrnnTrainingConfig::from_yaml("./training_config.yaml");
    run::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0), &config);
}
