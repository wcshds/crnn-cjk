use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use crnn::training::{run, CrnnTrainingConfig};

fn main() {
    let config = CrnnTrainingConfig::from_yaml("./training_config.yaml");
    run::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0), &config);
}
