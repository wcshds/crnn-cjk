use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use crnn::training::run;

fn main() {
    run::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0));
}
