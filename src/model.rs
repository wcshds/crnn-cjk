use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Linear, LinearConfig, ReLU,
    },
    tensor::{
        activation,
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

use crate::burn_ext::lstm::{BiLstm, BiLstmConfig};

#[derive(Module, Debug)]
pub struct BidirectionalLSTM<B: Backend> {
    rnn: BiLstm<B>,
    embedding: Linear<B>,
}

impl<B: Backend> BidirectionalLSTM<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (_, recurrent) = self.rnn.forward(input, None);
        let [batch_size, seq_length, _] = recurrent.dims();
        let t_rec = recurrent.reshape([(batch_size * seq_length) as i32, -1]);

        let output = self.embedding.forward(t_rec);
        let output = output.reshape([batch_size as i32, seq_length as i32, -1]);

        output
    }
}

#[derive(Config, Debug)]
struct BidirectionalLSTMConfig {
    num_channel_in: usize,
    hidden_size: usize,
    num_channel_out: usize,
}

impl BidirectionalLSTMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BidirectionalLSTM<B> {
        BidirectionalLSTM {
            rnn: BiLstmConfig::new(self.num_channel_in, self.hidden_size, true).init(device),
            embedding: LinearConfig::new(self.hidden_size * 2, self.num_channel_out).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct CRNN<B: Backend> {
    // 1. cnn
    conv0: Conv2d<B>,
    pooling0: MaxPool2d,
    conv1: Conv2d<B>,
    pooling1: MaxPool2d,
    conv2: Conv2d<B>,
    batchnorm0: BatchNorm<B, 2>,
    pooling2: MaxPool2d,
    conv3: Conv2d<B>,
    batchnorm1: BatchNorm<B, 2>,
    pooling3: MaxPool2d,
    conv4: Conv2d<B>,
    batchnorm2: BatchNorm<B, 2>,
    pooling4: MaxPool2d,
    conv5: Conv2d<B>,
    batchnorm3: BatchNorm<B, 2>,
    pooling5: MaxPool2d,
    conv6: Conv2d<B>,
    relu: ReLU,
    // 2. rnn
    pub rnn0: BidirectionalLSTM<B>,
    pub rnn1: BidirectionalLSTM<B>,
}

impl<B: Backend> CRNN<B> {
    /// # Shapes
    ///   - Images [batch_size, channels, height, width]
    ///   - Output [batch_size, seq_length, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let conv = self.conv0.forward(images);
        let conv = self.relu.forward(conv);
        let conv = self.pooling0.forward(conv);
        let conv = self.conv1.forward(conv);
        let conv = self.relu.forward(conv);
        let conv = self.pooling1.forward(conv);
        let conv = self.conv2.forward(conv);
        let conv = self.batchnorm0.forward(conv);
        let conv = self.relu.forward(conv);
        let conv = self.pooling2.forward(conv);
        let conv = self.conv3.forward(conv);
        let conv = self.batchnorm1.forward(conv);
        let conv = self.relu.forward(conv);
        let conv = self.pooling3.forward(conv);
        let conv = self.conv4.forward(conv);
        let conv = self.batchnorm2.forward(conv);
        let conv = self.relu.forward(conv);
        let conv = self.pooling4.forward(conv);
        let conv = self.conv5.forward(conv);
        let conv = self.batchnorm3.forward(conv);
        let conv = self.relu.forward(conv);
        let conv = self.pooling5.forward(conv);
        let conv = self.conv6.forward(conv);

        let conv = conv.squeeze::<3>(2);
        // [batch_size, seq_length, channels]
        let conv = conv.swap_dims(1, 2);

        let features = self.rnn0.forward(conv);
        let output = self.rnn1.forward(features);
        let output = activation::log_softmax(output, 2);

        output
    }
}

impl<B: AutodiffBackend> CRNN<B> {
    pub fn no_grad_cnn(self) -> Self {
        Self {
            conv0: self.conv0.no_grad(),
            conv1: self.conv1.no_grad(),
            conv2: self.conv2.no_grad(),
            conv3: self.conv3.no_grad(),
            conv4: self.conv4.no_grad(),
            conv5: self.conv5.no_grad(),
            conv6: self.conv6.no_grad(),
            batchnorm0: self.batchnorm0.no_grad(),
            batchnorm1: self.batchnorm1.no_grad(),
            batchnorm2: self.batchnorm2.no_grad(),
            batchnorm3: self.batchnorm3.no_grad(),
            ..self
        }
    }

    pub fn no_grad_rnn(self) -> Self {
        Self {
            rnn0: self.rnn0.no_grad(),
            rnn1: self.rnn1.no_grad(),
            ..self
        }
    }
}

#[derive(Config, Debug)]
pub struct CRNNConfig {
    num_channel: usize,
    num_classes: usize,
    rnn_hidden_size: usize,
}

impl CRNNConfig {
    fn generate_conv<B: Backend>(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        device: &B::Device,
    ) -> Conv2d<B> {
        Conv2dConfig::new(channels, kernel_size)
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
            .init(device)
    }

    fn generate_pooling(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> MaxPool2d {
        MaxPool2dConfig::new(kernel_size)
            .with_strides(stride)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
            .init()
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> CRNN<B> {
        CRNN {
            conv0: CRNNConfig::generate_conv(
                [self.num_channel, 64],
                [3, 3],
                [1, 1],
                [1, 1],
                device,
            ),
            pooling0: CRNNConfig::generate_pooling([2, 2], [2, 2], [1, 1]),
            conv1: CRNNConfig::generate_conv([64, 128], [3, 3], [1, 1], [1, 1], device),
            pooling1: CRNNConfig::generate_pooling([2, 2], [2, 2], [1, 1]),
            conv2: CRNNConfig::generate_conv([128, 256], [3, 3], [1, 1], [1, 1], device),
            batchnorm0: BatchNormConfig::new(256).init(device),
            pooling2: CRNNConfig::generate_pooling([2, 1], [2, 1], [1, 0]),
            conv3: CRNNConfig::generate_conv([256, 512], [3, 3], [1, 1], [1, 1], device),
            batchnorm1: BatchNormConfig::new(512).init(device),
            pooling3: CRNNConfig::generate_pooling([2, 1], [2, 1], [1, 0]),
            conv4: CRNNConfig::generate_conv([512, 512], [3, 3], [1, 1], [1, 1], device),
            batchnorm2: BatchNormConfig::new(512).init(device),
            pooling4: CRNNConfig::generate_pooling([2, 1], [2, 1], [1, 0]),
            conv5: CRNNConfig::generate_conv([512, 512], [3, 3], [1, 1], [1, 1], device),
            batchnorm3: BatchNormConfig::new(512).init(device),
            pooling5: CRNNConfig::generate_pooling([2, 1], [2, 1], [1, 0]),
            conv6: CRNNConfig::generate_conv([512, 512], [2, 2], [1, 1], [0, 0], device),
            relu: ReLU::new(),
            rnn0: BidirectionalLSTMConfig::new(512, self.rnn_hidden_size, self.rnn_hidden_size)
                .init(device),
            rnn1: BidirectionalLSTMConfig::new(
                self.rnn_hidden_size,
                self.rnn_hidden_size,
                self.num_classes,
            )
            .init(device),
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        tensor::{Int, Tensor},
    };

    use super::*;

    #[test]
    fn test_forward() {
        type Mybackend = NdArray;
        let device = NdArrayDevice::Cpu;
        let crnn = CRNNConfig::new(1, 1000, 256).init::<Mybackend>(&device);

        let input = Tensor::<Mybackend, 1, Int>::arange(0..(5 * 64 * 1000), &device)
            .reshape([5, 1, 64, 1000])
            .float();

        println!("{}", crnn.forward(input));
    }
}
