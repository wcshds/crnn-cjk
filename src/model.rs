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

use crate::{
    burn_ext::lstm::{BiLstm, BiLstmConfig},
    parse_config::CNNElement,
};

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
    pub convs: Vec<Conv2d<B>>,
    pub batchnorms: Vec<BatchNorm<B, 2>>,
    pub poolings: Vec<MaxPool2d>,
    relu: ReLU,
    // 2. rnn
    pub rnn0: BidirectionalLSTM<B>,
    pub rnn1: BidirectionalLSTM<B>,
}

impl<B: Backend> CRNN<B> {
    /// # Shapes
    ///   - Images [batch_size, channels, height, width]
    ///   - Output [batch_size, seq_length, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>, cnn_structure: &Vec<CNNElement>) -> Tensor<B, 3> {
        let mut cnns_iter = self.convs.iter();
        let mut batchnorms_iter = self.batchnorms.iter();
        let mut poolings_iter = self.poolings.iter();

        let conv = cnn_structure
            .iter()
            .fold(images, |conv, cnn_element| match cnn_element {
                CNNElement::Conv(_, _, _, _) => cnns_iter.next().unwrap().forward(conv),
                CNNElement::Batchnorm(_) => batchnorms_iter.next().unwrap().forward(conv),
                CNNElement::Pooling(_, _, _) => poolings_iter.next().unwrap().forward(conv),
                CNNElement::Relu => self.relu.forward(conv),
            });

        let conv = conv.squeeze::<3>(2);
        // [batch_size, seq_length(image width), channels]
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
            convs: self.convs.into_iter().map(|each| each.no_grad()).collect(),
            batchnorms: self
                .batchnorms
                .into_iter()
                .map(|each| each.no_grad())
                .collect(),
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
    // num_channel: usize,
    num_classes: usize,
    rnn_hidden_size: usize,
}

impl CRNNConfig {
    fn generate_conv<B: Backend>(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        strides: [usize; 2],
        paddings: [usize; 2],
        device: &B::Device,
    ) -> Conv2d<B> {
        Conv2dConfig::new(channels, kernel_size)
            .with_stride(strides)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(
                paddings[0],
                paddings[1],
            ))
            .init(device)
    }

    fn generate_pooling(
        kernel_size: [usize; 2],
        strides: [usize; 2],
        paddings: [usize; 2],
    ) -> MaxPool2d {
        MaxPool2dConfig::new(kernel_size)
            .with_strides(strides)
            .with_padding(burn::nn::PaddingConfig2d::Explicit(
                paddings[0],
                paddings[1],
            ))
            .init()
    }

    pub fn init<B: Backend>(&self, cnn_structure: &Vec<CNNElement>, device: &B::Device) -> CRNN<B> {
        let mut convs = Vec::new();
        let mut batchnorms = Vec::new();
        let mut poolings = Vec::new();

        for cnn_element in cnn_structure {
            match cnn_element {
                CNNElement::Conv(channels, kernel_size, strides, paddings) => {
                    let conv = CRNNConfig::generate_conv(
                        *channels,
                        *kernel_size,
                        *strides,
                        *paddings,
                        device,
                    );
                    convs.push(conv);
                }
                CNNElement::Batchnorm(num_features) => {
                    let batchnorm = BatchNormConfig::new(*num_features).init(device);
                    batchnorms.push(batchnorm);
                }
                CNNElement::Pooling(kernel_size, strides, paddings) => {
                    let pooling = CRNNConfig::generate_pooling(*kernel_size, *strides, *paddings);
                    poolings.push(pooling);
                }
                _ => (),
            }
        }

        CRNN {
            convs,
            batchnorms,
            poolings,
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

    fn generate_cnn_structure() -> Vec<CNNElement> {
        vec![
            CNNElement::Conv([1, 64], [3, 3], [1, 1], [1, 1]),
            CNNElement::Relu,
            CNNElement::Pooling([2, 2], [2, 2], [1, 1]),
            CNNElement::Conv([64, 128], [3, 3], [1, 1], [1, 1]),
            CNNElement::Relu,
            CNNElement::Pooling([2, 2], [2, 2], [1, 1]),
            CNNElement::Conv([128, 256], [3, 3], [1, 1], [1, 1]),
            CNNElement::Batchnorm(256),
            CNNElement::Relu,
            CNNElement::Pooling([2, 1], [2, 1], [1, 0]),
            CNNElement::Conv([256, 512], [3, 3], [1, 1], [1, 1]),
            CNNElement::Batchnorm(512),
            CNNElement::Relu,
            CNNElement::Pooling([2, 1], [2, 1], [1, 0]),
            CNNElement::Conv([512, 512], [3, 3], [1, 1], [1, 1]),
            CNNElement::Batchnorm(512),
            CNNElement::Relu,
            CNNElement::Pooling([2, 1], [2, 1], [1, 0]),
            CNNElement::Conv([512, 512], [3, 3], [1, 1], [1, 1]),
            CNNElement::Batchnorm(512),
            CNNElement::Relu,
            CNNElement::Pooling([2, 1], [2, 1], [1, 0]),
            CNNElement::Conv([512, 512], [2, 2], [1, 1], [0, 0]),
        ]
    }

    #[test]
    fn test_forward() {
        type Mybackend = NdArray;
        let device = NdArrayDevice::Cpu;
        let cnn_structure = generate_cnn_structure();
        let crnn = CRNNConfig::new(1000, 256).init::<Mybackend>(&cnn_structure, &device);

        let input = Tensor::<Mybackend, 1, Int>::arange(0..(5 * 64 * 1000), &device)
            .reshape([5, 1, 64, 1000])
            .float();

        println!("{}", crnn.forward(input, &cnn_structure));
    }
}
