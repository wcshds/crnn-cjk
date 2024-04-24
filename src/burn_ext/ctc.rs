#![allow(clippy::single_range_in_vec_init)]
use core::marker::PhantomData;

use burn::{
    nn::loss::Reduction,
    tensor::{ElementConversion, Int, Tensor},
};

use super::backend::Backend;

/// The Connectionist Temporal Classification loss.
#[derive(Clone, Debug)]
pub struct CTCLoss<B: Backend> {
    blank: usize,
    backend: PhantomData<B>,
}

impl<B: Backend> Default for CTCLoss<B> {
    fn default() -> Self {
        CTCLoss::new(0)
    }
}

impl<B: Backend> CTCLoss<B> {
    /// Create the criterion.
    pub fn new(blank: usize) -> Self {
        Self {
            blank,
            backend: PhantomData,
        }
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Parameters:
    ///
    /// - log_probs: The logarithmized probabilities of the outputs. Shape:
    ///   `[batch_size, input_length, num_classes]`
    /// - targets: It represent the concatenated  target sequences. Each
    ///   element in the target sequence is a class index. And the target
    ///   index cannot be blank. Shape: `[target_lengths_sum]`
    /// - input_lengths: It represent the lengths of the inputs. And the
    ///   lengths are specified for each sequence to achieve masking under
    ///   the assumption that sequences are padded to equal lengths. Shape:
    ///   `[batch_size]`
    /// - target_lengths:  It represent lengths of the targets. Shape:
    ///   `[batch_size]`
    /// - reduction: Specifies the reduction to apply to the output. None:
    ///   no reduction will be applied; Some(Reduction::Mean): the output
    ///   losses will be divided by the target lengths and then the mean
    ///   over the batch is taken; Some(Reduction::Sum): the output losses
    ///   will be summed.
    ///
    /// # Reference
    ///
    /// - [PyTorch implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/LossCTC.cpp)
    /// - [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
    pub fn forward(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
        reduction: Option<Reduction>,
    ) -> Tensor<B, 1> {
        Self::assertions(
            log_probs.clone(),
            targets.clone(),
            input_lengths.clone(),
            target_lengths.clone(),
        );

        let loss = B::ctc_loss(
            log_probs.into_primitive(),
            targets.into_primitive(),
            input_lengths.into_primitive(),
            target_lengths.into_primitive(),
            reduction,
            self.blank,
        );

        Tensor::from_primitive(loss)
    }

    fn assertions(
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) {
        let [log_probs_batch_size, input_seq_length, _] = log_probs.dims();
        let [targets_size] = targets.dims();
        let [input_lengths_size] = input_lengths.dims();
        let [target_lengths_size] = target_lengths.dims();

        assert!(
            log_probs_batch_size == input_lengths_size,
            "Batch size of log_probs ({}) should correspond to size of input_lengths ({}).",
            log_probs_batch_size,
            input_lengths
        );

        assert!(
            log_probs_batch_size == target_lengths_size,
            "Batch size of log_probs ({}) should correspond to size of target_lengths ({}).",
            log_probs_batch_size,
            target_lengths_size
        );

        assert!(
            target_lengths.sum().into_scalar().elem::<u32>() == targets_size as u32,
            "Batch size of targets ({}) should correspond to sum of target_lengths ({}).",
            log_probs_batch_size,
            target_lengths_size
        );

        let max_input_length = input_lengths.max().into_scalar().elem::<u32>() as usize;
        assert!(
            max_input_length <= input_seq_length,
            "The maximum value of input_lengths ({}) must not be greater than the sequence length of log_probs ({}).",
            max_input_length, input_seq_length
        );
    }
}

#[cfg(test)]
mod test {

    use burn::{
        backend::{Autodiff, LibTorch, NdArray, Wgpu},
        tensor::Data,
    };

    use super::*;

    type TestBackend = Autodiff<LibTorch>;

    #[test]
    fn test_ctc_loss() {
        let device = Default::default();

        let input = Tensor::<TestBackend, 3>::from_data(
            [[
                [
                    -0.785, -3.471, -2.531, -3.948, -2.373, -3.042, -2.029, -2.255, -4.228, -3.810,
                ],
                [
                    -3.548, -1.692, -0.967, -2.519, -2.806, -2.760, -2.434, -2.762, -3.638, -3.669,
                ],
                [
                    -3.904, -1.799, -1.312, -2.530, -2.267, -3.169, -3.838, -2.073, -2.484, -2.418,
                ],
                [
                    -0.890, -2.506, -3.405, -3.038, -2.483, -2.861, -2.749, -3.086, -1.960, -3.336,
                ],
                [
                    -1.113, -3.557, -2.580, -1.465, -3.884, -1.993, -3.574, -3.466, -2.669, -2.985,
                ],
                [
                    -3.948, -0.828, -1.805, -2.842, -2.767, -3.891, -2.825, -1.783, -5.566, -5.072,
                ],
                [
                    -1.677, -1.703, -4.191, -3.862, -1.726, -2.616, -2.366, -2.324, -2.767, -2.418,
                ],
                [
                    -1.511, -1.125, -3.526, -3.007, -2.975, -3.358, -2.037, -2.093, -4.137, -3.900,
                ],
                [
                    -1.850, -2.767, -1.718, -2.185, -2.890, -1.998, -3.661, -3.997, -2.738, -1.671,
                ],
                [
                    -2.621, -1.234, -3.499, -3.494, -1.612, -1.713, -2.179, -2.884, -4.122, -4.581,
                ],
                [
                    -1.519, -3.283, -1.287, -3.217, -2.544, -3.128, -2.061, -3.039, -2.388, -3.272,
                ],
                [
                    -1.112, -1.258, -3.206, -3.103, -3.918, -2.577, -4.399, -4.488, -2.187, -2.663,
                ],
                [
                    -1.889, -2.344, -3.232, -2.781, -3.312, -0.911, -2.864, -4.825, -3.180, -2.243,
                ],
                [
                    -4.368, -1.471, -1.308, -2.950, -3.211, -2.692, -1.923, -2.020, -3.859, -3.601,
                ],
                [
                    -4.254, -3.291, -1.539, -2.622, -2.281, -1.427, -1.712, -3.082, -2.653, -3.809,
                ],
                [
                    -3.322, -2.904, -0.942, -3.157, -2.987, -3.736, -1.208, -4.155, -4.383, -2.583,
                ],
                [
                    -2.827, -2.293, -3.109, -3.196, -3.297, -2.451, -2.136, -3.423, -1.012, -2.146,
                ],
                [
                    -1.803, -1.666, -1.780, -4.024, -3.083, -4.520, -2.674, -2.527, -3.365, -1.516,
                ],
                [
                    -2.199, -2.340, -2.009, -3.736, -3.363, -2.721, -2.350, -1.951, -1.815, -2.009,
                ],
                [
                    -1.721, -3.726, -1.701, -3.503, -2.153, -3.242, -2.284, -1.838, -2.646, -2.329,
                ],
                [
                    -3.655, -2.916, -2.913, -1.197, -3.060, -2.154, -1.776, -3.404, -1.823, -3.310,
                ],
                [
                    -2.671, -2.592, -2.929, -1.416, -2.007, -2.886, -2.781, -2.597, -1.738, -2.862,
                ],
                [
                    -1.686, -4.173, -0.884, -5.493, -5.498, -1.707, -3.573, -5.085, -2.060, -3.352,
                ],
                [
                    -2.114, -2.478, -2.178, -3.457, -3.264, -2.659, -2.653, -1.222, -2.375, -2.475,
                ],
                [
                    -2.136, -3.563, -2.325, -3.081, -2.035, -3.154, -1.122, -3.486, -1.951, -3.270,
                ],
                [
                    -3.206, -3.031, -3.913, -2.652, -2.985, -2.635, -1.153, -3.122, -3.256, -1.203,
                ],
                [
                    -2.104, -1.719, -2.141, -2.695, -2.448, -2.991, -1.542, -2.646, -3.090, -3.066,
                ],
                [
                    -3.320, -5.098, -1.085, -1.335, -2.588, -3.098, -2.466, -2.951, -3.911, -2.538,
                ],
                [
                    -3.756, -1.814, -2.752, -2.410, -3.305, -2.387, -2.112, -1.720, -2.616, -1.843,
                ],
                [
                    -3.985, -2.489, -2.305, -1.454, -2.533, -5.091, -1.759, -2.180, -3.673, -1.779,
                ],
            ]],
            &device,
        )
        .require_grad();
        let target = Tensor::<TestBackend, 1, Int>::from_data([1, 9, 6, 9, 4], &device);
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([30], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([5], &device);
        let expected_res = Data::from([50.3788948059082]);

        let ctc_loss = CTCLoss::<TestBackend>::new(0);
        let res = ctc_loss.forward(
            input.clone(),
            target,
            input_lengths,
            target_lengths,
            Some(Reduction::Sum),
        );

        let grads = res.backward();
        let grad = input.grad(&grads).unwrap();
        println!("{}", grad);

        // 3.1628532
        res.to_data().assert_approx_eq(&expected_res, 3);
    }
}
