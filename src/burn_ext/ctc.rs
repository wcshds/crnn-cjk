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

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_ctc_loss() {
        let device = Default::default();

        let input = Tensor::<TestBackend, 3>::from_data(
            [
                [
                    [
                        -3.721, -3.121, -1.217, -2.477, -3.491, -2.504, -3.056, -1.760, -2.698,
                        -1.879,
                    ],
                    [
                        -3.670, -4.571, -2.960, -0.860, -2.467, -2.524, -4.121, -3.128, -3.016,
                        -1.535,
                    ],
                    [
                        -1.855, -4.322, -4.390, -3.226, -0.798, -3.623, -3.068, -3.056, -2.575,
                        -2.027,
                    ],
                    [
                        -3.053, -2.753, -1.644, -2.342, -3.161, -1.687, -2.997, -4.425, -1.264,
                        -3.579,
                    ],
                    [
                        -2.740, -2.938, -2.689, -2.188, -3.110, -3.703, -2.417, -3.724, -1.205,
                        -1.513,
                    ],
                    [
                        -3.172, -1.466, -1.488, -2.865, -2.540, -2.191, -1.549, -3.688, -4.994,
                        -4.668,
                    ],
                    [
                        -3.141, -1.485, -2.210, -2.150, -3.578, -2.868, -3.659, -2.362, -3.298,
                        -1.338,
                    ],
                    [
                        -3.460, -2.865, -3.720, -2.077, -4.724, -2.267, -2.960, -2.761, -1.852,
                        -0.974,
                    ],
                    [
                        -2.367, -2.115, -1.519, -2.756, -1.994, -1.456, -4.914, -3.610, -3.062,
                        -2.945,
                    ],
                    [
                        -3.147, -2.434, -2.744, -3.516, -3.278, -3.918, -2.653, -2.377, -1.398,
                        -1.179,
                    ],
                    [
                        -1.597, -2.974, -1.855, -1.547, -3.343, -3.375, -2.403, -2.681, -3.552,
                        -2.120,
                    ],
                    [
                        -0.953, -2.601, -2.824, -2.295, -3.080, -4.893, -3.511, -1.899, -2.218,
                        -3.272,
                    ],
                    [
                        -2.654, -2.303, -3.368, -3.299, -3.960, -3.058, -2.725, -1.889, -5.190,
                        -0.755,
                    ],
                    [
                        -0.675, -1.887, -3.804, -2.546, -4.452, -2.963, -4.189, -3.135, -2.407,
                        -3.620,
                    ],
                    [
                        -3.143, -3.560, -3.201, -2.838, -2.845, -3.149, -1.954, -2.902, -2.685,
                        -0.769,
                    ],
                    [
                        -4.125, -2.617, -3.222, -1.773, -0.737, -4.640, -4.207, -3.514, -1.957,
                        -3.617,
                    ],
                    [
                        -2.559, -3.473, -3.213, -1.569, -1.641, -2.502, -3.472, -1.776, -2.016,
                        -3.384,
                    ],
                    [
                        -2.725, -1.552, -1.782, -2.172, -3.238, -2.419, -1.853, -2.781, -3.559,
                        -2.736,
                    ],
                    [
                        -3.725, -2.072, -2.942, -3.798, -1.666, -3.500, -1.758, -1.391, -3.476,
                        -2.268,
                    ],
                    [
                        -1.325, -2.264, -1.752, -3.558, -2.593, -2.502, -5.127, -1.880, -3.459,
                        -2.503,
                    ],
                    [
                        -1.383, -2.984, -1.752, -3.004, -3.909, -2.129, -2.796, -1.758, -2.497,
                        -3.873,
                    ],
                    [
                        -2.911, -1.777, -3.344, -2.651, -4.550, -4.031, -2.759, -1.856, -1.014,
                        -2.820,
                    ],
                    [
                        -2.186, -1.946, -0.893, -3.437, -2.165, -3.717, -3.438, -4.379, -2.336,
                        -3.795,
                    ],
                    [
                        -2.579, -3.879, -4.042, -4.817, -2.385, -2.664, -3.233, -3.038, -0.470,
                        -5.549,
                    ],
                    [
                        -3.115, -2.280, -3.806, -1.614, -3.552, -3.376, -2.064, -1.728, -2.164,
                        -1.898,
                    ],
                    [
                        -3.028, -2.860, -2.736, -3.190, -2.444, -0.780, -1.967, -3.530, -3.278,
                        -3.322,
                    ],
                    [
                        -2.687, -2.101, -2.325, -3.540, -1.480, -2.480, -3.354, -2.172, -1.860,
                        -2.704,
                    ],
                    [
                        -2.078, -3.080, -1.082, -2.431, -2.610, -2.662, -3.242, -3.930, -2.361,
                        -2.246,
                    ],
                    [
                        -3.197, -2.949, -2.725, -2.023, -1.343, -3.275, -3.092, -1.394, -2.931,
                        -2.764,
                    ],
                    [
                        -2.798, -3.394, -5.272, -2.175, -3.604, -1.108, -3.765, -2.778, -2.007,
                        -1.563,
                    ],
                ],
                [
                    [
                        -3.552, -3.311, -1.988, -1.033, -3.665, -2.866, -3.322, -2.586, -2.399,
                        -1.851,
                    ],
                    [
                        -1.791, -2.651, -2.299, -2.050, -2.828, -4.741, -1.926, -1.631, -2.545,
                        -3.081,
                    ],
                    [
                        -0.770, -3.851, -1.969, -2.980, -3.260, -2.693, -4.771, -2.333, -4.674,
                        -2.260,
                    ],
                    [
                        -3.134, -2.428, -3.606, -2.530, -2.375, -2.437, -3.255, -1.131, -4.250,
                        -1.583,
                    ],
                    [
                        -2.472, -3.634, -2.937, -1.647, -3.449, -3.102, -4.248, -1.303, -2.249,
                        -1.741,
                    ],
                    [
                        -2.827, -3.469, -1.523, -1.751, -1.904, -2.383, -3.416, -2.087, -4.248,
                        -2.248,
                    ],
                    [
                        -3.128, -2.631, -1.544, -1.824, -5.210, -2.099, -2.083, -2.104, -2.773,
                        -2.626,
                    ],
                    [
                        -1.541, -2.154, -2.171, -1.866, -1.719, -4.948, -2.834, -2.867, -2.779,
                        -3.302,
                    ],
                    [
                        -1.149, -3.529, -2.398, -2.174, -3.018, -3.500, -3.961, -4.169, -1.246,
                        -3.039,
                    ],
                    [
                        -2.462, -3.399, -2.576, -2.339, -1.642, -3.267, -2.579, -1.753, -2.054,
                        -2.304,
                    ],
                    [
                        -2.867, -3.076, -3.785, -1.289, -2.187, -1.583, -3.750, -2.035, -3.088,
                        -2.507,
                    ],
                    [
                        -0.811, -4.624, -2.889, -2.638, -2.892, -3.518, -3.181, -5.623, -4.044,
                        -1.305,
                    ],
                    [
                        -3.463, -3.448, -5.942, -0.653, -2.533, -2.173, -2.892, -4.582, -2.896,
                        -2.309,
                    ],
                    [
                        -3.431, -2.446, -3.841, -4.531, -3.151, -1.145, -3.475, -1.044, -2.970,
                        -2.933,
                    ],
                    [
                        -5.061, -3.031, -3.829, -2.755, -0.617, -4.196, -2.557, -2.180, -2.992,
                        -2.735,
                    ],
                    [
                        -2.570, -2.160, -2.272, -1.902, -2.955, -3.381, -3.059, -2.159, -1.756,
                        -2.005,
                    ],
                    [
                        -1.928, -3.874, -1.866, -2.267, -1.484, -1.958, -2.137, -2.960, -4.489,
                        -3.632,
                    ],
                    [
                        -2.227, -1.825, -3.089, -1.928, -2.935, -2.806, -1.813, -2.194, -2.563,
                        -2.594,
                    ],
                    [
                        -1.369, -3.556, -1.927, -4.358, -3.608, -1.372, -1.727, -3.372, -3.007,
                        -4.117,
                    ],
                    [
                        -4.439, -2.757, -4.094, -2.824, -2.173, -3.567, -0.715, -2.492, -4.456,
                        -2.097,
                    ],
                    [
                        -1.426, -2.435, -2.582, -2.875, -3.552, -1.800, -2.151, -2.291, -2.451,
                        -3.158,
                    ],
                    [
                        -2.665, -1.936, -2.270, -2.454, -2.047, -2.775, -1.779, -2.971, -2.165,
                        -2.649,
                    ],
                    [
                        -1.982, -1.780, -3.832, -2.599, -2.409, -3.019, -3.191, -3.323, -2.174,
                        -1.317,
                    ],
                    [
                        -2.971, -1.017, -3.567, -2.698, -2.621, -4.147, -4.522, -2.920, -1.241,
                        -3.016,
                    ],
                    [
                        -2.923, -2.917, -2.979, -4.009, -1.935, -2.803, -4.254, -1.314, -1.749,
                        -1.824,
                    ],
                    [
                        -2.931, -1.312, -4.946, -2.748, -3.854, -4.801, -1.160, -2.516, -2.125,
                        -2.760,
                    ],
                    [
                        -1.889, -2.751, -2.617, -1.616, -1.283, -3.162, -4.608, -4.141, -2.021,
                        -3.347,
                    ],
                    [
                        -3.496, -2.573, -4.179, -2.908, -1.715, -1.558, -2.855, -2.014, -3.842,
                        -1.512,
                    ],
                    [
                        -2.939, -2.866, -1.881, -2.003, -1.215, -2.056, -4.053, -3.262, -2.591,
                        -3.048,
                    ],
                    [
                        -2.730, -0.939, -2.276, -2.807, -2.710, -2.508, -1.895, -2.912, -4.367,
                        -4.190,
                    ],
                ],
            ],
            &device,
        )
        .require_grad();
        let target = Tensor::<TestBackend, 1, Int>::from_data(
            [
                8, 7, 8, 9, 8, 4, 6, 8, 5, 6, 8, 6, 5, 7, 2, 8, 4, 4, 6, 4, 8, 3, 2, 8, 4, 6, 1, 1,
                8, 5, 7,
            ],
            &device,
        );
        let input_lengths = Tensor::<TestBackend, 1, Int>::from_data([30, 30], &device);
        let target_lengths = Tensor::<TestBackend, 1, Int>::from_data([19, 12], &device);
        let expected_res = Data::from([3.162853240966797]);

        let ctc_loss = CTCLoss::<TestBackend>::new(0);
        let res = ctc_loss.forward(
            input.clone(),
            target,
            input_lengths,
            target_lengths,
            Some(Reduction::Mean),
        );

        let grads = res.backward();
        let grad = input.grad(&grads).unwrap();
        println!("{}", grad);

        // 3.1628532
        res.to_data().assert_approx_eq(&expected_res, 3);
    }
}
