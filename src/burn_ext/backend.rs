use burn::{
    backend::{
        autodiff::{
            grads::Gradients,
            ops::{unary, Backward, Ops, OpsKind},
        },
        libtorch::{TchElement, TchTensor},
        ndarray::FloatNdArrayElement,
        Autodiff, LibTorch, NdArray, Wgpu,
    },
    nn::loss::Reduction,
    tensor::{
        ops::{FloatTensor, IntTensor, IntTensorOps, TensorOps},
        ElementConversion, Shape,
    },
};

const NEG_INF: f32 = -1e5;
// a small value used to prevent the occurrence of log(0)
const EPSILON: f32 = 1e-8;

fn unsqueeze_dim<const D1: usize, const D2: usize, B: burn::tensor::backend::Backend>(
    tensor: B::TensorPrimitive<D1>,
    dim: usize,
) -> B::TensorPrimitive<D2> {
    let mut dims = [1; D2];
    let shape = B::shape(&tensor);

    dims[0..dim].copy_from_slice(&shape.dims[0..dim]);

    if dim < D1 {
        dims[dim] = 1;
        dims[(dim + 1)..].copy_from_slice(&shape.dims[dim..]);
    } else {
        dims[dim] = 1;
    }

    let shape = Shape::from(dims);
    B::reshape(tensor, shape)
}

fn stack<const D1: usize, const D2: usize, B: burn::tensor::backend::Backend>(
    tensors: Vec<B::TensorPrimitive<D1>>,
    dim: usize,
) -> B::TensorPrimitive<D2> {
    let tensors = tensors
        .into_iter()
        .map(|t| unsqueeze_dim::<D1, D2, B>(t, dim))
        .collect();

    B::cat(tensors, dim)
}

fn pad<const D: usize, B: burn::tensor::backend::Backend>(
    tensor: B::TensorPrimitive<D>,
    pad_width: [(usize, usize); D],
    fill_value: B::FloatElem,
) -> B::TensorPrimitive<D> {
    let device = B::device(&tensor);
    let origin_shape = B::shape(&tensor).dims;

    let mut pad_shape = [0; D];
    let mut assign_range = Vec::with_capacity(D);
    for (idx, (&origin_len, (left_pad, right_pad))) in
        origin_shape.iter().zip(pad_width).enumerate()
    {
        pad_shape[idx] = origin_len + left_pad + right_pad;
        assign_range.push(left_pad..(left_pad + origin_len));
    }

    let padded = B::full(Shape::from(pad_shape), fill_value, &device);

    B::slice_assign::<D, D>(padded, assign_range.try_into().unwrap(), tensor)
}

fn one_hot<B: burn::tensor::backend::Backend>(
    tensor: B::IntTensorPrimitive<2>,
    num_classes: usize,
) -> B::TensorPrimitive<3> {
    let device = B::int_device(&tensor);
    let [dim0, dim1] = B::int_shape(&tensor).dims;

    let labels = B::int_repeat(
        B::int_reshape(tensor, Shape::from([dim0, dim1, 1])),
        2,
        num_classes,
    );
    let indices = B::int_repeat(
        B::int_repeat(
            B::int_reshape(
                B::arange(0..num_classes, &device),
                Shape::from([1, 1, num_classes]),
            ),
            1,
            dim1,
        ),
        0,
        dim0,
    );

    B::bool_into_float(B::int_equal(labels, indices))
}

fn pad_target<B: burn::tensor::backend::Backend>(
    targets: B::IntTensorPrimitive<1>,
    target_lengths: B::IntTensorPrimitive<1>,
    max_target_length: usize,
    blank: usize,
    device: &B::Device,
) -> B::IntTensorPrimitive<2> {
    let [batch_size] = B::int_shape(&target_lengths).dims;

    let mut targets_pad = B::int_full(
        Shape::from([batch_size, max_target_length]),
        (blank as i32).elem(),
        device,
    );
    let mut start = 0usize;
    for batch in 0..batch_size {
        let length = B::int_slice(target_lengths.clone(), [batch..(batch + 1)]);
        let length = B::int_into_data(length).read().value[0].elem::<u32>() as usize;

        targets_pad = B::int_slice_assign(
            targets_pad,
            [batch..(batch + 1), 0..length],
            B::int_reshape(
                B::int_slice(targets.clone(), [start..(start + length)]),
                Shape::from([1, length]),
            ),
        );

        start += length
    }

    targets_pad
}

pub trait Backend: burn::tensor::backend::Backend {
    fn apply_reduction(
        neg_log_likelihood: FloatTensor<Self, 1>,
        target_lengths: IntTensor<Self, 1>,
        reduction: Option<Reduction>,
    ) -> FloatTensor<Self, 1> {
        match reduction {
            Some(Reduction::Mean) | Some(Reduction::Auto) => Self::mean(Self::div(
                neg_log_likelihood,
                Self::int_into_float(target_lengths),
            )),
            Some(Reduction::Sum) => Self::sum(neg_log_likelihood),
            None => neg_log_likelihood,
        }
    }

    fn ctc_loss(
        log_probs: FloatTensor<Self, 3>,
        targets: IntTensor<Self, 1>,
        input_lengths: IntTensor<Self, 1>,
        target_lengths: IntTensor<Self, 1>,
        reduction: Option<Reduction>,
        blank: usize,
    ) -> FloatTensor<Self, 1> {
        let neg_log_likelihood = Self::ctc_loss_internal(
            log_probs,
            targets,
            input_lengths,
            target_lengths.clone(),
            blank,
        );

        Self::apply_reduction(neg_log_likelihood, target_lengths, reduction)
    }

    fn ctc_loss_internal(
        log_probs: Self::TensorPrimitive<3>,
        targets: Self::IntTensorPrimitive<1>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
    ) -> Self::TensorPrimitive<1> {
        let (neg_log_likelihood, _, _) = Self::ctc_loss_internal_with_alphas_targetspad(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank,
        );

        neg_log_likelihood
    }

    fn ctc_loss_internal_with_alphas_targetspad(
        log_probs: Self::TensorPrimitive<3>,
        targets: Self::IntTensorPrimitive<1>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
    ) -> (
        Self::TensorPrimitive<1>,
        Self::TensorPrimitive<3>,
        Self::IntTensorPrimitive<2>,
    ) {
        // make sure tensors are on the same device
        let device = Self::device(&log_probs);
        let input_lengths = Self::int_to_device(input_lengths, &device);
        let target_lengths = Self::int_to_device(target_lengths, &device);

        let [batch_size, seq_length, num_classes] = Self::shape(&log_probs).dims;
        let max_target_length = Self::int_into_data(Self::int_max(target_lengths.clone()))
            .read()
            .value[0]
            .elem::<u32>() as usize;
        let target_with_blank_length = 2 * max_target_length + 1;

        let targets_pad = pad_target::<Self>(
            targets,
            target_lengths.clone(),
            max_target_length,
            blank,
            &device,
        );
        let targets_one_hot = one_hot::<Self>(targets_pad.clone(), num_classes);

        let log_alphas = Self::empty::<3>(
            Shape::from([batch_size, seq_length, target_with_blank_length]),
            &device,
        );
        // initialize value at t0
        let log_alphas = Self::slice_assign(
            log_alphas,
            [0..batch_size, 0..1, 0..target_with_blank_length],
            Self::full(
                Shape::from([batch_size, 1, target_with_blank_length]),
                NEG_INF.elem(),
                &device,
            ),
        );
        let log_alphas = Self::slice_assign(
            log_alphas,
            [0..batch_size, 0..1, 0..1],
            Self::slice(log_probs.clone(), [0..batch_size, 0..1, blank..(blank + 1)]),
        );
        let target_primes = Self::int_reshape(
            Self::int_slice(targets_pad.clone(), [0..batch_size, 0..1]),
            Shape::from([batch_size, 1, 1]),
        );
        let mut log_alphas = Self::slice_assign(
            log_alphas,
            [0..batch_size, 0..1, 1..2],
            Self::gather(
                2,
                Self::slice(log_probs.clone(), [0..batch_size, 0..1, 0..num_classes]),
                target_primes,
            ),
        );

        // Shape: [batch_size, seq_length, max_target_length]
        let log_probs_letter_available = Self::swap_dims(
            Self::matmul(targets_one_hot, Self::swap_dims(log_probs.clone(), 1, 2)),
            1,
            2,
        );
        // Shape: [batch_size, seq_length, 1]
        let log_probs_blank_available = Self::slice(
            log_probs,
            [0..batch_size, 0..seq_length, blank..(blank + 1)],
        );
        // Shape: [batch_size, seq_length, 2 * max_target_length + 1]
        let log_probs_available = Self::empty(
            Shape::from([batch_size, seq_length, target_with_blank_length]),
            &device,
        );
        let log_probs_available = Self::slice_assign(
            log_probs_available,
            [0..batch_size, 0..seq_length, 0..1],
            log_probs_blank_available.clone(),
        );
        let log_probs_available = Self::slice_assign(
            log_probs_available,
            [0..batch_size, 0..seq_length, 1..target_with_blank_length],
            Self::reshape(
                stack::<3, 4, Self>(
                    [
                        log_probs_letter_available,
                        Self::repeat(log_probs_blank_available, 2, max_target_length),
                    ]
                    .to_vec(),
                    3,
                ),
                Shape::from([batch_size, seq_length, 2 * max_target_length]),
            ),
        );

        // s != s-2
        let mask_la3_letter = Self::bool_into_float(Self::bool_not(Self::int_equal(
            Self::int_slice(
                targets_pad.clone(),
                [0..batch_size, 0..(max_target_length - 1)],
            ),
            Self::int_slice(targets_pad.clone(), [0..batch_size, 1..max_target_length]),
        )));
        let mask_la3_blank = Self::zeros(Shape::from([batch_size, max_target_length - 1]), &device);
        let mask_la3 = unsqueeze_dim::<2, 3, Self>(
            pad::<2, Self>(
                // interlace mask_la3_letter and mask_la3_blank
                Self::reshape(
                    stack::<2, 3, Self>([mask_la3_letter, mask_la3_blank].to_vec(), 2),
                    Shape::from([batch_size, 2 * (max_target_length - 1)]),
                ),
                [(0, 0), (3, 0)],
                0.0.elem(),
            ),
            1,
        );

        for t in 1..seq_length {
            // \alpha_{t-1}(s)
            let la1 = Self::slice(
                log_alphas.clone(),
                [0..batch_size, (t - 1)..t, 0..target_with_blank_length],
            );
            // \alpha_{t-1}(s-1)
            let la2 = Self::clamp_min(
                Self::slice(
                    la1.clone(),
                    [0..batch_size, 0..1, 0..(target_with_blank_length - 1)],
                ),
                NEG_INF.elem(),
            );
            let la2 = pad::<3, Self>(la2, [(0, 0), (0, 0), (1, 0)], NEG_INF.elem());
            // \alpha_{t-1}(s-2)
            let la3 = Self::clamp_min(
                Self::slice(
                    la1.clone(),
                    [0..batch_size, 0..1, 0..(target_with_blank_length - 2)],
                ),
                NEG_INF.elem(),
            );
            let la3 = pad::<3, Self>(la3, [(0, 0), (0, 0), (2, 0)], NEG_INF.elem());
            // for the logsumexp calculation
            let lamax = Self::reshape(
                Self::max_dim(
                    stack::<3, 4, Self>([la1.clone(), la2.clone(), la3.clone()].to_vec(), 3),
                    3,
                ),
                Shape::from([batch_size, 1, target_with_blank_length]),
            );

            let la_sum = Self::log(Self::add_scalar(
                Self::add(
                    Self::add(
                        Self::exp(Self::sub(la1, lamax.clone())),
                        Self::exp(Self::sub(la2, lamax.clone())),
                    ),
                    Self::mul(Self::exp(Self::sub(la3, lamax.clone())), mask_la3.clone()),
                ),
                EPSILON.elem(),
            ));
            log_alphas = Self::slice_assign(
                log_alphas,
                [0..batch_size, t..(t + 1), 0..target_with_blank_length],
                Self::clamp_min(
                    Self::add(
                        Self::add(la_sum, lamax),
                        Self::slice(
                            log_probs_available.clone(),
                            [0..batch_size, t..(t + 1), 0..target_with_blank_length],
                        ),
                    ),
                    NEG_INF.elem(),
                ),
            );
        }

        let l1 = Self::reshape(
            Self::gather(
                2,
                Self::gather(
                    1,
                    log_alphas.clone(),
                    Self::int_repeat(
                        Self::int_reshape(
                            Self::int_sub_scalar(input_lengths.clone(), 1.elem()),
                            Shape::from([batch_size, 1, 1]),
                        ),
                        2,
                        target_with_blank_length,
                    ),
                ),
                Self::int_reshape(
                    Self::int_mul_scalar(target_lengths.clone(), 2.elem()),
                    Shape::from([batch_size, 1, 1]),
                ),
            ),
            Shape::from([batch_size]),
        );
        let l2 = Self::reshape(
            Self::gather(
                2,
                Self::gather(
                    1,
                    log_alphas.clone(),
                    Self::int_repeat(
                        Self::int_reshape(
                            Self::int_sub_scalar(input_lengths, 1.elem()),
                            Shape::from([batch_size, 1, 1]),
                        ),
                        2,
                        target_with_blank_length,
                    ),
                ),
                Self::int_reshape(
                    Self::int_sub_scalar(
                        Self::int_mul_scalar(target_lengths.clone(), 2.elem()),
                        1.elem(),
                    ),
                    Shape::from([batch_size, 1, 1]),
                ),
            ),
            Shape::from([batch_size]),
        );

        let m = Self::max(Self::cat([l1.clone(), l2.clone()].to_vec(), 0));
        let m = Self::clamp_min(m, NEG_INF.elem());
        let log_likelihood = Self::add(
            Self::log(Self::add_scalar(
                Self::add(
                    Self::exp(Self::sub(l1, m.clone())),
                    Self::exp(Self::sub(l2, m.clone())),
                ),
                EPSILON.elem(),
            )),
            m,
        );
        let neg_log_likelihood = Self::neg(log_likelihood);

        (neg_log_likelihood, log_alphas, targets_pad)
    }

    fn ctc_loss_internal_backward(
        log_probs: Self::TensorPrimitive<3>,
        targets_pad: Self::IntTensorPrimitive<2>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
        neg_log_likelihood: Self::TensorPrimitive<1>,
        log_alphas: Self::TensorPrimitive<3>,
    ) -> Self::TensorPrimitive<3> {
        let device = Self::device(&log_probs);
        let [batch_size, max_input_length, num_classes] = Self::shape(&log_probs).dims;
        let mut grad = Self::full(
            Shape::from([batch_size, max_input_length, num_classes]),
            NEG_INF.elem(),
            &device,
        );
        let mut log_betas = Self::full(Self::shape(&log_alphas), NEG_INF.elem(), &device);

        for b in 0..batch_size {
            let input_length =
                Self::int_into_data(Self::int_slice(input_lengths.clone(), [b..(b + 1)]))
                    .read()
                    .value[0]
                    .elem::<u32>() as usize;
            let target_length =
                Self::int_into_data(Self::int_slice(target_lengths.clone(), [b..(b + 1)]))
                    .read()
                    .value[0]
                    .elem::<u32>() as usize;
            let targets_data = Self::int_reshape(
                Self::int_slice(targets_pad.clone(), [b..(b + 1), 0..target_length]),
                Shape::from([target_length]),
            );
            let nll = Self::reshape(
                Self::slice(neg_log_likelihood.clone(), [b..(b + 1)]),
                Shape::from([1, 1, 1]),
            );

            if input_length > 0 {
                log_betas = Self::slice_assign(
                    log_betas,
                    [
                        b..(b + 1),
                        (input_length - 1)..input_length,
                        (2 * target_length)..(2 * target_length + 1),
                    ],
                    Self::slice(
                        log_probs.clone(),
                        [
                            b..(b + 1),
                            (input_length - 1)..input_length,
                            blank..(blank + 1),
                        ],
                    ),
                );
                grad = Self::slice_assign(
                    grad,
                    // grad_a[input_length-1][BLANK]
                    [
                        b..(b + 1),
                        (input_length - 1)..input_length,
                        blank..(blank + 1),
                    ],
                    // log_alpha_a[input_length-1][2*target_length] + log_beta_a[input_length-1][2*target_length]
                    Self::add(
                        Self::slice(
                            log_alphas.clone(),
                            [
                                b..(b + 1),
                                (input_length - 1)..input_length,
                                (2 * target_length)..(2 * target_length + 1),
                            ],
                        ),
                        Self::slice(
                            log_betas.clone(),
                            [
                                b..(b + 1),
                                (input_length - 1)..input_length,
                                (2 * target_length)..(2 * target_length + 1),
                            ],
                        ),
                    ),
                );

                if target_length > 0 {
                    let current_target_prime =
                        Self::get_target_prime(targets_data.clone(), 2 * target_length - 1, blank);

                    log_betas = Self::slice_assign(
                        log_betas,
                        [
                            b..(b + 1),
                            (input_length - 1)..input_length,
                            (2 * target_length - 1)..(2 * target_length),
                        ],
                        Self::slice(
                            log_probs.clone(),
                            [
                                b..(b + 1),
                                (input_length - 1)..input_length,
                                current_target_prime..(current_target_prime + 1),
                            ],
                        ),
                    );

                    grad = Self::slice_assign(
                        grad,
                        // grad_a[input_length-1][current_target_prime]
                        [
                            b..(b + 1),
                            (input_length - 1)..input_length,
                            current_target_prime..(current_target_prime + 1),
                        ],
                        // log_alpha_a[input_length-1][2*target_length] + log_beta_a[input_length-1][2*target_length]
                        Self::add(
                            Self::slice(
                                log_alphas.clone(),
                                [
                                    b..(b + 1),
                                    (input_length - 1)..input_length,
                                    (2 * target_length - 1)..(2 * target_length),
                                ],
                            ),
                            Self::slice(
                                log_betas.clone(),
                                [
                                    b..(b + 1),
                                    (input_length - 1)..input_length,
                                    (2 * target_length - 1)..(2 * target_length),
                                ],
                            ),
                        ),
                    );
                }
            }

            for t in (0..=(input_length - 2)).rev() {
                for s in (0..=(2 * target_length)).rev() {
                    let lb1 = Self::slice(
                        log_betas.clone(),
                        [b..(b + 1), (t + 1)..(t + 2), s..(s + 1)],
                    );
                    let mut lbmax = lb1.clone();
                    let (lb2, lb3);
                    let current_target_prime =
                        Self::get_target_prime(targets_data.clone(), s, blank);
                    if s < 2 * target_length {
                        lb2 = Self::slice(
                            log_betas.clone(),
                            [b..(b + 1), (t + 1)..(t + 2), (s + 1)..(s + 2)],
                        );
                        if Self::into_data(lb2.clone()).read().value[0].elem::<f32>()
                            > Self::into_data(lbmax.clone()).read().value[0].elem::<f32>()
                        {
                            lbmax = lb2.clone();
                        }
                    } else {
                        lb2 = Self::full(Shape::from([1, 1, 1]), NEG_INF.elem(), &device);
                    }

                    if (s < 2 * target_length - 1)
                        && (Self::get_target_prime(targets_data.clone(), s + 2, blank)
                            != current_target_prime)
                    {
                        lb3 = Self::slice(
                            log_betas.clone(),
                            [b..(b + 1), (t + 1)..(t + 2), (s + 2)..(s + 3)],
                        );

                        if Self::into_data(lb3.clone()).read().value[0].elem::<f32>()
                            > Self::into_data(lbmax.clone()).read().value[0].elem::<f32>()
                        {
                            lbmax = lb3.clone();
                        }
                    } else {
                        lb3 = Self::full(Shape::from([1, 1, 1]), NEG_INF.elem(), &device);
                    }

                    if Self::into_data(lbmax.clone()).read().value[0].elem::<f32>() == NEG_INF {
                        lbmax = Self::full(Shape::from([1, 1, 1]), 0.elem(), &device);
                    }

                    log_betas = Self::slice_assign(
                        log_betas,
                        [b..(b + 1), t..(t + 1), s..(s + 1)],
                        // std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax + log_probs_a[t][current_target_prime]
                        Self::add(
                            Self::add(
                                Self::log(Self::add_scalar(
                                    Self::add(
                                        Self::add(
                                            Self::exp(Self::sub(lb1, lbmax.clone())),
                                            Self::exp(Self::sub(lb2, lbmax.clone())),
                                        ),
                                        Self::exp(Self::sub(lb3, lbmax.clone())),
                                    ),
                                    EPSILON.elem(),
                                )),
                                lbmax.clone(),
                            ),
                            Self::slice(
                                log_probs.clone(),
                                [
                                    b..(b + 1),
                                    t..(t + 1),
                                    current_target_prime..(current_target_prime + 1),
                                ],
                            ),
                        ),
                    );

                    let log_alpha_beta = Self::add(
                        Self::slice(log_alphas.clone(), [b..(b + 1), t..(t + 1), s..(s + 1)]),
                        Self::slice(log_betas.clone(), [b..(b + 1), t..(t + 1), s..(s + 1)]),
                    );
                    let lcab = Self::slice(
                        grad.clone(),
                        [
                            b..(b + 1),
                            t..(t + 1),
                            current_target_prime..(current_target_prime + 1),
                        ],
                    );

                    if Self::into_data(lcab.clone()).read().value[0].elem::<f32>() <= NEG_INF {
                        grad = Self::slice_assign(
                            grad.clone(),
                            [
                                b..(b + 1),
                                t..(t + 1),
                                current_target_prime..(current_target_prime + 1),
                            ],
                            log_alpha_beta,
                        );
                    } else {
                        let max = if Self::into_data(lcab.clone()).read().value[0].elem::<f32>()
                            > Self::into_data(log_alpha_beta.clone()).read().value[0].elem::<f32>()
                        {
                            lcab.clone()
                        } else {
                            log_alpha_beta.clone()
                        };

                        grad = Self::slice_assign(
                            grad.clone(),
                            [
                                b..(b + 1),
                                t..(t + 1),
                                current_target_prime..(current_target_prime + 1),
                            ],
                            Self::add(
                                Self::log(Self::add_scalar(
                                    Self::add(
                                        Self::exp(Self::sub(lcab.clone(), max.clone())),
                                        Self::exp(Self::sub(log_alpha_beta.clone(), max.clone())),
                                    ),
                                    EPSILON.elem(),
                                )),
                                max.clone(),
                            ),
                        );
                    }
                }
            }

            for t in 0..input_length {
                for c in 0..num_classes {
                    let res = Self::slice(grad.clone(), [b..(b + 1), t..(t + 1), c..(c + 1)]);
                    let lp = Self::slice(log_probs.clone(), [b..(b + 1), t..(t + 1), c..(c + 1)]);
                    grad = Self::slice_assign(
                        grad.clone(),
                        [b..(b + 1), t..(t + 1), c..(c + 1)],
                        Self::sub(
                            Self::exp(lp.clone()),
                            Self::exp(Self::sub(Self::add(res, nll.clone()), lp)),
                        ),
                    )
                }
            }
        }

        grad
    }

    fn get_target_prime(
        target_data: Self::IntTensorPrimitive<1>,
        idx: usize,
        blank: usize,
    ) -> usize {
        if idx % 2 == 0 {
            blank
        } else {
            let prime = Self::int_slice(target_data, [(idx / 2)..(idx / 2 + 1)]);
            Self::int_into_data(prime).read().value[0].elem::<u32>() as usize
        }
    }
}

impl<E: FloatNdArrayElement> Backend for NdArray<E> {}

impl Backend for Wgpu {}

// impl Backend for Candle {}

impl<E: TchElement> Backend for LibTorch<E> {
    fn ctc_loss_internal_with_alphas_targetspad(
        log_probs: Self::TensorPrimitive<3>,
        targets: Self::IntTensorPrimitive<1>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
    ) -> (
        Self::TensorPrimitive<1>,
        Self::TensorPrimitive<3>,
        Self::IntTensorPrimitive<2>,
    ) {
        let device = Self::device(&log_probs);

        let max_target_length = Self::int_into_data(Self::int_max(target_lengths.clone()))
            .read()
            .value[0]
            .elem::<u32>() as usize;
        let targets_pad = pad_target::<Self>(
            targets,
            target_lengths.clone(),
            max_target_length,
            blank,
            &device,
        );
        let log_probs = Self::swap_dims(log_probs, 0, 1);

        let (neg_log_likelihood, log_alphas) = tch::Tensor::f_internal_ctc_loss_tensor(
            &log_probs.tensor,
            &targets_pad.clone().tensor,
            &input_lengths.tensor,
            &target_lengths.tensor,
            blank as i64,
            false,
        )
        .expect("libtorch fail to compute ctc loss forward");

        (
            TchTensor::new(neg_log_likelihood),
            TchTensor::new(log_alphas),
            targets_pad,
        )
    }

    fn ctc_loss_internal_backward(
        log_probs: Self::TensorPrimitive<3>,
        targets_pad: Self::IntTensorPrimitive<2>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
        neg_log_likelihood: Self::TensorPrimitive<1>,
        log_alphas: Self::TensorPrimitive<3>,
    ) -> Self::TensorPrimitive<3> {
        let [batch_size, _, _] = Self::shape(&log_probs).dims;
        let grad = Self::ones(Shape::new([batch_size]), &Self::device(&log_probs));
        let log_probs = Self::swap_dims(log_probs, 0, 1);

        let res = tch::Tensor::f_internal_ctc_loss_backward_tensor(
            &grad.tensor,
            &log_probs.tensor,
            &targets_pad.tensor,
            &input_lengths.tensor,
            &target_lengths.tensor,
            &neg_log_likelihood.tensor,
            &log_alphas.tensor,
            blank as i64,
            false,
        )
        .expect("libtorch fail to compute ctc loss backward");

        Self::swap_dims(TchTensor::new(res), 0, 1)
    }
}

impl<B: Backend> Backend for Autodiff<B> {
    fn ctc_loss_internal(
        log_probs: Self::TensorPrimitive<3>,
        targets: Self::IntTensorPrimitive<1>,
        input_lengths: Self::IntTensorPrimitive<1>,
        target_lengths: Self::IntTensorPrimitive<1>,
        blank: usize,
    ) -> Self::TensorPrimitive<1> {
        #[derive(Debug)]
        struct CTCLoss;

        impl<B: Backend> Backward<B, 1, 1> for CTCLoss {
            // type State = (B::TensorPrimitive<2>, IntTensor<B, 2>);
            type State = (
                B::TensorPrimitive<3>,
                B::IntTensorPrimitive<2>,
                B::IntTensorPrimitive<1>,
                B::IntTensorPrimitive<1>,
                usize,
                B::TensorPrimitive<1>,
                B::TensorPrimitive<3>,
            );

            fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients) {
                let (
                    log_probs,
                    targets_pad,
                    input_lengths,
                    target_lengths,
                    blank,
                    neg_log_likelihood,
                    log_alphas,
                ) = ops.state;

                unary::<B, 1, 3, _>(ops.parents, ops.node, grads, |grad| {
                    let [batch_size] = B::shape(&grad).dims;

                    let res = B::ctc_loss_internal_backward(
                        log_probs,
                        targets_pad,
                        input_lengths,
                        target_lengths,
                        blank,
                        neg_log_likelihood,
                        log_alphas,
                    );

                    B::mul(
                        res.clone(),
                        B::reshape(grad, Shape::from([batch_size, 1, 1])),
                    )
                });
            }
        }

        match CTCLoss
            .prepare([log_probs.clone().node], [log_probs.clone().graph])
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let (neg_log_likelihood, log_alphas, targets_pad) =
                    B::ctc_loss_internal_with_alphas_targetspad(
                        log_probs.clone().primitive,
                        targets.clone(),
                        input_lengths.clone(),
                        target_lengths.clone(),
                        blank,
                    );
                prep.finish(
                    (
                        log_probs.primitive,
                        targets_pad,
                        input_lengths,
                        target_lengths,
                        blank,
                        neg_log_likelihood.clone(),
                        log_alphas,
                    ),
                    neg_log_likelihood,
                )
            }
            OpsKind::UnTracked(prep) => {
                let neg_log_likelihood = B::ctc_loss_internal(
                    log_probs.primitive,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank,
                );
                prep.finish(neg_log_likelihood)
            }
        }
    }
}
