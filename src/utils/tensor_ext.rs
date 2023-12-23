use burn::tensor::{backend::Backend, Element, ElementConversion, Numeric, Tensor};

pub fn pad<const D: usize, K, E, B>(
    tensor: Tensor<B, D, K>,
    pad_width: [(usize, usize); D],
    fill_value: E,
) -> Tensor<B, D, K>
where
    B: Backend,
    K: Numeric<B>,
    K::Elem: Element,
    E: ElementConversion,
{
    let device = tensor.device();
    let origin_shape = tensor.dims();

    let mut pad_shape = [0; D];
    let mut assign_range = Vec::with_capacity(D);
    for (idx, (&origin_len, (left_pad, right_pad))) in
        origin_shape.iter().zip(pad_width).enumerate()
    {
        pad_shape[idx] = origin_len + left_pad + right_pad;
        assign_range.push(left_pad..(left_pad + origin_len));
    }

    let padded = Tensor::<B, D, K>::full(pad_shape, fill_value, &device);

    padded.slice_assign::<D>(assign_range.try_into().unwrap(), tensor)
}
