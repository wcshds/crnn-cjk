use conv::ValueInto;
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer, Pixel, Primitive};
use imageproc::{
    definitions::Clamp,
    drawing::{draw_hollow_rect_mut, Canvas},
    geometric_transformations::{Interpolation, Projection},
    rect::Rect,
};
use nalgebra::{Matrix3, Matrix4, Matrix4x2, Matrix4x3, SMatrix, SVector, Vector4};

type Matrix8 = SMatrix<f32, 8, 8>;
type Vector8 = SVector<f32, 8>;

/// Performs the perspective matrix transformation of vectors
///
/// ## Reference:
/// [OpenCV documentation](https://docs.opencv.org/4.9.0/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7)
pub fn perspective_transform(
    points: &Matrix4x3<f32>,
    transform_mat: &Matrix4<f32>,
) -> Matrix4x3<f32> {
    #[rustfmt::skip]
    let points_pad_one: Matrix4<f32> = Matrix4::new(
        points.m11, points.m21, points.m31, points.m41,
        points.m12, points.m22, points.m32, points.m42,
        points.m13, points.m23, points.m33, points.m43,
        1., 1., 1., 1.,
    );

    let mul: Matrix4<f32> = transform_mat * points_pad_one;

    let row0: Vector4<f32> = mul.column(0) / mul.m41;
    let row1: Vector4<f32> = mul.column(1) / mul.m42;
    let row2: Vector4<f32> = mul.column(2) / mul.m43;
    let row3: Vector4<f32> = mul.column(3) / mul.m44;

    #[rustfmt::skip]
    let res = Matrix4x3::new(
        row0.x, row0.y, row0.z,
        row1.x, row1.y, row1.z,
        row2.x, row2.y, row2.z,
        row3.x, row3.y, row3.z,
    );

    res
}

/// ## Reference:
/// [OpenCV implementation](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/imgwarp.cpp#L3408-L3459)
pub fn get_perspective_transform(
    points_in: &Matrix4x2<f32>,
    points_out: &Matrix4x2<f32>,
) -> Matrix3<f32> {
    #[rustfmt::skip]
    let left = Matrix8::from_vec(vec![
        points_in.m11, points_in.m12, 1., 0., 0., 0., -points_in.m11 * points_out.m11, -points_in.m12 * points_out.m11,
        points_in.m21, points_in.m22, 1., 0., 0., 0., -points_in.m21 * points_out.m21, -points_in.m22 * points_out.m21,
        points_in.m31, points_in.m32, 1., 0., 0., 0., -points_in.m31 * points_out.m31, -points_in.m32 * points_out.m31,
        points_in.m41, points_in.m42, 1., 0., 0., 0., -points_in.m41 * points_out.m41, -points_in.m42 * points_out.m41,
        0., 0., 0., points_in.m11, points_in.m12, 1., -points_in.m11 * points_out.m12, -points_in.m12 * points_out.m12,
        0., 0., 0., points_in.m21, points_in.m22, 1., -points_in.m21 * points_out.m22, -points_in.m22 * points_out.m22,
        0., 0., 0., points_in.m31, points_in.m32, 1., -points_in.m31 * points_out.m32, -points_in.m32 * points_out.m32,
        0., 0., 0., points_in.m41, points_in.m42, 1., -points_in.m41 * points_out.m42, -points_in.m42 * points_out.m42,
    ]).transpose();

    #[rustfmt::skip]
    let right  = Vector8::from_vec(vec![
        points_out.m11,
        points_out.m21,
        points_out.m31,
        points_out.m41,
        points_out.m12,
        points_out.m22,
        points_out.m32,
        points_out.m42,
    ]);

    let decomp = left.lu();
    let x = decomp.solve(&right).expect("Linear resolution failed.");

    unsafe {
        Matrix3::new(
            *x.get_unchecked(0),
            *x.get_unchecked(1),
            *x.get_unchecked(2),
            *x.get_unchecked(3),
            *x.get_unchecked(4),
            *x.get_unchecked(5),
            *x.get_unchecked(6),
            *x.get_unchecked(7),
            1.0,
        )
    }
}

pub fn warp_perspective<I, P, S>(
    src: &I,
    transform_mat: &Matrix3<f32>,
    side_length: u32,
    default: P,
) -> ImageBuffer<P, Vec<S>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S> + 'static + Sync + Send,
    S: Primitive + 'static + Sync + Send + ValueInto<f32> + Clamp<f32>,
{
    #[rustfmt::skip]
    let projection = Projection::from_matrix([
        transform_mat.m11, transform_mat.m12, transform_mat.m13,
        transform_mat.m21, transform_mat.m22, transform_mat.m23,
        transform_mat.m31, transform_mat.m32, transform_mat.m33, 
    ]).unwrap();

    let mut padded_image = ImageBuffer::from_pixel(side_length, side_length, default);
    padded_image.copy_from(src, 0, 0).unwrap();

    imageproc::geometric_transformations::warp(
        &padded_image,
        &projection,
        Interpolation::Bilinear,
        default,
    )
}

/// Draws the outline of a rectangle on an image in place.
///
/// Draws as much of the boundary of the rectangle as lies inside the image bounds.
pub fn rectangle<C>(canvas: &mut C, rect: Rect, color: C::Pixel, thickness: u32)
where
    C: Canvas,
{
    let left = rect.left();
    let right = rect.right();
    let top = rect.top();
    let bottom = rect.bottom();

    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, top).of_size((right - left + 1) as u32, thickness),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, bottom).of_size((right - left + 1) as u32, thickness),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(left, top).of_size(thickness, (bottom - top + 1) as u32),
        color,
    );
    draw_hollow_rect_mut(
        canvas,
        Rect::at(right, top).of_size(thickness, (bottom - top + 1) as u32),
        color,
    );
}

/// The code snippet is copied from [fastblur](https://github.com/fschutt/fastblur).
pub struct GaussBlur;

impl GaussBlur {
    pub fn gaussian_blur(img: GrayImage, sigma_x: f32, sigma_y: f32) -> GrayImage {
        let width = img.width();
        let height = img.height();
        let mut data = img.into_vec();

        GaussBlur::gaussian_blur_asymmetric_single_channel(
            &mut data,
            width as usize,
            height as usize,
            sigma_x,
            sigma_y,
        );

        GrayImage::from_vec(width, height, data).unwrap()
    }

    #[inline]
    /// If there is no valid size (e.g. radius is negative), returns `vec![1; len]`
    /// which would translate to blur radius of 0
    fn create_box_gauss(sigma: f32, n: usize) -> Vec<i32> {
        if sigma > 0.0 {
            let n_float = n as f32;

            // Ideal averaging filter width
            let w_ideal = (12.0 * sigma * sigma / n_float).sqrt() + 1.0;
            let mut wl: i32 = w_ideal.floor() as i32;

            if wl % 2 == 0 {
                wl -= 1;
            };

            let wu = wl + 2;

            let wl_float = wl as f32;
            let m_ideal = (12.0 * sigma * sigma
                - n_float * wl_float * wl_float
                - 4.0 * n_float * wl_float
                - 3.0 * n_float)
                / (-4.0 * wl_float - 4.0);
            let m: usize = m_ideal.round() as usize;

            let mut sizes = Vec::<i32>::new();

            for i in 0..n {
                if i < m {
                    sizes.push(wl);
                } else {
                    sizes.push(wu);
                }
            }

            sizes
        } else {
            vec![1; n]
        }
    }

    #[inline]
    fn box_blur_single_channel(
        backbuf: &mut [u8],
        frontbuf: &mut [u8],
        width: usize,
        height: usize,
        blur_radius_horz: usize,
        blur_radius_vert: usize,
    ) {
        GaussBlur::box_blur_horz_single_channel(backbuf, frontbuf, width, height, blur_radius_horz);
        GaussBlur::box_blur_vert_single_channel(frontbuf, backbuf, width, height, blur_radius_vert);
    }

    #[inline]
    fn box_blur_vert_single_channel(
        backbuf: &[u8],
        frontbuf: &mut [u8],
        width: usize,
        height: usize,
        blur_radius: usize,
    ) {
        if blur_radius == 0 {
            frontbuf.copy_from_slice(backbuf);
            return;
        }

        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;

        for i in 0..width {
            let col_start = i; //inclusive
            let col_end = i + width * (height - 1); //inclusive
            let mut ti: usize = i;
            let mut li: usize = ti;
            let mut ri: usize = ti + blur_radius * width;

            let fv: u8 = backbuf[col_start];
            let lv: u8 = backbuf[col_end];

            let mut val_r: isize = (blur_radius as isize + 1) * isize::from(fv);

            // Get the pixel at the specified index, or the first pixel of the column
            // if the index is beyond the top edge of the image
            let get_top = |i: usize| {
                if i < col_start {
                    fv
                } else {
                    backbuf[i]
                }
            };

            // Get the pixel at the specified index, or the last pixel of the column
            // if the index is beyond the bottom edge of the image
            let get_bottom = |i: usize| {
                if i > col_end {
                    lv
                } else {
                    backbuf[i]
                }
            };

            for j in 0..std::cmp::min(blur_radius, height) {
                let bb = backbuf[ti + j * width];
                val_r += isize::from(bb);
            }
            if blur_radius > height {
                val_r += (blur_radius - height) as isize * isize::from(lv);
            }

            for _ in 0..std::cmp::min(height, blur_radius + 1) {
                let bb = get_bottom(ri);
                ri += width;
                val_r += isize::from(bb) - isize::from(fv);

                frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                ti += width;
            }

            if height > blur_radius {
                // otherwise `(height - blur_radius)` will underflow
                for _ in (blur_radius + 1)..(height - blur_radius) {
                    let bb1 = backbuf[ri];
                    ri += width;
                    let bb2 = backbuf[li];
                    li += width;

                    val_r += isize::from(bb1) - isize::from(bb2);

                    frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                    ti += width;
                }

                for _ in 0..std::cmp::min(height - blur_radius - 1, blur_radius) {
                    let bb = get_top(li);
                    li += width;

                    val_r += isize::from(lv) - isize::from(bb);

                    frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                    ti += width;
                }
            }
        }
    }

    #[inline]
    fn box_blur_horz_single_channel(
        backbuf: &[u8],
        frontbuf: &mut [u8],
        width: usize,
        height: usize,
        blur_radius: usize,
    ) {
        if blur_radius == 0 {
            frontbuf.copy_from_slice(backbuf);
            return;
        }

        let iarr = 1.0 / (blur_radius + blur_radius + 1) as f32;

        for i in 0..height {
            let row_start: usize = i * width; // inclusive
            let row_end: usize = (i + 1) * width - 1; // inclusive
            let mut ti: usize = i * width; // VERTICAL: $i;
            let mut li: usize = ti;
            let mut ri: usize = ti + blur_radius;

            let fv: u8 = backbuf[row_start];
            let lv: u8 = backbuf[row_end]; // VERTICAL: $backbuf[ti + $width - 1];

            let mut val_r: isize = (blur_radius as isize + 1) * isize::from(fv);

            // Get the pixel at the specified index, or the first pixel of the row
            // if the index is beyond the left edge of the image
            let get_left = |i: usize| {
                if i < row_start {
                    fv
                } else {
                    backbuf[i]
                }
            };

            // Get the pixel at the specified index, or the last pixel of the row
            // if the index is beyond the right edge of the image
            let get_right = |i: usize| {
                if i > row_end {
                    lv
                } else {
                    backbuf[i]
                }
            };

            for j in 0..std::cmp::min(blur_radius, width) {
                let bb = backbuf[ti + j]; // VERTICAL: ti + j * width
                val_r += isize::from(bb);
            }

            if blur_radius > width {
                val_r += (blur_radius - height) as isize * isize::from(lv);
            }

            // Process the left side where we need pixels from beyond the left edge
            for _ in 0..std::cmp::min(width, blur_radius + 1) {
                let bb = get_right(ri);
                ri += 1;
                val_r += isize::from(bb) - isize::from(fv);

                frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                ti += 1; // VERTICAL : ti += width, same with the other areas
            }

            if width > blur_radius {
                // otherwise `(width - blur_radius)` will underflow
                // Process the middle where we know we won't bump into borders
                // without the extra indirection of get_left/get_right. This is faster.
                for _ in (blur_radius + 1)..(width - blur_radius) {
                    let bb1 = backbuf[ri];
                    ri += 1;
                    let bb2 = backbuf[li];
                    li += 1;

                    val_r += isize::from(bb1) - isize::from(bb2);

                    frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                    ti += 1;
                }

                // Process the right side where we need pixels from beyond the right edge
                for _ in 0..std::cmp::min(width - blur_radius - 1, blur_radius) {
                    let bb = get_left(li);
                    li += 1;

                    val_r += isize::from(lv) - isize::from(bb);

                    frontbuf[ti] = GaussBlur::round(val_r as f32 * iarr) as u8;
                    ti += 1;
                }
            }
        }
    }

    #[inline]
    /// Fast rounding for x <= 2^23.
    /// This is orders of magnitude faster than built-in rounding intrinsic.
    ///
    /// Source: https://stackoverflow.com/a/42386149/585725
    fn round(mut x: f32) -> f32 {
        x += 12582912.0;
        x -= 12582912.0;
        x
    }

    /// Same as gaussian_blur, but allows using different blur radii for vertical and horizontal passes
    fn gaussian_blur_asymmetric_single_channel(
        data: &mut Vec<u8>,
        width: usize,
        height: usize,
        blur_radius_horizontal: f32,
        blur_radius_vertical: f32,
    ) {
        let boxes_horz = GaussBlur::create_box_gauss(blur_radius_horizontal, 3);
        let boxes_vert = GaussBlur::create_box_gauss(blur_radius_vertical, 3);
        let mut backbuf = data.clone();

        for (box_size_horz, box_size_vert) in boxes_horz.iter().zip(boxes_vert.iter()) {
            let radius_horz = ((box_size_horz - 1) / 2) as usize;
            let radius_vert = ((box_size_vert - 1) / 2) as usize;
            GaussBlur::box_blur_single_channel(
                &mut backbuf,
                data,
                width,
                height,
                radius_horz,
                radius_vert,
            );
        }
    }
}
