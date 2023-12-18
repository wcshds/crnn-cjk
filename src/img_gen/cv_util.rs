use image::{imageops::FilterType, GenericImage, GrayImage, Luma};
use nalgebra::{Matrix3, Matrix4, Matrix4x2, Matrix4x3};

use super::cv_helper;

fn get_rotate_matrix(x: f32, y: f32, z: f32) -> Matrix4<f32> {
    let x = x.to_radians();
    let y = y.to_radians();
    let z = z.to_radians();

    let (sin_x, cos_x) = x.sin_cos();
    let (sin_y, cos_y) = y.sin_cos();
    let (sin_z, cos_z) = z.sin_cos();

    #[rustfmt::skip]
    let matrix_x = Matrix4::new(
        1., 0., 0., 0.,
        0., cos_x, -sin_x, 0.,
        0., sin_x, cos_x, 0.,
        0., 0., 0., 1.
    );

    #[rustfmt::skip]
    let matrix_y = Matrix4::new(
        cos_y, 0., sin_y, 0.,
        0., 1., 0., 0.,
        -sin_y, 0., cos_y, 0.,
        0., 0., 0., 1.
    );

    #[rustfmt::skip]
    let matrix_z = Matrix4::new(
        cos_z, -sin_z, 0., 0.,
        sin_z, cos_z, 0., 0.,
        0., 0., 1., 0.,
        0., 0., 0., 1.
    );

    matrix_x * matrix_y * matrix_z
}

fn get_warped_pnts(
    points_in: &Matrix4x3<f32>,
    points_out: &Matrix4x3<f32>,
    width: f32,
    height: f32,
    side_length: f32,
) -> (Matrix4x2<f32>, Matrix4x2<f32>) {
    let width_half = width * 0.5;
    let height_half = height * 0.5;
    let side_length_half = side_length * 0.5;

    (
        Matrix4x2::new(
            points_in.m11 + width_half,
            points_in.m12 + height_half,
            points_in.m21 + width_half,
            points_in.m22 + height_half,
            points_in.m31 + width_half,
            points_in.m32 + height_half,
            points_in.m41 + width_half,
            points_in.m42 + height_half,
        ),
        Matrix4x2::new(
            (points_out.m11 + 1.) * side_length_half,
            (points_out.m12 + 1.) * side_length_half,
            (points_out.m21 + 1.) * side_length_half,
            (points_out.m22 + 1.) * side_length_half,
            (points_out.m31 + 1.) * side_length_half,
            (points_out.m32 + 1.) * side_length_half,
            (points_out.m41 + 1.) * side_length_half,
            (points_out.m42 + 1.) * side_length_half,
        ),
    )
}

/// # Parameter:
/// - rotate_angle: (x, y, z)
/// - fovy: field of view along y axis
///
/// # Return:
/// A tutle contains 4 object
/// - 2D transformation matrix
/// - final side length
/// - points input
/// - points output
///
/// # Reference:
/// - [https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles]
/// - [https://nbviewer.org/github/manisoftwartist/perspectiveproj/blob/master/perspective.ipynb]
fn get_warp_matrix(
    width: usize,
    height: usize,
    rotate_angle: (f32, f32, f32),
    scale: f32,
    fovy: f32,
) -> (Matrix3<f32>, f32, Matrix4x2<f32>, Matrix4x2<f32>) {
    let (width, height) = (width as f32, height as f32);
    let (x, y, z) = rotate_angle;

    let fovy_half = ((fovy * 0.5) as f32).to_radians(); // fvHalf
    let distance = (width * width + height * height).sqrt(); // d
    let side_length = scale * distance / fovy_half.cos(); // sideLength
    let hypotenuse = distance / (2.0 * fovy_half.sin()); // h
    let rest = hypotenuse - distance * 0.5; // n
    let full = hypotenuse + distance * 0.5; // f

    let mut translation_mat: Matrix4<f32> = Matrix4::identity();
    translation_mat.m34 = -hypotenuse;

    let rotate_mat: Matrix4<f32> = get_rotate_matrix(x, y, z);

    // initialization should use zeros?
    let mut projection_mat = Matrix4::identity();
    projection_mat.m11 = 1.0 / fovy_half.tan();
    projection_mat.m22 = projection_mat.m11;
    projection_mat.m33 = -(full + rest) / (full - rest);
    projection_mat.m34 = -(2.0 * full * rest) / (full - rest);
    projection_mat.m43 = -1.0;

    let perspective_transform_mat: Matrix4<f32> = projection_mat * translation_mat * rotate_mat;

    let width_half = width * 0.5;
    let height_half = height * 0.5;

    #[rustfmt::skip]
    let points_in: Matrix4x3<f32> = Matrix4x3::new(
        -width_half, height_half, 0.,
        width_half, height_half, 0.,
        width_half, -height_half, 0.,
        -width_half, -height_half, 0.,
    );

    let points_out: Matrix4x3<f32> =
        cv_helper::perspective_transform(&points_in, &perspective_transform_mat);

    let (points_in, points_out) =
        get_warped_pnts(&points_in, &points_out, width, height, side_length);

    (
        cv_helper::get_perspective_transform(&points_in, &points_out),
        side_length,
        points_in,
        points_out,
    )
}

/// Perform a perspective transform and crop the transformed text area.
pub fn warp_perspective_transform(img: &GrayImage, rotate_angle: (f32, f32, f32)) -> GrayImage {
    let (raw_height, raw_width) = (img.height(), img.width());

    let (transform_mat, side_length, _, points_out) = get_warp_matrix(
        raw_width as usize,
        raw_height as usize,
        rotate_angle,
        1.0,
        50.,
    );

    let (raw_height, raw_width) = (raw_height as f32, raw_width as f32);
    let side_length = side_length.ceil() as u32;

    let mut warp_img = cv_helper::warp_perspective(img, &transform_mat, side_length, Luma([0]));

    let (min_x, max_x, min_y, max_y) = (
        points_out.column(0).min(),
        points_out.column(0).max(),
        points_out.column(1).min(),
        points_out.column(1).max(),
    );
    let (min_x, min_y, max_x, max_y) = (
        min_x.floor() as u32,
        min_y.floor() as u32,
        max_x.ceil() as u32,
        max_y.ceil() as u32,
    );
    let crop_img = warp_img
        .sub_image(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        .to_image();

    let (new_height, new_width) = (crop_img.height() as f32, crop_img.width() as f32);
    let (resize_width, resize_height) = (
        (new_width * raw_height / new_height).ceil() as u32,
        raw_height as u32,
    );
    let resize_img = if resize_width <= raw_width as u32 && resize_height <= raw_height as u32 {
        image::imageops::resize(&crop_img, resize_width, resize_height, FilterType::Triangle)
    } else {
        let (resize_width, resize_height) = (
            raw_width as u32,
            (new_height * raw_width / new_width).ceil() as u32,
        );
        image::imageops::resize(&crop_img, resize_width, resize_height, FilterType::Triangle)
    };

    resize_img
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_warp_perspective_transform() {
        let img = image::open("./test.png").unwrap();
        let gray = image::imageops::grayscale(&img);

        let res = warp_perspective_transform(&gray, (-30., -30., -30.));

        res.save("warp.png").unwrap();
    }
}
