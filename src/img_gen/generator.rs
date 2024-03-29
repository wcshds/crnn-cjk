use std::{
    fs,
    path::Path,
    sync::{Arc, RwLock},
};

use cosmic_text::{
    Attrs, AttrsList, AttrsOwned, Buffer, BufferLine, Family, FontSystem, Metrics, Shaping,
    Stretch, Style, SwashCache, Weight,
};
use image::{GenericImage, GenericImageView, GrayImage, ImageBuffer};
use indexmap::IndexMap;
use rand::seq::SliceRandom;
use rand_distr::WeightedAliasIndex;

use super::{
    corpus::wrap_text_with_font_list,
    cv_util::CvUtil,
    font_util::FontUtil,
    init::init_ch_dict_and_weight,
    merge_util::{BgFactory, MergeUtil},
    parse_config::GeneratorConfig,
};

/// Generate images of Chinese characters, including rare Chinese characters
pub struct Generator<P: AsRef<Path> + Clone> {
    font_util: FontUtil,
    cv_util: CvUtil,
    merge_util: MergeUtil,
    bg_factory: BgFactory,
    editor: Arc<RwLock<Buffer>>,
    swash_cache: Arc<RwLock<SwashCache>>,
    chinese_ch_dict: IndexMap<String, Vec<AttrsOwned>>,
    chinese_ch_weights: WeightedAliasIndex<f64>,
    config: GeneratorConfig<P>,
}

impl<P: AsRef<Path> + Clone> Clone for Generator<P> {
    fn clone(&self) -> Self {
        let font_system_origin = self.font_util.font_system.read().unwrap();
        let mut font_system = FontSystem::new_with_locale_and_db(
            font_system_origin.locale().to_owned(),
            font_system_origin.db().clone(),
        );
        // create one per application
        let swash_cache = SwashCache::new();

        let mut editor = Buffer::new(
            &mut font_system,
            Metrics::new(self.config.font_size as f32, self.config.line_height as f32),
        );
        editor.set_size(
            &mut font_system,
            self.config.font_img_width as f32,
            self.config.font_img_height as f32,
        );

        let font_util = FontUtil::new(font_system);
        Self {
            font_util,
            cv_util: self.cv_util.clone(),
            merge_util: self.merge_util.clone(),
            bg_factory: self.bg_factory.clone(),
            editor: Arc::new(RwLock::new(editor)),
            swash_cache: Arc::new(RwLock::new(swash_cache)),
            chinese_ch_dict: self.chinese_ch_dict.clone(),
            chinese_ch_weights: self.chinese_ch_weights.clone(),
            config: self.config.clone(),
        }
    }
}

// todo: image enhancement
impl<P: AsRef<Path> + Clone> Generator<P> {
    pub fn new(config: GeneratorConfig<P>) -> Self {
        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir(config.font_dir.as_ref());

        // create one per application
        let swash_cache = SwashCache::new();

        let mut editor = Buffer::new(
            &mut font_system,
            Metrics::new(config.font_size as f32, config.line_height as f32),
        );
        editor.set_size(
            &mut font_system,
            config.font_img_width as f32,
            config.font_img_height as f32,
        );

        let font_util = FontUtil::new(font_system);

        let chinese_character_file_data =
            fs::read_to_string(config.chinese_ch_file_path.as_ref()).unwrap();
        let full_font_list = font_util.get_full_font_list();
        let (chinese_ch_dict, chinese_ch_weights) =
            init_ch_dict_and_weight(&font_util, &full_font_list, &chinese_character_file_data);

        let cv_util = CvUtil {
            box_prob: config.box_prob,
            perspective_prob: config.perspective_prob,
            perspective_x: config.perspective_x,
            perspective_y: config.perspective_y,
            perspective_z: config.perspective_z,
            blur_prob: config.blur_prob,
            blur_sigma: config.blur_sigma,
            filter_prob: config.filter_prob,
            emboss_prob: config.emboss_prob,
            sharp_prob: config.sharp_prob,
        };
        let merge_util = MergeUtil {
            height_diff: config.height_diff,
            bg_alpha: config.bg_alpha,
            bg_beta: config.bg_beta,
            font_alpha: config.font_alpha,
            reverse_prob: config.reverse_prob,
        };
        let bg_factory = BgFactory::new(config.bg_dir.clone(), config.bg_height, config.bg_width);

        Self {
            font_util,
            cv_util,
            merge_util,
            bg_factory,
            editor: Arc::new(RwLock::new(editor)),
            swash_cache: Arc::new(RwLock::new(swash_cache)),
            chinese_ch_dict,
            chinese_ch_weights,
            config,
        }
    }

    fn gen_img(
        &self,
        foreground_color: [u8; 3],
        background_color: [u8; 3],
    ) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let mut raw_image = ImageBuffer::from_pixel(
            self.config.font_img_width as u32,
            self.config.font_img_height as u32,
            image::Rgb(background_color),
        );
        let mut right_border = 0;

        let mut font_system = self.font_util.font_system.write().unwrap();
        let editor = self.editor.write().unwrap();
        let mut swash_cache = self.swash_cache.write().unwrap();
        // Draw the buffer (for performance, instead use SwashCache directly)
        editor.draw(
            &mut font_system,
            &mut swash_cache,
            cosmic_text::Color::rgb(
                foreground_color[0],
                foreground_color[1],
                foreground_color[2],
            ),
            |x, y, _, _, color| {
                if x < 0
                    || x >= self.config.font_img_width as i32
                    || y < 0
                    || y >= self.config.font_img_height as i32
                    || (x == 0 && y == 0)
                {
                    return;
                }
                if x > right_border {
                    right_border = x
                }

                let (r, g, b, a) = (
                    color.r() as u32,
                    color.g() as u32,
                    color.b() as u32,
                    color.a() as u32,
                );
                let (raw_image_r, raw_image_g, raw_image_b) = unsafe {
                    let tmp = raw_image.unsafe_get_pixel(x as u32, y as u32).0;
                    (tmp[0] as u32, tmp[1] as u32, tmp[2] as u32)
                };
                let red = r * a / 255 + raw_image_r * (255 - a) / 255;
                let green = g * a / 255 + raw_image_g * (255 - a) / 255;
                let blue = b * a / 255 + raw_image_b * (255 - a) / 255;
                let rgb = image::Rgb([red as u8, green as u8, blue as u8]);

                unsafe {
                    raw_image.unsafe_put_pixel(x as u32, y as u32, rgb);
                }
            },
        );
        std::mem::drop(font_system);
        std::mem::drop(editor);
        std::mem::drop(swash_cache);

        raw_image
            .sub_image(
                0,
                0,
                (right_border + 1) as u32,
                self.config.font_img_height as u32,
            )
            .to_image()
    }

    // CJK Unified Ideographs
    pub fn gen_image_from_cjk(
        &self,
        text: &str,
        foreground_color: [u8; 3],
        background_color: [u8; 3],
    ) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let chinese_text_with_font_list = wrap_text_with_font_list(text, &self.chinese_ch_dict);
        let mut editor = self.editor.write().unwrap();
        let mut font_system = self.font_util.font_system.write().unwrap();

        editor.lines.clear();

        let attrs = Attrs::new()
            .family(Family::Name("Gandhari Unicode"))
            .style(Style::Normal)
            .weight(Weight::NORMAL)
            .stretch(Stretch::Normal);

        let mut line_text = String::new();
        let mut attrs_list = AttrsList::new(attrs);
        for (ch, font_list) in chinese_text_with_font_list {
            let start = line_text.len();
            line_text.push_str(ch);
            let end = line_text.len();
            if let Some(attrs_vec) = font_list {
                if attrs_vec.len() > 0 {
                    let attrs = attrs_vec
                        .choose(&mut rand::thread_rng())
                        .unwrap()
                        .as_attrs();
                    attrs_list.add_span(start..end, attrs);
                }
            }
        }
        editor
            .lines
            .push(BufferLine::new(&line_text, attrs_list, Shaping::Advanced));
        editor.shape_until_scroll(&mut font_system);
        std::mem::drop(font_system);
        std::mem::drop(editor);
        let img = self.gen_img(foreground_color, background_color);

        img
    }

    pub fn get_chinese_ch_dict(&self) -> &IndexMap<std::string::String, Vec<AttrsOwned>> {
        &self.chinese_ch_dict
    }
    pub fn get_chinese_ch_weights(&self) -> &WeightedAliasIndex<f64> {
        &self.chinese_ch_weights
    }

    pub fn apply_effect(&self, font_img: GrayImage) -> GrayImage {
        let font_img = self.cv_util.apply_effect(font_img);
        let bg_img = self.bg_factory.random();
        let merge = self.merge_util.poisson_edit(&font_img, bg_img);

        merge
    }
}

#[cfg(test)]
mod test {
    use super::{Generator, GeneratorConfig};

    #[test]
    fn test_chinese_image_gen() {
        let config = GeneratorConfig::default();
        let generator = Generator::new(config);
        let img = generator.gen_image_from_cjk("今天天氣眞好", [255, 105, 105], [166, 208, 221]);
        img.save("test.png").unwrap();
    }

    #[test]
    fn test_effect() {
        let config = GeneratorConfig::default();
        let generator = Generator::new(config);
        let img = generator.gen_image_from_cjk("今天天氣眞好", [255, 255, 255], [0, 0, 0]);
        let gray = image::imageops::grayscale(&img);

        let res = generator.apply_effect(gray);
        res.save("./test-img/generator.png").unwrap();
    }

    #[test]
    fn test_config() {
        let res = GeneratorConfig::from_yaml("./synth_text/config.yaml");
        println!("{:?}", res);
        println!();
        println!("{:?}", GeneratorConfig::default());
    }
}
