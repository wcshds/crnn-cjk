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
use serde::{Deserialize, Serialize};

use super::{
    corpus::wrap_text_with_font_list,
    cv_util::CvUtil,
    effect_helper::math::Random,
    font_util::FontUtil,
    init::init_ch_dict_and_weight,
    merge_util::{BgFactory, MergeUtil},
};

#[derive(Copy, Clone, Debug)]
pub struct GeneratorConfig<P: AsRef<Path> + Clone> {
    // 1. font_util
    pub font_dir: P,
    pub chinese_ch_file_path: P,
    pub font_size: usize,
    pub line_height: usize,
    pub font_img_height: usize,
    pub font_img_width: usize,
    // 2. cv_util
    // draw box
    pub box_prob: f64,
    // perspective transform
    pub perspective_prob: f64,
    pub perspective_x: Random,
    pub perspective_y: Random,
    pub perspective_z: Random,
    // gaussian blur
    pub blur_prob: f64,
    pub blur_sigma: Random,
    // filter: emboss/sharp
    pub filter_prob: f64,
    pub emboss_prob: f64,
    pub sharp_prob: f64,
    // 3. merge_util
    pub bg_dir: P,
    pub bg_height: usize,
    pub bg_width: usize,
    pub height_diff: Random,
    pub bg_alpha: Random,
    pub bg_beta: Random,
    pub font_alpha: Random,
    pub reverse_prob: f64,
}

impl Default for GeneratorConfig<String> {
    fn default() -> Self {
        GeneratorConfig {
            font_dir: "./font".to_string(),
            chinese_ch_file_path: "./ch.txt".to_string(),
            font_size: 50,
            line_height: 64,
            font_img_width: 2000,
            font_img_height: 64,
            box_prob: 0.1,
            perspective_prob: 0.2,
            perspective_x: Random::new_gaussian(-15.0, 15.0),
            perspective_y: Random::new_gaussian(-15.0, 15.0),
            perspective_z: Random::new_gaussian(-3.0, 3.0),
            blur_prob: 0.1,
            blur_sigma: Random::new_uniform(0.0, 1.5),
            filter_prob: 0.01,
            emboss_prob: 0.4,
            sharp_prob: 0.6,
            bg_dir: "./synth_text/background".to_string(),
            bg_height: 64,
            bg_width: 1000,
            height_diff: Random::new_uniform(2.0, 10.0),
            bg_alpha: Random::new_gaussian(0.5, 1.5),
            bg_beta: Random::new_gaussian(-50.0, 50.0),
            font_alpha: Random::new_uniform(0.2, 1.0),
            reverse_prob: 0.5,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct FontYaml {
    pub font_dir: String,
    pub chinese_ch_file_path: String,
    font_size: usize,
    line_height: usize,
    font_img_height: usize,
    font_img_width: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct RandomYaml(f64, f64, String);

impl RandomYaml {
    fn to_random(&self) -> Random {
        if self.2 == "g" {
            Random::new_gaussian(self.0, self.1)
        } else if self.2 == "u" {
            Random::new_uniform(self.0, self.1)
        } else {
            panic!("distribution parameter in config file should be `g` or `u`");
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct CvYaml {
    box_prob: f64,
    perspective_prob: f64,
    perspective_x: RandomYaml,
    perspective_y: RandomYaml,
    perspective_z: RandomYaml,
    blur_prob: f64,
    blur_sigma: RandomYaml,
    filter_prob: f64,
    emboss_prob: f64,
    sharp_prob: f64,
}

#[derive(Serialize, Deserialize, Debug)]
struct MergeYaml {
    pub bg_dir: String,
    pub bg_height: usize,
    pub bg_width: usize,
    // make it into Random(2.0, height_diff) later
    pub height_diff: f64,
    pub bg_alpha: RandomYaml,
    pub bg_beta: RandomYaml,
    pub font_alpha: RandomYaml,
    pub reverse_prob: f64,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "UPPERCASE")]
struct GeneratorConfigYaml {
    font: FontYaml,
    cv: CvYaml,
    merge: MergeYaml,
}

impl<P1: AsRef<Path> + Clone> GeneratorConfig<P1> {
    pub fn from_yaml(path: P1) -> GeneratorConfig<String> {
        let yaml_str = fs::read_to_string(path).expect("the config file does not exist");
        let yaml: GeneratorConfigYaml = serde_yaml::from_str(&yaml_str).expect("fail to parse config file");

        GeneratorConfig {
            font_dir: yaml.font.font_dir,
            chinese_ch_file_path: yaml.font.chinese_ch_file_path,
            font_size: yaml.font.font_size,
            line_height: yaml.font.line_height,
            font_img_width: yaml.font.font_img_width,
            font_img_height: yaml.font.font_img_height,
            box_prob: yaml.cv.box_prob,
            perspective_prob: yaml.cv.perspective_prob,
            perspective_x: yaml.cv.perspective_x.to_random(),
            perspective_y: yaml.cv.perspective_y.to_random(),
            perspective_z: yaml.cv.perspective_z.to_random(),
            blur_prob: yaml.cv.blur_prob,
            blur_sigma: yaml.cv.blur_sigma.to_random(),
            filter_prob: yaml.cv.filter_prob,
            emboss_prob: yaml.cv.emboss_prob,
            sharp_prob: yaml.cv.sharp_prob,
            bg_dir: yaml.merge.bg_dir,
            bg_height: yaml.merge.bg_height,
            bg_width: yaml.merge.bg_width,
            height_diff: Random::new_uniform(2.0, yaml.merge.height_diff),
            bg_alpha: yaml.merge.bg_alpha.to_random(),
            bg_beta: yaml.merge.bg_beta.to_random(),
            font_alpha: yaml.merge.font_alpha.to_random(),
            reverse_prob: yaml.merge.reverse_prob,
        }
    }
}

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
