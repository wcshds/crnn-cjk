use std::{
    fs,
    sync::{Arc, RwLock},
};

use cosmic_text::{
    Attrs, AttrsList, AttrsOwned, Buffer, BufferLine, Family, FontSystem, Metrics, Shaping,
    Stretch, Style, SwashCache, Weight,
};
use image::{GenericImage, GenericImageView, ImageBuffer};
use indexmap::IndexMap;
use rand::seq::SliceRandom;
use rand_distr::WeightedAliasIndex;

use super::{corpus::wrap_text_with_font_list, font_util::FontUtil, init::init_ch_dict_and_weight};

#[derive(Copy, Clone)]
pub struct GeneratorConfig {
    pub font_size: usize,
    pub line_height: usize,
    pub image_width: usize,
    pub image_height: usize,
}

/// Generate images of Chinese characters, including rare Chinese characters
pub struct Generator {
    font_util: FontUtil,
    editor: Arc<RwLock<Buffer>>,
    swash_cache: Arc<RwLock<SwashCache>>,
    chinese_ch_dict: IndexMap<String, Vec<AttrsOwned>>,
    chinese_ch_weights: WeightedAliasIndex<f64>,
    config: GeneratorConfig,
}

impl Clone for Generator {
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
            self.config.image_width as f32,
            self.config.image_height as f32,
        );

        let font_util = FontUtil::new(font_system);
        Self {
            font_util,
            editor: Arc::new(RwLock::new(editor)),
            swash_cache: Arc::new(RwLock::new(swash_cache)),
            chinese_ch_dict: self.chinese_ch_dict.clone(),
            chinese_ch_weights: self.chinese_ch_weights.clone(),
            config: self.config.clone(),
        }
    }
}

// todo: image enhancement
impl Generator {
    pub fn new(font_dir: &str, chinese_ch_file: &str, config: GeneratorConfig) -> Self {
        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir(font_dir);

        // create one per application
        let swash_cache = SwashCache::new();

        let mut editor = Buffer::new(
            &mut font_system,
            Metrics::new(config.font_size as f32, config.line_height as f32),
        );
        editor.set_size(
            &mut font_system,
            config.image_width as f32,
            config.image_height as f32,
        );

        let font_util = FontUtil::new(font_system);

        let chinese_character_file_data = fs::read_to_string(chinese_ch_file).unwrap();
        let full_font_list = font_util.get_full_font_list();
        let (chinese_ch_dict, chinese_ch_weights) =
            init_ch_dict_and_weight(&font_util, &full_font_list, &chinese_character_file_data);

        Self {
            font_util,
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
            self.config.image_width as u32,
            self.config.image_height as u32,
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
                    || x >= self.config.image_width as i32
                    || y < 0
                    || y >= self.config.image_height as i32
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
                self.config.image_height as u32,
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
        // img.save(format!(
        //     "./imgs/{}.png",
        //     std::time::UNIX_EPOCH.elapsed().unwrap().as_secs_f64()
        // ))
        // .unwrap();

        img
    }

    pub fn get_chinese_ch_dict(&self) -> &IndexMap<std::string::String, Vec<AttrsOwned>> {
        &self.chinese_ch_dict
    }
    pub fn get_chinese_ch_weights(&self) -> &WeightedAliasIndex<f64> {
        &self.chinese_ch_weights
    }
}

#[cfg(test)]
mod test {
    use super::{Generator, GeneratorConfig};

    #[test]
    fn test_chinese_image_gen() {
        let font_dir = "./font";
        let chinese_ch_file = "./ch.txt";
        let config = GeneratorConfig {
            font_size: 50,
            line_height: 64,
            image_width: 2000,
            image_height: 64,
        };
        let generator = Generator::new(font_dir, chinese_ch_file, config);
        let img = generator.gen_image_from_cjk("今天天氣眞好", [255, 105, 105], [166, 208, 221]);
        img.save("test.png").unwrap();
    }
}
