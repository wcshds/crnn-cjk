use std::{
    fs,
    sync::{Arc, RwLock},
};

use cosmic_text::{Attrs, AttrsOwned, Family, FontSystem};
use once_cell::sync::Lazy;
use rand_distr::WeightedAliasIndex;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct FontUtil {
    pub(crate) font_system: Arc<RwLock<FontSystem>>,
}

impl FontUtil {
    pub fn new(font_system: FontSystem) -> FontUtil {
        FontUtil {
            font_system: Arc::new(RwLock::new(font_system)),
        }
    }

    pub fn get_full_font_list(&self) -> Vec<AttrsOwned> {
        let mut res = vec![];
        let font_system = self.font_system.write().unwrap();
        for face in font_system.db().faces() {
            let font_name = &face.families.iter().next().unwrap().0;
            let font_style = face.style;
            let font_weight = face.weight;
            let font_stretch = face.stretch;

            let attrs = Attrs::new()
                .family(Family::Name(&font_name))
                .style(font_style)
                .weight(font_weight)
                .stretch(font_stretch);
            res.push(AttrsOwned::new(attrs))
        }

        res
    }

    pub fn is_font_contain_ch(&self, font_attrs: Attrs, character: char) -> bool {
        let query = cosmic_text::fontdb::Query {
            families: &[font_attrs.family],
            weight: font_attrs.weight,
            stretch: font_attrs.stretch,
            style: font_attrs.style,
        };
        let mut font_system = self.font_system.write().unwrap();

        let db = font_system.db();
        let id = db.query(&query).unwrap();
        let font = font_system.get_font(id).unwrap();
        let codepoint = character as u32;

        let rustybuzz_face = font.rustybuzz();
        let cmap = rustybuzz_face.tables().cmap.unwrap();
        for subtable in cmap.subtables.into_iter() {
            let glyph_id = match subtable.glyph_index(codepoint) {
                Some(content) => content,
                None => continue,
            };

            return matches!(rustybuzz_face.glyph_bounding_box(glyph_id), Some(_));
        }

        return false;
    }

    // pub fn map_chinese_corpus_with_attrs<'a, S1, S3, V>(
    //     &mut self,
    //     ch_list_with_font_name_list: &'a Vec<(S1, Option<&Vec<AttrsOwned>>)>,
    //     main_font_list: &'a V,
    // ) -> Vec<(&'a S1, Attrs<'a>)>
    // where
    //     S1: AsRef<str> + Sized,
    //     S3: AsRef<str> + 'a,
    //     V: AsRef<[S3]>,
    // {
    //     let main_font = main_font_list
    //         .as_ref()
    //         .choose(&mut rand::thread_rng())
    //         .unwrap();

    //     let mut res = vec![];

    //     for (text, font_name_list) in ch_list_with_font_name_list {
    //         if let Some(content) = font_name_list {
    //             if content.len() != 0 {
    //                 res.push((text, *content.choose(&mut rand::thread_rng()).unwrap()));
    //             } else {
    //                 res.push((text, self.font_name_to_attrs(main_font)));
    //             }
    //         } else {
    //             res.push((text, self.font_name_to_attrs(main_font)));
    //         }
    //     }

    //     res
    // }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FontConfig {
    font_list: Vec<String>,
    weight: f64,
}

const FONT_CONFIG: Lazy<Vec<FontConfig>> = Lazy::new(|| {
    let data = fs::read_to_string("./config.json").unwrap();
    let font_weight: Vec<FontConfig> = serde_json::from_str(&data).unwrap();

    font_weight
});

pub static TOTAL_FONT_NAME_LIST: Lazy<Vec<String>> = Lazy::new(|| {
    FONT_CONFIG
        .iter()
        .map(|each| &each.font_list)
        .flatten()
        .map(|each| each.to_string())
        .collect()
});

pub static TOTAL_FONT_WEIGHT_DIST: Lazy<WeightedAliasIndex<f64>> = Lazy::new(|| {
    let weight = FONT_CONFIG
        .iter()
        .flat_map(|font_config| {
            std::iter::repeat(font_config.weight).take(font_config.font_list.len())
        })
        .collect();

    WeightedAliasIndex::new(weight).unwrap()
});

#[cfg(test)]
mod test {
    // use crate::img_gen::{
    //     corpus::get_random_chinese_text_with_font_list, init::init_ch_dict_and_weight,
    // };

    // use super::*;

    // #[test]
    // fn test_corpus_with_attrs_chinese() {
    //     let mut font_system = FontSystem::new();
    //     let db = font_system.db_mut();
    //     db.load_fonts_dir("./font");
    //     let mut fu = FontUtil::new(font_system);
    //     let full_font_list = fu.get_full_font_list();
    //     let character_file_data = fs::read_to_string("./ch.txt").unwrap();
    //     let (ch_list, ch_list_weights) =
    //         init_ch_dict_and_weight(&mut fu, &full_font_list, &character_file_data);
    //     // 加載 symbol 文件
    //     let symbol = fs::read_to_string("symbol")
    //         .unwrap()
    //         .trim()
    //         .split("\n")
    //         .map(String::from)
    //         .collect();

    //     let ch_list_with_font_name_list = get_random_chinese_text_with_font_list(
    //         &ch_list,
    //         &ch_list_weights,
    //         Some(&symbol),
    //         50..=60,
    //     );
    //     // let corpus_info = CorpusInfo::new("這是一……個——測 (試");
    //     let main_font_list = vec!["SimSun"];

    //     let a = fu.map_chinese_corpus_with_attrs(
    //         // &full_font_list,
    //         &ch_list_with_font_name_list,
    //         &main_font_list,
    //     );

    //     println!("{a:#?}")
    // }
}
