mod corpus;
pub mod cv_util;
pub(super) mod effect_helper;
mod font_util;
pub mod generator;
mod init;
pub mod merge_util;
pub mod parse_config;

#[cfg(test)]
mod test {
    use cosmic_text::FontSystem;

    #[test]
    fn test_cosmic() {
        let mut font_system = FontSystem::new();
        let db = font_system.db_mut();
        db.load_fonts_dir("./font");
    }
}
