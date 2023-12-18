mod corpus;
pub(super) mod cv_helper;
pub mod cv_util;
mod font_util;
pub mod generator;
mod init;

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
