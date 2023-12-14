mod font_util;
mod corpus;
mod init;
pub mod generator;


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
