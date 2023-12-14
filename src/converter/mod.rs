use indexmap::IndexMap;

use crate::utils::string_utils::StringUtil;

// Converter is used to convert text to label.
#[derive(Debug, Clone)]
pub struct Converter {
    pub lexicon: IndexMap<String, i32>,
}

impl<'a> Converter {
    pub fn new(raw: &'a str) -> Self {
        let bytes_length = raw.trim().as_bytes().len();

        let mut lexicon_dict = IndexMap::with_capacity(bytes_length / 3);

        for (ch, idx) in raw.strs().zip(1i32..) {
            lexicon_dict.insert(ch.to_string(), idx);
        }

        Converter {
            lexicon: lexicon_dict,
        }
    }

    pub fn encode_single<S: AsRef<str>>(&self, text: S) -> (Vec<i32>, i32) {
        let raw = text.as_ref();

        let encoded: Vec<_> = raw.strs().map(|each| self.lexicon[each]).collect();
        let length = encoded.len() as i32;
        (encoded, length)
    }

    pub fn encode_multi<S: AsRef<str>>(&self, texts: &[S]) -> (Vec<i32>, Vec<i32>) {
        let mut length = vec![];
        let mut encoded = vec![];

        for text in texts {
            let mut count = 0;
            let tmp = text.as_ref().strs().map(|each| {
                count += 1;
                self.lexicon[each]
            });
            encoded.extend(tmp);
            length.push(count)
        }

        (encoded, length)
    }

    pub fn decode(&self, encoded_texts: &[i32], length: &[i32], raw: bool) -> Vec<String> {
        let mut start = 0usize;
        let mut res = vec![];
        for &len in length {
            let len = len as usize;
            let encoded_text = &encoded_texts[start..start + len];
            let mut decoded_text = String::with_capacity(len);
            for (&ch_encoded, idx) in encoded_text.iter().zip(0..) {
                if raw {
                    if ch_encoded == 0 {
                        decoded_text.push('-');
                    } else {
                        let lex_pos = ch_encoded - 1;
                        let (decoded_ch, _) = self.lexicon.get_index(lex_pos as usize).unwrap();
                        decoded_text.push_str(decoded_ch);
                    }
                } else {
                    if ch_encoded != 0 && (!(idx > 0 && encoded_text[idx - 1] == ch_encoded)) {
                        let lex_pos = ch_encoded - 1;
                        let (decoded_ch, _) = self.lexicon.get_index(lex_pos as usize).unwrap();
                        decoded_text.push_str(decoded_ch);
                    }
                }
            }
            res.push(decoded_text);
            start += len as usize;
        }

        res
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use super::*;

    #[test]
    fn test_converter() {
        let raw = fs::read_to_string("./lexicon.txt").unwrap();
        let converter = Converter::new(&raw);
        let res = converter.encode_multi(&["0 ignored; 0 measured;", "principal axis"]);
        println!("{:#?}", res)
    }
}
