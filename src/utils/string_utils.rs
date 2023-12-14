use core::str::from_utf8_unchecked;

pub struct CharInStrRefIter<'a> {
    data_bytes: &'a [u8],
    start: usize,
}

impl<'a> Iterator for CharInStrRefIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes_length = self.data_bytes.len();

        while self.start < bytes_length {
            let ch_byte = self.data_bytes[self.start];
            if !utf8_width::is_width_0(ch_byte) {
                let ch_len = unsafe { utf8_width::get_width_assume_valid(ch_byte) };
                let ch_in_str = unsafe {
                    from_utf8_unchecked(&self.data_bytes[self.start..self.start + ch_len])
                };
                self.start += ch_len;

                return Some(ch_in_str);
            }

            self.start += 1;
        }

        return None;
    }
}

pub trait StringUtil {
    fn strs(&self) -> CharInStrRefIter;
}

impl StringUtil for str {
    fn strs(&self) -> CharInStrRefIter {
        CharInStrRefIter {
            data_bytes: self.as_bytes(),
            start: 0,
        }
    }
}
