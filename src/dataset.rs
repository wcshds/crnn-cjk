use std::{
    collections::HashMap,
    ops::Range,
    sync::{Arc, Mutex, RwLock},
    thread::{self, ThreadId},
};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, Int, Tensor},
};
use rand::Rng;
use rand_distr::Distribution;
use serde::{Deserialize, Serialize};

use crate::{
    converter::Converter,
    img_gen::generator::{Generator, GeneratorConfig},
    utils::tensor_ext::pad,
};

pub struct TextImgBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TextImgBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TextImgBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
    pub target_lengths: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<TextImgItem, TextImgBatch<B>> for TextImgBatcher<B> {
    fn batch(&self, items: Vec<TextImgItem>) -> TextImgBatch<B> {
        let images = items
            .iter()
            .map(|item| {
                let data = Data::<u8, 1>::from(item.image_raw.as_slice());
                let tensor = Tensor::<B, 1, Int>::from_data(data.convert(), &self.device).float();
                let tensor = tensor.reshape([1, 1, item.image_height, item.image_width]);
                let pad_left = (1000 - item.image_width) / 2;
                let pad_right = 1000 - item.image_width - pad_left;
                let tensor = pad(tensor, [(0, 0), (0, 0), (0, 0), (pad_left, pad_right)], 0);
                // range: [-1.0, 1.0]
                let tensor = ((tensor / 255) - 0.5) / 0.5;
                tensor
            })
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                let data = Data::<i32, 1>::from(item.target.as_slice());
                let tensor = Tensor::<B, 1, Int>::from_data(data.convert(), &self.device);
                tensor
            })
            .collect();

        let target_lengths: Vec<_> = items.iter().map(|item| item.target_len).collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        let target_lengths = Tensor::<B, 1, Int>::from_data(
            Data::<i32, 1>::from(target_lengths.as_slice()).convert(),
            &self.device,
        );

        TextImgBatch {
            images,
            targets,
            target_lengths,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TextImgItem {
    pub image_raw: Vec<u8>,
    pub image_height: usize,
    pub image_width: usize,
    pub target: Vec<i32>,
    pub target_len: i32,
}

pub struct TextImgDataset {
    generator: Generator,
    converter: Converter,
    num_char_range: Range<usize>,
    dataset_size: usize,
    thread_ids: Arc<Mutex<Vec<ThreadId>>>,
    generators: Arc<RwLock<HashMap<ThreadId, Generator>>>,
}

impl TextImgDataset {
    pub fn new(
        num_char_range: Range<usize>,
        dataset_size: usize,
        font_dir: &str,
        chinese_ch_file: &str,
        config: GeneratorConfig,
        converter: Converter,
    ) -> Self {
        let generator = Generator::new(font_dir, chinese_ch_file, config);

        Self {
            generator,
            converter,
            num_char_range,
            dataset_size,
            thread_ids: Arc::new(Mutex::new(Vec::with_capacity(8))),
            generators: Arc::new(RwLock::new(HashMap::with_capacity(8))),
        }
    }
}

impl Dataset<TextImgItem> for TextImgDataset {
    fn get(&self, index: usize) -> Option<TextImgItem> {
        // make sure each thread has a unique Generator
        let current_thread_id = thread::current().id();
        {
            let mut thread_ids = self.thread_ids.lock().unwrap();
            if !thread_ids.contains(&current_thread_id) {
                thread_ids.push(current_thread_id);
                self.generators
                    .write()
                    .unwrap()
                    .insert(current_thread_id, self.generator.clone());
            }
        }
        let generator = self.generators.read().unwrap();
        let generator = generator.get(&current_thread_id).unwrap();

        if index >= self.dataset_size {
            return None;
        }

        let chinese_ch_dict = generator.get_chinese_ch_dict();
        let chinese_ch_weights = generator.get_chinese_ch_weights();

        let mut rng = rand::thread_rng();
        let num_char = rng.gen_range(self.num_char_range.clone());
        let text: String = (0..num_char)
            .map(|_| {
                chinese_ch_dict
                    .get_index(chinese_ch_weights.sample(&mut rand::thread_rng()))
                    .unwrap()
                    .0
                    .as_str()
            })
            .collect();
        let img = generator.gen_image_from_cjk(&text, [255, 255, 255], [0, 0, 0]);
        let gray = image::imageops::grayscale(&img);
        let image_height = gray.height() as usize;
        let image_width = gray.width() as usize;
        let image_raw = gray.into_vec();
        let (target, target_len) = self.converter.encode_single(&text);

        Some(TextImgItem {
            image_raw,
            image_height,
            image_width,
            target,
            target_len,
        })
    }

    fn len(&self) -> usize {
        self.dataset_size
    }
}
