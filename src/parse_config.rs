use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum CNNElement {
    Conv([usize; 2], [usize; 2], [usize; 2], [usize; 2]),
    Pooling([usize; 2], [usize; 2], [usize; 2]),
    Batchnorm(usize),
    Relu,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelYaml {
    crnn_num_classes: usize,
    crnn_rnn_hidden_size: usize,
    cnn_structure: Vec<CNNElement>,
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingYaml {
    pretrained_model_path: String,
    lexicon_path: String,
    batch_size: usize,
    num_workers: usize,
    random_seed: u64,
    learning_rate: f64,
    generator_config_path: String,
    save_interval: usize,
    save_dir: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "UPPERCASE")]
struct CrnnTrainingConfigYaml {
    model: ModelYaml,
    training: TrainingYaml,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CrnnTrainingConfig {
    pub crnn_num_classes: usize,
    pub crnn_rnn_hidden_size: usize,
    pub cnn_structure: Vec<CNNElement>,
    pub pretrained_model_path: String,
    pub lexicon_path: String,
    pub batch_size: usize,
    pub num_workers: usize,
    pub random_seed: u64,
    pub learning_rate: f64,
    pub generator_config_path: String,
    pub save_interval: usize,
    pub save_dir: String,
}

impl CrnnTrainingConfig {
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Self {
        let path = fs::read_to_string(path).expect("training config does not exist");
        let yaml: CrnnTrainingConfigYaml =
            serde_yaml::from_str(&path).expect("fail to read training config");

        Self {
            crnn_num_classes: yaml.model.crnn_num_classes,
            crnn_rnn_hidden_size: yaml.model.crnn_rnn_hidden_size,
            cnn_structure: yaml.model.cnn_structure,
            pretrained_model_path: yaml.training.pretrained_model_path,
            lexicon_path: yaml.training.lexicon_path,
            batch_size: yaml.training.batch_size,
            num_workers: yaml.training.num_workers,
            random_seed: yaml.training.random_seed,
            learning_rate: yaml.training.learning_rate,
            generator_config_path: yaml.training.generator_config_path,
            save_interval: yaml.training.save_interval,
            save_dir: yaml.training.save_dir,
        }
    }
}
