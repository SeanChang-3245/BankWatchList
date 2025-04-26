import yaml
from dataclasses import dataclass

@dataclass
class SplitConfig:
    transaction: str
    accountInfo: str
    idInfo: str
    watchList: str

@dataclass
class DatasetConfig:
    train: SplitConfig
    test: SplitConfig
    validateSplit: float

@dataclass
class ParameterConfig:
    batchSize: int
    learningRate: float

@dataclass
class ModelConfig:
    textEmbeddingModel: str

@dataclass
class Config:
    dataset: DatasetConfig
    parameter: ParameterConfig
    model: ModelConfig

def load_config(path: str = "./config.yaml") -> Config:
    with open(path, 'r') as file:
        raw = yaml.safe_load(file)
    return Config(**raw)

