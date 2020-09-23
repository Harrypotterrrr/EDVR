import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

from data_loader.data_loader import DatasetGenerator
from models.EDVR import EDVR
from trainers.train import Trainer
from utils.config import process_config

# config_path = "configs/config.json"


def main():
    config = process_config()
    train_data = DatasetGenerator(config)()
    model = EDVR(config)

    trainer = Trainer(model, train_data, config)
    trainer.train()


if __name__ == '__main__':
    main()
