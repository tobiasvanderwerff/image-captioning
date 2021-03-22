# TODO: this file is still a work in progress (WIP).

import yaml

from trainer import Trainer, TrainerConfig

# load hyperparameters from a configuration file
with open('hyperparams.yaml', 'r') as f:
    config_data = yaml.load_all(f, Loader=yaml.Loader)

    for experiment in config_data: 
        save_folder = experiment['save_folder']
        params = experiment['parameters']

        """
        config = TrainerConfig(**params)
        trainer = Trainer(config, ...)
        trainer.train()
        """
