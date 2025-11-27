import random
import pprint
import time
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import cloudpickle as pickle

import numpy as np

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb

class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = True
        config.prefix = ''
        config.project = os.environ.get('WANDB_PROJECT', 'PI-Model')
        config.output_dir = './logs'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        # Add group field to prevent locked config error
        config.group = config_dict.placeholder(str) 

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            name=self.config.experiment_id,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            # For lists or other types, define as string and parse later if needed
            # or skip definition if not supported by simple absl flags
            pass 
    return kwargs

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed) # Imported only where needed to avoid dependency

def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output[key] = val.to_dict()
        else:
            output[key] = val
    return output

def prefix_metrics(metrics, prefix):
    return {
        "{}/{}".format(prefix, key): val for key, val in metrics.items()
    }

class EarlyStopper(object):
    def __init__(self, patience: int = 5, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Annealer:
    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        self.total_steps = total_steps
        self.current_step = 0
        self.shape = shape
        self.baseline = baseline
        self.cyclical = cyclical
        self.disable = disable

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1

    def slope(self):
        if self.disable:
            return 1.0

        if self.cyclical:
            step = self.current_step % self.total_steps
        else:
            step = self.current_step

        if self.shape == 'linear':
            y = step / self.total_steps
        elif self.shape == 'cosine':
            y = (np.cos(np.pi * (step / self.total_steps - 1)) + 1) / 2
        elif self.shape == 'logistic':
            exponent = (self.total_steps / 2) - step
            y = 1 / (1 + np.exp(exponent))
        else:
            raise NotImplementedError

        return y * (1 - self.baseline) + self.baseline
