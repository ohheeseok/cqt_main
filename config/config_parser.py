import os
import json
import logging

from collections import OrderedDict
from pathlib import Path, PurePosixPath
from functools import reduce
from operator import getitem
from datetime import datetime

from logger.logger import setup_logging

class ConfigParser:
    def __init__(self, args, options='', timestamp=True, train=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()
        display_config(args)

        # parse device option
        if args.device:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICE'] = args.device

        # parse whether resume or not
        if args.resume:
            self.resume = args.resume
            self.cfg_fname = Path(self.resume).parent / 'config.json'
        else:
            assert args.config is not None, '[!] Configuration file need to be specified!'
            self.resume = None
            self.cfg_fname = Path(args.config)

        # parse config.json
        config = read_json(self.cfg_fname)
        self.config = _update_config(config, options, args)

        if train:
            save_dir = Path(self.config['trainer']['save_dir'])
            timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
            exper_name = self.config['name']
            self._ckpt_dir = save_dir / 'ckpt' / exper_name / timestamp
            self._log_dir = save_dir / 'log' / exper_name / timestamp
            self._out_dir = save_dir / 'out' / exper_name / timestamp
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # save updated config file to the checkpoint dir
            write_json(self.config, self.ckpt_dir / 'config.json')
            # configuration logging module
            setup_logging(self.log_dir)

        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(
            verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    # setting read-only attributes
    @property
    def ckpt_dir(self):
        return self._ckpt_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def out_dir(self):
        return self._out_dir
# eoc


def display_config(args):
    print('#####################################################################')
    print('## Pytorch - Convolved Quality Transformer  (ohhs@hansung.ac.kr) ####')
    print('#####################################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" %(str(arg), str(getattr(args, arg))))
    print('')


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested opject in tree by sequence of keys."""
    return reduce(getitem, keys, tree)