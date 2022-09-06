import argparse
import collections

from config.config_parser import ConfigParser
from data_loader.data_loader import Kadid10kDataLoader
from model.model_main import Model
from trainer.trainer import Trainer
import model.metric as module_metric

def main(config):
    logger = config.get_logger('train', verbosity=2)

    # setup data_loader instances
    dataloader_args = dict(config['kadid_dataloader']['args'])
    dataloader = Kadid10kDataLoader(is_train=True, **dataloader_args)
    valid_dataloader = dataloader.split_validation()

    # build model architecture, then print to console
    model_args = dict(config['model_main']['args'])
    model = Model(**model_args)
    # logger.info(model)

    # setup metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # setup trainer
    trainer = Trainer(model=model,
                      metrics=metrics,
                      dataloader=dataloader,
                      config=config,
                      valid_dataloader=valid_dataloader)

    # train the model
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Convolved Quality Transformer')
    args.add_argument('-c', '--config', default='./config/config.json', type=str,
                      help='config file path (default: config.json)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPU to enable (efault: None)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]

    config = ConfigParser(args, options=options, timestamp=True, train=True)
    main(config)