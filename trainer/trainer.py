import torch
from torchvision.utils import make_grid
import numpy as np
from logger.visualization import TensorboardWriter
from utils.util import inf_loop, data_to_device

class Trainer:
    def __init__(self, model, metrics, dataloader, config, valid_dataloader=None, n_batches=None):
        self.model = model
        self.metrics = metrics
        self.dataloader = dataloader
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.epochs = config['trainer']['epochs']
        self.save_period = config['trainer']['save_period']
        self.monitor = config['trainer'].get('monitor', 'off')
        self.start_epoch = 1
        self.ckpt_dir = config.ckpt_dir
        self.writer = TensorboardWriter(config.log_dir, self.logger, config['trainer']['tensorboard'])
        # epoch-based training
        if n_batches is None:
            self.n_batches = len(self.dataloader)
        # iteration-based training
        else:
            self.dataloader = inf_loop(dataloader)
            self.n_batches = n_batches
        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.train_step = (len(self.dataloader) // 5) + 1
        self.valid_step = (len(self.valid_dataloader) // 1) + 1 if self.do_validation else None

        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = self.model.to(self.device)
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        self.optimizer = torch.optim.Adam(params, **config['optimizer']['args'])

        # set learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0.001)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = config['trainer'].get('early_stop', np.inf)

        # resume training from checkpoint when the option is activated
        if config.resume is not None:
            self.resume_checkpoint(config.resume)


    def train(self):
        self.model.train()
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self.train_epoch(epoch)

            #  construct log dictionary
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'corr_train':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'corr_test':
                    log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            #  print log
            for key, value in log.items():
                self.logger.info('       {:15s}:  {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("[!] Warning: Metric '{}' is not found. "
                                        "[!] Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                if not_improved_count > self.early_stop:
                    self.logger.info("[!] Validation performance didn\'t improve for {} epochs. "
                                     "[!] Training stops.".format(self.early_stop))
                    break

            #  save checkpoint at each epoch
            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, save_best=best)

        self.model.eval()
        # self.scheduler.step()


    def train_epoch(self, epoch):
        """Training logic for an epoch"""
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (img_ref, img_dist, mos, mos_norm) in enumerate(self.dataloader):
            self.model.train()
            img_ref, img_dist, mos, mos_norm = data_to_device(self.device, img_ref, img_dist,
                                                              mos.type(torch.FloatTensor),
                                                              mos_norm.type(torch.FloatTensor))
            pred, logs = self.model.backward(img_ref, img_dist, mos, optimizer=self.optimizer)

            # writer: first excute self.writer.set_step()
            self.writer.set_step((epoch - 1) * self.n_batches + batch_idx)
            self._writer_log(**logs)
            total_metrics += self._writer_metrics(pred, mos, name='_coef')

            if batch_idx % self.train_step == 0:
                self.logger.debug('Train Epoch: {} {}'.format(epoch, self._progress(batch_idx)))

                if self.do_validation:
                    test_log = {}
                    it = (epoch - 1) * self.n_batches + batch_idx
                    test_corrs = self.test_iter(it)
                    test_log.update({'test_' + mtr.__name__: test_corrs[i] for i, mtr in enumerate(self.metrics)})
                    for key, value in test_log.items():
                        self.writer.add_scalar(key, value.item())
                    self.save_checkpoint_iter(it, save_best=False)

                if batch_idx == self.n_batches:
                    break

            self.scheduler.step()

        log = {'corr_train': (total_metrics / self.n_batches).tolist()}

        # if self.do_validation:
        #     val_log = self.test_epoch(epoch)
        #     log.update(val_log)

        return log


    def valid_epoch(self, epoch):
        """validate after training an epoch"""
        self.model.eval()
        total_val_metrics = np.zeros(len(self.metrics))

        with torch.no_grad():
            for batch_idx, (img_ref, img_dist, mos, mos_norm) in enumerate(self.valid_dataloader):
                img_ref, img_dist, mos, mos_norm = data_to_device(self.device, img_ref, img_dist,
                                                                  mos.type(torch.FloatTensor),
                                                                  mos_norm.type(torch.FloatTensor))
                pred, logs = self.model.backward(img_ref, img_dist, mos)

                # validation
                self.writer.set_step((epoch - 1) * len(self.valid_dataloader) + batch_idx, 'valid')
                self._writer_log(**logs)
                total_val_metrics += self._writer_metrics(pred, mos, name='_coef')

                if batch_idx % self.valid_step == 0:
                    save_image = {}
                    save_image['0_img_ref'] = img_ref
                    save_image['1_img_dist'] = img_dist
                    self._save_image(img_ref.size()[0], **save_image)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {'val_metrics': (total_val_metrics / len(self.valid_dataloader)).tolist()}


    def test_epoch(self):
        """validate & test after training an epoch"""
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch_idx, (img_ref, img_dist, mos, mos_norm) in enumerate(self.valid_dataloader):
                img_ref, img_dist, mos, mos_norm = data_to_device(self.device, img_ref, img_dist,
                                                                  mos.type(torch.FloatTensor),
                                                                  mos_norm.type(torch.FloatTensor))
                pred, logs = self.model.backward(img_ref, img_dist, mos)
                preds.append(pred)
                targets.append(mos)

                if batch_idx % self.valid_step == 0:
                    save_image = {}
                    save_image['0_img_ref'] = img_ref
                    save_image['1_img_dist'] = img_dist
                    self._save_image(img_ref.size()[0], **save_image)

            # calculate performances
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            total_test_metrics = self._writer_metrics(preds, targets, name='_test')
            print(total_test_metrics)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {'corr_test': total_test_metrics}

    def test_iter(self, it):
        """validate & test at assigned iteration"""
        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch_idx, (img_ref, img_dist, mos, mos_norm) in enumerate(self.valid_dataloader):
                # mos: Double to Float type casting
                img_ref, img_dist, mos, mos_norm = data_to_device(self.device, img_ref, img_dist,
                                                                  mos.type(torch.FloatTensor),
                                                                  mos_norm.type(torch.FloatTensor))
                pred, logs = self.model.backward(img_ref, img_dist, mos)
                preds.append(pred)
                targets.append(mos)

                if batch_idx % self.valid_step == 0:
                    save_image = {}
                    save_image['0_img_ref'] = img_ref
                    save_image['1_img_dist'] = img_dist
                    self._save_image(img_ref.size()[0], **save_image)

            # calculate performances
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            total_test_metrics = self._writer_metrics(preds, targets, name='_test')
            self.logger.info(f"Performance at {it}th iter, plcc:{total_test_metrics[0]}, srcc:{total_test_metrics[1]}, krcc:{total_test_metrics[2]}")

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return total_test_metrics


    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth
        """
        arch = type(self.model).__name__
        state={
            'arch': arch,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = str(self.ckpt_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("... Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.ckpt_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("... Saving current best: model_best.pth")
        self.logger.info("[*] Checkpoint saved: {} ...".format(filename))


    def save_checkpoint_iter(self, it, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth
        """
        arch = type(self.model).__name__
        state={
            'arch': arch,
            'iteration': it,
            'model': self.model.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = str(self.ckpt_dir / 'checkpoint-iteration{}.pth'.format(it))
        torch.save(state, filename)
        self.logger.info("... Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.ckpt_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("... Saving current best: model_best.pth")
        self.logger.info("[*] Checkpoint saved: {} ...".format(filename))


    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("... Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        self.model.load_state_dict(checkpoint['model'])

        self.logger.info("[*] Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.dataloader, 'n_samples'):
            current = batch_idx * self.dataloader.batch_size
            total = self.dataloader.n_samples
        else:
            current = batch_idx
            total = self.n_batches
        return base.format(current, total, 100.0 * current / total)

    def _writer_log(self, **logs):
        for key, value in logs.items():
            if key[0] == 'l':
                self.writer.add_scalar(key, value.item())

    def _writer_metrics(self, image_gt, image_pred, name=None):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(image_gt, image_pred)
            self.writer.add_scalar('{}'.format(metric.__name__+name), acc_metrics[i])
        return acc_metrics

    def _save_image(self, nrow, **log):
        for key, image in log.items():
            grid = make_grid(image.cpu(), nrow=nrow, normalize=True)
            self.writer.add_image(key, grid)


    def _prepare_device(self, n_gpu_use):
        """setup GPU device if available, move model into configured device"""
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("[!] Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                           "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids