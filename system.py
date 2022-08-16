import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.nn.modules.loss import _Loss
import pytorch_lightning as pl
from argparse import Namespace
from typing import Callable, Optional
import soundfile as sf

from asteroid.utils import flatten_dict
from asteroid.utils.deprecation_utils import DeprecationMixin

from model import com_sisdr_loss1

class System(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        self.hparams = Namespace(**self.config_to_hparams(config))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, single_speaker = batch

        est_targets, pha_src = self(inputs)
        loss, loss_dict = self.loss_func(est_targets, pha_src, targets, single_speaker)

        return loss, loss_dict
      
    def training_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = loss_dict
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        loss, loss_dict = self.common_step(batch, batch_nb, train=False)
        tensorboard_logs = loss_dict
        return {"val_loss": loss, "log": tensorboard_logs}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        return {"val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def optimizer_step(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            for sched in self.scheduler:
                if isinstance(sched, dict) and sched["interval"] == "batch":
                    sched["scheduler"].step()  # call step on each batch scheduler
            super().optimizer_step(*args, **kwargs)
    def configure_optimizers(self):
        """ Required by pytorch-lightning. """

        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            epoch_schedulers = []
            for sched in self.scheduler:
                if not isinstance(sched, dict):
                    epoch_schedulers.append(sched)
                else:
                    assert sched["interval"] in [
                        "batch",
                        "epoch",
                    ], "Scheduler interval should be either batch or epoch"
                    if sched["interval"] == "epoch":
                        epoch_schedulers.append(sched)
            return [self.optimizer], epoch_schedulers
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
      
    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic
      
