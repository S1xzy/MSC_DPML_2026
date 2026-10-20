"""Lightning wrapper for Module 3 (temporal-aware detection)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision,
                                         BinaryRecall, MulticlassF1Score)

from src.models.components.rppg.detector import TemporalAwareDetector


def _metrics():
    # Positive class = 1 = real news (paper: y = 0 fake, y = 1 real); acc/pre/rec/f1 as in Tables II-IV.
    return MetricCollection({"acc": BinaryAccuracy(), "pre": BinaryPrecision(), "rec": BinaryRecall(),
                             "f1": BinaryF1Score(), "macro_f1": MulticlassF1Score(num_classes=2, average="macro")})


class RPPGDetectorModule(LightningModule):
    def __init__(self, user_dim: int = 128, content_dim: int = 1024, dim: int = 128, heads: int = 4,
                 dropout: float = 0.2, max_steps: int = 64, lr: float = 5e-3, weight_decay: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.net = TemporalAwareDetector(user_dim, content_dim, dim, heads, dropout, max_steps)
        self.m = torch.nn.ModuleDict({k: _metrics() for k in ("tr", "va", "te")})
        self.auc = torch.nn.ModuleDict({k: BinaryAUROC() for k in ("va", "te")})
        self.loss_avg = torch.nn.ModuleDict({k: MeanMetric() for k in ("tr", "va", "te")})
        self.val_acc_best = MaxMetric()

    def forward(self, batch):
        return self.net(batch)

    def _step(self, batch, key):
        logit = self(batch)
        y = batch["y"]
        loss = F.binary_cross_entropy_with_logits(logit, y)  # Eq. 22
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).long()
        yl = y.long()
        self.loss_avg[key](loss)
        for name, metric in self.m[key].items():
            metric.update(pred if name == "macro_f1" else prob, yl)
        if key in self.auc:
            self.auc[key].update(prob, yl)
        return loss

    def _log_epoch(self, key, prefix):
        out = {f"{prefix}/{k}": v for k, v in self.m[key].compute().items()}
        out[f"{prefix}/loss"] = self.loss_avg[key].compute()
        if key in self.auc:
            out[f"{prefix}/auc"] = self.auc[key].compute()
            self.auc[key].reset()
        self.m[key].reset()
        self.loss_avg[key].reset()
        self.log_dict(out, prog_bar=prefix != "test")
        return out

    def on_train_start(self):
        # the sanity-check validation (2 batches) must not count as the best validation accuracy
        self.val_acc_best.reset()
        self.m["va"].reset(), self.auc["va"].reset(), self.loss_avg["va"].reset()

    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch, "tr")

    def on_train_epoch_end(self):
        self._log_epoch("tr", "train")

    def validation_step(self, batch: Any, batch_idx: int):
        self._step(batch, "va")

    def on_validation_epoch_end(self):
        out = self._log_epoch("va", "val")
        self.val_acc_best(out["val/acc"])
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        self._step(batch, "te")

    def on_test_epoch_end(self):
        self._log_epoch("te", "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
