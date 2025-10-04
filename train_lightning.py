from collections import defaultdict
from dataclasses import asdict
import gc
from typing import Optional
import itertools

import matplotlib
import numpy as np

from model import PositionalEncoding

matplotlib.use("Agg")
import lightning as L
import optuna
import torch
from lightning.pytorch.utilities import grad_norm
from optuna.integration import PyTorchLightningPruningCallback
import torch.nn.functional as F
from lightning import seed_everything, Callback
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from torch import optim, nn
from torch.optim.swa_utils import get_ema_avg_fn

from dataset.lightning import DataModule
from evallib import (
    peak_pick_max_mean,
    calculate_pr,
    calculate_f_score,
    evaluate_onset_stats,
    combine_onsets,
)
from model.CRNN import CRNN, CRNN_Vogl
from model.cnn import CNN
from model.cnnA import CNNAttention
from model.cnnM2 import CNNMambaFast
from model.lightning_weight_averaging import WeightAveraging
from settings import (
    CNNSettings,
    CNNAttentionSettings,
    CNNMambaSettings,
    CRNNSettings,
    Config,
    asdict as settings_asdict,
    flatten_dict,
    ModelSettingsBase,
)


def get_model(
    model_type: str, n_classes: int, n_mels: int, settings: Optional[ModelSettingsBase]
):
    match model_type:
        case "cnn":
            settings = CNNSettings() if settings is None else settings
            return CNN(**asdict(settings), n_classes=n_classes, n_mels=n_mels)
        case "cnn_attention":
            settings = CNNAttentionSettings() if settings is None else settings
            return CNNAttention(**asdict(settings), n_classes=n_classes, n_mels=n_mels)
        case "mamba" | "mamba_fast":
            settings = CNNMambaSettings() if settings is None else settings
            return CNNMambaFast(**asdict(settings), n_classes=n_classes, n_mels=n_mels)
        case "crnn":
            settings = CRNNSettings() if settings is None else settings
            return CRNN(**asdict(settings), n_classes=n_classes, n_mels=n_mels)
        case "vogl" | "crnn_vogl":
            settings = CRNNSettings() if settings is None else settings
            return CRNN_Vogl(n_classes=n_classes, n_mels=n_mels, causal=settings.causal)
        case _:
            raise ValueError("Invalid model setting")


class LitModel(L.LightningModule):
    def __init__(self, config: dict[str, int | bool | float | str]):
        super().__init__()

        self.conf = Config.from_flat_dict(config)
        settings = self.conf.model
        n_classes, n_mels = (
            self.conf.dataset.annotation_settings.n_classes,
            self.conf.dataset.audio_settings.n_mels,
        )

        self.model = get_model(config["model_settings"], n_classes, n_mels, settings)

        self.conf.model = settings

        self.save_hyperparameters(flatten_dict(asdict(self.conf)))
        self.hparams.n_classes = n_classes

        self.strict_loading = False

        self.pos_weight = None
        self.thresholds = nn.Parameter(torch.ones(n_classes) * 0.1, requires_grad=False)

        self.preds = defaultdict(list)
        self.gts = defaultdict(list)

    def configure_optimizers(self):
        # Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/training/src/optim/param_grouping.py
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.Embedding, PositionalEncoding)
        blacklist_weight_modules += (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LazyBatchNorm1d,
            nn.LazyBatchNorm2d,
            nn.LazyBatchNorm3d,
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.LocalResponseNorm,
        )
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # In case of parameter sharing, some parameters show up here but are not in
                # param_dict.keys()
                if not p.requires_grad or fpn not in param_dict:
                    continue  # frozen weights
                elif getattr(p, "_no_weight_decay", False):
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        decay |= param_dict.keys() - no_decay
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            f"Parameters {str(inter_params)} made it into both decay/no_decay sets!"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {str(param_dict.keys() - union_params)}  were not separated into either decay/no_decay set!"
        )

        if self.hparams.weight_decay == 0.0 or not no_decay:
            param_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay | decay))],
                    "weight_decay": self.hparams.weight_decay,
                }
            ]
        else:
            # We need sorted(list()) so that the order is deterministic. Otherwise, when we resume
            # the order could change and resume will fail.
            param_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                    "weight_decay": 0.0,
                },
            ]

        opt_class = optim.RAdam if self.hparams.optimizer == "radam" else optim.Adam
        optimizer = opt_class(
            param_groups,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            decoupled_weight_decay=self.hparams.decoupled_weight_decay,
            betas=(self.hparams.beta_1, self.hparams.beta_2),
            eps=self.hparams.epsilon,
        )
        if not self.hparams.scheduler:
            return optimizer

        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='min',
        #     factor=0.2,
        #     patience=10,
        #     min_lr=1e-6
        # )
        # scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "epoch",
        #     "frequency": 1,
        #     "monitor": "val_loss",
        #     "strict": True,
        # }
        # lr_scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=2e-3,
        #     div_factor=2e-3 / self.hparams.learning_rate,
        #     final_div_factor=self.hparams.learning_rate / 8e-6,  # min_lr = 8e-6
        #     total_steps=self.trainer.estimated_stepping_batches,
        # )
        # scheduler_config = {
        #     "scheduler": lr_scheduler,
        #     "interval": "step",
        #     "frequency": 1,
        #     "strict": True,
        # }

        # warmup-stable-decay schedule
        cycles = 1  # int(np.ceil(self.hparams.epochs / 20))
        schedulers = []
        milestones = []
        cycle_length = self.trainer.estimated_stepping_batches // cycles
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.hparams.epochs

        warmup_steps = 100
        decay_length = cycle_length // 2

        for cycle in range(cycles):
            schedulers.append(
                # LinearLR keeps the LR constant after the warmup phase if end_factor = 1.0
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
            )
            schedulers.append(
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_length)
            )
            milestones.append((cycle_length - decay_length) + cycle_length * cycle)
            if cycle != cycles - 1:
                milestones.append(cycle_length * (cycle + 1))

        # lr_scheduler = optim.lr_scheduler.SequentialLR(
        #     optimizer,
        #     [
        #         # LinearLR keeps the LR constant after the warmup phase if end_factor = 1.0
        #         optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
        #                                     total_iters=self.trainer.estimated_stepping_batches // 20),
        #         optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches // 10)
        #     ],
        #     [self.trainer.estimated_stepping_batches * 9 // 10]
        # )

        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones
        )

        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer=optimizer,
        #     T_0=steps_per_epoch * 50,
        #     T_mult=1,
        #     eta_min=8e-6
        # )
        scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [scheduler_config]

    def on_train_start(self):
        summary = L.pytorch.utilities.model_summary.summarize(self, max_depth=2)
        model_size = {
            "Params/Total": summary.total_parameters,
            "Params/Trainable": summary.trainable_parameters,
        }
        for name, params in zip(summary.layer_names, summary.param_nums):
            model_size[f"Params/{name}"] = params
        self.log_dict(model_size, prog_bar=False, on_step=False, on_epoch=True)

        if not self.hparams.use_pos_weight:
            self.pos_weight = None
            return

        num_pos, num_neg = self.trainer.datamodule.sample_distribution

        total = (num_pos + num_neg)[0]
        # if dataset_settings.annotation_settings.pad_annotations:
        #     num_pos = num_pos * (1 + 2 * dataset_settings.annotation_settings.pad_value)
        p_i = num_pos / total
        weight: torch.Tensor = 1 / (-p_i * p_i.log() - (1 - p_i) * (1 - p_i).log())
        weight = weight / 4  # shift weight closer to the ones used by Zheren and Vogl
        weight[0] = 1.0  # don't weigh kick as it is easy and common
        # if (weight > 10).any():
        #     print(f"High pos_weight detected: {weight}, applying compression")
        #     # compress weights to avoid extreme values
        #     log_weights = (weight * 2).log1p()
        #     log_weights = log_weights / log_weights.max()
        #     weight = log_weights.expm1() * weight.quantile(q=0.5) / 2
        # else:
        #     weight[0] = 1.0  # don't weigh kick as it is easy and common
        # weight = None
        # outliers = weight > 10
        # weight[outliers] = weight[outliers].clamp(1, 100)
        # weight[outliers] = weight[outliers] / weight[outliers].max() * 10

        # weight = torch.ones_like(weight)
        print(f"Using pos_weight: {weight}")

        self.pos_weight = weight.unsqueeze(-1).to(self.device)  # .clamp(1, 10)

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy_with_logits(
            pred, y, reduction="none", pos_weight=self.pos_weight
        )

        loss = loss[y != -1].mean()

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def on_before_optimizer_step(self, *args, **kwargs) -> None:
        grd_norm = grad_norm(self.model, norm_type=2.0)
        total_key = [key for key in grd_norm.keys() if "total" in key][0]
        grd_norm_total = grd_norm[total_key]
        grd_norm.pop(total_key)
        self.log_dict(
            grd_norm,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )
        self.log(
            total_key,
            grd_norm_total,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        weight_norms = {
            "weight_norm/" + name: p.detach().norm(p=2)
            for name, p in self.model.named_parameters()
        }
        total_weight_norm = torch.tensor(list(weight_norms.values())).norm(p=2)
        weight_norms["weight_norm/total"] = total_weight_norm
        self.log_dict(weight_norms, prog_bar=False, on_step=False, on_epoch=True)

    def evaluate_step(self, batch, set_key):
        x, y, gts = batch
        pred = self.model(x)
        loss = F.binary_cross_entropy_with_logits(
            pred, y, reduction="none", pos_weight=self.pos_weight
        )

        loss = loss[y != -1].mean()

        self.log(
            f"{set_key}_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.test_batch_size,
        )

        filtered_pred = pred.sigmoid() * (y != -1)
        peaks = peak_pick_max_mean(
            filtered_pred.cpu().detach().float(),
            self.hparams.sample_rate,
            self.hparams.hop_size,
            self.hparams.fft_size,
            self.hparams.peak_mean_range,
            self.hparams.peak_max_range,
        )

        for song_idx in range(len(peaks)):
            for cls_idx in range(len(peaks[song_idx])):
                peaks[song_idx][cls_idx] -= self.hparams.time_shift
                # filter out predictions before the start
                peaks[song_idx][cls_idx] = peaks[song_idx][cls_idx][
                    :, peaks[song_idx][cls_idx][0, :] >= 0
                ]
        self.preds[set_key].extend(peaks)
        self.gts[set_key].extend(gts)

        return loss

    def evaluate_epoch_end(self, set_key):
        precisions, recalls, thresholds, f_sum, f_avg, best_thresholds = calculate_pr(
            peaks=self.preds[set_key],
            groundtruth=self.gts[set_key],
            onset_cooldown=self.hparams.onset_cooldown,
            detection_window=self.hparams.detect_tolerance,
            ignore_beats=self.hparams.ignore_beats,
            pr_points=self.hparams.pr_points,
        )

        if set_key == "val":
            self.thresholds[:] = best_thresholds

        f_scores = [
            calculate_f_score(precision, recall).detach().cpu()
            for precision, recall in zip(precisions, recalls)
        ]

        combined_metrics = torch.stack(
            [
                torch.stack(precisions, dim=0),
                torch.stack(recalls, dim=0),
                torch.stack(f_scores, dim=0),
                torch.stack(thresholds, dim=0),
            ],
            dim=1,
        )

        top_idx = torch.stack(f_scores, dim=0).argmax(dim=-1)

        combined_metrics = [
            metrics[:, idx] for metrics, idx in zip(combined_metrics, top_idx)
        ]

        stat_names = ["Precision", "Recall", "F-Score", "Threshold"]

        per_class_metrics = {
            f"{name}/Sum/{self.hparams.mapping.get_name(cls_idx)}": value.item()
            for cls_idx, stats in enumerate(combined_metrics)
            for name, value in zip(stat_names, stats)
        }

        if set_key != "val":
            per_class_metrics = {
                f"{set_key}/{key}": value for key, value in per_class_metrics.items()
            }

        self.log_dict(per_class_metrics, on_epoch=True, prog_bar=False)

        if set_key == "val":
            self.log("F-Score/Sum/Total", f_sum, on_epoch=True, prog_bar=True)
            self.log("F-Score/Avg", f_avg, on_epoch=True, prog_bar=True)
        else:
            self.log(
                f"{set_key}/F-Score/Sum/Total", f_sum, on_epoch=True, prog_bar=True
            )
            self.log(f"{set_key}/F-Score/Avg", f_avg, on_epoch=True)

        self.evaluate_at_thresholds(set_key, best_thresholds.tolist(), False)

        self.preds[set_key].clear()
        self.gts[set_key].clear()
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate_at_thresholds(
        self, set_key: str, thresholds: list[float], calculate_scores: bool = False
    ):
        stats_counter = torch.zeros(self.hparams.n_classes, 3)
        assert len(self.preds[set_key]) == len(self.gts[set_key]), (
            f"{len(self.preds[set_key])} != {len(self.gts[set_key])}"
        )
        detection_errors = defaultdict(list)
        for song_index in range(len(self.preds[set_key])):
            offset = 0
            if self.hparams.ignore_beats:
                offset = 2

            assert len(self.preds[set_key][song_index]) == len(
                self.gts[set_key][song_index][offset:]
            )
            for cls, (cls_onset, cls_gt) in enumerate(
                zip(
                    self.preds[set_key][song_index],
                    self.gts[set_key][song_index][offset:],
                )
            ):
                cls_onset = combine_onsets(
                    cls_onset[0, :][cls_onset[1, :] >= thresholds[cls]],
                    cool_down=self.hparams.onset_cooldown,
                )
                tps, fps, fns = evaluate_onset_stats(
                    cls_onset, cls_gt, window=self.hparams.detect_tolerance
                )
                detection_errors[cls].extend([tp[1] for tp in tps])
                stats_counter[cls] += torch.tensor([len(tps), len(fps), len(fns)])

        prefix = "" if set_key == "val" else f"{set_key}/"

        mean_errors = {
            f"{prefix}Mean_Deviation/{self.hparams.mapping.get_name(cls_idx)}": np.mean(
                detection_errors[cls_idx]
            )
            if len(detection_errors[cls_idx]) > 0
            else 0.0
            for cls_idx in detection_errors.keys()
        }

        all_errors = list(itertools.chain.from_iterable(detection_errors.values()))

        mean_errors[f"{prefix}Mean_Deviation/Total"] = (
            np.mean(all_errors) if len(all_errors) > 0 else 0.0
        )

        self.log_dict(
            mean_errors, on_epoch=True, prog_bar=False, add_dataloader_idx=False
        )

        if not calculate_scores:
            return

        precisions = stats_counter[:, 0] / (
            stats_counter[:, 0] + stats_counter[:, 1] + 1e-8
        )
        recalls = stats_counter[:, 0] / (
            stats_counter[:, 0] + stats_counter[:, 2] + 1e-8
        )
        f_scores = (
            2
            * stats_counter[:, 0]
            / (2 * stats_counter[:, 0] + stats_counter[:, 1] + stats_counter[:, 2])
        )
        f_avg = f_scores.mean().item()
        total_stats = stats_counter.sum(dim=0)
        prec_sum = (total_stats[0] / (total_stats[0] + total_stats[1])).item()
        rec_sum = (total_stats[0] / (total_stats[0] + total_stats[2])).item()
        f_sum = (
            2 * total_stats[0] / (2 * total_stats[0] + total_stats[1] + total_stats[2])
        ).item()

        per_class_precisions = {
            f"{set_key}/Precision/Sum/{self.hparams.mapping.get_name(i)}": precision
            for i, precision in enumerate(precisions)
        }
        self.log_dict(
            per_class_precisions,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        per_class_recalls = {
            f"{set_key}/Recall/Sum/{self.hparams.mapping.get_name(i)}": recall
            for i, recall in enumerate(recalls)
        }
        self.log_dict(
            per_class_recalls, on_epoch=True, prog_bar=False, add_dataloader_idx=False
        )

        per_class_f_scores = {
            f"{set_key}/F-Score/Sum/{self.hparams.mapping.get_name(i)}": f_score
            for i, f_score in enumerate(f_scores)
        }
        self.log_dict(
            per_class_f_scores, on_epoch=True, prog_bar=False, add_dataloader_idx=False
        )

        ## self.log(f"Stats/{test_sets[key]}", stats_counter, on_epoch=True, prog_bar=True)
        ## self.log(f"F-Score/Class/{test_sets[key]}", f_scores, on_epoch=True, prog_bar=True)

        self.log(
            f"{set_key}/Precision/Sum/Total",
            prec_sum,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{set_key}/Recall/Sum/Total",
            rec_sum,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{set_key}/F-Score/Sum/Total",
            f_sum,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{set_key}/F-Score/Avg",
            f_avg,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

    def on_validation_epoch_start(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def validation_step(self, batch, batch_idx):
        return self.evaluate_step(batch, "val")

    def on_validation_epoch_end(self):
        self.evaluate_epoch_end("val")

    def on_test_epoch_start(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        test_set = self.hparams.test_sets[dataloader_idx]
        return self.evaluate_step(batch, test_set)

    def on_test_epoch_end(self):
        for test_set in self.hparams.test_sets:
            self.evaluate_epoch_end(test_set)
            continue


def main(
    config=None,
    trial: optuna.Trial = None,
    metric_to_track="F-Score/Sum/Total",
    seed=69,
    experiment_name="debug",
    comment="test",
):
    if config is None:
        config = flatten_dict(settings_asdict(Config()))
    config_class = Config.from_flat_dict(config)
    print(config_class)
    seed_everything(seed, workers=True)

    data_module = DataModule(
        config["batch_size"], config["test_batch_size"], config_class.dataset
    )

    model = LitModel(config)
    # model = LitModel.load_from_checkpoint("models/mamba_fast/last.ckpt", strict=False, config=config)

    # find_learning_rate(model, data_module, min_lr=5e-6, max_lr=1e-4, num_training=1500, type="linear")

    tags = {
        "model": config_class.training.model_settings,
        "time_shift": str(config_class.dataset.annotation_settings.time_shift),
        "mapping": str(config_class.dataset.annotation_settings.mapping),
        # "comment": comment,
        "a2md_cutoff": str(config_class.dataset.a2md_penalty_cutoff),
    }
    if trial is not None:
        tags["trial"] = str(trial.number)

    loggers = [
        TensorBoardLogger(
            save_dir="./runs",
            name=experiment_name,
            version=config_class.training.model_settings,
            sub_dir=comment,
        ),
        MLFlowLogger(
            experiment_name=experiment_name,
            tags=tags,
            tracking_uri="sqlite:///mlruns.db",
            artifact_location="./models",
            run_name=comment,
        ),
    ]

    directions = {
        "train_loss": "min",
        "val_loss": "min",
        "F-Score/Sum/Total": "max",
    }

    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor="F-Score/Sum/Total",
            filename=f"{config_class.training.model_settings}/epoch={{epoch}}-val_loss={{val_loss}}-val_score={{F-Score/Sum/Total}}",
            dirpath="./models",
            mode=directions["F-Score/Sum/Total"],
            auto_insert_metric_name=False,
            save_top_k=1,
            verbose=False,
        ),
        ModelCheckpoint(
            monitor="train_loss_epoch",
            mode="min",
            dirpath="./models",
            save_top_k=1,
            save_last="link",
            filename=f"{config_class.training.model_settings}/last",
            auto_insert_metric_name=False,
            save_on_train_epoch_end=True,
            verbose=False,
            enable_version_counter=False,
        ),
        EarlyStopping(
            "train_loss_epoch",
            check_finite=True,
            mode="min",
            patience=999999,
            check_on_train_epoch_end=True,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        # DeviceStatsMonitor(cpu_stats=True),
    ]

    if trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(trial, metric_to_track))
    if config["ema"] is not None and str(config["ema"]).lower() != "none":
        callbacks.append(WeightAveraging(avg_fn=get_ema_avg_fn(float(config["ema"]))))
    if (
        config["early_stopping"] is not None
        and str(config["early_stopping"]).lower() != "none"
    ):
        callbacks.append(
            EarlyStopping(
                metric_to_track,
                mode=directions[metric_to_track],
                patience=config["early_stopping"],
            )
        )

    # profiler = PyTorchProfiler(
    #     profile_memory=True,
    # )

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=config["epochs"],
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config["gradient_clip_norm"],
        gradient_clip_algorithm="norm",
        benchmark=True,
        accumulate_grad_batches=1,
        deterministic=True,
        # max_steps=20_000,
        log_every_n_steps=10,
        num_sanity_val_steps=-1,
        # fast_dev_run=30,
        # profiler=profiler,
        # overfit_batches=10,
        # check_val_every_n_epoch=20,
    )

    torch.set_float32_matmul_precision("medium")

    # tuner = Tuner(trainer)
    # batch size doesn't work on Windows/WSL due to RAM being automatically allocated as VRAM
    # bs_suggestion = tuner.scale_batch_size(model, datamodule=data_module, batch_arg_name="batch_size")
    # print(bs_suggestion)
    # lr_finder = tuner.lr_find(
    #     model,
    #     datamodule=data_module,
    #     min_lr=1e-4,
    #     max_lr=9e-4,
    #     num_training=100,
    #     update_attr=True,
    #     mode="linear",
    # )
    # if lr_finder is not None:
    #     lr_finder.plot(show=True, suggest=True)

    trainer.fit(model, datamodule=data_module) #, ckpt_path="models/13-7714.ckpt")

    if trial is not None and not torch.isfinite(trainer.callback_metrics["train_loss_epoch"]):
        raise optuna.TrialPruned()

    scores = trainer.validate(model, datamodule=data_module, ckpt_path="best")[0]

    # trainer.test(model, datamodule=data_module, ckpt_path="last")
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    return model, scores[metric_to_track]

if __name__ == '__main__':
    _model, _trainer = main()