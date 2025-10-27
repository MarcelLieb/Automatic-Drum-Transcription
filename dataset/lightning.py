import lightning as L
import torch

from dataset.A2MD import A2MD, get_splits as get_a2md_splits, get_fold
from dataset.MDB_Drums import MDBDrums
from dataset.RBMA13 import RBMA13
from dataset.TMIDT import TMIDT
from dataset.datasets import SongSampler
from dataset.generics import ConcatADTDataset
from dataset import audio_collate

from settings import DatasetSettings, flatten_dict
from dataclasses import asdict


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        test_batch_size: int,
        settings: DatasetSettings,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.n_mels = settings.audio_settings.n_mels
        self.n_classes = settings.annotation_settings.n_classes

        self.settings = settings
        self.save_hyperparameters(flatten_dict(asdict(settings)))
        self.hparams.seed = seed

        self.train = None
        self.val = None
        self.test_sets = []

        self.sample_distribution = None

    def setup(self, stage: str | None = None):
        path = "./data/a2md_public/"
        if stage == "fit" or stage is None:
            if self.hparams.k_folds is not None:
                train_split, val_split, test_split = get_fold(
                    self.hparams.a2md_penalty_cutoff,
                    path=path,
                    n_folds=self.hparams.k_folds,
                    fold=self.hparams.fold,
                    seed=self.hparams.seed,
                )
            else:
                train_split, val_split, test_split = get_a2md_splits(
                    self.hparams.a2md_penalty_cutoff,
                    self.hparams.splits,
                    path,
                    seed=self.hparams.seed,
                )
            match self.hparams.train_set:
                case "a2md_train":
                    self.train = A2MD(
                        split=train_split,
                        settings=self.settings,
                        path=path,
                        use_dataloader=True,
                        is_train=True,
                        segment=True,
                    )
                case "tmidt" | "midi":
                    self.train = TMIDT(
                        path="./data/midi",
                        settings=self.settings,
                        use_dataloader=True,
                        is_train=True,
                        segment=True,
                    )
                case "all":
                    a2md = A2MD(
                        split=None,
                        settings=self.settings,
                        path=path,
                        use_dataloader=True,
                        is_train=True,
                        segment=True,
                    )
                    rbma = RBMA13(
                        path="./data/rbma_13",
                        settings=self.settings,
                        use_dataloader=True,
                        is_train=True,
                        segment=True,
                    )
                    mdb = MDBDrums(
                        path="./data/MDB Drums",
                        settings=self.settings,
                        use_dataloader=True,
                        is_train=True,
                        segment=True,
                    )
                    self.train = ConcatADTDataset(self.settings, [a2md, rbma, mdb])

            self.sample_distribution = self.train.get_sample_distribution()
        if stage == "validate":
            if self.hparams.k_folds is not None:
                train_split, val_split, test_split = get_fold(
                    self.hparams.a2md_penalty_cutoff,
                    path=path,
                    n_folds=self.hparams.k_folds,
                    fold=self.hparams.fold,
                    seed=self.hparams.seed,
                )
            else:
                train_split, val_split, test_split = get_a2md_splits(
                    self.hparams.a2md_penalty_cutoff,
                    self.hparams.splits,
                    path,
                    seed=self.hparams.seed,
                )
            self.val = A2MD(
                split=val_split,
                settings=self.settings,
                path=path,
                use_dataloader=True,
                is_train=False,
                segment=not self.hparams.full_length_test,
            )

        if stage == "test" or stage is None:
            self.test_sets = []
            for test_set in self.hparams.test_sets:
                match test_set:
                    case "RBMA":
                        test_rbma = RBMA13(
                            path="./data/rbma_13",
                            settings=self.settings,
                            use_dataloader=True,
                            is_train=False,
                            segment=not self.hparams.full_length_test,
                        )
                        self.test_sets.append(test_rbma)
                    case "MDB":
                        test_mdb = MDBDrums(
                            path="./data/MDB Drums",
                            settings=self.settings,
                            use_dataloader=True,
                            is_train=False,
                            segment=not self.hparams.full_length_test,
                        )
                        self.test_sets.append(test_mdb)
                    case "A2MD":
                        _, _, test_split = get_a2md_splits(
                            self.hparams.dataset_version, self.hparams.splits, path
                        )
                        test_a2md = A2MD(
                            split=None,
                            settings=self.settings,
                            path="./data/a2md_public/",
                            use_dataloader=True,
                            is_train=False,
                            segment=not self.hparams.full_length_test,
                        )
                        self.test_sets.append(test_a2md)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=not self.hparams.per_song_sampling,
            num_workers=self.hparams.num_workers,
            collate_fn=audio_collate,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
            persistent_workers=False,  # self.hparams.num_workers > 0,
            sampler=SongSampler(self.train) if self.hparams.per_song_sampling else None,
        )
        # return get_dataloader(self.train, self.batch_size, self.hparams.num_workers, is_train=True)

    def val_dataloader(self):
        prefetch_factor = 4 if self.hparams.num_workers > 0 else None
        prefetch_factor = (
            4 // self.test_batch_size
            if self.hparams.full_length_test
            else prefetch_factor
        )
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.test_batch_size
            if self.hparams.full_length_test
            else self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=audio_collate,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )
        # return get_dataloader(
        #     self.val,
        #     self.test_batch_size if self.hparams.full_length_test else self.batch_size,
        #     self.hparams.num_workers,
        #     is_train=False
        # )

    def test_dataloader(self):
        test_loaders = {}
        prefetch_factor = 4 if self.hparams.num_workers > 0 else None
        prefetch_factor = (
            4 // self.test_batch_size
            if self.hparams.full_length_test
            else prefetch_factor
        )
        for test_set in self.test_sets:
            loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=self.test_batch_size
                if self.hparams.full_length_test
                else self.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=audio_collate,
                drop_last=False,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                persistent_workers=False,
            )
            test_loaders[test_set.get_identifier()] = loader
        return test_loaders

