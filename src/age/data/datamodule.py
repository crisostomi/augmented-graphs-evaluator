import json
import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from nn_core.common import PROJECT_ROOT

from age.data.dataset import GraphDataset
from age.data.io_utils import load_data

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_to_label_dict, feature_dim):
        """The data information the Lightning Module will be provided with.
        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.
        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.
        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.
        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.
        Args:
            class_vocab: association between class names and their indices
        """
        self.classes_to_label_dict = class_to_label_dict
        self.feature_dim = feature_dim
        self.num_classes = len(class_to_label_dict)

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.
        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        data = {
            "classes_to_label_dict": self.classes_to_label_dict,
            "feature_dim": self.feature_dim,
        }

        (dst_path / "data.json").write_text(json.dumps(data, indent=4, default=lambda x: x.__dict__))

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.
        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint
        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        data = json.loads((src_path / "data.json").read_text(encoding="utf-8"))

        return MetaData(
            class_to_label_dict=data["classes_to_label_dict"],
            feature_dim=data["feature_dim"],
        )


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        data_dir,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.data_list, self.class_to_label_dict = load_data(self.data_dir, self.dataset_name, attr_to_consider="both")

        self.split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        self.train_val_test_split()

        self.feature_dim = self.data_list[0].x.shape[-1]

    @property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.
        Examples are vocabularies, number of classes...
        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        metadata = MetaData(
            class_to_label_dict=self.class_to_label_dict,
            feature_dim=self.feature_dim,
        )

        return metadata

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):

        split_indices = self.train_val_test_split()

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):

            train_data_list = [self.data_list[idx] for idx in split_indices["train"]]
            self.train_dataset = GraphDataset(train_data_list)

            val_data_list = [self.data_list[idx] for idx in split_indices["val"]]
            self.val_datasets = [GraphDataset(val_data_list)]

        if stage is None or stage == "test":
            test_data_list = [self.data_list[idx] for idx in split_indices["val"]]
            self.test_datasets = [GraphDataset(test_data_list)]

    def train_val_test_split(self):
        idxs = np.arange(len(self.data_list))
        random.shuffle(idxs)

        train_upperbound = math.ceil(self.split_ratios["train"] * len(idxs))
        train_idxs = idxs[:train_upperbound]
        val_test_idxs = idxs[train_upperbound:]

        val_over_valtest_ratio = self.split_ratios["val"] / (self.split_ratios["val"] + self.split_ratios["test"])

        val_upperbound = math.ceil(val_over_valtest_ratio * len(val_test_idxs))
        val_idxs = val_test_idxs[:val_upperbound]
        test_idxs = val_test_idxs[val_upperbound:]

        return {"train": train_idxs, "val": val_idxs, "test": test_idxs}

    def train_dataloader(self) -> DataLoader:
        collate_fn = Batch.from_data_list
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        collate_fn = Batch.from_data_list

        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                collate_fn=collate_fn,
                pin_memory=self.pin_memory,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        collate_fn = Batch.from_data_list

        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                collate_fn=collate_fn,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)


if __name__ == "__main__":
    main()
