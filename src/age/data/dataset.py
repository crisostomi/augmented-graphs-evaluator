import hydra
import omegaconf
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT


class GraphDataset(Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        return self.data_list[index]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: GraphDataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, split="train", _recursive_=False)


if __name__ == "__main__":
    main()
