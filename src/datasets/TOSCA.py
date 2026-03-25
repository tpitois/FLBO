import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_off
from tqdm.auto import tqdm


class TOSCA(InMemoryDataset):
    def __init__(self, root, category='cat', transform=None, pre_transform=None):
        self.category = category.lower()
        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"{self.category}{i}.off" for i in range(11)]

    @property
    def processed_file_names(self):
        return [f'{self.category}_processed.pt']

    def download(self):
        raise RuntimeError(
            f"Fichiers introuvables. Place tes fichiers {self.category}X.off dans {self.raw_dir}"
        )

    def process(self):
        data_list = []

        for path in tqdm(self.raw_paths, desc='Reading data'):
            data = read_off(path)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list_processed = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc='Processing data')
            ]

        self.save(data_list_processed, self.processed_paths[0])
