import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_off


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

        for path in self.raw_paths:
            print(f"Lecture de {os.path.basename(path)}...")
            data = read_off(path)
            data_list.append(data)

        if self.pre_transform is not None:
            print("Application du pré-traitement (FLBO, WKS...)...")
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        print("Dataset traité et sauvegardé avec succès !")