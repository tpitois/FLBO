import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.io import read_off
from tqdm.auto import tqdm


class TOSCA(InMemoryDataset):
    def __init__(self, root, categories=["cat"], transform=None, pre_transform=None):
        if isinstance(categories, str):
            categories = [categories]

        self.categories = [c.lower() for c in categories]

        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        names = []
        for cat in self.categories:
            for i in range(11):
                names.append(f"{cat}{i}.off")
        return names

    @property
    def processed_file_names(self):
        cats_str = "_".join(self.categories)
        return [f"{cats_str}_processed.pt"]

    def download(self):
        raise RuntimeError(
            f"Files not found. Please place your .off files for {', '.join(self.categories)} in {self.raw_dir}"
        )

    def process(self):
        data_list = []

        cat_to_idx = {cat: i for i, cat in enumerate(self.categories)}

        for path in tqdm(self.raw_paths, desc="Reading data"):
            data = read_off(path)

            filename = os.path.basename(path)
            for cat in self.categories:
                if filename.startswith(cat):
                    data.y = torch.tensor([cat_to_idx[cat]], dtype=torch.long)
                    break

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(data)
                for data in tqdm(data_list, desc="Processing data")
            ]

        self.save(data_list, self.processed_paths[0])
