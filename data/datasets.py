from torch.utils.data import Dataset, DataLoader


class FeatureEncoder:
    pass


class BillionDataset(Dataset):
    def __init__(self, split):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class ACE2005Dataset(Dataset):
    def __init__(self, split):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class DrugGeneDataset(Dataset):
    def __init__(self, split):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def get_dataset(dataset, split, batch_size):
    if dataset == "ace":
        data = ACE2005Dataset(split)
    elif dataset == "drug":
        data = DrugGeneDataset(split)
    elif dataset == "billion":
        data = BillionDataset(split)

    shuffle = split != "test"
    return DataLoader(data, batch_size, shuffle)
