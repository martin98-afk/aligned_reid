from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    def __init__(self, array, transform):
        self.array = array
        self.transform = transform

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, item):
        image = self.array[item, ...]
        image = self.transform(image)
        return image