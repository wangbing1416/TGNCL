from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_helper):
        self.content = data_helper.content
        self.label = data_helper.label

    def __getitem__(self, item):
        return self.content[item], self.label[item]

    def __len__(self):
        return len(self.label)
