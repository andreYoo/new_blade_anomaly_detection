from .tmp_sample import MyData
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def load_dataset(data_path,batch_size,shuffle_train,num_workers):
    """Loads the dataset."""

    dataset = None

    transform = transforms.ToTensor()
    dataset = MyData(path=data_path,transforms=transform)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, drop_last=True)
    return train_loader
