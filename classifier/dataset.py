import os
import random
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
random.seed(42)

def obtain_shuffled_dataset(list_1, path1, list_2, path2):
    transform= ToTensor()
    retlist=[]
    for val in list_1:
        img_tensor= transform(Image.open(os.path.join(path1, val)))
        retlist.append((img_tensor, 1))
    for val in list_2:
        img_tensor= transform(Image.open(os.path.join(path2, val)))
        retlist.append((img_tensor, 0))
    random.shuffle(retlist)
    return retlist

class DementiaDataset(Dataset):
    def __init__(self, arr):
        super().__init__()
        self.arr= arr
    def __len__(self):
        return len(self.arr)
    def __getitem__(self, index):
        return self.arr[index]

def get_loaders(dementia_path, normal_path, batch_size=16, TRAIN_SIZE=0.8):
    dem_class= os.listdir(dementia_path)
    norm_class= os.listdir(normal_path)
    random.shuffle(norm_class)
    random.shuffle(dem_class)
    train_len= int(TRAIN_SIZE*len(dem_class))

    train_dem= dem_class[:train_len]
    test_dem= dem_class[train_len:]
    train_norm= norm_class[:train_len]
    test_norm= norm_class[train_len:]

    train_arr= obtain_shuffled_dataset(train_dem, dementia_path, train_norm, normal_path)
    test_arr= obtain_shuffled_dataset(test_dem, dementia_path, test_norm, normal_path)

    train_set= DementiaDataset(train_arr)
    test_set= DementiaDataset(test_arr)

    train_loader= DataLoader(train_set, batch_size=batch_size)
    test_loader= DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader
