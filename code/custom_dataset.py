#Create a custom_dataset.py class which given a list of tensors returns a pytorch dataset object

#importing libraries
import torch
import torchvision
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_tensors, transform=None):
        self.transform = transform
        # if the item is not none and is of size > 20,000, add it to the list of tensors
        self.data = [tensor for tensor in list_of_tensors if tensor is not None and tensor.shape[1] == 60]


    def __getitem__(self, idx):
        item = self.data[idx].reshape(1, 1, 128, 60)
        # add gaussian noise blur
        transform = torchvision.transforms.GaussianBlur((3,3))
        item = transform(item)
        item = item.reshape(1, 128, 60)
        # make 3 channels of img
        item = torch.stack((item, item, item), dim=1)
        item = item.squeeze(0)
        return item
    
    def __len__(self):
        return len(self.data)