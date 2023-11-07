#Create a custom_dataset.py class which given a list of tensors returns a pytorch dataset object

#importing libraries
import torch
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_tensors):
        self.data = list_of_tensors
    def __getitem__(self, idx):
        item = self.data[idx]
        #Truncate all of the tensors to a shape of 660000
        item = item[:330000].reshape(1,330000).to(device)
        return item
    def __len__(self):
        return len(self.data)