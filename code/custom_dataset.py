#Create a custom_dataset.py class which given a list of tensors returns a pytorch dataset object

#importing libraries
import torch
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_tensors, min_size=20000, transform=None):
        self.transform = transform
        # if the item is not none and is of size > 20,000, add it to the list of tensors
        self.data = [item[:min_size] for item in list_of_tensors if item is not None and item.shape[0] > 20000]
        self.min_size = min_size


    def __getitem__(self, idx):
        item = self.data[idx]
        #Truncate all of the tensors to a shape of 660000
        item = item[:self.min_size].reshape(1, -1)
        # tile the tensor to a shape of batch_size, 3, 20000
        item = torch.cat((item, item, item), 0)

        # data is super small numbers, so we need to scale it up
        item = item * 100
        # print("item shape: " + str(item.shape))

        item = item.to(device)
        return item
    
    def __len__(self):
        return len(self.data)