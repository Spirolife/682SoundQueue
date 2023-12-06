#Create a custom_dataset.py class which given a list of tensors returns a pytorch dataset object

#importing libraries
import torch
import torch.utils.data as data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_tensors, min_size=None, transform=None):
        self.transform = transform
        self.data = list_of_tensors
        self.min_size = min_size
        # get smallest tensor size
        if self.min_size is None:
            # list comp. if item is not none, get the size of the tensor, else return the min_size
            self.min_size = min([item.shape[0] if item is not None else torch.inf for item in self.data])

            # print("Min size: " + str(min_size)) 
        else:
            self.min_size = min_size


    def __getitem__(self, idx):
        item = self.data[idx]
        #Truncate all of the tensors to a shape of 660000
        item = item[:self.min_size].reshape(1, -1).to(device)
        # tile the tensor to a shape of batch_size, 3, 20000
        item = torch.cat((item, item, item), 0)
        return item
    
    def __len__(self):
        return len(self.data)