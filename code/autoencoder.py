# PURPOSE OF FILE: Autoencoder model for the soundqueue project.

import os

import custom_dataset
import torch
import utils
from torch import nn


# Called by main.py, trains the autoencoder and returns the encoded dataset
def get_encoded_data(train_set, test_set, validation_set,wandb=None):
	# Check if encoded_train, encoded_test already exist
	try: 
		encoded_train = torch.load("encoded_train.pt")
		encoded_test = torch.load("encoded_test.pt")
		utils.diagnostic_print("Encoded train and test sets found, loading...")
		return encoded_train, encoded_test
	
	except:
		utils.diagnostic_print("Encoded train and test sets not found, training autoencoder...")

		encoder = get_encoder(train_set, test_set, validation_set,wandb=wandb)

		encoded_train, validation_set, encoded_test = encode_dataset(encoder, train_set, validation_set, test_set)
		# encode the train and test set and return them
		return encoded_train, encoded_test

# Gets the optimal encoder, load or train
def get_encoder(train_set,validation_set,wandb=None):
	# 1. iteratively change autoencoder parameters
	# 2. train autoencoder on train set with parameters
	# 3. use validation set to evaluate autoencoder
	# 4. save autoencoder or load autoencoder if the above is done
	if os.path.exists("encoder.pt"):
		utils.diagnostic_print("encoder found, loading...")
		return torch.load("encoder.pt")
	else:
		utils.diagnostic_print("Autoencoder not found, training...")
		#create train dataset object
		train_dataset =  custom_dataset.CustomDataset(train_set)
		#create validation dataset object
		validation_dataset = custom_dataset.CustomDataset(validation_set)
		return train_autoencoder(train_dataset,validation_set=validation_dataset,wandb=wandb)
	

# Sends the dataset through the encoder
def encode_dataset(encoder, train_set, test_set):
	# 1. encode the train set
	# 2. encode the test set
	# 3. save the encoded train and test set for later use
	encoded_train = []
	for data in train_set:
		tensor_arr = data
		output = encoder(tensor_arr)
		encoded_train.append(output)
	encoded_test = []
	for data in test_set:
		tensor_arr = data
		output = encoder(tensor_arr)
		encoded_test.append(output)
	utils.diagnostic_print("Encoded train and test sets, saving...")
	torch.save(encoded_train, "encoded_train.pt")
	torch.save(encoded_test, "encoded_test.pt")
	return encoded_train, encoded_test


# class CAE:
#     def __init__(self, train_set, test_set, validation_set):
#         self.train_set = train_set
#         self.test_set = test_set
#         self.validation_set = validation_set
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=2, padding=5) #Shape after conv1:  torch.Size([32, 330000])
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=330006, out_features=128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.relu3 = nn.ReLU()

        # Decoder
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=128, out_features=330006)
        self.relu5 = nn.ReLU()
        self.conv_transpose = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=11, stride=2, padding=5, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        print("Shape after conv1: ", x.shape)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        print("Shape after fc1: ", x.shape)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        print("Shape after fc2: ", x.shape)
        x = self.relu3(x)

        # Decoder
        x = self.fc3(x)
        print("Shape after fc3: ", x.shape)
        x = self.relu4(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        print("Shape after fc4: ", x.shape)
        x = self.relu5(x)
        x = x.view(x.size(0), 32, 550)
        x = self.conv_transpose(x)
        print("Shape after conv_transpose: ", x.shape)
        return x


	
	


def train_autoencoder(train_set, validation_set, wandb=None):
	"""
	train_set: Dataset Object of training data
	validation_set: Dataset Object of validation data
	test_set: Dataset Object of test data
	Returns encoder from autoencoder model trained on train_set and validated on validation_set
	"""
	#Define the Model!
	model = Autoencoder()
	
	#Define the dataloader
	train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=1,shuffle=True)
	validation_loader = torch.utils.data.DataLoader(dataset=validation_set,batch_size=1,shuffle=True)
	utils.diagnostic_print("Dataloaders created")
	# Move the model to GPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	model.to(device)
	learning_rate = 0.001
	num_epochs = 50
	# Define the loss function and optimizer
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	# wandb.log({"optimizer": optimizer})
	# wandb.log({"criterion": criterion})
	# wandb.log({"device": device})
	# wandb.log({"model": model})
	wandb.log({"train_config": {'lr':learning_rate,'num_epochs':num_epochs}})
	
	utils.diagnostic_print("Loaded model, defined loss function and optimizer,Now training...")
	print(f'TRAINING')
	# Train the autoencoder
	for epoch in range(num_epochs):
		for data in train_loader:
			tensor_arr = data
			optimizer.zero_grad()
			output = model(tensor_arr)
			loss = criterion(output, tensor_arr)
			loss.backward()
			optimizer.step()
			wandb.log({"loss": loss})
		if epoch % 5== 0:
			print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
			#Run the validation set and check the loss..
			for data_val in validation_loader:
				tensor_arr_val = data_val
				optimizer.zero_grad()
				output_val = model(tensor_arr_val)
				val_loss = criterion(output_val, tensor_arr_val)
				wandb.log({"validation_loss": val_loss})
				print('Ran Validation Set, Loss: {:.4f}'.format(val_loss.item()))
				
	utils.diagnostic_print("Finished training, now saving both autoencoder and encoder...")
	#Save the autoencoder..
	torch.save(model.state_dict(), 'conv_autoencoder.pth')
	#Save the encoder
	encoder = model.encoder
	torch.save(encoder, 'encoder.pt')
	return encoder
	
	
	
	
	
	