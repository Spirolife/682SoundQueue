# PURPOSE OF FILE: Autoencoder model for the soundqueue project.

import os

import custom_dataset
import torch
import tqdm
import utils
from torch import nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

		encoder = get_encoder(train_set, validation_set,wandb)

		utils.diagnostic_print("Encoder found, encoding train and test sets...")

		encoded_train, encoded_test = encode_dataset(encoder, train_set, test_set)
		# encode the train and test set and return them

		utils.diagnostic_print("Encoded train and test sets, returning Encoded Train and encoded test...")
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
def encode_dataset(encoder, train_set, test_set, K=500):
	# 1. encode the train set
	# 2. encode the test set
	# 3. save the encoded train and test set for later use

	train_set = custom_dataset.CustomDataset(train_set)
	test_set = custom_dataset.CustomDataset(test_set)

	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False)
	test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

	encoded_train = torch.zeros((len(train_set), K))
	encoded_test = torch.zeros((len(test_set), K))
	# encode the train set
	for i, data in enumerate(tqdm(train_loader)):
		# print(data.shape, encoder(data).shape)
		encoded_train[i] = encoder(data).detach()

	# encode the test set
	for i, data in enumerate(tqdm(test_loader)):
		encoded_test[i] = encoder(data).detach()

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
	def __init__(self, K):
		super(Autoencoder, self).__init__()
		
		# recreate above in pytorch
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(64*32*15, K)
		)


		self.decoder = nn.Sequential(
			nn.Linear(K, 64*32*15),
			nn.ReLU(),
			nn.Unflatten(1, (64, 32, 15)),
			nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
		)

	def forward(self, x):

		x = self.encoder(x)
		x = self.decoder(x)
		print(x.shape)

		return x


def train_autoencoder(train_set, validation_set, wandb=None):
	"""
	train_set: Dataset Object of training data
	validation_set: Dataset Object of validation data
	test_set: Dataset Object of test data
	Returns encoder from autoencoder model trained on train_set and validated on validation_set
	"""
	model = Autoencoder(K=5000).to(device)

	# Move the model to GPU
	
	print(f'Inside Train_autoencoder: device: {device}; Checking cuda is available:{torch.cuda.is_available()}')
	model.to(device)
	learning_rate = 1e-7
	num_epochs = 100
	batch_size = 50

	# Define the loss function and optimizer
	criterion =	nn.MSELoss()
	optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=100, max_eval=100, history_size=100, line_search_fn='strong_wolfe')
	# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	# wandb.log({"optimizer": optimizer})
	# wandb.log({"criterion": criterion})
	# wandb.log({"device": device})
	# wandb.log({"model": model})
	wandb.log({"train_config": {'lr':learning_rate,'num_epochs':num_epochs}})
	
	#Define the dataloader
	train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
	validation_loader = torch.utils.data.DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True)
	utils.diagnostic_print("Dataloaders created")
	
	utils.diagnostic_print("Loaded model, defined loss function and optimizer,Now training...")
	print(f'TRAINING')

	# Train the autoencoder and add tqdm.tqdm to see progress
	for epoch in tqdm(range(num_epochs)):
		loss_avg = 0
		for data in train_loader:	
			def closure():
				tensor_arr = data
				optimizer.zero_grad()
				output = model(tensor_arr)
				loss = criterion(output, tensor_arr)
				loss.backward()
				return loss
			loss = optimizer.step(closure)
			loss_avg += loss.item()
		wandb.log({"avg loss": loss_avg/len(train_loader)})
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}',end='\n')
		if epoch % 1== 0:
			# print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
			#Run the validation set and check the loss..
			val_loss_avg = 0
			for data_val in validation_loader:
				tensor_arr_val = data_val
				optimizer.zero_grad()
				output_val = model(tensor_arr_val)
				val_loss = criterion(output_val, tensor_arr_val)
				val_loss_avg += val_loss.item()
			wandb.log({"avg validation_loss": val_loss_avg/len(validation_loader)})
			# print('Ran Validation Set, Loss: {:.4f}'.format(val_loss.item()))
				
	utils.diagnostic_print("Finished training, now saving both autoencoder and encoder...")
	#Save the autoencoder..
	torch.save(model.state_dict(), 'conv_autoencoder.pt')
	#Save the encoder
	encoder = model.encoder
	torch.save(encoder, 'encoder.pt')
	return encoder
	
	
	
	
	
	