import os
import torch
import hydra
import time 
from PIL import Image
import numpy as np
from torchvision import transforms
import torchvision.datasets as datasets

from skimage.metrics import structural_similarity as ssim
import cv2

from torch.nn import MSELoss
from torchvision.utils import save_image
from torch.optim import Adam
from torch.utils.data import DataLoader
from autoencoder import Autoencoder
import warnings


# Ignora gli avvisi specifici
warnings.filterwarnings("ignore")

def load_dataset(dataset: str, batch_size:int, transform):
	try:
		if dataset == "MNIST": 		
			trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
			testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
		elif dataset == "FashionMNIST":
			trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
			testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

		elif dataset == "EMNIST":
			trainset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
			trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
			testloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

		return trainset, trainloader, testloader
	except: 
		print("Dataset is not in these values: [MNIST, FashionMNIST, EMNIST]")
		
		


# Definisco funzione di test 
def save(model, dataloader, name, save_dir):
	# Salvataggio delle immagini ricostruite
	image = iter(dataloader).next()[0]
	image = image.view(image.size(0), -1)
	output = model(image)
	#output = output.view(28, 28)

	print("Saving images")
	image = np.array(image.detach().cpu() * 255, dtype=np.uint8).reshape(28,28)
	output = np.array(output.detach().cpu() * 255, dtype=np.uint8).reshape(28,28)

	saving_file_name = save_dir + f'{name}.png'
	saving_file_name_out = save_dir + f'{name}_reconstruded.png'

	# Save the output to the disk
	image = Image.fromarray(image)
	output = Image.fromarray(output)

	image.save(saving_file_name)
	output.save(saving_file_name_out)


def compute_SSIM(save_dir, dataset):
	# Leggi le immagini
	fn_reconstruced = save_dir + f"{dataset}_reconstruded.png"
	fn_original = save_dir + f"{dataset}.png"

	img1 = cv2.imread(fn_original)
	img2 = cv2.imread(fn_reconstruced)

	# Calcola l'indice SSIM
	ssim_index = ssim(img1, img2, multichannel=True)

	# Stampa il valore SSIM
	print(f"Indice SSIM:{ssim_index:.2f}")
	


# Definire funzione di training
def train(model, dataloader, criterion, optimizer, num_epochs, save_path):
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = "cpu"
	min_mse = 100
	start_time = time.time()

	for ep in range(num_epochs):
		loss_ep = 0
		outputs = list()
		names = list()
		count = 0
		for i, data in enumerate(dataloader):
			# Leggi immagini dataset
			image, _ = data
			image = image.view(image.size(0), -1).to(device)
			
			output = model(image)
			loss = criterion(output, image)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			count += 1
			loss_ep += loss.item()
			outputs.append(output)

			if loss_ep < min_mse:
				min_mse = loss_ep
				torch.save(model.state_dict(), save_path)

		print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(ep+1, num_epochs, i+1, len(dataloader), loss_ep/count))
	end_time = time.time()
	print(f"Time for compression: {end_time - start_time} and min MSE: {min_mse:.2f}")
		



@hydra.main(config_path='cfg', config_name='config')
def main(cfg):
	# variabili da definire
	
	batch_size = cfg.batch_size
	learning_rate = cfg.learning_rate
	num_epochs = cfg.num_epochs
	dataset = cfg.dataset
	save_path = cfg.save_path + f"compressed_model_{dataset}.pth"
	
	
	
	# Definisci trasformazioni per le immagini
	transform = transforms.Compose([
        transforms.ToTensor(),  # converte l'immagine in un tensore
    ])
	
	# read dataset 
	trainset, trainloader, testloader = load_dataset(dataset, batch_size, transform)
	name = dataset


	# Crea Modello, Loss e Optimizer
	model = Autoencoder()
	criterion = MSELoss()
	optimizer = Adam(model.parameters(), lr= learning_rate)

	# Train the model
	if cfg.train:
		train(model, trainloader, criterion, optimizer, num_epochs, save_path)

	else:
		model = Autoencoder()
		model.load_state_dict(torch.load(save_path))
		save(model, testloader, name, cfg.save_dir)
		compute_SSIM(cfg.save_dir, name)

if __name__ == '__main__':
	main()