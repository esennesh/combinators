import os
import git
import math
import torch
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torch.distributions.uniform import Uniform
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.vision import VisionDataset
from torch.nn.functional import affine_grid, grid_sample

# this returns the root directory of the repository, so that we can locate the ../root/data/
PARENT_DIR = git.Repo(os.path.abspath(os.curdir), search_parent_directories=True).git.rev_parse("--show-toplevel")

def data_loader_indices(train, timesteps, num_digits, frame_size, dv):
    if train:
        split = 'train_t={}_d={}_v={}_fs={}'.format(timesteps, num_digits, dv, frame_size) 
    else:
        split = 'test_t={}_d={}_v={}_fs={}'.format(timesteps, num_digits, dv, frame_size) 
    
    data_paths = []    
    data_path = os.path.join(PARENT_DIR, "data", 'MovingMNIST', split) 
    indices = torch.load(os.path.join(data_path, 'loader_indices.pt'))
    for index in indices:
        data_paths.append(os.path.join(data_path, '{}.pt'.format(index)))
    return data_paths
    
def setup_data_loader(data_path, batch_size, shuffle=True, **kwargs):
    """
    This method constructs a dataloader for several commonly-used image datasets
    arguments:
        data_dir: path of the dataset
        num_shots: if positive integer, it means the number of labeled examples per class
                   if value is -1, it means the full dataset is labeled
        batch_size: batch size 
        train: if True return the training set;if False, return the test set
    """

    dataset = MovingMNIST(data_path)
        
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            drop_last=True)
    
    return dataloader


class MovingMNIST(VisionDataset):
    """
    simulate multi mnist using the training dataset in mnist
    """
    def __init__(self, data_path, transform=None):
        super(MovingMNIST, self).__init__(data_path, transform=transform)

        self.data = torch.load(data_path)
        
    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

class Sim_MovingMNIST():
    """
    simulate multi mnist using the mnist
    """
    def __init__(self, timesteps, num_digits, frame_size, dv, root=os.path.os.path.join(PARENT_DIR, "data"), transform=None, target_transform=None, download=True, chunksize=1000):
        super(Sim_MovingMNIST, self).__init__()
        self.timesteps = timesteps
        self.num_digits = num_digits
        self.frame_size = frame_size
        self.dv = dv
        self.mnist_size = 28 ## by default
        self.chunksize = chunksize
        self.root = root
        self.data_path = os.path.join(self.root, 'MovingMNIST')
        self.files = {'train': 'train_t={}_d={}_v={}_fs={}'.format(self.timesteps, self.num_digits, self.dv, self.frame_size), 
                      'test': 'test_t={}_d={}_v={}_fs={}'.format(self.timesteps, self.num_digits, self.dv, self.frame_size), 
                      }
        # generate datasets if not exist
        dataset_mnist_train = datasets.MNIST(self.root, train=True, transform=transforms.ToTensor(), download=True)
        dataset_mnist_test = datasets.MNIST(self.root, train=False, transform=transforms.ToTensor(), download=True)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        for k, v in self.files.items(): 
            if not os.path.exists(os.path.join(self.data_path, v)):
                os.makedirs(os.path.join(self.data_path, v))
                
                
        self.generate_movingmnist(dataset_mnist_train, 'train')
        self.generate_movingmnist(dataset_mnist_test, 'test')    
    
    def sim_trajectory(self, init_xs):
        ''' Generate a random sequence of a MNIST digit '''
        v_norm = Uniform(0, 1).sample() * 2 * math.pi
        #v_norm = torch.ones(1) * 2 * math.pi
        v_y = torch.sin(v_norm).item()
        v_x = torch.cos(v_norm).item()
        V0 = torch.Tensor([v_x, v_y]) * self.dv
        X = torch.zeros((self.timesteps, 2))
        V = torch.zeros((self.timesteps, 2))
        X[0] = init_xs
        V[0] = V0
        for t in range(0, self.timesteps -1):
            X_new = X[t] + V[t] 
            V_new = V[t]

            if X_new[0] < -1.0:
                X_new[0] = -1.0 + torch.abs(-1.0 - X_new[0])
                V_new[0] = - V_new[0]
            if X_new[0] > 1.0:
                X_new[0] = 1.0 - torch.abs(X_new[0] - 1.0)
                V_new[0] = - V_new[0]
            if X_new[1] < -1.0:
                X_new[1] = -1.0 + torch.abs(-1.0 - X_new[1])
                V_new[1] = - V_new[1]
            if X_new[1] > 1.0:
                X_new[1] = 1.0 - torch.abs(X_new[1] - 1.0)
                V_new[1] = - V_new[1]
            V[t+1] = V_new
            X[t+1] = X_new
        return X, V

    def sim_trajectories(self, num_tjs):
        Xs = []
        Vs = []
        x0 = Uniform(-1, 1).sample((num_tjs, 2))
        a2 = 0.5
        while(True):
            if ((x0[0] - x0[1])**2).sum() > a2 and ((x0[2] - x0[1])**2).sum() > a2 and ((x0[0] - x0[2])**2).sum() > a2:
                break
            x0 = Uniform(-1, 1).sample((num_tjs, 2))
        for i in range(num_tjs):
            x, v = self.sim_trajectory(init_xs=x0[i])
            Xs.append(x.unsqueeze(0))
            Vs.append(v.unsqueeze(0))
        return torch.cat(Xs, 0), torch.cat(Vs, 0)
    

    def generate_movingmnist(self, dataset_mnist, split):
        s_factor = self.frame_size / self.mnist_size
        t_factor = (self.frame_size - self.mnist_size) / self.mnist_size
        data_loader = DataLoader(dataset=dataset_mnist, 
                                 batch_size=self.num_digits,
                                 shuffle=True, 
                                 drop_last=True)
        canvases = []
        indices = []
        index = 0
        if split == 'train':
            iterations = self.num_digits
        else:
            iterations = self.num_digits
        for i in tqdm(range(iterations)):
            for (images, labels) in tqdm(data_loader):
                canvas = []
                locations, _ = self.sim_trajectories(num_tjs=self.num_digits)
                for k in range(self.num_digits):
                    S = torch.Tensor([[s_factor, 0], [0, s_factor]]).repeat(self.timesteps, 1, 1)
                    Thetas = torch.cat((S, locations[k][..., None] * t_factor), -1)
                    grid = affine_grid(Thetas, torch.Size((self.timesteps, 1, self.frame_size, self.frame_size)), align_corners=True)
                    canvas.append(grid_sample(images[k].repeat(self.timesteps, 1, 1)[:, None, :, :], grid, mode='nearest', align_corners=True))

                canvas = torch.cat(canvas, 1).sum(1).clamp(min=0.0, max=1.0)
                canvases.append(canvas.unsqueeze(0))
                if len(canvases) == self.chunksize:
                    canvases = torch.cat(canvases, 0)
                    torch.save(canvases, os.path.join(self.data_path, self.files[split], '{}.pt'.format(index)))
                    indices.append(index)
                    index += 1
                    canvases = []
        if canvases:
            canvases = torch.cat(canvases, 0)
            torch.save(canvases, os.path.join(self.data_path, self.files[split], '{}.pt'.format(index)))
            indices.append(index)
            
        torch.save(indices, os.path.join(self.data_path, self.files[split], 'loader_indices.pt'))