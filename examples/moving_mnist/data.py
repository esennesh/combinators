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

def setup_data_loader(data, batch_size, train, normalize=False, shuffle=True, data_dir=os.path.os.path.join(PARENT_DIR, "data"), **kwargs):
    """
    This method constructs a dataloader for several commonly-used image datasets
    arguments:
        data: name of the dataset
        data_dir: path of the dataset
        num_shots: if positive integer, it means the number of labeled examples per class
                   if value is -1, it means the full dataset is labeled
        batch_size: batch size 
        train: if True return the training set;if False, return the test set
        normalize: if True, rescale the pixel values to [-1, 1]
    """

    if data in ['mnist', 'fashionmnist']:
        img_h, img_w, n_channels = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    elif data == 'movingmnist':
        img_h, img_w, n_channels = kwargs['frame_size'], kwargs['frame_size'], 1
        transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))]) 
        
    if not normalize:
        del transform.transforms[-1]

    if data == 'mnist':
        dataset = datasets.MNIST(data_dir, train=train, transform=transform, download=True)
    elif data == 'fashionmnist':
        dataset = datasets.FashionMNIST(data_dir, train=train, transform=transform, download=True)

    elif data == 'movingmnist':
        dataset = MovingMNIST(data_dir, timesteps= kwargs['timesteps'], num_digits=kwargs['num_digits'], frame_size=kwargs['frame_size'], dv=kwargs['dv'], train=train, transform=transform, download=True)
        
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            drop_last=True)
    
    return dataloader, img_h, img_w, n_channels


class MovingMNIST(VisionDataset):
    """
    simulate multi mnist using the training dataset in mnist
    """
    def __init__(self, root, timesteps, num_digits, frame_size, dv, train, transform=None, target_transform=None, download=True):
        super(MovingMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.timesteps = timesteps
        self.num_digits = num_digits
        self.frame_size = frame_size
        self.dv = dv
        self.mnist_size = 28 ## by default
        self.files = {'train': 'train_t={}_d={}_v={}_fs={}.pt'.format(self.timesteps, self.num_digits, self.dv, self.frame_size), 
                      'test': 'test_t={}_d={}_v={}_fs={}.pt'.format(self.timesteps, self.num_digits, self.dv, self.frame_size), 
                      }
    
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if train:
            filepath = os.path.join(self.data_path, self.files['train'])
        else:
            filepath = os.path.join(self.data_path, self.files['test'])
        self.data = torch.load(filepath)
        
    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
    
    @property
    def data_path(self):
        return os.path.join(self.root, self.__class__.__name__)

    def download(self):
        if self._check_exists():
            return
        # generate datasets if not exist
        dataset_mnist_train = datasets.MNIST(self.root, train=True, transform=transforms.ToTensor(), download=True)
        dataset_mnist_test = datasets.MNIST(self.root, train=False, transform=transforms.ToTensor(), download=True)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            
        self.generate_movingmnist(dataset_mnist_train, os.path.join(self.data_path, self.files['train']))
        self.generate_movingmnist(dataset_mnist_test, os.path.join(self.data_path, self.files['test']))    
            
    def _check_exists(self):
        for k, v in self.files.items():
            if not os.path.exists(os.path.join(self.data_path, v)):
                return False
        return True
    
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
    

    def generate_movingmnist(self, dataset_mnist, output_path):
        s_factor = self.frame_size / self.mnist_size
        t_factor = (self.frame_size - self.mnist_size) / self.mnist_size
        
        data_loader = DataLoader(dataset=dataset_mnist, 
                                 batch_size=self.num_digits,
                                 shuffle=True, 
                                 drop_last=True)
        canvases = []
        for i in tqdm(range(self.num_digits)):
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
        canvases = torch.cat(canvases, 0)
        torch.save(canvases, output_path)