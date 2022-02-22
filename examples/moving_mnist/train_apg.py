import os
import git
import torch
import argparse
from objectives import apg_objective
from data import data_loader_indices, Sim_MovingMNIST
from resampler import Resampler
from utils import set_seed, create_exp_name, init_models, save_models, load_models, Trainer

# this returns the root directory of the repository, so that we can locate the ../root/data/
PARENT_DIR = git.Repo(os.path.abspath(os.curdir), search_parent_directories=True).git.rev_parse("--show-toplevel")

class Train_APG(Trainer):
    def __init__(self, models, data_paths, num_epochs, batch_size, device, exp_name, lr, sample_size, num_sweeps, resampler, mnist_mean):
        super().__init__(models, data_paths, num_epochs, batch_size, device, exp_name)
        params= []
        for m in models.values():
            params += list(m.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
        
        self.metric_names = ['loss_phi', 'loss_theta', 'ess', 'density']
        
        self.sample_size = sample_size
        self.num_sweeps = num_sweeps
        self.resampler = resampler
        self.mnist_mean = mnist_mean.repeat(self.sample_size, 1, 1, 1, 1).to(device)
        
    def train_epoch(self, epoch, train_loader):
        metric_epoch = dict.fromkeys(self.metric_names, 0.0)
        for b, images in enumerate(train_loader):
            self.optimizer.zero_grad()
            images = images.repeat(self.sample_size, 1, 1, 1, 1).to(self.device)
            metrics, _ = apg_objective(self.models, images, self.num_sweeps, self.resampler, self.mnist_mean, self.metric_names)
            (metrics['loss_phi'].sum()+metrics['loss_theta'].sum()).backward()
            self.optimizer.step()
            for k, v in metrics.items():
                metric_epoch[k] += v[-1].mean().detach()
            
        return {k: (v.item() / (b+1)) for k, v in metric_epoch.items()}
    
  
    
def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    exp_name = create_exp_name(args)
    
    dataset_args = {'data': args.data,
                    'batch_size': args.batch_size,
                    'train': True,
                    'timesteps': args.timesteps,
                    'num_digits': args.num_digits,
                    'frame_size': args.frame_size,
                    'dv': args.dv,
                   }
    # re-generate dataset
    if args.sim_data:
        Sim_MovingMNIST(dataset_args['timesteps'], dataset_args['num_digits'], dataset_args['frame_size'], dataset_args['dv'])
    
    data_paths  = data_loader_indices(train=True, 
                                      timesteps=dataset_args['timesteps'], 
                                      num_digits=dataset_args['num_digits'], 
                                      frame_size=dataset_args['frame_size'], 
                                      dv=dataset_args['dv'])
#     breakpoint()
    network_args = {'model_name': args.model_name, 
                    'device': device, 
                    'frame_size': args.frame_size, 
                    'digit_size': args.digit_size, 
                    'num_hidden_location': args.num_hidden_location, 
                    'num_hidden_digit': args.num_hidden_digit, 
                    'z_where_dim': args.z_where_dim, 
                    'z_what_dim': args.z_what_dim,
                    }
    models = init_models(**network_args)
    if args.restore:
        load_models(models, exp_name)
        
    print('start training..')
    sample_size = int(args.budget / args.num_sweeps)      
    
    trainer_args = {'models': models, 
                  'data_paths': data_paths, 
                  'num_epochs': args.num_epochs, 
                  'batch_size': args.batch_size,
                  'device': device, 
                  'exp_name': exp_name, 
                  'lr': args.lr, 
                  'sample_size': sample_size,
                  'num_sweeps': args.num_sweeps, 
                  'resampler': Resampler(args.resample_strategy, sample_size, device), 
                  'mnist_mean': torch.load('mnist_mean.pt').repeat(args.batch_size, args.num_digits, 1, 1),
                 }
    
    trainer = Train_APG(**trainer_args)
    trainer.train()
    
def parse_args():
    parser = argparse.ArgumentParser('Moving MNIST')
    parser.add_argument('--model_name', default='APG', choices=['APG', 'RWS'])
    parser.add_argument('--restore', default=False, action='store_true')
    parser.add_argument('--sim_data', default=False, action='store_true')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--data', default='movingmnist')
    parser.add_argument('--timesteps', default=10, type=int)
    parser.add_argument('--num_digits', default=3, type=int)
    parser.add_argument('--frame_size', default=96, type=int)
    parser.add_argument('--digit_size', default=28, type=int)
    parser.add_argument('--dv', default=0.1, type=float)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--budget', default=20, type=int)
    parser.add_argument('--num_sweeps', default=2, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--resample_strategy', default='systematic', choices=['systematic', 'multinomial'])
    parser.add_argument('--num_hidden_digit', default=400, type=int)
    parser.add_argument('--num_hidden_location', default=400, type=int)
    parser.add_argument('--z_where_dim', default=2, type=int)
    parser.add_argument('--z_what_dim', default=10, type=int)
    return parser.parse_args()  
    
if __name__ == '__main__':
    args = parse_args()
    main(args)