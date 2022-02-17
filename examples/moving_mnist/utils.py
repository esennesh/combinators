import os
import git
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from affine_transformer import Affine_Transformer
from models import Enc_location, Enc_digit, Decoder
    
PARENT_DIR = git.Repo(os.path.abspath(os.curdir), search_parent_directories=True).git.rev_parse("--show-toplevel")

def expandSdim(x, sample_size):
    """
    expand tensor x with sample size as the new dim0
    """
    ndims = tuple([1] * x.dim())
    return x[None,...].repeat(sample_size, *ndims)
        
def plot_samples(images, fs=.5):
    num_rows, num_cols, _, _ = images.shape
    images = images.squeeze(1).cpus().detach()
    images = torch.clamp(images, min=-1, max=1)
    gs = gridspec.GridSpec(num_rows, num_cols)
    gs.update(left=0.0 , bottom=0.0, right=1.0, top=1.0, wspace=0.1, hspace=0.1)
    fig = plt.figure(figsize=(fs*num_cols, fs*num_rows))        
    for r in range(num_rows):
        for c in range(num_cols):
            ax = fig.add_subplot(gs[r, c])
            try:
                ax.imshow(images[r, c], cmap='gray', vmin=0, vmax=1.0)
            except:
                ax.imshow(np.transpose(images[i], (1,2,0)), vmin=0, vmax=1.0)
            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])

def set_seed(seed):
    import torch
    import numpy
    import random
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def create_exp_name(args): 
    exp_name = '{}_d={}_ts={}_objs={}_dv={}_budget={}_sweeps={}_z={}_lr={}_seed={}'.format(args.model_name, args.data, args.timesteps, args.num_digits, args.dv, args.budget, args.num_sweeps, args.z_what_dim, args.lr, args.seed)        
    print("Experiment: %s" % exp_name)
    return exp_name

def init_models(model_name, device, frame_size, digit_size, num_hidden_location, num_hidden_digit, z_where_dim, z_what_dim):  
    #FIXME: currently I add reference to AT in each network model for convenience, but this may be redundant
    AT = Affine_Transformer(frame_size, digit_size, device)
    enc_location = Enc_location(num_pixels=(frame_size-digit_size+1)**2, 
                        num_hidden=num_hidden_location, 
                        z_where_dim=z_where_dim, 
                        AT=AT).to(device)
    
    enc_digit = Enc_digit(num_pixels=digit_size**2, 
                          num_hidden=num_hidden_digit, 
                          z_what_dim=z_what_dim, 
                          AT=AT).to(device)
    
    decoder = Decoder(num_pixels=digit_size**2, 
                      num_hidden=num_hidden_digit, 
                      z_where_dim=z_where_dim, 
                      z_what_dim=z_what_dim, 
                      AT=AT, 
                      device=device).to(device)
        
    return {'enc_location': enc_location,
            'enc_digit': enc_digit,
            'dec': decoder}


def save_models(models, filename, weights_dir=os.path.join(PARENT_DIR, "examples/moving_mnist", "weights")):
    checkpoint = {k: v.state_dict() for k, v in models.items()}
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(checkpoint, f'{weights_dir}/{filename}')

def load_models(models, filename, weights_dir=os.path.join(PARENT_DIR, "examples/moving_mnist", "weights"), **kwargs):
    checkpoint = torch.load(f'{weights_dir}/{filename}', **kwargs)
    {k: v.load_state_dict(checkpoint[k]) for k, v in models.items()}  

def print_path(path = os.path.join(PARENT_DIR, 'examples/moving_mnist', 'test_folder')):
    print(path)
    
class Trainer():
    """
    A generic model trainer
    """
    def __init__(self, models, train_loader, num_epochs, device, exp_name):
        super().__init__()
        self.models = models
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.device = device
        self.exp_name = exp_name
        self.logging_path = os.path.join(PARENT_DIR, "examples/moving_mnist", "logging")
        
    def train_epoch(self, epoch):
        pass
    
    def train(self):
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:            
            metric_epoch = self.train_epoch(epoch)
            pbar.set_postfix(ordered_dict=metric_epoch)
            self.logging(metric_epoch, epoch)
            save_models(self.models, self.exp_name)
        
    def logging(self, metrics, epoch):
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path)
        fout = open(os.path.join(self.logging_path, self.exp_name + '.txt'), mode='w+' if epoch==0 else 'a+')
        metric_print = ",  ".join(['{:s}={:.2e}'.format(k, v) for k, v in metrics.items()])
        print("Epoch={:d}, ".format(epoch+1) + metric_print, file=fout)
        fout.close()