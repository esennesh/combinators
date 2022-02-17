import torch
import probtorch
import torch.nn.functional as F
from torch.distributions.normal import Normal

def resample_variables(resampler, q, log_weights):
    ancestral_index = resampler.sample_ancestral_index(log_weights)
    q_new = probtorch.Trace()
    for key, node in q.items():
        resampled_loc = resampler.resample_4dims(var=node.dist.loc, ancestral_index=ancestral_index)
        resampled_scale = resampler.resample_4dims(var=node.dist.scale, ancestral_index=ancestral_index)
        resampled_value = resampler.resample_4dims(var=node.value, ancestral_index=ancestral_index)
        q_new.normal(loc=resampled_loc, scale=resampled_scale, value=resampled_value, name=key)
    return q_new
        
def apg_objective(models, frames, num_sweeps, resampler, mnist_mean, metric_names, algo='apg', **kwargs):
    metrics = {'loss_phi': [], 
               'loss_theta': [], 
               'ess': [], 
               'density': []}
    
    modes = {'E_where' : [], 
             'E_recon' : [], 
            }
    
    # one-shot prediction
    log_w, q, metrics, modes = oneshot(models, frames, mnist_mean, metrics, modes)
    q = resample_variables(resampler, q, log_weights=log_w)
    if algo == 'hmc':
        z_what = q['z_what'].value
        z_where = q['z_where'].value
        metrics = kwargs['hmc_sampler'].hmc_sampling(frames, z_where, z_what, metrics)
        metrics['density'] = torch.cat(metrics['density'], 0)
        return metrics
    # if we are running either APG or BPG (i.e. bootstrapped population Gibbs, where we sample z_what from the prior)
    else:
        T = frames.shape[2]
        for m in range(num_sweeps-1):
            for t in range(T):
                log_w, q, metrics, modes = apg_where_t(models, frames, q, t, metrics, modes)
                q = resample_variables(resampler, q, log_weights=log_w)
            log_w, q, metrics, modes = apg_what(models, frames, q, metrics, modes, apg=True if algo == 'apg' else False)
            q = resample_variables(resampler, q, log_weights=log_w)
        if algo == 'apg':
            metrics['loss_phi'] = torch.cat(metrics['loss_phi'], 0) 
            metrics['loss_theta'] = torch.cat(metrics['loss_theta'], 0) 
            metrics['ess'] = torch.cat(metrics['ess'], 0) 

            modes['E_where'] = torch.cat(modes['E_where'], 0)  
            modes['E_recon'] = torch.cat(modes['E_recon'], 0)
            metrics['density'] = torch.cat(metrics['density'], 0) 

    return metrics, modes


def oneshot(models, frames, conv_kernel, metrics, modes):
    """
    doing initial one-shot prediction (i.e. this step is the same as when using a RWS-style vae model)
    """
    S, B, K, DP, DP = conv_kernel.shape
    T = frames.shape[2]
    # propose for z_what and z_where
    q = probtorch.Trace()
    for t in range(T):
        q = models['enc_location'](q, frames, t, conv_kernel, extend_dir='forward')
    q = models['enc_digit'](q, frames, extend_dir='forward')
    # reonstruct the frame   
    p, ll, recons = models['dec'](q, frames, recon_level='frames')
    log_q = q.log_joint(sample_dims=0, batch_dim=1, reparameterized=False)
    log_p = p.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) + ll
    log_w = (log_p - log_q).detach()
    w = F.softmax(log_w, 0).detach()
    
    loss_phi = (w * (- log_q)).sum(0).mean()
    loss_theta = (w * (-log_p)).sum(0).mean()
    metrics['loss_phi'].append(loss_phi.unsqueeze(0))
    metrics['loss_theta'].append(loss_theta.unsqueeze(0))
    ess = (1. /(w**2).sum(0))
    metrics['ess'].append(ess.unsqueeze(0))
    metrics['density'].append(log_p.detach().unsqueeze(0))

    E_where = []
    for t in range(T):
        E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
    E_where = torch.cat(E_where, 2)
    modes['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * K * 2
    modes['E_recon'].append(recons.mean(0).unsqueeze(0).cpu().detach()) # 1 * B * T * FP * FP
    return log_w, q, metrics, modes

def apg_where_t(models, frames, q, timestep, metrics, modes):
    T = frames.shape[2]
    conv_kernel = models['dec'](q, frames, recon_level='object')
    # forward
    q_f = models['enc_location'](q, frames, timestep, conv_kernel, extend_dir='forward')
    p_f, ll_f, _ = models['dec'](q_f, frames, recon_level='frame', timestep=timestep)
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) + ll_f
    log_q_f = q_f['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_f = log_p_f - log_q_f
    # backward
    q_b = models['enc_location'](q, frames, timestep, conv_kernel, extend_dir='backward')
    p_b, ll_b, _ = models['dec'](q_b, frames, recon_level='frame', timestep=timestep)
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) + ll_b
    log_q_b = q_b['z_where_%d' % (timestep+1)].log_prob.sum(-1).sum(-1) ## equivanlent to call .log_joint, but not sure which one is computationally efficient
    log_w_b = log_p_b - log_q_b
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()          
    
    metrics['loss_phi'].append((w * (- log_q_f)).sum(0).mean().unsqueeze(0))
    metrics['loss_theta'].append((w * (- log_p_f)).sum(0).mean().unsqueeze(0))
    return log_w, q_f, metrics, modes


def apg_what(models, frames, q, metrics, modes, apg=True):
    T = frames.shape[2]
    if apg:
        q_f = models['enc_digit'](q, frames, extend_dir='forward')  
    else:
        q_f = models['enc_digit'].bpg(models['dec'], q, frames, extend_dir='forward')  
    p_f, ll_f, recons = models['dec'](q_f, frames, recon_level='frames')
    log_p_f = p_f.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) + ll_f
    log_q_f = q_f['z_what'].log_prob.sum(-1).sum(-1)
    log_w_f = log_p_f - log_q_f
    
    if apg:
        q_b = models['enc_digit'](q, frames, extend_dir='backward')  
    else:    
        q_b = models['enc_digit'].bpg(models['dec'], q, frames, extend_dir='backward')  
    p_b, ll_b, _ = models['dec'](q_b, frames, recon_level='frames')
    log_p_b = p_b.log_joint(sample_dims=0, batch_dim=1, reparameterized=False) + ll_b
    log_q_b = q_b['z_what'].log_prob.sum(-1).sum(-1)
    log_w_b = log_p_b - log_q_b
    
    log_w = (log_w_f - log_w_b).detach()
    w = F.softmax(log_w, 0).detach()
    
    loss_phi = (w * (-log_q_f)).sum(0).mean()
    loss_theta = (w * (-log_p_f)).sum(0).mean()
    metrics['loss_phi'][-1] = metrics['loss_phi'][-1] + loss_phi.unsqueeze(0)
    metrics['loss_theta'][-1] = metrics['loss_theta'][-1] + loss_theta.unsqueeze(0)
    
    ess = (1. / (w**2).sum(0))
    metrics['ess'].append(ess.unsqueeze(0))
    
    metrics['density'].append(log_p_f.detach().unsqueeze(0))
    
    E_where = []
    for t in range(T):
        E_where.append(q['z_where_%d' % (t+1)].dist.loc.unsqueeze(2))
    E_where = torch.cat(E_where, 2)
    modes['E_where'].append(E_where.mean(0).unsqueeze(0).cpu().detach())
    modes['E_recon'].append(recons.mean(0).unsqueeze(0).detach().cpu())

    return log_w, q_f, metrics, modes


def hmc_objective(models, AT, frames, result_flags, hmc_sampler, mean_shape):
    """
    HMC objective
    """
    metrics = {'density' : []} 
    S, B, T, FP, _ = frames.shape
    log_w, q, metrics = oneshot(models, frames, mean_shape, metrics, result_flags)
    z_where = []
    for t in range(frames.shape[2]):
        z_where.append(q['z_where_%d' % (t+1)].value.unsqueeze(2))
    z_where = torch.cat(z_where, 2)
    z_what = q['z_what'].value
    metrics = hmc_sampler.hmc_sampling(frames, z_where, z_what, metrics)
    metrics['density'] = torch.cat(metrics['density'], 0)
    return metrics

def bpg_objective(models, AT, frames, result_flags, num_sweeps, resampler, mnist_mean):
    """
    bpg objective
    """
    metrics = {'density' : []} ## a dictionary that tracks things needed during the sweeping
    S, B, T, FP, _ = frames.shape
    (enc_coor, dec_coor, enc_digit, dec_digit) = models
    log_w, z_where, z_what, metrics = oneshot(enc_coor, dec_coor, enc_digit, dec_digit, AT, frames, mnist_mean, metrics, result_flags)
    z_where, z_what = resample_variables(resampler, z_where, z_what, log_weights=log_w)
    for m in range(num_sweeps-1):
        z_where, metrics = apg_where(enc_coor, dec_coor, dec_digit, AT, resampler, frames, z_what, z_where, metrics, result_flags)
        log_w, z_what, metrics = bpg_what(dec_digit, AT, frames, z_where, z_what, metrics)
        z_where, z_what = resample_variables(resampler, z_where, z_what, log_weights=log_w)
    metrics['density'] = torch.cat(metrics['density'], 0) 
    return metrics

def bpg_what(dec_digit, AT, frames, z_where, z_what_old, metrics):
    S, B, T, K, _ = z_where.shape
    z_what_dim = z_what_old.shape[-1]
    cropped = AT.frame_to_digit(frames=frames, z_where=z_where)
    DP = cropped.shape[-1]
    q = Normal(dec_digit.prior_mu, dec_digit.prior_std)
    z_what = q.sample((S, B, K, ))
    cropped = cropped.view(S, B, T, K, int(DP*DP))
    log_p_f, ll_f, recon = dec_digit(frames=frames, z_what=z_what, z_where=z_where, AT=AT)
    log_prior = log_p_f.sum(-1)
    ## backward
    _, ll_b, _ = dec_digit(frames=frames, z_what=z_what_old, z_where=z_where, AT=AT)
    log_w = (ll_f.sum(-1) - ll_b.sum(-1)).detach()
    metrics['density'][-1] = metrics['density'][-1] + (ll_f.sum(-1) + log_prior).unsqueeze(0).detach()
    return log_w, z_what, metrics