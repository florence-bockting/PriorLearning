import patsy as pa
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

tfd = tfp.distributions

from PriorLearning.elicitation_wrapper import expert_model
from PriorLearning.training import trainer
from PriorLearning.helper_functions import group_obs, Exponential_unconstrained, Normal_unconstrained, TruncatedNormal_unconstrained
from PriorLearning.user_config import target_config, target_input
from PriorLearning.helper_functions import _print_restab
from PriorLearning._case_studies import sleep_data_predictor, plot_expert_pred, tab_expert_pred, plot_learned_prior_mlm, plot_diagnostics_mlm, print_target_info
import PriorLearning.combine_losses 

from tabulate import tabulate
import pprint
pp = pprint.PrettyPrinter(depth=4)

tf.random.set_seed(2024)

user_config = dict(                    
        B = 2**7,                          
        rep = 300,                         
        epochs = 300,                      
        view_ep = 20,
        lr_decay = True,
        lr0 = 0.1, 
        lr_min = 0.01, 
        loss_dimensions = "m,n:B",   
        loss_discrepancy = "energy", 
        loss_scaling = "unscaled",         
        method = "hyperparameter_learning"  
        )

# further case study specific variables
N_subj = 300 # number of participants
N_days = 10  # number of total days
selected_days = [0,2,4,6,8] # days for which the expert is queried
# get design matrix
dmatrix, cmatrix = sleep_data_predictor(scaling = "standardize", N_days = N_days, N_subj = N_subj, selected_days = selected_days)

# true hyperparameter values for ideal_expert
true_values = dict({
    "mu": [5.52, 0.1],
    "sigma": [0.03, 0.02],
    "omega": [0.15, 0.09],
    "alpha_lkj": 1.,
    "nu": 0.069
})

# model parameters
exp_dist = Exponential_unconstrained(user_config["rep"])
parameters_dict = {
    "beta_0": {
        "family":  Normal_unconstrained(),
        "true": tfd.Normal(true_values["mu"][0], true_values["sigma"][0]),
        "initialization": [tfd.Normal(4.,0.1), tfd.Uniform(-6.,-5.)]
        },
    "beta_1": {
        "family":  Normal_unconstrained(),
        "true": tfd.Normal(true_values["mu"][1], true_values["sigma"][1]),
        "initialization": [tfd.Normal(4.,0.1), tfd.Uniform(-6.,-5.)]
        },
    "tau_0": {
        "family": TruncatedNormal_unconstrained(loc = 0.,low = 0.,high = 500),
        "true":  tfd.TruncatedNormal(0., true_values["omega"][0], low=0., high=500),
        "initialization": [tfd.Uniform(-4.,-3.)]
        },
    "tau_1": {
        "family": TruncatedNormal_unconstrained(loc = 0.,low = 0.,high = 500),
        "true":  tfd.TruncatedNormal(0., true_values["omega"][1], low=0., high=500),
        "initialization": [tfd.Uniform(-4.,-3.)]
        },
    "sigma": {
        "family": exp_dist,
        "true": exp_dist(tf.math.log(true_values["nu"])),
        "initialization": [tfd.Uniform(-4., -3.)]
        }
    }

# generative model
class GenerativeModel(tf.Module):
    def __call__(self, 
                 parameters,        # obligatory: samples from prior distributions; tf.Tensor
                 dmatrix,           # required: design matrix; tf.Tensor
                 alpha_lkj,
                 N_subj,
                 N_days,
                 model,
                 **kwargs           # obligatory: possibility for further keyword arguments is needed 
                 ):  
        if model == "expert":
            B = 1
        else:
            B = user_config["B"]
            
        rep = user_config["rep"]
        # correlation matrix
        corr_matrix = tfd.LKJ(2, alpha_lkj).sample((B, rep))
        
        # SD matrix
        # shape = (B, 2)
        taus_m = tf.reduce_mean(
            tf.gather(parameters, indices=[2,3], axis=-1),
            axis=1)
        
        # shape = (B, 2, 2)
        S = tf.linalg.diag(taus_m)
        
        # covariance matrix: Cov=S*R*S
        # shape = (B, 2, 2)
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), 
                                  padding_value=tf.reduce_mean(corr_matrix))
        # compute cov matrix
        # shape = (B, 2, 2)
        cov_mx_subj = tf.matmul(tf.matmul(S,corr_mat), S)
        
        # generate by-subject random effects: T0s, T1s
        # shape = (B, N_subj, 2)
        subj_rfx = tfd.Sample(
            tfd.MultivariateNormalTriL(
                loc= [0,0], 
                scale_tril=tf.linalg.cholesky(cov_mx_subj)), 
            N_subj).sample()
        
        # broadcast by-subject random effects
        # shape = (B, N_obs, 2) with N_obs = N_subj*N_days
        taus = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(subj_rfx, axis=2), 
                shape=(B, N_subj, N_days, 2)), 
            shape=(B, N_subj*N_days, 2))
        
        # reshape coefficients
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas_reshaped = tf.broadcast_to(
            tf.expand_dims(
                tf.gather(parameters, indices=[0,1], axis=-1),
                axis=2), 
            shape=(B, rep, N_subj*N_days, 2))
        
        ## compute betas_s
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas = tf.add(betas_reshaped, tf.expand_dims(taus, axis=1)) 
        
        # compute linear predictor term
        # shape = (B, rep, N_obs) with N_obs = N_subj*N_days
        epred = tf.add(betas[:,:,:,0]*dmatrix[:,0], 
                       betas[:,:,:,1]*dmatrix[:,1])

        theta = tf.exp(epred)
        # for the weibull model
        shape = tf.expand_dims(parameters[:,:,-1],-1)
        scale = tf.subtract(theta, tf.math.lgamma(1+1/shape))
        # define likelihood
        likelihood = tfd.Weibull(concentration = shape,
                                 scale = scale)
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # custom target quantities 
        ## epred averaged over individuals
        epred_days = tf.stack([epred[:,:,i::N_days] for i in range(N_days)], 
                              axis = -1)
        epred_days = tf.reduce_mean(epred_days, axis=2)
        
        ## R2 for initial day
        R2_day0 = tf.divide(
            tf.math.reduce_variance(theta[:,:,selected_days[0]::N_days], axis=-1),
            tf.math.reduce_variance(ypred[:,:,selected_days[0]::N_days], axis=-1))
        
        ## R2 for last day
        R2_day9 = tf.divide(
            tf.math.reduce_variance(theta[:,:,selected_days[-1]::N_days], axis=-1),
            tf.math.reduce_variance(ypred[:,:,selected_days[-1]::N_days], axis=-1))
        # compute standard deviation of linear predictor 
        mu0_sd_comp = tf.math.reduce_std(theta[:,:,selected_days[0]::N_days], axis=-1)
        mu9_sd_comp = tf.math.reduce_std(theta[:,:,selected_days[-1]::N_days], axis=-1)

        # compute sigma from scale parameter
        #tf.stop_gradient(scale**2)
        sigma = tf.sqrt(tf.stop_gradient(scale**2) * (tf.exp(tf.math.lgamma(1+2/shape)) - (tf.exp(tf.math.lgamma(1+1/shape)))**2))
        
        return dict(likelihood = likelihood,      # obligatory: likelihood; callable; 
                    ypred = ypred,                # obligatory: prior predictive data
                    epred = epred,                # obligatory: samples from linear predictor
                    sigma = sigma,                # optional: custom target quantity: sigma
                    epred_days = epred_days,      # optional: custom target quantity: epred avg. over individuals
                    R2_day0 = R2_day0,            # optional: custom target quantity: R2 first day
                    R2_day9 = R2_day9,            # optional: custom target quantity; R2 last day
                    mu0_sd_comp = mu0_sd_comp,
                    mu9_sd_comp = mu9_sd_comp
                    )

# specify target quantity, elicitation technique and loss combination
t1 = target_config(target="sigma", 
                   elicitation="moments",
                   combine_loss="by-stats",
                   moments_specs=("mean", "sd"))
t2 = target_config(target="epred_days", 
                   elicitation="quantiles",
                   combine_loss="by-group",
                   quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90))
t3 = target_config(target="mu0_sd_comp", 
                   elicitation="histogram",
                   combine_loss="all")
t4 = target_config(target="mu9_sd_comp", 
                   elicitation="histogram",
                   combine_loss="all")

target_info_expert = target_input(t1, t2, t3, t4)

# print summary of expert input
print_target_info(target_info_expert)


expert_res_list, prior_pred_res = expert_model(1, user_config["rep"],
                                   parameters_dict, GenerativeModel, target_info_expert,
                                   method = "ideal_expert",
                                   dmatrix = dmatrix,
                                   alpha_lkj = 1., N_subj = N_subj, N_days = len(selected_days),
                                   model = "expert")

