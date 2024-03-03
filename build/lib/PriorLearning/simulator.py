# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 10:20:08 2023

@author: flobo
"""

import tensorflow as tf
import tensorflow_probability as tfp
import time
tfd = tfp.distributions

from MakeMyPrior.helper_functions import create_cinn
from MakeMyPrior.global_config_params import _global_variables 
from MakeMyPrior.softmax_gumbel_trick import apply_softmax_gumbel_trick

class InitializePriorFamily:
    def __init__(self, rep):
        """Initialize the prior distribution family as well as the starting 
        value of each respective hyperparameter value (only for method: hyper-
                                                       paramter_learning).

        Parameters
        ----------
        rep : integer
            number of simulations per model parameter.
        parameters_dict : dictionary
            Dictionary as specified by the user providing information about the
            prior distribution family and the distribution used for sampling the 
            initial hyperparameter value.

        """
        self.rep = rep
        
    def __call__(self, parameters_dict):
        """Initialize the prior distribution families and sample from the init-
        ialization distribution the starting value.

        Returns
        -------
        initial_hypparam_list : list
            DESCRIPTION.

        """
        ## initialize hyperparameter values
        initial_hypparam_list = []
                
        n = -1
        for j, param_name in enumerate(parameters_dict.keys()):
            # prior distribution family of model parameter
            prior_family = parameters_dict[param_name]["family"]
            # hyperparameternames of prior distr. family
            hypparam_name = prior_family.keys
            
            initial_hypparam = dict()
            n += 1
            for i, hyperparam_name in enumerate(hypparam_name):
                initialization = parameters_dict[param_name]["initialization"][i]
                
                initial_hypparam[f"{hyperparam_name}"] = tf.Variable(
                    initial_value = initialization.sample(),
                    trainable = True,
                    name = f"{hyperparam_name}_{n}")
            
            initial_hypparam_list.append(initial_hypparam)
                
        return initial_hypparam_list

class Simulator(tf.Module):
    def __init__(self, B, rep, parameters_dict, method, **kwargs):
        """Initializes the trainable variables which are either the hyperparameter
        values of each prior distribution (method: hyperparameter_learning) or the
        weights and biases of the normalizing flow (method: normalizing_flow)
    
            Parameters
            ----------
            B : integer
                batch size.
            rep : integer
                number of simulations per model parameter.
            parameters_dict : dictionary
                Dictionary as specified by the user providing information about the
                prior distribution family and the distribution used for sampling the 
                initial hyperparameter value.
            method : string; either "ideal_expert", "hyperparameter_learning", or 
                "normalizing_flow".
                Determines the optimzation goal: (1) "hyperparameter_learning": goal is 
                to find the optimal hyperparamter values of a predfined prior
                distribution family; (2) "normalizing_flow": goal is to find an optimal
                joint prior distribution for all model parameters; (3) "ideal_expert": 
                prior distribution family and respective hyperparameter values are pre-
                determined (represents "ground truth").
            **kwargs : optional keyword arguments
                Additional keyword arguments needed for the generative model or the 
                target quantities (e.g., design matrix, contrast matrix).
    
            """
        self.B = B
        self.rep = rep
        self.method = method
        # inspect whether additional keyword arguments were specified
        kwargs = locals()
        self.opt_args = kwargs["kwargs"]
        
        if self.method == "hyperparameter_learning":
            # initialize the prior distribution family as specified by 
            # the user
            self.initialized_priors = InitializePriorFamily(
                self.rep)
            # get list with initial hyperparameter values which are subject of
            # the optimization algorithm
            self.initialized_priors = self.initialized_priors(parameters_dict)
           
        if self.method == "normalizing_flow":
            # prepare invertible neural network for prior distribution
            # learning via normalizing flows
            self.cINN = create_cinn(len(parameters_dict), 
                                    _global_variables["coupling_layers"])

    def __call__(self, generative_model, parameters_dict, **kwargs):
        """Samples first from the initialized prior distribution of each model 
        parameter and subsequently from the generative model.

        Parameters
        ----------
        generative_model : callable; (tf.Module)
            Generative model (as tf.Module) as specified by the user.
        **kwargs : optional keyword arguments
            Additional keyword arguments needed for the generative model or the 
            target quantities (e.g., design matrix, contrast matrix).

        Returns
        -------
        res_dict : dictionary
            Dictionary including: "likelihood" (callable), "ypred" (prior pred-
            ictions), "epred" (samples from the linear predictor), "priors" 
            (samples from the prior distributions), and if specified further 
            custom target quantities.

        """
        if (self.method == "ideal_expert_nf" or self.method == "ideal_expert"):
            # get "true" prior distributions with hyperparameter values that 
            # specify the ground truth
            priors = []
            # sample from each prior distribution (shape: (B,rep))
            if self.method == "ideal_expert_nf":
                for param_name in parameters_dict.keys():
                    if param_name == "sigma" or param_name.startswith("omega"):
                        priors.append(
                           tf.math.log(
                                parameters_dict[param_name]["true"].sample(
                                    (self.B,self.rep))
                                )
                            )  
                    else:
                        priors.append(
                            parameters_dict[param_name]["true"].sample(
                                (self.B,self.rep))
                            )  
            if self.method == "ideal_expert":
                for param_name in parameters_dict.keys():
                    priors.append(
                        parameters_dict[param_name]["true"].sample(
                            (self.B,self.rep))
                        )  
            # stack all prior distributions into one tf.Tensor of 
            # shape (B, rep, num_parameters)
            prior_samples = tf.stack(priors, axis = -1)
            
        if self.method == "hyperparameter_learning":
            priors = []
           
            for initial_hyperparameter, name in zip(self.initialized_priors,
                                                    parameters_dict.keys()):
                # get the prior distribution family as specified by the user
                prior_family = parameters_dict[name]["family"]
                # sample from the prior distribution using the initial (or updated) 
                # hyperparameter values
                priors.append(prior_family(**initial_hyperparameter
                                           ).sample((self.B,self.rep))
                              )
            # stack all prior distributions into one tf.Tensor of 
            # shape (B, rep, num_parameters)
            prior_samples = tf.stack(priors, axis = -1)
            
        if self.method == "normalizing_flow":
            # sample from the multivariate normal distribution (base distr.)
            # with number of dimensions equal number of parameters
            base_dist = tfd.Normal(
                loc = tf.zeros(len(parameters_dict)),
                scale = tf.ones(len(parameters_dict))
                ).sample((self.B, self.rep))
            
            # final tf.Tensor of prior samples has shape (B, rep, num_param)
            # (Note: the index [0] is used as the cINN has two outputs whereby
            # the second one is the log of the Jacobian determinant which is 
            # not needed here.)
            prior_samples = self.cINN(base_dist, condition = None)[0] 
        
        # after sampling from prior distributions is done, we can sample
        # prior predictions from the generative model 
        
        # initialize generative model as specified by the user
        generator = generative_model()
        # sample prior predictions from the generative model
        res_dict = generator(prior_samples, **kwargs)
        
        # in case the data model (likelihood) is discrete and thus no gradients
        # can be computed, the softmax gumbel trick is used which is a relax.
        # of discrete variables as continuous variables, which allows for the
        # computation of gradients
        if res_dict["likelihood"].reparameterization_type != tfd.FULLY_REPARAMETERIZED:
            # for the softmax-gumbel trick the design matrix and the upper bound of the 
            # distribution is needed. If no upper bound exist (e.g., Poisson distribution)
            # "total_count" represents an upper truncation threshold specified
            # by the user
            softmax_args = {key: self.opt_args[key] for key in ["total_count", "dmatrix"]}
      
            # apply the softmax-gumbel trick and return transformed prior pre-
            # dictions "ypred"
            res_dict["ypred"] = apply_softmax_gumbel_trick(
                res_dict["likelihood"], self.B, self.rep, 
                softmax_gumbel_temp = _global_variables["softmax_gumbel_temp"],
                **softmax_args)
        # save prior samples in results for later examination.
        res_dict["priors"] = prior_samples
        
        return res_dict

