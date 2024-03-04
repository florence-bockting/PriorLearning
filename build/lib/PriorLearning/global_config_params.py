import tensorflow_probability as tfp

tfd = tfp.distributions

# dictionary including default settings for hyperparameters in the learning
# algorithm
def default_configs():
    """
    Dictionary with default settings for hyperparameters of the learning
    algorithm incl.:
        
    Gradient estimation of discrete RVs:

    - method: 'softmax_gumbel_trick' (only method currently implemented)
    - softmax_gumbel_temp: 1.0 (temperature parameter in Softmax-Gumble function)
    
    Learning rate schedule for Adam optimizer 
    
    - lr_step: 5 (after how many epochs should the learning rate be adapted)
    - lr_perc: 0.90 (by what percentage is the learning rate reduced per step)
    - clipnorm_val: 1. (parameter of Adam optimizer: the gradient of each weight is individually clipped so that its norm is no higher than this value.)
    
    Multi-objective loss weighting function
    
    - task_balance_factor: 1.6 (temperature parameter in dynamic weight averaging function)
        
    """
    global_variables = dict(
        method = "softmax_gumbel_trick", 
        softmax_gumbel_temp = 1.0,
        lr_step = 5,
        lr_perc = 0.90,
        clipnorm_val = 1.,
        task_balance_factor = 1.6
        )     
    
    return global_variables

_global_variables = default_configs()