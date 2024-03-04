import tensorflow as tf
import tensorflow_probability as tfp
import inspect

tfd = tfp.distributions

from PriorLearning.helper_functions import group_obs


def method_grand_mean(ypred):
    """
    Averages prior predictions (ypred) over all observations 

    Parameters
    ----------
    ypred : tf.tensor of shape (B,rep,Nobs)
        prior predictions of generative model.

    Returns
    -------
    grand_mean : tf.tensor of shape (B,rep)
        Prior predictions averaged over all observations.

    """
    grand_mean = tf.reduce_mean(ypred, -1)
    return grand_mean


def method_group_means(ypred, dmatrix_fct, cmatrix, **kwargs):
    """
    Computes the mean of groups within one factor.

    Parameters
    ----------
    ypred : tf.tensor of shape (B,rep,Nobs)
        Prior predictions of generative model.
    dmatrix_fct : tf.tensor
        Design matrix incl. only factors.
    cmatrix : tf.tensor
        Contrast matrix (needed to allocate observations to groups).
    **kwargs : optional
        additional keyword arguments.

    Returns
    -------
    group_mean : tf.tensor of shape (B,rep,Ngr)
        Group means.

    """
    # reshape tf.Tensor from (B, rep, Nobs) to (B, rep, Ngr, num_gr)
    samples_grouped = group_obs(ypred, dmatrix_fct, cmatrix)
    # average over individual obs (i.e., Ngr)
    group_mean = tf.reduce_mean(samples_grouped, axis=2)
    # return group means
    return group_mean


def method_R2(ypred, epred):
    """
    Computes R2 by dividing the variance of the linear predictor (epred) by the variance of the prior predictions (ypred).

    Parameters
    ----------
    ypred : tf.tensor of shape (B,rep,Nobs)
        Prior predictions of the generative model.
    epred : tf.tensor of shape (B,rep,Nobs)
        Linear predictor of the generative model.

    Returns
    -------
    R2 : tf.tensor of shape (B,rep)
        R2 computed from model simulations.

    """
    R2 = tf.math.divide(
        tf.math.reduce_variance(epred, -1),
        tf.math.reduce_variance(ypred, -1),
    )
    return R2

class TargetQuantities(tf.Module):
    def __init__(self):
        super(TargetQuantities).__init__()

    def __call__(self, simulations, target_info, **kwargs):
        """
        Computes the prespecificed target quantities based on the simulations
        from the generative model.

        Parameters
        ----------
        simulations : dict
            Simulations from generative model incl. ypred and epred.
        target_info : dict
            User specification wrt type of target quantity and/or custom function for computing the target quantity
        **kwargs : optional; additional keyword arguments
            For example design matrix or contrast matrix, when group means should be specified.

        Returns
        -------
        target_dict : dict of tf.tensors
            Dictionary consisting of all target quantities.

        """
        ypred = simulations["ypred"]
        epred = simulations["epred"]
        
        user_input = locals()
        opt_args = user_input["kwargs"]
        
        target_dict = dict()
        for i, target in enumerate(target_info["target"]):
            
            # if custom target function has been specified use it
            if target_info["custom_target_function"][i] is not None:
                
                get_function_args = str(inspect.signature(target_info["custom_target_function"][i])).removeprefix("(").removesuffix(")").split(",")
                function_args = [get_function_args[i].removeprefix(" ") for i in range(len(get_function_args))]
                if "**kwargs" in function_args: 
                    function_args = set(function_args).difference(set(["**kwargs"]))
         
                simulation_subset = dict()
                for argument in function_args:
                    simulation_subset[argument] = simulations[argument]
                
                target_dict[target] = target_info["custom_target_function"][i](**simulation_subset)
           
            # use prior predictions as is 
            if target == "y_obs" and target_info["custom_target_function"][i] is None:
                target_dict[target] = ypred
            
            # compute R2 from epred and ypred
            if target == "R2" and target_info["custom_target_function"][i] is None:
                target_dict[target] = method_R2(ypred, epred)
            
            # compute group means of factors
            if target == "group_means" and target_info["custom_target_function"][i] is None:
                
                if not set(["dmatrix_fct", "cmatrix"]).issubset(set(opt_args.keys())):
                    raise ValueError("The elicitation method 'group_means' requires the additional arguments dmatrix_fct (i.e., design matrix incl. only factors) and cmatrix (i.e., contrast matrix).")
                
                target_dict[target] = method_group_means(ypred, **kwargs)
            
            # average over all prior predictions 
            if target == "grand_mean" and target_info["custom_target_function"][i] is None:
                target_dict[target] = method_grand_mean(ypred)
                
            if (not(target in set(["y_obs", "R2", "group_means", "grand_mean"])) and
                target_info["custom_target_function"][i] is None):
                target_dict[target] = simulations[target]
             
        return target_dict
    
    


