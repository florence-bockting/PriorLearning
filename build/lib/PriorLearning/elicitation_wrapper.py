import tensorflow_probability as tfp

tfd = tfp.distributions

from PriorLearning.target_quantities import TargetQuantities
from PriorLearning.elicitation_techniques import ElicitationTechnique
from PriorLearning.combine_losses import combine_loss_components
from PriorLearning.simulator import Simulator

def elicitation_wrapper(B, rep, parameters_dict, Simulator, GenerativeModel, 
                        target_info, method, **kwargs):
    """
    Wrapper function that takes as input the initialized simulator and returns
    the final quantities that are used as input for each loss component.

    Parameters
    ----------
    B : integer
        batch size.
    rep : integer
        number of simulations per model parameter.
    parameters_dict : dictionary
        Dictionary as specified by the user providing information about the
        prior distribution family and the distribution used for sampling the 
        initial hyperparameter value (in case of hyperparameter-learning).
    Simulator : Callable; tf.Module
        initialized simulator which initializes the hyperparameter values (if
        hyperparameter_learning) or the base distribution (if normalizing_flow).
    GenerativeModel : Callable; tf.Module
        Generative model as specified by the user.
    target_info : dictionary
        User specification of target quantities, elicitation technique, and 
        single loss components.
    **kwargs : optional keyword arguments
        Additional keyword arguments needed for the generative model (e.g., 
        design matrix, contrast matrix).

    Returns
    -------
    res_list : list
        list of loss components (tf.Tensors) used as input for the discrepancy 
        loss.
    predictive_simulations : dictionary
        Output of the generative model with fixed keys: likelihood, ypred, 
        epred, priors. Additional keys might result due to the specification of
        custom quantities.
    """
    # simulate predictions from the generative model
    predictive_simulations = Simulator(GenerativeModel, parameters_dict, **kwargs)
    
    # compute the target quantities
    targets_ini = TargetQuantities()                          
    targets = targets_ini(predictive_simulations, target_info, **kwargs)
    
    # apply elicitation technique to target quantities
    # in order to get elicited statistics
    elicit_ini = ElicitationTechnique()
    elicits = elicit_ini(target_info, targets)
    
    # compute final loss components used as input for the discrepancy loss
    res_dict = combine_loss_components(target_info = target_info, 
                                       elicits = elicits)
    
    return res_dict, predictive_simulations


def expert_model(B, rep, parameters_dict, GenerativeModel, target_info, method,
                 **kwargs):
    """
    wrapper function for computing the loss components of the ideal expert
    as specified by the user.

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
    GenerativeModel : Callable; tf.Module
        Generative model as specified by the user.
    target_info : dictionary
        User specification of target quantities, elicitation technique, and 
        single loss components.
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

    Returns
    -------
    res_list : list
        list of loss components (tf.Tensors) used as input for the discrepancy 
        loss.
    predictive_simulations : dictionary
        Output of the generative model with fixed keys: likelihood, ypred, 
        epred, priors. Additional keys might result due to the specification of
        custom quantities.

    """
    
    # initialize the simulator with the true prior distributions and 
    # corresponding hyperparameter values
    simulator = Simulator(B, rep, parameters_dict, method, **kwargs)
    
    # compute the loss components of the ideal expert
    res_list, predictive_sims = elicitation_wrapper(
        B, rep, parameters_dict, simulator, GenerativeModel, target_info, 
        method, **kwargs)
    
    return res_list, predictive_sims






