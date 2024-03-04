import tensorflow as tf
import tensorflow_probability as tfp
import time

tfd = tfp.distributions

from tensorflow import keras

from PriorLearning.loss_helpers import compute_loss
from PriorLearning.simulator import Simulator
from PriorLearning.global_config_params import _global_variables
from PriorLearning.multi_loss_weighting import dynamic_weight_averaging
from PriorLearning.target_quantities import TargetQuantities
from PriorLearning.combine_losses import combine_loss_components
from PriorLearning.elicitation_techniques import ElicitationTechnique

@tf.function
def _body(simulation_model,expert_res_list,user_config, _global_variables,
         loss_balancing, epoch, optimizer, lr,
         B, rep, parameters_dict, GenerativeModel, 
         target_info, method, targets_ini, elicit_ini, 
         initial_loss_components, save_vals, num_loss_comp, **kwargs):
    
    # simulate from generative model
    predictive_simulations = simulation_model(
        GenerativeModel, parameters_dict, **kwargs) 
  
    # compute target quantities
    targets = targets_ini(predictive_simulations, target_info, **kwargs) 
    # compute elicited statistics by applying selected elicitation technique
    # on target quantities
    elicits = elicit_ini(target_info, targets)
    # combine all elicited statistics for input to loss function
    res_dict = combine_loss_components(
        target_info = target_info, elicits = elicits)
    
    # compute loss: discrepancy between expert and model-implied quantities 
    loss, components = compute_loss(
             expert_res_dict = tf.nest.flatten(expert_res_list), 
             model_res_dict = tf.nest.flatten(res_dict),
             loss_scaling = user_config["loss_scaling"],
             loss_dimensions = user_config["loss_dimensions"], 
             loss_discrepancy = user_config["loss_discrepancy"],
             num_loss_comp = num_loss_comp
             )
     
    # apply a multi-loss weighting scheme
    # get initial loss per component
    if epoch == 0:
       initial_loss_components = loss
  
    loss_sum, _ = dynamic_weight_averaging(
          epoch, 
          current_loss_components=loss, 
          initial_loss_components=initial_loss_components, 
          task_balance_factor=_global_variables["task_balance_factor"]
       )
    return_values = []
    
    if "targets" in save_vals:
        return_values.append(targets)
    if "elicits" in save_vals:
        return_values.append(elicits)
    if "prior_preds" in save_vals:
        return_values.append(predictive_simulations)
    
    return loss_sum, loss, return_values

@tf.function
def _gradient_computation(
        simulation_model,expert_res_list,user_config, _global_variables,
        loss_balancing, epoch, optimizer, lr, B, rep, parameters_dict, 
        GenerativeModel, target_info, method, targets_ini, elicit_ini, 
        initial_loss_components, save_vals, num_loss_comp, **kwargs):
    
    with tf.GradientTape() as tape:   
        
        # compute loss and simulate from data generating model
        loss_sum, loss, predictive_simulations = _body(
            simulation_model, expert_res_list,user_config, _global_variables,
            loss_balancing, epoch, optimizer, lr,
            B, rep, parameters_dict, GenerativeModel, 
            target_info, method, targets_ini, elicit_ini, 
            initial_loss_components, save_vals, num_loss_comp, **kwargs)
    
    # compute gradient of loss wrt trainable_variables
    g = tape.gradient(loss_sum, simulation_model.trainable_variables)

    return loss_sum, loss, g, predictive_simulations


def trainer(expert_res_list, B, rep, parameters_dict, method, GenerativeModel, 
            target_info, user_config, loss_balancing, save_vals, **kwargs):
    """
    Performs training over all epochs and saves results.

    Parameters
    ----------
    expert_res_list : list
        Elicited statistics of expert as input for loss computation.
    B : int
        batch size.
    rep : int
        number of simulations from prior distribution.
    parameters_dict : dict
        User specification of prior distribution family and hyperparameter initialization.
    method : string; 'hyperparameter_learning'
        Goal of learning algorithm. Currently only hyperparameter learning is implemented.
    GenerativeModel : Callable; Class object
        User specification of generative model 
    target_info : dict
        User specification of type of target quantity, elicitation technique, and further optional arguments.
    user_config : dict
        User specification of hyperparameter of the learning algorithm such as batch size, number of sim. from priors, etc..
    loss_balancing : Boolean; default = True
        Whether loss_balancing should be applied or not. If not weights for each loss component are set to one.
    save_vals : list of strings; ['targets', 'elicits', 'prior_preds']
        Whether quantities from substeps in analyses should be savd for later analysis.
        Possibility to save for the last epoch: target quantities ('targets'), elicited statistics ('elicits'), and predictions from generative model ('prior_preds')
    **kwargs : optional add. keywords arguments
        For example design matrix, contrast matrix, total_count.

    Returns
    -------
    res_return : dict
        results of learning algorithm with the following keys:
            
        - 'loss_info' : list of total loss per epoch
        - 'epoch_time' : list of time needed per epoch in sec 
        - 'priors_info' : list
            - priors_info[0]: dict with the following keys :
                - 'likelihood', 'ypred', 'epred' (+ custom target quantities)
                - 'priors' (depending on the save_vals argument)    
            - 'hyperparam_info' : list of model hyperparameter values per epoch

    """
    
    # initalize schedule for learning rate decay
    if user_config["lr_decay"]:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=user_config["lr0"], 
            decay_steps=_global_variables["lr_step"], 
            decay_rate=_global_variables["lr_perc"],
            staircase=True)
    # set user defined initial learning rate
    lr = user_config["lr0"]
    # initialize classes: simulator, computation of target quantities, and 
    # computation of elicitet statistics
    simulation_model = Simulator(B, rep, parameters_dict, method, **kwargs)
    targets_ini = TargetQuantities() 
    elicit_ini = ElicitationTechnique()
    # compute number of loss components 
    num_loss_comp = int(tf.reduce_sum(
        [len(expert_res_list[key]) for key in expert_res_list.keys()]))
    # initialize initial_loss for the dynamic weight averaging algo. in epoch=0
    initial_loss_components = [tf.Variable(0., trainable=None)]*num_loss_comp
    # initialize lists for saving progress during learning
    if user_config["method"] == "hyperparameter_learning":
        vars_list = []
        varnames = []
    loss_list = []
    epoch_time = []

    # initialize the adam optimizer
    optimizer = keras.optimizers.legacy.Adam(
        learning_rate = lr,
        clipnorm = _global_variables["clipnorm_val"])
    
    # run epochs
    for epoch in tf.range(user_config["epochs"]):
        # runtime of one epoch
        start = time.time()
        
        # compute updated learning rate (according to lr schedule)
        if user_config["lr_decay"]:
            lr = lr_schedule(epoch)
            if lr < user_config["lr_min"]:
                lr = user_config["lr_min"]
        
        params = []
        # compute loss, gradients, and simulate model predictions
        loss_sum, loss , g, predictive_simulations  = _gradient_computation(
            simulation_model,expert_res_list,user_config, _global_variables,
            loss_balancing, epoch, optimizer, lr, B, rep, parameters_dict, 
            GenerativeModel, target_info, method, targets_ini, elicit_ini,
            initial_loss_components, save_vals, num_loss_comp, **kwargs)
        
        # set initial loss component such that dynamic weight averaging works
        if epoch.numpy() == 0:
          initial_loss_components = loss

        # update trainable_variables using gradient info with adam optimizer
        optimizer.apply_gradients(zip(g, simulation_model.trainable_variables))
          
        # time end of epoch
        end = time.time()
        epoch_time.append(end-start)
        
        # print information for user during training
        if epoch % user_config["view_ep"] == 0:
            print(f"epoch_time: {(end-start)*60:.3f}ms")
            print(f"Epoch: {epoch}, loss: {loss_sum:.5f}, lr: {lr:.6f}")
             
        # save results per epoch        
        if user_config["method"] == "hyperparameter_learning":
            [params.append(simulation_model.trainable_variables[i].numpy()) 
             for i in range(len(simulation_model.trainable_variables))]
            vars_list.append(params)
            
            [varnames.append(simulation_model.trainable_variables[i].name) 
             for i in range(len(simulation_model.trainable_variables))]
        
        loss_list.append(loss_sum)
    
    # clean up final results for user return 
    res_return = dict()
    res_return["loss_info"] = loss_list
    res_return["epoch_time"] = epoch_time
    res_return["priors_info"] = predictive_simulations 
    
    if user_config["method"] == "hyperparameter_learning":
        res_return["hyperparam_info"] = [vars_list, varnames]
    
    return res_return
