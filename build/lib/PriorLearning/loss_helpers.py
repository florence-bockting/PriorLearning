import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from PriorLearning.discrepancy_measures import energy_loss


def scale_loss_component(method, loss_comp_exp, loss_comp_sim):
    """
    Function for scaling the expert and model elicited statistic of
    a particular loss component

    Parameters
    ----------
    method : string; either 'scale-by-std', 'scale-to-unity', or 'unscaled'
        Scaling method applied to the elicited statistics of the loss component.
    loss_comp_exp : tf.tensor of shape (B, stats)
        Elicited statistics from the expert.
    loss_comp_sim : tf.tensor of shape (B, stats)
        Elicited statistics as simulated by the model.

    Returns
    -------
    loss_comp_exp_std : tf.tensor of shape (B, stats)
        if not 'unscaled': scaled elicited statistics from the expert otherwise
        unmodified elicited statistics
    loss_comp_sim_std : tf.tensor of shape (B, stats)
        if not 'unscaled': scaled elicited statistics from the model otherwise
        unmodified elicited statistics

    """
    if method == "unscaled":
        return loss_comp_exp, loss_comp_sim

    if method == "scale-by-std":
        emp_std_sim = tf.expand_dims(tf.math.reduce_std(loss_comp_sim, axis=-1), -1)

        loss_comp_exp_std = tf.divide(loss_comp_exp, emp_std_sim)
        loss_comp_sim_std = tf.divide(loss_comp_sim, emp_std_sim)
        
        return loss_comp_exp_std, loss_comp_sim_std

    if method == "scale-to-unity":
        emp_min_sim = tf.expand_dims(tf.math.reduce_min(loss_comp_sim, axis=-1), -1)
        emp_max_sim = tf.expand_dims(tf.math.reduce_max(loss_comp_sim, axis=-1), -1)

        loss_comp_exp_std = tf.math.divide(
            loss_comp_exp - emp_min_sim, emp_max_sim - emp_min_sim
        )
        loss_comp_sim_std = tf.math.divide(
            loss_comp_sim - emp_min_sim, emp_max_sim - emp_min_sim
        )
        
        return loss_comp_exp_std, loss_comp_sim_std

def compute_loss(expert_res_dict, model_res_dict, loss_scaling, loss_dimensions, 
                 loss_discrepancy, num_loss_comp):
    """
     Computes loss values based on model-implied and expert elicited statistics 
     and given the loss specification by the user.

     Parameters
     ----------
     expert_res_dict : dict
        expert elicited statitics.
     model_res_dict : dict
        model-implied elicited statistics.
     loss_scaling : string; 'scale-by-std', 'scale-to-unity', or 'unscaled'
        whether the loss component (i.e., elicited statistics from expert and model) should be scaled.
     loss_dimensions : string; 'm,n:B', 'B:m,n'
        Defines axis 0 and 1 of the 2-dim. tensor representing a loss component.
     loss_discrepancy : string; 'energy'
        Loss function used for the loss component. Currently we use the energy loss for all loss components
     num_loss_comp : int
        Number of loss components in total weighted loss

     Returns
     -------
     loss_list, (expert_loss_comp, model_loss_comp) : lists
        
         - loss_value: list with final loss value of each loss component
         - expert/model_loss_comp: list with tensors representing expert and model-implied loss component
        
    """
    
    loss = []
    expert_components = []
    model_components = []
    
    assert loss_discrepancy == "energy", "Currently on the energy loss is implemented"
    
    if loss_discrepancy == "energy":
        # use the energy loss for all loss components
        loss_discrepancy_list = [energy_loss()]*num_loss_comp
       
    for i in range(len(model_res_dict)):
        # initialize the energy loss
        loss_measure = loss_discrepancy_list[i]   
        # get model-implied loss component from dict
        m = model_res_dict[i]        
        # get expert elicited loss component from dict
        e_raw = expert_res_dict[i]
        
        # reshape loss components such that they match in shapes
        if len(m.shape) == 1:
            m = tf.expand_dims(m, axis=-1)
        if len(e_raw.shape) == 1:
            e_raw = tf.expand_dims(e_raw, axis=-1)
        
        # broadcasting such that first dimension is batch size
        e = tf.broadcast_to(e_raw, shape = (m.shape[0], e_raw.shape[1]))
        
        # apply scaling of elicited statistics of loss component
        if e.shape[1] != 1:
            e, m = scale_loss_component(loss_scaling, e, m)
       
        # reshape loss dimensions if specified by user
        if loss_dimensions == "m,n:B":
            e = tf.transpose(e)
            m = tf.transpose(m)
            
        # create list with final loss per loss component
        loss.append(loss_measure(
                        e, m, 
                        m.shape[0],         # B
                        e.shape[1],         # m
                        m.shape[1])         # n
                        )
        # save each expert and model-implied loss component in a list
        expert_components.append(e)
        model_components.append(m)
    
    loss_list = loss
    expert_loss_comp = expert_components
    model_loss_comp = model_components
    
    return loss_list, (expert_loss_comp, model_loss_comp)

            

   


