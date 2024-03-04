import tensorflow as tf


def dynamic_weight_averaging(epoch, current_loss_components, 
                             initial_loss_components, task_balance_factor):
    """DWA determines the weights based on the learning speed of each component
    
    The Dynamic Weight Averaging (DWA) method proposed by 
    Liu, Johns, & Davison (2019) determines the weights based on the learning 
    speed of each component, aiming to achieve a more balanced learning process. 
    Specifically, the weight of a component exhibiting a slower learning speed 
    is increased, while it is decreased for faster learning components.
    
    Liu, S., Johns, E., & Davison, A. J. (2019). End-To-End Multi-Task Learning With Attention. In
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 1871–1880).
    doi: https://doi.org/10.1109/CVPR.2019.00197

    Parameters
    ----------
    epoch : int
        How often should the hyperparameter values be updated?
    current_loss_components : list of floats
        List of loss values per loss component for the current epoch.
    initial_loss_components : list of floats
        List of loss values per loss component for the initial epoch (epoch = 0).
    task_balance_factor : float
        temperature parameter that controls the softness of the loss weighting 
        in the softmax operator. Setting the temperature 𝑎 to a large value 
        results in the weights approaching unity.

    Returns
    -------
    total_loss : float
        Weighted sum of all loss components. Loss used for gradient computation.
    weight_loss_component : list of floats
        List of computed weight values per loss component for current epoch.

    """
    # get number of loss components
    num_loss_components = len(current_loss_components)
    # initialize weights
    
    if epoch < 2:
        rel_weight_descent = tf.ones(num_loss_components)
    # w_t (epoch-1) = L_t (epoch-1) / L_t (epoch_0)
    else:
        rel_weight_descent = tf.math.divide(current_loss_components, 
                             initial_loss_components)
    
    # T*exp(w_t(epoch-1)/a)
    numerator = tf.math.multiply(
        tf.cast(num_loss_components, dtype = tf.float32), 
        tf.exp(tf.math.divide(rel_weight_descent, task_balance_factor)))

    # softmax operator
    weight_loss_component = tf.math.divide(numerator, 
                                           tf.math.reduce_sum(numerator))
    
    # total loss: L = sum_t lambda_t*L_t
    total_loss = tf.math.reduce_sum(tf.math.multiply(weight_loss_component, 
                                                     current_loss_components)) 

    return total_loss, weight_loss_component
