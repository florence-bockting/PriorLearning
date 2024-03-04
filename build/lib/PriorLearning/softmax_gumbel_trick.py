import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def apply_softmax_gumbel_trick(likelihood, B, rep, dmatrix, total_count, 
                               softmax_gumbel_temp):
    """
    Function to apply softmax gumbel trick to compute gradients for
    discrete RV

    Parameters
    ----------
    B : int
        batch size.
    rep : int
        Number of model parameter simulations within one batch.
    dmatrix : tf.Tensor of shape (Nobs, num_coef)
        Design matrix with length equal to number of obs (Nobs) and width
        equal to number of coefficients (num_coef).
    likelihood : callable
        initialized likelihood according to specified distribution family.
    temp : float
        Temperature parameter of softmax function in gumbel trick. A
        meaningful default value is 1.0.

    Returns
    -------
    ypred: tf.Tensor of shape (B, rep, Nobs)
        Prior predictive samples.

    """
    number_obs = 0
    # get number of observations
    if len(dmatrix.shape) == 1:
        number_obs = len(dmatrix)
    else:
        number_obs = dmatrix.shape[-2]

    # get total observed count
    size = total_count
    # constant outcome vector (including zero outcome)
    c = tf.range(size+1, delta=1, dtype=tf.float32)
    # broadcast to shape (B, rep, outcome-length)
    c_brct = tf.broadcast_to(c[None, None, None, :], 
                             shape=(B, rep, number_obs, len(c)))
    # compute pmf value
    pi = likelihood.prob(c_brct)
    # prevent underflow
    pi = tf.where(pi < 1.8 * 10 ** (-30), 1.8 * 10 ** (-30), pi)
    # sample from uniform
    u = tfd.Uniform(0, 1).sample((B, rep, number_obs, len(c)))
    # generate a gumbel sample from uniform sample
    g = -tf.math.log(-tf.math.log(u))
    # softmax gumbel trick
    w = tf.nn.softmax(
        tf.math.divide(tf.math.add(tf.math.log(pi), g), softmax_gumbel_temp)
    )
    # reparameterization/linear transformation
    ypred = tf.reduce_sum(tf.multiply(w, c), axis=-1)
    
    return ypred