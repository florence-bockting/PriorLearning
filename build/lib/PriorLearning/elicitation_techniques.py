import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def method_quantiles(target_samples, q):
    """
    Computes quantiles from model-implied target quantities

    Parameters
    ----------
    target_samples : tf.tensor of rank 2 or 3
        Samples from target quantity.
    q : list of floats
        Quantiles that should be computed.
        Example: q = [0.25, 0.5, 0.75].

    Returns
    -------
    quantiles : tf.tensor of shape (B,...,len(q)) 
        Quantiles computed from target quantity.

    """
   
    # compute quantiles
    quants = tfp.stats.percentile(x=target_samples, q=q, axis=1)
    # depending on rank of tensor reshape tensor such that quantile axis 
    # is last (not first)
    if len(quants.shape) == 2:
        quantiles = tf.transpose(quants)
    if len(quants.shape) == 3:
        quantiles = tf.transpose(quants, perm=[1, 0, 2])
    
    return quantiles

def method_moments(target_samples, m):
    """
    Computes moments from model-implied target quantity. 

    Parameters
    ----------
    target_samples : tf.tensor of rank 2 or 3
        Samples from target quantity.
    m : list of strings
        Specification of moments that should be computed. Currently supported
        are "mean", "sd",and "variance".
        Example: m = ["mean", "sd"]

    Returns
    -------
    moments : tf.tensor of shape (B,...,len(m))
        Moments computed from target quantity.

    """
    # check whether moment specification is supported
    assert m in ["mean", "sd", "variance"], "Currently supported moments are 'mean', 'sd', and 'variance'"
    
    mom_list = []
    for i in m:
        if i == "mean":
            mom_list.append(tf.reduce_mean(target_samples, axis=1))
        if i == "sd":
            mom_list.append(tf.math.reduce_std(target_samples, axis=1))
        if i == "variance":
            mom_list.append(tf.math.reduce_variance(target_samples, axis=1))
    
    moments = tf.stack(mom_list,1)
    
    return moments

class ElicitationTechnique(tf.Module):
    def __init__(self):
        super(ElicitationTechnique).__init__()
    
    def __call__(self, elicit_info, targets):
        """
        Computes the elicited statistics for each target quantity based on the 
        elicitation technique specified by the user

        Parameters
        ----------
        elicit_info : dict
            User specified dict incl. information about elicitation technique 
            and opt. additional information.
        targets : dict
            Dict including all target quantities.

        Returns
        -------
        elicited_statistics : dict
            Elicited statistics computed from target quantities.

        """
        
        elicit_dict = dict()
        # compute elicited statistics from target quantities according to
        # specified elicitation technique
        
        # additional specification for quantile elicitation: list of quantiles
        if "quantiles_specs" in elicit_info.keys(): 
            quant = 0
        
        # additional specification for moment specification: list of moments
        if "moments_specs" in elicit_info.keys(): 
            mom = 0
       
        for i, (elicit_method, target) in enumerate(
                zip(elicit_info["elicitation"], targets.keys())):
           
            # histogram-based elicitation
            # uses the target quantity as is (target quantity and elicited 
            # statistics are identical)
            if elicit_method == "histogram":
                elicit_dict[f"{target}_hist_{i}"] = targets[target]
    
            # quantile-based elicitation
            # computes pre-specified quantiles from target quantity 
            if elicit_method == "quantiles":
                q = elicit_info["quantiles_specs"][quant]
                quant += 1
                
                elicit_dict[f"{target}_quant_{i}"] = method_quantiles(
                    targets[target], list(q))
            
            # moment-based elicitation
            # computes pre-specified moments from target quantity
            if elicit_method == "moments":
                m = elicit_info["moments_specs"][mom]
                mom += 1
                
                elicit_dict[f"{target}_mom_{i}"] = method_moments(
                    targets[target], list(m))
                
        elicited_statistics = elicit_dict
        return elicited_statistics
        
    
