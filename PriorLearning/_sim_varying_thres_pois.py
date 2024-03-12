import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
tfd = tfp.distributions

from PriorLearning.elicitation_wrapper import expert_model
from PriorLearning.training import trainer
from PriorLearning.helper_functions import Normal_unconstrained
from PriorLearning.user_config import target_config, target_input
from PriorLearning._case_studies import antidiscr_laws_dmatrix

def sim_varying_thres_pois(threshold):
    # setting of hyperparameter of learning algorithm
    user_config = dict(                    
            B = 2**8,                          
            rep = 300,                         
            epochs = 400,                      
            view_ep = 40,
            lr_decay = True,
            lr0 = 0.1, 
            lr_min = 0.0001, 
            loss_dimensions = "m,n:B",   
            loss_discrepancy = "energy", 
            loss_scaling = "unscaled",         
            method = "hyperparameter_learning"  
            )
    # case-specific variables
    selected_states = [1, 11, 27, 33, 17, 15]
    names_states = ["Vermont", "Hawaii", "West Virginia", "Alabama", "Wisconsin", "Iowa"]
    # get design matrix
    dmatrix_exp, _, _ = antidiscr_laws_dmatrix(scaling = "standardize", selected_obs = selected_states, B = 1, rep = user_config["rep"])
    dmatrix, dmatrix_fct, cmatrix = antidiscr_laws_dmatrix(scaling = "standardize", selected_obs = selected_states, 
                                                           B = user_config["B"], rep = user_config["rep"])
    
    # true hyperparameter values for ideal_expert
    true_values = dict({
        "mu": [2.91, 0.23, -1.51, -0.610],
        "sigma": [0.07, 0.05, 0.135, 0.105]
    })
    
    # model parameters
    parameters_dict = dict()
    for i in range(4):
        parameters_dict[f"beta_{i}"] = {
                "family":  Normal_unconstrained(),
                "true": tfd.Normal(true_values["mu"][i], true_values["sigma"][i]),
                "initialization": [tfd.Uniform(0.,1.), tfd.Normal(tf.math.log(0.1), 0.001)]
                }
        
    # generative model
    class GenerativeModel(tf.Module):
        def __call__(self, 
                     parameters,        # obligatory: samples from prior distributions; tf.Tensor
                     dmatrix,           # required: design matrix; tf.Tensor
                     **kwargs           # obligatory: possibility for further keyword arguments is needed 
                     ):  
    
            # linear predictor
            theta = tf.matmul(dmatrix, parameters[:,:,:,None])
    
            # map linear predictor to theta
            epred = tf.exp(theta)
           
            # define likelihood
            likelihood = tfd.Poisson(
                rate = epred
            )
            
            return dict(likelihood = likelihood,      # obligatory: likelihood; callable
                        ypred = None,                 # obligatory: prior predictive data
                        epred = epred                 # obligatory: samples from linear predictor
                        )
        
    # upper threshold value
    u_thres = threshold
    # specify target quantity, elicitation technique and loss combination
    t1 = target_config(target="group_means", 
                       elicitation = "quantiles", 
                       combine_loss = "by-group", 
                       quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90))
    t2 = target_config(target="y_obs", 
                       elicitation = "histogram", 
                       combine_loss = "by-group")
    
    target_info = target_input(t1, t2)
    
    # simulate from expert
    expert_res_list, prior_pred_res = expert_model(1, user_config["rep"],
                                               parameters_dict, GenerativeModel, target_info,
                                               method = "ideal_expert",
                                               dmatrix = dmatrix_exp,
                                               dmatrix_fct = dmatrix_fct,
                                               cmatrix = cmatrix,
                                               total_count = u_thres)
    
    # learn prior distributions
    res_dict = trainer(expert_res_list, user_config["B"], user_config["rep"],
                   parameters_dict, user_config["method"], GenerativeModel,
                   target_info, user_config, loss_balancing = True, save_vals = ["prior_preds", "elicits"],
                   dmatrix = dmatrix, dmatrix_fct = dmatrix_fct, cmatrix = cmatrix, total_count = u_thres)
    
    return dict({"user_config": user_config, 
                 "res_dict": res_dict, 
                 "expert_res_list": expert_res_list, 
                 "names_states": names_states, 
                 "selected_states": selected_states})

def get_res_tab(res, threshold, last_vals = 30):
    param = tf.stack([res["res_dict"]["hyperparam_info"][0][i] for i in range(res["user_config"]["epochs"])], -1)
    
    df = pd.DataFrame({
        "x": [0,1,2,3],
        "avg_m": tf.reduce_mean(tf.gather(param[:, -last_vals:], [0,2,4,6], axis = 0), axis = -1),
        "avg_sd": tf.math.reduce_mean(tf.gather(tf.exp(param[:, -last_vals:]), [1,3,5,7], axis = 0), axis = 1),
        "std_m": tf.math.reduce_std(tf.gather(param[:, -last_vals:], [0,2,4,6], axis = 0), axis = 1),
        "std_sd": tf.math.reduce_std(tf.gather(tf.exp(param[:, -last_vals:]), [1,3,5,7], axis = 0), axis = 1),
        "t=": [threshold]*4
    })
    return df

def print_res_vary_thres(df, save_fig):
    true_values = dict({
        "mu": [2.91, 0.23, -1.51, -0.610],
        "sigma": [0.07, 0.05, 0.135, 0.105]
    })
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    
    _, axs = plt.subplots(1,2, constrained_layout = True, figsize = (6,2))
    axs[1].scatter([0,1,2,3], true_values["sigma"], s = 30, c = "red", marker = "x", zorder = 10)
    axs[0].scatter([0,1,2,3], true_values["mu"], s = 30, c = "red", marker = "x", zorder = 10)
    sns.pointplot(data = df, x="x", y="avg_m", ax = axs[0], markers = "D", scale = 0.5, linestyles="none", hue = "t=")
    sns.pointplot(data = df, x="x", y="avg_sd", ax = axs[1], markers = "D", scale = 0.5, linestyles="none", hue = "t=")
    axs[1].legend([],[], frameon=False)
    axs[0].legend(handlelength = 0.6, labelspacing = 0.2, loc = "upper left", framealpha = 0.2, fontsize = "small", 
                  ncol = 6, columnspacing = 0.5, handletextpad = 0.5, alignment = "left", title = r"$t_u=$", title_fontsize = "small")
    axs[0].set_ylim(-2, 7)
    axs[0].set_xticks([0,1,2,3], [rf"$\mu_{i}$" for i in range(4)])
    axs[1].set_xticks([0,1,2,3], [rf"$\sigma_{i}$" for i in range(4)])
    [axs[i].set_xlabel(" ") for i in range(2)]
    [axs[i].set_ylabel(" ") for i in range(2)]
    if save_fig:
        plt.savefig('graphics/vary_thres_pois_res.png', dpi = 300)
    else:
        plt.show()
        
def print_res_vary_time(res_list, names_list, save_fig):
    [plt.plot(res["res_dict"]["epoch_time"], label = l) for res,l in zip(res_list, names_list)]
    plt.legend(handlelength = 0.6, labelspacing = 0.2, loc = "lower left", framealpha = 0.2, fontsize = "small", 
                  ncol = 6, columnspacing = 0.5, handletextpad = 0.5, alignment = "left", title = r"$t_u=$", title_fontsize = "small")
    plt.ylim(0,8)
    plt.xlabel("epochs")
    plt.ylabel("time per epoch in sec")
    if save_fig:
        plt.savefig('graphics/vary_thres_pois_time.png', dpi = 300)
    else:
        plt.show()