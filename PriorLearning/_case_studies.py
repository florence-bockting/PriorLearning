import pandas as pd
import tensorflow as tf
import seaborn as sns
import polars as pl
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import patsy as pa

tfd = tfp.distributions

def truth_data_dmatrix(fct1, fct2, N_group):
    # construct design matrix
    X =  pa.dmatrix("a*b", pa.balanced(a = fct1, b = fct2, repeat = N_group), return_type="dataframe")
    dmatrix = tf.cast(X, dtype = tf.float32)
    # extract contrast matrix from design matrix (for allocating observations to groups)
    cmatrix = dmatrix[0:dmatrix.shape[1], :]
    
    return dmatrix, cmatrix

def haberman_data_predictor(scaled, selected_obs):
    d = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/haberman_prep.csv')
    X = tf.constant(d["no_axillary_nodes"], dtype=tf.float32)
    x_sd = tf.math.reduce_std(X)
    if scaled:
        # scale predictor
        X_scaled = tf.constant(X, dtype=tf.float32)/x_sd
        # select only data points that were selected from expert
        dmatrix = tf.gather(X_scaled, selected_obs) 
    else:
        dmatrix = tf.gather(tf.constant(X, dtype=tf.float32), selected_obs)
    
    return dmatrix

def antidiscr_laws_dmatrix(scaling, selected_obs, B, rep):
    X = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/antidis_laws.csv')
    # sort by group and perc_urban in decreasing order
    X_sorted = pd.DataFrame(tf.squeeze(X)).sort_values(by=[2,3,1])
    if scaling == "standardize":
        # standardize metric predictor
        X_sorted[1] = (X_sorted[1] - X_sorted[1].mean())/X_sorted[1].std() 
    else:
        print("currently only scaling = 'standardize' is supported")
    dmatrix = tf.cast(tf.gather(X_sorted, selected_obs, axis = 0), tf.float32)
    dmatrix_fct = tf.gather(dmatrix, indices = [0,2,3], axis = 1)
    cmatrix = X_sorted[[0,2,3]].drop_duplicates()
    
    return dmatrix, dmatrix_fct, cmatrix

def sleep_data_predictor(scaling, N_days, N_subj, selected_days):
    X = tf.cast(tf.tile(tf.range(0., N_days, 1.), [N_subj]), tf.float32)
    if scaling == "standardize":
        X_scaled = (X - tf.reduce_mean(X))/tf.math.reduce_std(X)
    dmatrix_full = tf.stack([tf.ones(len(X_scaled)), X_scaled], axis = -1)
    dmatrix_list = [dmatrix_full[day::N_days] for day in selected_days]
    dmatrix = tf.stack(dmatrix_list, axis=1)
    dmatrix = tf.reshape(dmatrix, (N_subj*len(selected_days), 2))
    cmatrix = pd.DataFrame(dmatrix).drop_duplicates()
    
    return dmatrix, cmatrix

def plot_expert_pred_poisson(expert_res_list, names_states):
    q_exp = expert_res_list["group_means_quant_0"]
    q_exp1 = expert_res_list["y_obs_hist_1"]
    
    df = pl.DataFrame( ) 
    df = df.with_columns(
        q = np.arange(0.1,1., 0.1),
        democrats = pl.Series(q_exp[0][0,:].numpy()),
        swing = pl.Series(q_exp[1][0,:].numpy()),
        republican = pl.Series(q_exp[2][0,:].numpy())
    )
    df = df.melt(id_vars = "q", value_vars = ["democrats", "swing", "republican"], variable_name = "group")
    df_pd = df.to_pandas()
    
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    figs = fig.subfigures(1,2)
    
    axs0 = figs[0].subplots(1,1)
    axs1 = figs[1].subplots(2,3, sharex = True, sharey = True)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    [sns.scatterplot(y = i, x = q_exp[i][0,:], color = c, ax = axs0, zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    sns.boxplot(x = df_pd["value"], y = df_pd["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs0) 
    axs0.set_xlabel("expected #LGBTQ+ anti-discrimination laws")
    [sns.histplot(x=q_exp1[i][0,:], ax = axs1[0,i], stat = "proportion") for i in range(3)]
    [sns.histplot(x=q_exp1[i+3][0,:], ax = axs1[1,i], stat = "proportion") for i in range(3)]
    [axs1[0,i].set_title(t) for i,t in zip(range(3), names_states[:3])]
    [axs1[1,i].set_title(t) for i,t in zip(range(3), names_states[3:])]
    axs1[1,1].set_xlabel("#LGBTQ+ anti-discrimination laws")
    figs[0].suptitle("Quantile-based elicitation", fontweight = "bold" )
    figs[1].suptitle("Histogram-based elicitation", fontweight = "bold" )
    plt.show()
    
def plot_expert_pred(expert_res_list, selected_obs):
    q_exp = expert_res_list
    
    df = pl.DataFrame( ) 
    df = df.with_columns(
        q = np.arange(0.1,1., 0.1)
    )
    for i,j in zip(range(len(selected_obs)),selected_obs):
        df = df.with_columns(pl.Series(f"{j}", q_exp[i][0,:].numpy())) 
    df = df.melt(id_vars = "q", value_vars = [f"{j}" for i,j in zip(range(len(selected_obs)), selected_obs)], variable_name = "no.nodes")
    
    df_pd = df.to_pandas()
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    
    fig, axs = plt.subplots(1,1, layout='constrained', figsize=(6, 4))
    [sns.scatterplot(y = j, x = q_exp[j][0,:], ax = axs, zorder = 1) for j in range(len(selected_obs))]
    sns.boxplot(x = df_pd["value"], y = df_pd["no.nodes"], color=".99",  linewidth=.75, zorder = 0) 
    axs.set_ylabel("number of nodes")
    axs.set_xlabel("$Q_p^G$")
    axs.set_title("Quantile-based elicitation", loc = "left", pad = 10., fontdict = {'fontsize': 14}) 
    plt.show()
    
def tab_expert_pred(expert_res_list, selected_obs):
    return pd.DataFrame({
            "no.axillary.nodes": selected_obs,
            "no.pat.died": np.round(tf.reduce_mean(tf.stack(expert_res_list, -1), (0,1)), 2)
            })

def plot_loss(res_dict_loss):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    axs.plot(np.arange(len(res_dict_loss)), res_dict_loss, color = "#44546A", lw = 3) 
    axs.set_xlabel("epochs")
    axs.set_ylabel(r"$L(\lambda)$")
    axs.set_title("Loss function",pad = 10., fontdict = {'fontsize': 14}, loc = "left")
    plt.show()
    
def plot_priors_binom(avg_res, true_vals):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (6,3))
    x = tf.range(-0.8,0.5,0.01)
    sns.lineplot(x=x,y=tfd.Normal(avg_res[0], avg_res[1]).prob(x), lw = 3, label = rf"$\beta_0 \sim$ N({avg_res[0]:.2f}, {avg_res[1]:.2f})")
    sns.lineplot(x=x,y=tfd.Normal(avg_res[2], avg_res[3]).prob(x), lw = 3, label = rf"$\beta_1 \sim$ N({avg_res[2]:.2f}, {avg_res[3]:.2f})")
    sns.lineplot(x=x,y=tfd.Normal(true_vals[0], true_vals[1]).prob(x), linestyle= "dashed", color = "black")
    sns.lineplot(x=x,y=tfd.Normal(true_vals[2], true_vals[3]).prob(x), linestyle = "dashed", color = "black")
    axs.set_xlabel(r"$\beta_0, \beta_1 \sim$ Normal($\cdot$, $\cdot$)")
    axs.set_ylabel("density") 
    axs.legend(handlelength = 0.4, fontsize = "medium") 
    plt.show()
    
def plot_priors(avg_res, true_vals, xrange, loc):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (6,3))
    x = tf.range(xrange[0], xrange[1], 0.01)
    for j,i in enumerate(np.arange(0, len(avg_res), 2)):
        sns.lineplot(x=x,y=tfd.Normal(avg_res[i], avg_res[i+1]).prob(x), lw = 3, label = rf"$\beta_{j} \sim$ N({avg_res[i]:.2f}, {avg_res[i+1]:.2f})")
        sns.lineplot(x=x,y=tfd.Normal(true_vals[i], true_vals[i+1]).prob(x), linestyle= "dashed", color = "black")
    axs.set_xlabel(r"$\beta_k \sim$ Normal($\cdot$, $\cdot$)")
    axs.set_ylabel("density") 
    axs.legend(handlelength = 0.4, fontsize = "medium", loc = loc, labelspacing = 0.1) 
    plt.show()
    
def plot_priors_normal(avg_res, true_values):
    _, axes = plt.subplots(2,4, constrained_layout = True, figsize = (8,4))
    [sns.kdeplot(tfd.Normal(avg_res[i], avg_res[i+1]).sample(1000), 
                 ax = axes[0,s], lw=3) for s,i in enumerate([0,2,4,6])]
    [sns.kdeplot(tfd.Normal(true_values["mu"][s], true_values["sigma"][s]).sample(1000), 
                 ax = axes[0,s], color = "black", linestyle = "dashed") for s in range(4)]
    [sns.kdeplot(tfd.Normal(avg_res[i], avg_res[i+1]).sample(1000), 
                 ax = axes[1,s], lw=3) for s,i in enumerate([8,10])]
    [sns.kdeplot(tfd.Normal(true_values["mu"][s], true_values["sigma"][s]).sample(1000), 
                 ax = axes[1,i], color = "black", linestyle = "dashed") for i,s in enumerate(range(4,6))]
    sns.kdeplot(tfd.Exponential(avg_res[-1]).sample(1000), lw = 3, ax = axes[1,2])
    sns.kdeplot(tfd.Exponential(true_values["nu"]).sample(1000), ax = axes[1,2], 
                color = "black", linestyle = "dashed")
    [axes[0,i].set_xlabel(rf"$\beta_{i}$") for i in range(4)]
    [axes[1,j].set_xlabel(rf"$\beta_{i}$") for j,i in enumerate(range(4,6))]
    axes[1,2].set_xlabel(r"$s$")
    axes[1,3].set_axis_off()


def expert_pred_elicits_normal(expert_res_list, save_fig): 
    q_exp = expert_res_list["marginal_EnC_quant_1"]
    q_exp2 = expert_res_list["marginal_ReP_quant_0"]
    q_exp3 = expert_res_list["mean_effects_quant_2"]
    
    df = pl.DataFrame( ) 
    df = df.with_columns(
        q = np.arange(0.1,1., 0.1),
        deep = pl.Series(q_exp[0][0,:].numpy()),
        standard = pl.Series(q_exp[1][0,:].numpy()),
        shallow = pl.Series(q_exp[2][0,:].numpy())
    )
    df = df.melt(id_vars = "q", value_vars = ["deep", "standard", "shallow"], variable_name = "group")
    df_pd = df.to_pandas()
    
    df2 = pl.DataFrame( ) 
    df2 = df2.with_columns(
        q = np.arange(0.1,1., 0.1),
        repeated = pl.Series(q_exp2[0][0,:].numpy()),
        new = pl.Series(q_exp2[1][0,:].numpy())
    )
    df2 = df2.melt(id_vars = "q", value_vars = ["repeated", "new"], variable_name = "group")
    df_pd2 = df2.to_pandas()
    
    df3 = pl.DataFrame( ) 
    df3 = df3.with_columns(
        q = np.arange(0.1,1., 0.1),
        deep = pl.Series(q_exp3[0][0,:].numpy()),
        standard = pl.Series(q_exp3[1][0,:].numpy()),
        shallow = pl.Series(q_exp3[1][0,:].numpy())
    )
    df3 = df3.melt(id_vars = "q", value_vars = ["deep", "standard", "shallow"], variable_name = "group")
    df_pd3 = df3.to_pandas()
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })
    
    fig, axs = plt.subplots(1,4, layout='constrained', figsize=(6.5, 2))
    sns.histplot(expert_res_list["R2_hist_3"][0,:], ax = axs[0], stat = "proportion", color = "#44546A")
    [sns.scatterplot(y = i, x = q_exp[i][0,:], color = c, ax = axs[1], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp2[i][0,:], color = c, ax = axs[2], zorder = 1) for i,c in zip(range(2), ["#020024", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp3[i][0,:], color = c, ax = axs[3], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    sns.boxplot(x = df_pd["value"], y = df_pd["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[1]) 
    sns.boxplot(x = df_pd2["value"], y = df_pd2["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[2])
    sns.boxplot(x = df_pd3["value"], y = df_pd3["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[3]) 
    axs[0].set_xlim(0,1)
    [axs[i].set_yticklabels(labels = l, fontsize = 10, rotation=45, ha='right') for l,i in zip([["dep", "std", "shw"],["rep", "new"],["dep", "std", "shw"]],[1,2,3])]
    axs[0].set_xlabel(r"${R^2}^{(s)}$")
    [axs[i].set_title(l, size = "small", loc = "left") for i,l in zip([1,2,3], ["Encoding depth", "Repetition", r"$\Delta$PTJ (Rep-New)"])]
    [axs[i].set_xlabel("$Q_p^G$") for i in [1,2,3]]
    [axs[i].set_ylabel(" ") for i in [1,2,3]]
    [axs[i].set_title(t, pad = 10., size = "medium") for i,t in zip([0,2], ["Histogram-based elicitation \n", " Quantile-based elicitation \n"])]
    [axs[i].tick_params(axis='x', labelsize=10) for i in range(4)]
    if save_fig:
        plt.savefig('graphics/linreg_target_quant.png', dpi = 300)
    else:
        plt.show()    
    
def print_target_info(target_info):
    print_tab  = target_info.copy()
    for specs in ["moments_specs", "quantiles_specs"]:
        if specs in list(target_info.keys()):
            index_mom = tf.where(tf.equal(target_info["elicitation"], specs[0:-6]))[:,0].numpy()
            print_tab[specs] = [None]*len(target_info["elicitation"])
            for i,j in enumerate(index_mom):
             print_tab[specs][j] = target_info[specs][i]
    
    return pd.DataFrame(print_tab)
        
def plot_hyp_pois(res_dict, user_config):
    
    betas = tf.stack([res_dict["hyperparam_info"][0][i][0::2] for i in range(user_config["epochs"])], -1)
    sigmas =  tf.stack([tf.exp(res_dict["hyperparam_info"][0][i][1::2]) for i in range(user_config["epochs"])], -1)
    
    _, axs = plt.subplots(1,2, constrained_layout = True, figsize = (8,3), sharex = True)
    [axs[0].plot(betas[i,:], lw = 3, color = c, label = fr"$\mu_{i}$") for i,c in zip(range(4), ["#020024", "#44546A", "#00d4ff", "#00d4ff"])]
    [axs[1].plot(sigmas[i,:], lw = 3, color = c, label = fr"$\sigma_{i}$") for i,c in zip(range(4), ["#020024", "#44546A", "#00d4ff", "#00d4ff"])]
    [axs[i].legend(handlelength = 0.4, fontsize = "medium", ncol = 2, columnspacing = 0.5) for i in range(2)]
    axs[0].set_title(r"$\mu_0, \mu_1, \mu_2, \mu_3$",pad = 10., fontdict = {'fontsize': 14}, loc = "left")
    axs[1].set_title(r"$\sigma_0, \sigma_1, \sigma_2, \sigma_3$",pad = 10., fontdict = {'fontsize': 14}, loc = "left")
    axs[1].set_xlabel("epochs")
    plt.show()
    
def plot_learned_prior_linreg(res_dict, true_vals, avg_res, epochs, save_fig):
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_nu = "#a6b7c6"
    true_vals_raw = []
    for i in range(len(true_vals)):
        if i in [0,2,4,6,8,10]:
            true_vals_raw.append(true_vals[i])
        else:
            true_vals_raw.append(tf.math.log(true_vals[i]))

    err = tf.stack([tf.subtract(res_dict["hyperparam_info"][0][i], true_vals_raw) for i in range(epochs)], -1)

    fig = plt.figure(layout='constrained', figsize=(6, 4))
    figs = fig.subfigures(2,1)
    
    axs0 = figs[0].subplots(1,2, gridspec_kw = {"width_ratios": (2,1)})
    axs1 = figs[1].subplots(1,3, sharex = True)
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Computer Modern")
    
    x_rge = np.arange(-0.4, 0.3, 0.01)
    [axs0[0].plot(x_rge, tfd.Normal(true_vals[i],true_vals[i+1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black") for i in [0,2,4,6,8,10]]
    [axs0[0].plot(x_rge, tfd.Normal(avg_res[i],avg_res[i+1]).prob(x_rge), lw = 3, color = col_betas[j], 
                  alpha = 0.6, label = fr"$\beta_{j} \sim N$({avg_res[i]:.2f}, {avg_res[i+1]:.2f})") for j,i in enumerate([0,2,4,6,8,10])]
    axs0[0].legend(handlelength = 0.6, labelspacing = 0.2, loc = (0.02,0.15), frameon = False, fontsize = "x-small")
    axs0[0].set_title("model parameters", fontfamily = "Computer Modern", size = "small")
    axs0[0].set_xlabel(r"$\beta_k$")
    
    x_rge = np.arange(0, 5, 0.01)
    axs0[1].plot(x_rge, tfd.Exponential(true_vals[-1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black")
    axs0[1].plot(x_rge, tfd.Exponential(avg_res[-1]).prob(x_rge), lw = 3, color = col_nu, 
                  alpha = 0.6, label = fr"$s \sim Exp$({avg_res[-1]:.2f})")
    axs0[1].legend(handlelength = 0.6, labelspacing = 0.2, loc = "upper right", frameon = False, fontsize = "x-small")
    axs0[1].set_title("random noise", size = "small")
    axs0[1].set_xlabel(r"$s$")
    
    x_rge = np.arange(0, epochs)
    [axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") for i in range(3)]
    axs1[0].plot(x_rge, err[-1], color = col_nu, lw = 3, alpha = 0.6)
    [axs1[1].plot(x_rge, err[i], color = col_betas[j], lw = 3, alpha = 0.6,
                 label = rf"$\mu_{j}$") for j,i in enumerate([0,2,4,6,8,10])]
    [axs1[2].plot(x_rge, err[i+1], color = col_betas[j], lw = 3, alpha = 0.6,
                 label = rf"$\sigma_{j}$") for j,i in enumerate([0,2,4,6,8,10])]
    [axs1[i].legend(handlelength = 0.6, labelspacing = 0.2, loc = (0.5,0.6), frameon = False, fontsize = "x-small", ncol = 2, 
                    columnspacing = 1.) for i in [1,2]]
    [axs1[i].set_title(t, fontsize = "medium") for i,t in zip(range(3), [r"$\nu$", r"$\mu_k$", r"$\sigma_k$"])]
    axs1[1].set_xlabel("epochs", size = "small")
    figs[1].suptitle("Error between true and learned hyperparameter", x = 0.05, ha = "left", size = "medium")
    figs[0].suptitle("Learned prior distributions", x = 0.05, ha = "left", size = "medium")
    if save_fig:
        plt.savefig('graphics/linreg_summary_results.png', dpi = 300)
    else:
        plt.show()        

def plot_diagnostics_linreg(res_dict, user_config, save_fig):
    col_nu = "#a6b7c6"
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    grads_total = tf.stack([res_dict["grads_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss = tf.stack([res_dict["loss_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss_indiv = tf.stack([res_dict["loss_indiv_info"][i] for i in range(user_config["epochs"])]  , -1)
    learned_hyp = tf.stack([res_dict["hyperparam_info"][0][i] for i in range(user_config["epochs"])]  , -1)
    
    x_rge = np.arange(0,user_config["epochs"],1)
    
    _, axs = plt.subplots(2,4, constrained_layout = True, sharex = True, 
                          figsize = (6.5,2.5), gridspec_kw = {"hspace": 0.2})
    
    axs[0,0].plot(x_rge, loss)
    axs[0,0].set_title("Total loss", loc = "left", size = "medium")
    
    sns.scatterplot(x = x_rge, y = grads_total[-1], ax = axs[0,1], marker = "x", s = 5, color = "black", alpha =.6)
    axs[0,1].set_title(r"Gradients: $\nu$", loc = "left", size = "medium")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i], ax = axs[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6,8,10]]
    axs[0,2].set_title(r"$\mu_k$")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i+1], ax = axs[0,3], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6,8,10]]
    axs[0,3].set_title(r"$\sigma_k$")
    
    [axs[1,0].plot(x_rge, loss_indiv[i,:]) for i in range(loss_indiv.shape[0])]
    axs[1,0].set_title("Individual losses", size = "medium")
    
    axs[1,1].plot(x_rge, tf.exp(learned_hyp[-1,:]), color = col_nu, lw = 3, alpha = 0.6)
    axs[1,1].set_title(r"Convergence:", loc = "left", size = "medium")
    
    [axs[1,2].plot(x_rge, learned_hyp[j,:], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\mu_{i}$") for i,j in enumerate([0,2,4,6,8,10])]
    
    [axs[1,3].plot(x_rge, tf.exp(learned_hyp[j+1,:]), color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\sigma_{i}$") for i,j in enumerate([0,2,4,6,8,10])]
    
    [axs[1,i].legend(handlelength = 0.3, labelspacing = 0.1, loc = (0.4,0.6), frameon = False, fontsize = "x-small", ncol = 2, 
                        columnspacing = 0.2) for i in [2,3]]
    axs[1,2].set_ylim(-0.25, 0.7)
    [axs[1,i].set_xlabel("epochs", size = "small") for i in range(4)]
    [axs[1,i].tick_params(axis='x', labelsize=10) for i in range(4)]
    if save_fig:
        plt.savefig('graphics/linreg_diagnostics.png', dpi = 300)
    else:
        plt.show()  
        
def plot_learned_prior_binom(res_dict, user_config, true_vals, avg_res, 
                             expert_res_list, selected_obs, save_fig):
    
    epochs = user_config["epochs"]
    
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    q_cols_total = ["#49c1db", "#3faed9", "#499ad2", "#5c84c5", "#6f6db2", "#7c5697", "#833e78", "#822556", "#780d33"]
    
    true_vals_raw = []
    for i in range(len(true_vals)):
        if i in [0,2]:
            true_vals_raw.append(true_vals[i])
        else:
            true_vals_raw.append(tf.math.log(true_vals[i]))
    
    err = tf.stack([tf.subtract(res_dict["hyperparam_info"][0][i], true_vals_raw) for i in range(epochs)], -1)
    prior_pred_exp = tf.squeeze(tf.stack(expert_res_list['y_obs_quant_0'], -1))
    prior_pred_mod = tf.reduce_mean(res_dict["priors_info"][0]['y_obs_quant_0'], 0) 
    
    fig = plt.figure(layout='constrained', figsize=(6, 3.5))
    figs = fig.subfigures(2,1, height_ratios = (1,1.5))
    
    axs0 = figs[0].subplots(1,7, gridspec_kw = {"hspace": 0.2})
    axs1 = figs[1].subplots(1,3, width_ratios = (1,1,2))
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Computer Modern")
    
    [axs0[i].axline(xy1 = (prior_pred_mod[0,i], prior_pred_exp[0,i]), 
                xy2 = (prior_pred_mod[-1,i], prior_pred_exp[-1,i]), color = "black", lw = 2, 
                linestyle = "dashed", zorder = 1) for i in range(7)]
    [sns.scatterplot(x = prior_pred_mod[:,i], y = prior_pred_exp[:,i], color = q_cols_total, ax = axs0[i], zorder = 2) for i in range(7)]
    [axs0[i].tick_params(left = False, right = False , labelleft = False , 
                     labelbottom = False, bottom = False) for i in range(7)]
    [axs0[i].set_title(fr"$x = {selected_obs[i]}$") for i in range(7)]
    axs0[3].set_xlabel("model-based q. \n", size = "small")
    axs0[0].set_ylabel("expert-elicited q. \n", size = "small")
    figs[0].suptitle("Prior predictions: Model-based vs. expert-elicited quantiles", ha = "left", x = 0.06, size = "medium")
    
    x_rge = np.arange(-1.5, 0.9, 0.01)
    [axs1[2].plot(x_rge, tfd.Normal(true_vals[i],true_vals[i+1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black") for i in [0,2]]
    [axs1[2].plot(x_rge, tfd.Normal(avg_res[i],avg_res[i+1]).prob(x_rge), lw = 3, color = col_betas[j], 
              alpha = 0.6, label = fr"$\beta_{j} \sim N$({avg_res[i]:.2f}, {avg_res[i+1]:.2f})") for j,i in enumerate([0,2])]
    axs1[2].legend(handlelength = 0.6, labelspacing = 0.2, loc = (0.02, 0.7), frameon = False, fontsize = "small")
    axs1[2].set_title("Learned prior distributions \n", loc = "left", size = "medium")
    axs1[2].set_xlabel(r"model parameters $\beta_k$")
    
    x_rge = np.arange(0, epochs)
    [axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") for i in range(2)]
    [axs1[0].plot(x_rge, err[i], color = col_betas[j], lw = 3, alpha = 0.6,
             label = rf"$\mu_{j}$") for j,i in enumerate([0,2])]
    [axs1[1].plot(x_rge, err[i+1], color = col_betas[j], lw = 3, alpha = 0.6,
             label = rf"$\sigma_{j}$") for j,i in enumerate([0,2])]
    [axs1[i].legend(handlelength = 0.6, labelspacing = 0.2, loc = "upper right", frameon = False, fontsize = "small", 
                columnspacing = 1.) for i in [0,1]]
    [axs1[i].set_xlabel("epochs") for i in range(2)]
    axs1[0].set_title("Error between true and learned \n hyperparameter", ha = "left", x = 0., size = "medium")
    if save_fig:
        plt.savefig('graphics/binom_summary_results.png', dpi = 300)
    else:
        plt.show()

def plot_diagnostics_binom(res_dict, user_config, save_fig):

    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    grads_total = tf.stack([res_dict["grads_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss = tf.stack([res_dict["loss_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss_indiv = tf.stack([res_dict["loss_indiv_info"][i] for i in range(user_config["epochs"])]  , -1)
    learned_hyp = tf.stack([res_dict["hyperparam_info"][0][i] for i in range(user_config["epochs"])]  , -1)
    
    x_rge = np.arange(0,user_config["epochs"],1)
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })
    
    _, axs = plt.subplots(2,3, constrained_layout = True, sharex = True, figsize = (6,3.), gridspec_kw = {"hspace": 0.2})
    
    axs[0,0].plot(x_rge, loss)
    axs[0,0].set_title("Total loss", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i], ax = axs[0,1], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2]]
    axs[0,1].set_title(r"Gradients: $\mu_k$", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i+1], ax = axs[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2]]
    axs[0,2].set_title(r"$\sigma_k$")
    
    [axs[1,0].plot(x_rge, loss_indiv[i,:]) for i in range(loss_indiv.shape[0])]
    axs[1,0].set_title("Individual losses", loc = "left")
    
    [axs[1,1].plot(x_rge, learned_hyp[j,:], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\mu_{i}$") for i,j in enumerate([0,2])]
    axs[1,1].set_title(r"Convergence: $\mu_k$", loc = "left")
    
    [axs[1,2].plot(x_rge, learned_hyp[j+1,:], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\sigma_{i}$") for i,j in enumerate([0,2])]
    axs[1,2].set_title(r"$\sigma_k$")
    
    [axs[1,i].legend(handlelength = 0.6, labelspacing = 0.2, loc = l, frameon = False, fontsize = "small", ncol = 2, 
                        columnspacing = 1.) for i,l in zip([1,2], ["center right", "upper right"])]
    [axs[1,i].set_xlabel("epochs") for i in range(3)]
    if save_fig:
        plt.savefig('graphics/binom_diagnostics.png', dpi = 300)
    else:
        plt.show()
        
        

def plot_diagnostics_pois(res_dict, user_config, save_fig):

    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    grads_total = tf.stack([res_dict["grads_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss = tf.stack([res_dict["loss_info"][i] for i in range(user_config["epochs"])]  , -1)
    loss_indiv = tf.stack([res_dict["loss_indiv_info"][i] for i in range(user_config["epochs"])]  , -1)
    learned_hyp = tf.stack([res_dict["hyperparam_info"][0][i] for i in range(user_config["epochs"])]  , -1)
    
    x_rge = np.arange(0,user_config["epochs"],1)
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern"
    })
    
    _, axs = plt.subplots(2,3, constrained_layout = True, sharex = True, figsize = (6,3.), gridspec_kw = {"hspace": 0.2})
    
    axs[0,0].plot(x_rge, loss)
    axs[0,0].set_title("Total loss", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i], ax = axs[0,1], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6]]
    axs[0,1].set_title(r"Gradients: $\mu_k$", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = grads_total[i+1], ax = axs[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6]]
    axs[0,2].set_title(r"$\sigma_k$")
    
    [axs[1,0].plot(x_rge, loss_indiv[i,:]) for i in range(loss_indiv.shape[0])]
    axs[1,0].set_title("Individual losses", loc = "left")
    
    [axs[1,1].plot(x_rge, learned_hyp[j,:], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\mu_{i}$") for i,j in enumerate([0,2,4,6])]
    axs[1,1].set_title("Convergence:", loc = "left")
    axs[1,1].set_ylim(-2.,4.5)
    
    [axs[1,2].plot(x_rge, tf.exp(learned_hyp[j+1,:]), color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\sigma_{i}$") for i,j in enumerate([0,2,4,6])]
    axs[1,2].set_ylim(0.,0.4)
    
    [axs[1,i].legend(handlelength = 0.3, labelspacing = 0.1, loc = l, frameon = False, fontsize = "small", ncol = 4, 
                    columnspacing = 0.2) for i,l in zip([1,2], [(0.05,0.8), (0.05,0.8)])]
    [axs[1,i].set_xlabel("epochs", size = "small") for i in range(3)]
    [axs[1,i].tick_params(axis='x', labelsize=10) for i in range(3)]
    
    if save_fig:
        plt.savefig('graphics/pois_diagnostics.png', dpi = 300)
    else:
        plt.show()
        
def plot_learned_prior_pois(res_dict, user_config, true_vals, avg_res, 
                             expert_res_list, names_states, selected_states, save_fig):
    
    epochs = user_config["epochs"]
    
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    q_cols_total = ["#49c1db", "#3faed9", "#499ad2", "#5c84c5", "#6f6db2", "#7c5697", "#833e78", "#822556", "#780d33"]
    
    true_vals_raw = []
    for i in range(len(true_vals)):
        if i in [0,2,4,6]:
            true_vals_raw.append(true_vals[i])
        else:
            true_vals_raw.append(tf.math.log(true_vals[i]))
    
    err = tf.stack([tf.subtract(res_dict["hyperparam_info"][0][i], true_vals_raw) for i in range(epochs)], -1)
    prior_pred_exp = tf.squeeze(tf.stack(expert_res_list['group_means_quant_0'], -1))
    prior_pred_mod = tf.reduce_mean(res_dict["priors_info"][0]['group_means_quant_0'], 0) 
    prior_pred_exp_hist = tf.squeeze(tf.stack(expert_res_list['y_obs_hist_1'], -1))
    prior_pred_mod_hist = res_dict["priors_info"][0]['y_obs_hist_1'][-1,:,:] 
    
    fig = plt.figure(layout='constrained', figsize=(6., 4))
    figs = fig.subfigures(2,1, height_ratios = (1.4,1)) 
    
    axs0 = figs[0].subplots(2,5)
    axs1 = figs[1].subplots(1,3, width_ratios = (1,1,2.5))
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Computer Modern")
    
    [axs0[0,i].axline(xy1 = (prior_pred_mod[0,i], prior_pred_exp[0,i]), 
                    xy2 = (prior_pred_mod[-1,i], prior_pred_exp[-1,i]), color = "black", lw = 2, 
                    linestyle = "dashed", zorder = 1) for i in range(3)]
    [sns.scatterplot(x = prior_pred_mod[:,i], y = prior_pred_exp[:,i], color = q_cols_total, 
                     ax = axs0[0,i], zorder = 2) for i in range(3)]
    [axs0[0,i].tick_params(left = False, right = False , labelleft = False , 
                         labelbottom = False, bottom = False) for i in range(4)]
    axs0[0,4].tick_params(left = False, right = False , labelleft = False) 
    [axs0[0,i].set_title(f"{t}", size = "small") for i,t in zip(range(3),["Democrats","Swing","Republican"])]
    axs0[0,4].set_title(fr"{names_states[0]} ($x_{{{selected_states[0]}}}$)", size = "small")
    axs0[0,3].remove()
    axs0[0,1].set_xlabel("model-based q. \n", size = "small")
    axs0[0,0].set_ylabel("expert-elicited q. \n", size = "small")
    
    sns.histplot(prior_pred_exp_hist[:,0], bins = 30, ax = axs0[0,4])
    sns.histplot(prior_pred_mod_hist[:,0], bins = 30, ax = axs0[0,4], alpha = 0.6)
    [sns.histplot(prior_pred_exp_hist[:,j], bins = 30, ax = axs0[1,i]) for i,j in enumerate(range(1,6))]
    [sns.histplot(prior_pred_mod_hist[:,j], bins = 30, ax = axs0[1,i], alpha = 0.6) for i,j in enumerate(range(1,6))]
    [axs0[1,i].tick_params(left = False, right = False , labelleft = False) for i in range(5)]
    [axs0[1,i].set_ylabel(" ") for i in range(5)]
    [axs0[1,i].set_title(fr"{t} ($x_{{{selected_states[i+1]}}}$)", size = "small") for i,t in enumerate(names_states[1:])]
    axs0[1,2].set_xlabel("\# LGBTQ+ antidiscrimination laws ", size = "x-small")
    axs0[1,0].set_ylabel("counts \n", size = "small")
    figs[0].suptitle("Prior predictions: Model-based vs. expert-elicited quantiles", ha = "left", x = 0.06, size = "medium")

    x_rge = np.arange(-3, 4, 0.01)
    [axs1[2].plot(x_rge, tfd.Normal(true_vals[i],true_vals[i+1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black") for i in [0,2,4,6]]
    [axs1[2].plot(x_rge, tfd.Normal(avg_res[i],avg_res[i+1]).prob(x_rge), lw = 3, color = col_betas[j], 
              alpha = 0.6, label = fr"$\beta_{j} \sim N$({avg_res[i]:.2f}, {avg_res[i+1]:.2f})") for j,i in enumerate([0,2,4,6])]
    axs1[2].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.02, 0.35), frameon = False, fontsize = "x-small")
    axs1[2].set_title("Learned prior distributions \n", loc = "left", size = "medium")
    axs1[2].set_xlabel(r"model parameters $\beta_k$", size = "small")
    
    x_rge = np.arange(0, epochs)
    [axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") for i in range(2)]
    [axs1[0].plot(x_rge, err[i], color = col_betas[j], lw = 3, alpha = 0.6,
             label = rf"$\mu_{j}$") for j,i in enumerate([0,2,4,6])]
    [axs1[1].plot(x_rge, err[i+1], color = col_betas[j], lw = 3, alpha = 0.6,
             label = rf"$\sigma_{j}$") for j,i in enumerate([0,2,4,6])]
    [axs1[i].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.3, 0.5), ncol = 2, frameon = False, fontsize = "small", 
                columnspacing = .5) for i in [0,1]]
    [axs1[i].set_xlabel("epochs", size = "small") for i in range(2)]
    axs1[2].set_xlim((-3.5, 3.5))
    axs1[0].set_title("Error between true and learned \n hyperparameter", ha = "left", x = 0., size = "medium")
    if save_fig:
        plt.savefig('graphics/pois_summary_results.png', dpi = 300)
    else:
        plt.show()

        
