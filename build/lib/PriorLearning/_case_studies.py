import pandas as pd
import tensorflow as tf
import seaborn as sns
import polars as pl
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import patsy as pa

from statsmodels.graphics.boxplots import violinplot

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

def antidiscr_laws_dmatrix(standardize, selected_obs, B, rep):
    X = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/antidis_laws.csv')
    # sort by group and perc_urban in decreasing order
    df = pd.DataFrame(tf.squeeze(X)).sort_values(by=[2,3,1])
    if standardize:
        # standardize metric predictor
        df[1] = (df[1] - df[1].mean())/df[1].std() 
        # reshape model matrix and create tensor
        dmatrix = tf.cast(tf.gather(df, selected_obs), tf.float32)
    else:
        dmatrix = tf.cast(tf.gather(df, selected_obs), tf.float32)
        
    dmatrix = tf.broadcast_to(
        dmatrix[None,None,:,:], (B, rep, dmatrix.shape[0], dmatrix.shape[1] )
        )
    return dmatrix

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
    
def plot_priors(avg_res, true_vals, xrange):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (6,3))
    x = tf.range(xrange[0], xrange[1], 0.01)
    for j,i in enumerate(np.arange(0, len(avg_res), 2)):
        sns.lineplot(x=x,y=tfd.Normal(avg_res[i], avg_res[i+1]).prob(x), lw = 3, label = rf"$\beta_{j} \sim$ N({avg_res[i]:.2f}, {avg_res[i+1]:.2f})")
        sns.lineplot(x=x,y=tfd.Normal(true_vals[i], true_vals[i+1]).prob(x), linestyle= "dashed", color = "black")
    axs.set_xlabel(r"$\beta_k \sim$ Normal($\cdot$, $\cdot$)")
    axs.set_ylabel("density") 
    axs.legend(handlelength = 0.4, fontsize = "medium") 
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
    
def expert_pred_elicits_normal(expert_res_list): 
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
    
    fig, axs = plt.subplots(1,4, layout='constrained', figsize=(10, 3))
    sns.histplot(expert_res_list["R2_hist_3"][0,:], ax = axs[0], stat = "proportion", color = "#44546A")
    [sns.scatterplot(y = i, x = q_exp[i][0,:], color = c, ax = axs[1], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp2[i][0,:], color = c, ax = axs[2], zorder = 1) for i,c in zip(range(2), ["#020024", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp3[i][0,:], color = c, ax = axs[3], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    sns.boxplot(x = df_pd["value"], y = df_pd["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[1]) 
    sns.boxplot(x = df_pd2["value"], y = df_pd2["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[2])
    sns.boxplot(x = df_pd3["value"], y = df_pd3["group"], color=".99",  linewidth=.75, zorder = 0, ax = axs[3]) 
    axs[0].set_xlim(0,1)
    [axs[i].set_yticklabels(labels = l, fontsize = 9, rotation=45, ha='right') for l,i in zip([["deep", "standard", "shallow"],["repeated", "new"],["deep", "standard", "shallow"]],[1,2,3])]
    axs[0].set_xlabel(r"${R^2}^{(s)}$")
    [axs[i].set_title(l, fontdict = {'fontsize': 10}, loc = "left") for i,l in zip([1,2,3], ["Encoding depth", "Repetition", "Truth Effect"])]
    [axs[i].set_xlabel("$Q_p^G$") for i in [1,2,3]]
    [axs[i].set_ylabel(" ") for i in [1,2,3]]
    [axs[i].set_title(t, pad = 10., fontdict = {'fontsize': 12, 'fontweight': "bold"}) for i,t in zip([0,2], ["Histogram-based elicitation \n", " Quantile-based elicitation \n"])]
    plt.show()
    
def sleep_data_predictor(scaled, N_subj, selected_obs):
    X = tf.cast(tf.tile(tf.range(0., 10, 1.), [N_subj]), tf.float32)
    x_sd = tf.math.reduce_std(X)
    if scaled:
        # scale predictor
        X_scaled = tf.constant(X, dtype=tf.float32)/x_sd
        # select only data points that were selected from expert
        dmatrix = tf.gather(X_scaled, selected_obs) 
    else:
        dmatrix = tf.gather(tf.constant(X, dtype=tf.float32), selected_obs)
    
    return dmatrix    
    