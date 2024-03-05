.. Make-My-Prior documentation master file, created by
   sphinx-quickstart on Mon Oct 30 10:23:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Learn Prior distributions from expert knowledge
###############################################

Short overview
==============

We propose a new elicitation method for translating knowledge from a domain expert into an appropriate parametric prior distribution `(Bockting et al., 2023) <https://arxiv.org/abs/2308.11672>`_. 
In particular, our approach builds on recent contributions made by `Hartmann et al. (2020) <http://proceedings.mlr.press/v124/hartmann20a/hartmann20a.pdf>`_, `da Silva et al. (2019) <https://www.jmlr.org/papers/volume24/21-0623/21-0623.pdf>`_, and `Manderson & Goudie (2023) <https://arxiv.org/pdf/2303.08528>`_.
Their key commonality is the development of (more or less) model-agnostic methods in which the search for appropriate prior distributions is formulated as an optimization problem. 
Thus, the objective is to determine the optimal hyperparameters that minimize the discrepancy between model-implied and expert-elicited statistics. 
We also adopt this perspective and introduce a novel elicitation method that supports expert feedback in both the space of parameters and observable quantities (i.e., a hybrid approach) 
and minimizes human effort. The key ideas underlying our method are visualized in the following figure and outlined below.

.. image:: _static/conceptTemplate.png
  :width: 800
  :alt: Visual representation workflow


#. The analyst defines a generative model comprising a likelihood function :math:`p(y \mid \theta)` and a parametric prior distribution :math:`p(\theta \mid \lambda)` for the model parameters, where :math:`\lambda` represents the prior hyperparameters to be inferred from expert knowledge.
#. The analyst selects a set of target quantities, which may involve queries related to observable quantities (data), model parameters, or anything else in between.
#. The domain expert is queried using a specific elicitation technique for each target quantity (*expert-elicited statistics*).
#. From the generative model implied by likelihood and prior and a given value of :math:`\lambda`, parameters and (prior) predictive data are simulated, and the predefined set of target quantities is computed based on the simulations (*model-implied quantities*).
#. The discrepancy between the model-implied and the expert-elicited statistics is evaluated with a discrepancy measure (loss function). 
#. Stochastic gradient descent is employed to update the hyperparameters :math:`\lambda` so as to minimize the loss function.
#. Steps 4 to 6 are repeated iteratively until an optimal set of hyperparameters :math:`\lambda` is found that minimizes the discrepancy between the model-implied and the expert-elicited statistics.

References
==========

* `Bockting F., Radev, S. T., & Bürkner P. C. (2023). Simulation-Based Prior Knowledge Elicitation for Parametric Bayesian Models. ArXiv preprint. <https://arxiv.org/abs/2308.11672>`_
* `da Silva, E. D. S., Kuśmierczyk, T., Hartmann, M., & Klami, A. (2023). Prior Specification for Bayesian Matrix Factorization via Prior Predictive Matching. Journal of Machine Learning Research, 24(67), 1-51. <https://www.jmlr.org/papers/volume24/21-0623/21-0623.pdf>`_
* `Hartmann, M., Agiashvili, G., Bürkner, P., & Klami, A. (2020). Flexible prior elicitation via the prior predictive distribution. In Conference on Uncertainty in Artificial Intelligence (pp. 1129-1138). PMLR. <http://proceedings.mlr.press/v124/hartmann20a/hartmann20a.pdf>`_
* `Manderson, A. A., & Goudie, R. J. (2023). Translating predictive distributions into informative priors. ArXiv preprint. <https://arxiv.org/pdf/2303.08528>`_
* `Mikkola, P., Martin, O. A., Chandramouli, S., Hartmann, M., Pla, O. A., Thomas, O., ... & Klami, A. (2021). Prior knowledge elicitation: The past, present, and future. ArXiv preprint. <https://arxiv.org/pdf/2308.11672>`_

Contents
========

.. toctree::
   :maxdepth: 2

   Home <self>
   API <modules>
   Case Studies <case_studies>

