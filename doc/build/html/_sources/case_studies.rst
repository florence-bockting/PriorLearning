.. Make-My-Prior documentation master file, created by
   sphinx-quickstart on Mon Oct 30 10:23:38 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Case Studies
############

Normal regression model
=======================

.. image:: _static/icon_cs_normal.png
  :width: 400
  :alt: Normal regression model

Normal regression model with a 2 x 3 between-participant factorial design. The goal is to learn 13 hyperparameter values. 
Case study is inspired by the experimental design of truth-effect studies in the field of social cognition.

+ *Target quantities:*  Marginal distribution for both factors (1,2), distribution of effects for each level of the second factor (3), distribution of the expected R2 (4)
+ *Elicited statistics:* Quantile-based elicitation for (1-3) and histogram elicitation for (4). 

:doc:`Go to the case study <case_studies/cs_normal>`

Binomial regression model
==========================

.. image:: _static/icon_cs_binomial.png
  :width: 400
  :alt: Binomial regression model

Binomial regression model with logit link and one continuous predictor. The goal is to learn 4 hyperparameter values. 
Case study uses the Haberman’s survival dataset which contains cases from a study on the survival
of patients who had undergone surgery for breast cancer.

+ *Target quantities:*  expected number of patients who died within five years for different numbers of axillary nodes
+ *Elicited statistics:* Quantile-based elicitation for each selected design point

:doc:`Go to the case study <case_studies/cs_binomial>`

Poisson regression model
==========================

.. image:: _static/icon_cs_poisson.png
  :width: 400
  :alt: Poisson regression model

Poisson regression model with log link and one continuous and one categorical predictor with three levels. The goal is to learn 8 hyperparameter values. 
Case study uses a data set of a study which investigates the number of LGBTQ+ anti-discrimination laws in each US state.

+ *Target quantities:*  predictive distribution of the group means for the categorical variable (1) and the expected number of LGBTQ+ anti-discrimination laws for selected US states (2)
+ *Elicited statistics:* Quantile-based elicitation for (1) and histogram-based elicitation for each selected design point (2)

:doc:`Go to the case study <case_studies/cs_poisson>`

