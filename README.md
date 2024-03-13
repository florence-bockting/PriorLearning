[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

# Simulation-based prior knowledge elicitation

**See :bookmark_tabs: [preprint](https://arxiv.org/abs/2308.11672) for more information and the :globe_with_meridians: [website](https://florence-bockting.github.io/PriorLearning/) for implemented case studies.**

A central characteristic of Bayesian statistics is the ability to consistently incorporate prior
knowledge into various modeling processes. In this paper, we focus on translating domain
expert knowledge into corresponding prior distributions over model parameters, a process
known as prior elicitation. Expert knowledge can manifest itself in diverse formats, including
information about raw data, summary statistics, or model parameters. A major challenge for
existing elicitation methods is how to effectively utilize all of these different formats in order to
formulate prior distributions that align with the expert’s expectations, regardless of the model
structure. To address these challenges, we develop a simulation-based elicitation method that
can learn the hyperparameters of potentially any parametric prior distribution from a wide
spectrum of expert knowledge using stochastic gradient descent. We validate the effectiveness
and robustness of our elicitation method in four representative case studies covering linear
models, generalized linear models, and hierarchical models. Our results support the claim that
our method is largely independent of the underlying model structure and adaptable to various
elicitation techniques, including quantile-based, moment-based, and histogram-based methods.

