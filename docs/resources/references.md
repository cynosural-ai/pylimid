# References

The guides show how to build a decision problem, fit its policies and examine the possible outcomes. The sources below develop the ideas behind those steps, from representing uncertainty and available information to solving diagrams with continuous variables.

If we are new to decision analysis, the books and introductions are a useful starting point. The later sections group research papers by topic. For the behaviour implemented in PyLIMID, the [Building a diagram](../user_guide/building_a_diagram.md), [Solving](../user_guide/solving.md) and [Inference](../user_guide/inference.md) guides explain how these ideas appear in the library.

## Books and introductions

Before choosing a solving method, we need to decide what the choices are, what remains uncertain and how we value the outcomes. These books cover that modelling work as well as the probability and algorithms behind it.

- Clemen, R. T. (1996). [Making Hard Decisions: An Introduction to Decision Analysis](https://books.google.com/books?id=EMAnMQAACAAJ). 2nd edition. Duxbury Press. An introduction to structuring decisions, assessing uncertainty and comparing alternatives through their consequences.
- Kjærulff, U. B. and Madsen, A. L. (2013). [Bayesian Networks and Influence Diagrams: A Guide to Construction and Analysis](https://doi.org/10.1007/978-1-4614-5104-4). 2nd edition. Springer. Covers building and checking probabilistic models, with examples of sensitivity and value-of-information analysis.
- Koller, D. and Friedman, N. (2009). [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). MIT Press. A more mathematical treatment of graphical models, inference, learning and decision making.

For a shorter introduction, the *Introduction to Decision Theory* series develops the topic across [Part I](https://ferjorosa.github.io/blog/2025/06/08/decision-theory-I.html), [Part II](https://ferjorosa.github.io/blog/2025/07/04/decision-theory-II.html) and [Part III](https://ferjorosa.github.io/blog/2025/08/07/decision-theory-III.html).

## Influence diagrams and available information

In the shop example, the owner knows the weather before ordering, but demand is still uncertain. The promotion is available to the order decision only when its information arc is present. These papers explain the distinction between dependence and available information, and how it affects the policies we can use.

- Howard, R. A. and Matheson, J. E. (2005). [Influence Diagrams](https://doi.org/10.1287/deca.1050.0020). *Decision Analysis*, 2(3), 127–143. Introduces the representation of uncertainty, decisions and information through a diagram. This link points to the journal publication of their foundational work.
- Shachter, R. D. (1986). [Evaluating Influence Diagrams](https://doi.org/10.1287/opre.34.6.871). *Operations Research*, 34(6), 871–882. Shows how to evaluate a decision problem directly from its influence diagram without first expanding it into a decision tree.
- Jensen, F., Jensen, F. V. and Dittmer, S. L. (1994). [From Influence Diagrams to Junction Trees](https://arxiv.org/abs/1302.6824). In *Uncertainty in Artificial Intelligence (UAI 1994)*, pp. 367–373. Develops a junction-tree method for computing expected utilities and decision policies.
- Lauritzen, S. L. and Nilsson, D. (2001). [Representing and Solving Decision Problems with Limited Information](https://doi.org/10.1287/mnsc.47.9.1235.9779). *Management Science*, 47(9), 1235–1251. Introduces limited-memory influence diagrams, where later decisions need not remember every earlier observation or choice. It also develops single policy updating and studies when it finds an optimal strategy.

## Continuous and mixed variables

Demand, prices and order quantities need not be reduced to a few named states. The papers in this section study ways to represent and solve decision problems with continuous variables. Their assumptions differ, so the Gaussian and hybrid methods are useful comparisons when deciding how much structure to impose on a model.

- Shachter, R. D. and Kenley, C. R. (1989). [Gaussian Influence Diagrams](https://doi.org/10.1287/mnsc.35.5.527). *Management Science*, 35(5), 527–550. Develops inference and decision analysis for linear-quadratic-Gaussian models, exploiting that structure in the calculations.
- Cobb, B. R. and Shenoy, P. P. (2008). [Decision Making with Hybrid Influence Diagrams Using Mixtures of Truncated Exponentials](https://doi.org/10.1016/j.ejor.2007.01.036). *European Journal of Operational Research*, 186(1), 261–275. Uses mixtures of truncated exponentials to represent continuous distributions and approximate utilities, with discrete decisions and an oil-wildcatter example.
- Li, Y. and Shenoy, P. P. (2010). [Solving Hybrid Influence Diagrams with Deterministic Variables](https://proceedings.mlr.press/r8/li10a.html). In *Uncertainty in Artificial Intelligence (UAI 2010)*. Extends hybrid-diagram methods to include deterministic relationships alongside discrete and continuous chance and decision variables.

## Simulation and policy gradients

When we cannot calculate expected utility analytically, we can estimate it from simulated outcomes. Finding a policy then requires a method for using those estimates to improve the decisions. The first two papers below develop simulation methods for decision analysis. The third provides the gradient-estimation framework used by PyLIMID's optimizer.

- Bielza, C., Müller, P. and Ríos Insua, D. (1999). [Decision Analysis by Augmented Probability Simulation](https://doi.org/10.1287/mnsc.45.7.995). *Management Science*, 45(7), 995–1007. Constructs an artificial probability distribution whose marginal over alternatives can be used to search for a maximum-expected-utility decision through Monte Carlo simulation.
- Charnes, J. M. and Shenoy, P. P. (2004). [Multistage Monte Carlo Method for Solving Influence Diagrams Using Local Computation](https://doi.org/10.1287/mnsc.1030.0138). *Management Science*, 50(3), 405–418. Uses the diagram's structure to limit which chance variables need to be sampled together when solving a sequence of decisions.
- Schulman, J., Heess, N., Weber, T. and Abbeel, P. (2015). [Gradient Estimation Using Stochastic Computation Graphs](https://papers.nips.cc/paper_files/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html). In *Advances in Neural Information Processing Systems*, 28. Derives estimators for gradients of expected objectives in graphs containing random choices, including score-function estimators and methods for reducing their variance.

## Probabilistic programming and reinforcement learning

An influence diagram can also define a simulator: chance nodes generate outcomes, policies choose actions from the information they observe, and utility nodes score the result. These papers explore how probabilistic programming, reinforcement learning and neural policies can be used to represent or solve such problems.

- Prestwich, S. D., Toffano, F. and Wilson, N. (2017). [A Probabilistic Programming Language for Influence Diagrams](https://doi.org/10.1007/978-3-319-67582-4_18). In *Scalable Uncertainty Management (SUM 2017)*, Lecture Notes in Computer Science, vol. 10564, pp. 252–265. Springer. Presents IDLP, a logic-programming approach to representing and solving influence diagrams with sampling and reinforcement learning.
- Cooper, H. J., Iyengar, G. and Lin, C.-Y. (2018). [Interpretable Robust Decision Making](https://ifaamas.org/Proceedings/aamas2018/pdfs/p1912.pdf). Extended abstract in *Autonomous Agents and Multiagent Systems (AAMAS 2018)*, pp. 1912–1914. Describes DeepID, a neural-network approximation of influence diagrams with jointly trained decision policies and continuous relaxations for discrete nodes.
- Prestwich, S. D. (2024). [Solving Mixed Influence Diagrams by Reinforcement Learning](https://doi.org/10.1007/978-3-031-53966-4_19). In *Machine Learning, Optimization, and Data Science (LOD 2023)*, Lecture Notes in Computer Science, vol. 14506, pp. 255–269. Springer. Describes a reinforcement-learning solver for multistage discrete decision problems, including limited-memory influence diagrams and constrained or multi-objective models. An [accepted manuscript](https://cora.ucc.ie/items/e1725118-5ea9-4c04-93f5-8e588120e0ce) is available from University College Cork.

## Software and implementation background

PyLIMID builds on NumPyro's distributions and inference machinery, with JAX providing compilation, vectorisation and automatic differentiation. The following papers describe NumPyro, the Pyro system from which it draws its modelling interface, and aGrUM/pyAgrum as another approach to working with graphical models.

- Phan, D., Pradhan, N. and Jankowiak, M. (2019). [Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro](https://arxiv.org/abs/1912.11554). *arXiv:1912.11554*. Explains how effect handlers compose with JAX transformations to support probabilistic modelling and accelerated inference.
- Bingham, E. et al. (2019). [Pyro: Deep Universal Probabilistic Programming](https://jmlr.org/papers/v20/18-403.html). *Journal of Machine Learning Research*, 20(28), 1–6. Describes Pyro's modelling language, variational inference and composable effect-handling machinery.
- Ducamp, G., Gonzales, C. and Wuillemin, P.-H. (2020). [aGrUM/pyAgrum: A Toolbox to Build Models and Algorithms for Probabilistic Graphical Models in Python](https://proceedings.mlr.press/v138/ducamp20a.html). In *Probabilistic Graphical Models (PGM 2020)*, Proceedings of Machine Learning Research, 138. Presents the aGrUM framework and its Python interface for models including Bayesian networks and influence diagrams.

For current usage and installation details, see the [NumPyro documentation](https://num.pyro.ai/en/stable/), [JAX documentation](https://docs.jax.dev/en/latest/) and [pyAgrum documentation](https://pyagrum.readthedocs.io/).
