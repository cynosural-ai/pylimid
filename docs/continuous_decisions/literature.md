# Continuous decisions — literature map

A reading map for the mathematics and algorithms behind continuous decisions, grouped by field. Each entry says what the work is and why it matters *here*. The map is deliberately short per entry; it is a guide for deciding how far to reach, not a survey.

Two observations frame the list:

- The math needed to optimize non-Gaussian, non-linear expected utility is **established and published**, spread across stochastic control, decision analysis, and machine learning. Nothing about the continuous-decision extension requires inventing new theory.
- What is **not** established is a graph-native, probabilistic-programming treatment of continuous influence diagrams: the exact solvers in use (pyAgrum and friends) are discrete, and the control-as-inference literature does not model influence-diagram structure. That combination is where this library could differentiate.

## Influence diagrams and LIMIDs

- **Howard, R. A. and Matheson, J. E. (1984). Influence diagrams.** In *Readings on the Principles and Applications of Decision Analysis*, Vol. II, 719-762. The origin of the influence diagram, assuming full memory.
- **Shachter, R. D. (1986). Evaluating influence diagrams.** *Operations Research* 34(6), 871-882. The classical backward-induction/arc-reversal algorithm for regular influence diagrams; the foundation the current discrete solvers mirror.
- **Lauritzen, S. L. and Nilsson, D. (2001). Representing and solving decision problems with limited information.** *Management Science* 47(9), 1235-1251. The LIMID formalism decisionpy targets, plus single policy updating — the reference behind the implemented scan and backward induction, and the reason the library's decision semantics are "local policy" rather than full memory.
- **Shachter, R. D. and Kenley, C. R. (1989). Gaussian influence diagrams.** *Management Science* 35(5). The classical continuous case: Gaussian chance nodes with linear/quadratic structure, solved in closed form. This is the LQG class used here as the analytic test oracle, not the target scope.

## Continuous and simulation-based decision analysis

- **Bielza, C., Müller, P. and Ríos Insua, D. (2007). Decision analysis by augmented probability simulation.** *Management Science* 53(7). Monte-Carlo decision analysis: transform the expected-utility problem into sampling from an augmented probability model, find modes. The tradition the sampling solver family belongs to, and the closest published relative of continuous-decision optimization by simulation.
- The takeaway from this line: expectations without closed forms are handled by simulation as a matter of routine, so "no Gaussian/quadratic structure" is not a mathematical blocker.

## Gradient estimation and differentiable sampling

This is the toolbox that turns "sample and average" into "sample, average, and differentiate".

- **Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.** *Machine Learning* 8, 229-256. REINFORCE: the score-function estimator, unbiased for any distribution but high-variance. The fallback for discrete action selection and discrete chance descendants.
- **Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes.** *ICLR*. The reparameterization trick in its modern form: write a sample as a deterministic function of fixed noise, making gradients flow through the sample. The core enabler for continuous chance nodes.
- **Rezende, D. J., Mohamed, S. and Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models.** *ICML*. The same idea developed independently, with the variance argument.
- **Figurnov, M., Mohamed, S. and Mnih, A. (2018). Implicit reparameterization gradients.** *NeurIPS*. Extends pathwise gradients to distributions without a simple location-scale form (Gamma, Beta, Dirichlet, ...) via implicit differentiation of the CDF. This is how NumPyro differentiates many non-Gaussian distributions.
- **Mohamed, S., Rosca, M., Figurnov, M. and Mnih, A. (2020). Monte Carlo gradient estimation in machine learning.** *JMLR* 21(132), 1-62. The survey that unifies pathwise, score-function, and measure-valued estimators, including variance analysis and when each applies. The single best entry point for the estimator choices in [`solver_methods.md`](./solver_methods.md).

## Control as inference

The conceptual bridge: optimal decisions as probabilistic inference. These papers show that expected-cost minimization can be recast as inference in a graphical model — the move a probabilistic-programming backend is naturally positioned to make.

- **Kappen, H. J. (2005). Path integrals and symmetry breaking for optimal control theory.** *Journal of Statistical Mechanics: Theory and Experiment*. Stochastic optimal control as a path integral; the origin of the control-as-inference line for continuous state spaces.
- **Todorov, E. (2008). General duality between optimal control and estimation.** *IEEE Conference on Decision and Control*. The duality in its cleanest form: a class of optimal control problems is exactly an estimation problem.
- **Toussaint, M. (2009). Probabilistic inference as a model of planned behavior.** *Künstliche Intelligenz* 3/09. Planning and control framed directly as message passing / inference.
- **Kappen, H. J., Gómez, V. and Opper, M. (2012). Optimal control as a graphical model inference problem.** *Machine Learning* 87, 159-182. Control as inference on a factor graph, with approximate inference algorithms; the most directly transferable framing to an influence diagram.
- **Rawlik, K., Toussaint, M. and Vijayakumar, S. (2012). On stochastic optimal control and reinforcement learning by approximate inference.** *Robotics: Science and Systems*. Approximate inference for control, including the temporal-difference/expectation-maximization connection.
- **Levine, S. (2018). Reinforcement learning and control as probabilistic inference: tutorial and review.** arXiv:1805.00909. The accessible tutorial for this whole line, including the policy-gradient connections.

## Model-based policy search

- **Deisenroth, M. P. and Rasmussen, C. E. (2011). PILCO: A model-based and data-efficient approach to policy search.** *ICML*. Optimizes a parameterized policy through a probabilistic model to minimize expected long-run cost; the multi-step, learned-model cousin of what a continuous-decision solver does with a known model. Useful for the variance and policy-parameterization discussion.

## Probabilistic-programming systems

- **Cusumano-Towner, M. F., Saad, F. A., Lew, A. K. and Mansinghka, V. K. (2019). Gen: A general-purpose probabilistic programming system with programmable inference.** *PLDI*. A PPL designed with decision-making among its target applications; evidence that general-purpose PPLs can host planning, without providing influence-diagram solving out of the box.
- **van de Meent, J.-W., Paige, B., Yang, H. and Wood, F. (2018). An introduction to probabilistic programming.** *Foundations and Trends in Machine Learning* 11(4). Background for how PPL inference machinery (enumeration, HMC/NUTS, SVI) maps onto the needs here.

## LQG foundations

- **Kalman, R. E. (1960). Contributions to the theory of optimal control.** *Boletín de la Sociedad Matemática Mexicana* 5, 102-119. The linear-quadratic regulator.
- **Wonham, W. M. (1968). On the separation theorem of stochastic control.** *SIAM Journal on Control* 6(2), 312-326. The separation principle: estimation and control decompose, and the optimal control uses the conditional mean (certainty equivalence). This is why the thermostat example's affine policy is provably optimal, and why LQG is a trustworthy oracle.

## The gap this library could fill

Putting the sections together:

- Classical continuous influence diagrams are **Gaussian/linear/quadratic** (Shachter & Kenley), or handled by generic simulation (**augmented probability simulation**) without graph structure.
- **Control as inference** supplies the mathematics for general stochastic objectives but does not model influence-diagram structure, information sets, or limited memory.
- **Gradient estimation** supplies the estimators but not the diagram semantics.
- **PPL systems** supply the model/sampling/autodiff substrate but not influence-diagram solving.
- The existing external validation references (**pyAgrum**) are discrete.

A graph-native solver that composes these — LIMID information-set semantics, NumPyro's reparameterized sampling, Monte-Carlo expected-utility gradients, LQG and discretized-grid validation — would be an uncommon combination. That, rather than any single algorithmic novelty, is the case for the continuous-decision extension.
