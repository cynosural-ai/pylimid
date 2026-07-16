# DecisionPy (working name)

A Python library for modeling and solving **continuous and mixed-type influence diagrams**, built on top of **NumPyro**.

## Goal

Provide a high-level API for influence diagrams while delegating probabilistic inference to NumPyro.

Users define:

* Chance nodes
* Decision nodes
* Utility nodes

and solve the diagram without interacting directly with the inference engine.

---

## Architecture

Separate the project into three layers:

```
Influence Diagram
        │
        ▼
 Graph Representation
        │
        ▼
 NumPyro Backend
```

### Graph representation

Implement a backend-agnostic graph containing:

* Chance nodes
* Decision nodes
* Utility nodes

Each node specifies only its semantics:

* chance → probability distribution
* decision → policy or optimization variable
* utility → utility function

---

### NumPyro backend

Translate the graph into a NumPyro model.

Leverage NumPyro's inference capabilities depending on the model:

* NUTS for continuous latent variables
* SVI for scalable approximate inference
* Enumeration where applicable for discrete variables
* Importance sampling or other Monte Carlo methods when appropriate

The initial objective is not to invent new inference algorithms, but to build a robust influence diagram abstraction on top of an existing probabilistic programming framework.

---

## Initial scope

Support:

* Continuous variables
* Discrete variables
* Mixed discrete/continuous models
* Arbitrary probability distributions
* Arbitrary utility functions
* Continuous and discrete decision variables

The first implementation will focus on **single-agent influence diagrams**.

---

## Long-term vision

The graph abstraction should remain independent of NumPyro, allowing future backends such as:

* Exact discrete influence diagram solvers
* Linear Gaussian (LQG) solvers
* PyMC
* Pyro
* Other probabilistic programming systems

without changing the user-facing API.
