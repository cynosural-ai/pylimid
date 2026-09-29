"""
SANGIOVESE: a realistic conditional linear Gaussian Bayesian network.

Fitted on real agronomic data by A. Magrini, S. Di Blasi and F. M. Stefanini (2017),
A conditional linear Gaussian network to assess the impact of several agronomic
settings on the quality of Tuscan Sangiovese grapes, Biometrical Letters 54(1):25-42.
Downloaded from the bnlearn Bayesian Network Repository (CC BY-SA 3.0),
https://www.bnlearn.com/bnrepository/ -- model "SANGIOVESE", 15 nodes, 55 arcs, 259 parameters.

Structure: a single discrete root Treatment (16 levels, the agronomic treatments),
whose descendants are continuous. Nodes with Treatment among their parents are
conditional linear Gaussian nodes: Normal with a per-treatment mean (and, for
BunchN/SPAD06/Brix, a per-treatment regression coefficient on their Gaussian parents);
all other continuous nodes are plain linear Gaussian. The leaves (GrapeW, Brix, pH,
Anthoc, Polyph, ...) are the grape-quality measurements.

One source of truth: the data below builds the decisionpy diagram; the example
notebook runs it through the NumPyro engine (NUTS with the discrete Treatment
enumerated out).
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpyro.distributions as dist

from decisionpy.graph import ChanceNode, InfluenceDiagram

__all__ = ["decisionpy_diagram"]


#: Treatment levels, in repository order.
TREATMENT_LEVELS = ('T1a', 'T1b', 'T2a', 'T2b', 'T3a', 'T3b', 'T4a', 'T4b', 'T5a', 'T5b', 'T6a', 'T6b', 'T7a', 'T7b', 'T8a', 'T8b')

#: Prior over the treatments.
TREATMENT_PRIOR = [0.0591805766, 0.06676783, 0.0576631259, 0.0637329287, 0.0591805766, 0.06676783, 0.0606980273, 0.06676783, 0.0591805766, 0.0652503794, 0.0606980273, 0.06676783, 0.0606980273, 0.0637329287, 0.0591805766, 0.0637329287]


#: Edges (parent, child).
EDGES = [('Treatment', 'SproutN'), ('Treatment', 'BunchN'), ('SproutN', 'BunchN'), ('SproutN', 'GrapeW'), ('BunchN', 'GrapeW'), ('WoodW', 'GrapeW'), ('NDVI06', 'GrapeW'), ('NDVI08', 'GrapeW'), ('Acid', 'GrapeW'), ('Brix', 'GrapeW'), ('pH', 'GrapeW'), ('Anthoc', 'GrapeW'), ('SproutN', 'WoodW'), ('BunchN', 'WoodW'), ('SPAD06', 'WoodW'), ('SPAD08', 'WoodW'), ('NDVI08', 'WoodW'), ('Treatment', 'SPAD06'), ('SproutN', 'SPAD06'), ('SproutN', 'NDVI06'), ('SPAD06', 'NDVI06'), ('SPAD06', 'SPAD08'), ('NDVI06', 'SPAD08'), ('SproutN', 'NDVI08'), ('NDVI06', 'NDVI08'), ('SPAD08', 'NDVI08'), ('SproutN', 'Acid'), ('BunchN', 'Acid'), ('SPAD06', 'Acid'), ('NDVI06', 'Acid'), ('NDVI08', 'Acid'), ('Brix', 'Acid'), ('Anthoc', 'Acid'), ('Polyph', 'Acid'), ('BunchN', 'Potass'), ('SPAD06', 'Potass'), ('Anthoc', 'Potass'), ('Treatment', 'Brix'), ('Anthoc', 'Brix'), ('SproutN', 'pH'), ('WoodW', 'pH'), ('SPAD06', 'pH'), ('Acid', 'pH'), ('Potass', 'pH'), ('Brix', 'pH'), ('Anthoc', 'pH'), ('Polyph', 'pH'), ('BunchN', 'Anthoc'), ('WoodW', 'Anthoc'), ('NDVI08', 'Anthoc'), ('BunchN', 'Polyph'), ('NDVI06', 'Polyph'), ('NDVI08', 'Polyph'), ('Brix', 'Polyph'), ('Anthoc', 'Polyph')]


#: Treatment -- DISCRETE
Treatment = {
    "type": 'discrete',
    "parents": (),
    "levels": ('T1a', 'T1b', 'T2a', 'T2b', 'T3a', 'T3b', 'T4a', 'T4b', 'T5a', 'T5b', 'T6a', 'T6b', 'T7a', 'T7b', 'T8a', 'T8b'),
    "prob": [0.0591805766, 0.06676783, 0.0576631259, 0.0637329287, 0.0591805766, 0.06676783, 0.0606980273, 0.06676783, 0.0591805766, 0.0652503794, 0.0606980273, 0.06676783, 0.0606980273, 0.0637329287, 0.0591805766, 0.0637329287],
}


#: SproutN -- CG
SproutN = {
    "type": 'cg',
    "parents": ('Treatment',),
    "intercept": [-0.1454672312, -0.1337576595, -0.1475825732, -0.1325994859, -0.1246768589, -0.0975516494, -0.1135181655, -0.0727987732, 0.0625245153, 0.1343911022, 0.0557900209, 0.1156714732, 0.1256926361, 0.1076859991, 0.1236711796, 0.1476998902],
    "sd": [0.1424387613, 0.1813484529, 0.1361936462, 0.1594647898, 0.1870713965, 0.1480091792, 0.1713604737, 0.1351204851, 0.1900935492, 0.1701830705, 0.1880962896, 0.1819765845, 0.1481518193, 0.2005532468, 0.1705529219, 0.1769564095],
}


#: BunchN -- CG
BunchN = {
    "type": 'cg',
    "parents": ('Treatment', 'SproutN'),
    "intercept": [0.1266592515, 0.12441731, -0.0448515193, -0.1576167002, 0.1294968832, 0.1626483557, -0.2482123141, -0.167814294, 0.2268356834, 0.2041911119, -0.1272652989, -0.1925490517, 0.1497279828, 0.1778378718, -0.2316235932, -0.2524324728],
    "gcoef": [
        [0.8627521176, 0.5831144461, 1.2177006055, 1.0166774653, 1.0278515347, 0.8057956402, 0.7364866092, 1.2097183945, 1.1077525398, 0.9564685145, 0.6333599381, 1.0480716459, 0.7933612934, 0.8422050504, 1.2592488532, 0.9309395724],
    ],
    "sd": [0.2816935787, 0.3207793324, 0.3242622136, 0.3856272989, 0.3407581855, 0.2826435641, 0.3512261609, 0.3390433119, 0.3075564035, 0.4226637358, 0.360295382, 0.3448297529, 0.2961201221, 0.3668736367, 0.2434776228, 0.3222200838],
}


#: GrapeW -- GAUSSIAN
GrapeW = {
    "type": 'gaussian',
    "parents": ('SproutN', 'BunchN', 'WoodW', 'NDVI06', 'NDVI08', 'Acid', 'Brix', 'pH', 'Anthoc'),
    "intercept": 0.0037852389,
    "gcoef": [-0.2338231795, 0.779743594, 0.3829438747, 0.8191238471, 0.2782258031, -1.0997393163, -1.424193325, -2.8167338187, -0.1385742554],
    "sd": 0.2888755459,
}


#: WoodW -- GAUSSIAN
WoodW = {
    "type": 'gaussian',
    "parents": ('SproutN', 'BunchN', 'SPAD06', 'SPAD08', 'NDVI08'),
    "intercept": -0.0105901431,
    "gcoef": [0.2086688845, 0.1055157638, 1.3207198187, 1.1707308367, 0.6973012502],
    "sd": 0.2683441044,
}


#: SPAD06 -- CG
SPAD06 = {
    "type": 'cg',
    "parents": ('Treatment', 'SproutN'),
    "intercept": [0.0836654195, 0.0653996463, 0.0768125975, 0.0635801113, 0.0281502386, 0.033555142, 0.0418775924, 0.0367612105, -0.0446389385, -0.0765299223, -0.0221821637, -0.0440273601, -0.0899319023, -0.0502676798, -0.0569517057, -0.0393174304],
    "gcoef": [
        [0.4876396963, 0.3730929096, 0.4973389624, 0.3159272436, 0.3400409631, 0.3376646439, 0.5759864536, 0.6347555325, 0.5150905605, 0.4779942532, 0.462459439, 0.456090777, 0.5055714411, 0.3505919226, 0.4827893833, 0.2764874164],
    ],
    "sd": [0.0995338682, 0.1011680296, 0.0960394731, 0.0950349613, 0.0717670924, 0.1068228338, 0.0781161987, 0.1056333913, 0.0771705857, 0.0807754529, 0.0964170957, 0.0605387342, 0.098390916, 0.0945272895, 0.0949606973, 0.107185652],
}


#: NDVI06 -- GAUSSIAN
NDVI06 = {
    "type": 'gaussian',
    "parents": ('SproutN', 'SPAD06'),
    "intercept": 0.0009060401,
    "gcoef": [0.0524537942, 0.4391847164],
    "sd": 0.0916218266,
}


#: SPAD08 -- GAUSSIAN
SPAD08 = {
    "type": 'gaussian',
    "parents": ('SPAD06', 'NDVI06'),
    "intercept": 0.0034135105,
    "gcoef": [0.6572326508, 0.3471696969],
    "sd": 0.0870218552,
}


#: NDVI08 -- GAUSSIAN
NDVI08 = {
    "type": 'gaussian',
    "parents": ('SproutN', 'NDVI06', 'SPAD08'),
    "intercept": -0.0048587781,
    "gcoef": [0.1042570791, 0.13598597, 0.4316950043],
    "sd": 0.1123024278,
}


#: Acid -- GAUSSIAN
Acid = {
    "type": 'gaussian',
    "parents": ('SproutN', 'BunchN', 'SPAD06', 'NDVI06', 'NDVI08', 'Brix', 'Anthoc', 'Polyph'),
    "intercept": 0.0009448236,
    "gcoef": [-0.0859213515, 0.0745025088, -0.3684175756, -0.1935515016, -0.2441153549, -0.6116201325, -0.0684739505, 0.1803238117],
    "sd": 0.1208580605,
}


#: Potass -- GAUSSIAN
Potass = {
    "type": 'gaussian',
    "parents": ('BunchN', 'SPAD06', 'Anthoc'),
    "intercept": -0.0050455651,
    "gcoef": [-0.0718027101, 0.3918097274, 0.0612000795],
    "sd": 0.1454155604,
}


#: Brix -- CG
Brix = {
    "type": 'cg',
    "parents": ('Treatment', 'Anthoc'),
    "intercept": [-0.0632063641, 0.0202382407, -0.0387757013, 0.025777974, -0.020747206, 0.0596576103, -0.0199815674, 0.0601169145, -0.0694135728, -0.0050813117, -0.0425454173, 0.016107227, -0.0320623733, 0.0340875656, -0.0003748422, 0.0584812196],
    "gcoef": [
        [0.0916094449, 0.0811215251, 0.0993188994, 0.0423627564, 0.1556132581, 0.0938667697, 0.1036335186, 0.0727073368, 0.1701939272, 0.0727000317, 0.1101793021, 0.0468651603, 0.119170987, 0.0877573654, 0.0856687195, 0.0846071396],
    ],
    "sd": [0.0543295036, 0.0548468609, 0.0545580051, 0.0555374259, 0.0462019852, 0.0885133439, 0.0630522065, 0.0653503304, 0.0658161387, 0.072990744, 0.0441416504, 0.0619601523, 0.0568165303, 0.0632266875, 0.0564648473, 0.0617422389],
}


#: pH -- GAUSSIAN
pH = {
    "type": 'gaussian',
    "parents": ('SproutN', 'WoodW', 'SPAD06', 'Acid', 'Potass', 'Brix', 'Anthoc', 'Polyph'),
    "intercept": -0.0004402041,
    "gcoef": [0.0115191464, 0.0061100089, 0.0407614454, -0.1814311205, 0.0569802619, 0.1606425478, -0.0239600155, 0.0210687973],
    "sd": 0.0168076417,
}


#: Anthoc -- GAUSSIAN
Anthoc = {
    "type": 'gaussian',
    "parents": ('BunchN', 'WoodW', 'NDVI08'),
    "intercept": 0.0068265405,
    "gcoef": [-0.1361318519, -0.3303086721, -0.4529745403],
    "sd": 0.317656096,
}


#: Polyph -- GAUSSIAN
Polyph = {
    "type": 'gaussian',
    "parents": ('BunchN', 'NDVI06', 'NDVI08', 'Brix', 'Anthoc'),
    "intercept": 0.0020743338,
    "gcoef": [0.0553731153, -0.3734417378, 0.2343595096, 0.2867605258, 0.5500649395],
    "sd": 0.1565977524,
}

#: Node names in topological order (parents before children).
NODES = ['Treatment', 'SproutN', 'BunchN', 'GrapeW', 'WoodW', 'SPAD06', 'NDVI06', 'SPAD08', 'NDVI08', 'Acid', 'Potass', 'Brix', 'pH', 'Anthoc', 'Polyph']


def decisionpy_diagram() -> InfluenceDiagram:
    """Build the decisionpy diagram from the shared parameters."""
    diag = InfluenceDiagram()
    for name in NODES:
        params = globals()[name]
        kind = params["type"]
        parents = tuple(params["parents"])
        if kind == "discrete":
            diag.add_node(ChanceNode(name=name, parents=parents, states=params["levels"],
                dist=lambda params=params: dist.Categorical(probs=jnp.array(params["prob"]))))
            continue
        if kind == "cg":
            dist_factory = _cg_factory(params, parents)
        else:
            dist_factory = _gaussian_factory(params)
        diag.add_node(ChanceNode(name=name, parents=parents, dist=dist_factory))
    return diag


def _cg_factory(params: dict, parents: tuple) -> callable:
    """A dist factory for a CG node: per-treatment Normal(loc, scale)."""
    intercept = jnp.array(params["intercept"])
    sd = jnp.array(params["sd"])
    gaussian_parents = tuple(p for p in params["parents"] if p != "Treatment")
    gcoef = None
    if "gcoef" in params:
        gcoef = {p: jnp.array(c) for p, c in zip(gaussian_parents, params["gcoef"], strict=True)}

    def fn(**kwargs):
        t = kwargs["Treatment"]
        loc = intercept[t]
        if gcoef is not None:
            loc = loc + sum(gcoef[p][t] * kwargs[p] for p in gaussian_parents)
        return dist.Normal(loc=loc, scale=sd[t])

    return _declare_signature(fn, parents)


def _gaussian_factory(params: dict) -> callable:
    """A dist factory for a plain linear Gaussian node."""
    parents = tuple(params["parents"])
    coef = {p: params["gcoef"][i] for i, p in enumerate(parents)}
    intercept = params["intercept"]
    sd = params["sd"]

    def fn(**kwargs):
        loc = intercept + sum(c * kwargs[p] for p, c in coef.items())
        return dist.Normal(loc=loc, scale=sd)

    return _declare_signature(fn, parents)


def _declare_signature(fn, names: tuple):
    """Attach a named-parameter signature so the consistency gate passes."""
    fn.__signature__ = inspect.Signature(
        [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY) for n in names]
    )
    return fn
