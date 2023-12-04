import os

from math import fabs

from time import time

from compas.geometry import Point
from compas.geometry import Translation
from compas.geometry import scale_vector

from compas_cem.diagrams import TopologyDiagram

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import TrailEdgeForceConstraint

from compas_cem.optimization import TrailEdgeParameter
from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.plotters import Plotter
from compas_cem.viewers import Viewer

import jax
import jaxopt

from jax import jit
from jax import grad

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate


VIEW = True
OPTIMIZE = False
OPTIMIZE_JAX = True

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN = os.path.abspath(os.path.join(HERE, "data/stadium.json"))

# ------------------------------------------------------------------------------
# Load topology diagram from JSON
# ------------------------------------------------------------------------------

topology = TopologyDiagram.from_json(IN)

aux_trails = topology.attributes["_auxiliary_trails"]
for k, v in topology.auxiliary_trails(True):
    aux_trails[k] = tuple(v)

# ------------------------------------------------------------------------------
# Copy topology
# ------------------------------------------------------------------------------

topology0 = topology.copy()

# ------------------------------------------------------------------------------
# Form-finding
# ------------------------------------------------------------------------------

form = static_equilibrium(topology, tmax=1)

# ------------------------------------------------------------------------------
# Initialize optimizer
# ------------------------------------------------------------------------------

if OPTIMIZE:
    opt = Optimizer()

# ------------------------------------------------------------------------------
# Define constraints
# ------------------------------------------------------------------------------

    for edge in topology.auxiliary_trail_edges():
        if not topology.edge_attribute(edge, "is_constrained"):
            continue
        constraint = TrailEdgeForceConstraint(edge, 0.0)
        opt.add_constraint(constraint)

# ------------------------------------------------------------------------------
# Define optimization parameters
# ------------------------------------------------------------------------------

    for edge in topology.deviation_edges():
        bl = topology.edge_attribute(edge, "bound_low")
        bu = topology.edge_attribute(edge, "bound_up")
        opt.add_parameter(DeviationEdgeParameter(edge, bound_low=bl, bound_up=bu))

# for key, parameter in opt.parameters.items():
#     print(key, parameter, parameter.bound_low(topology), parameter.bound_up(topology))

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

    form_opt = opt.solve(topology=topology,
                         algorithm="SLSQP",
                         tmax=1,
                         iters=300,
                         eps=1e-6,
                         verbose=True)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology0)
model = EquilibriumModel.from_topology_diagram(topology0)
eqstate = model(structure, tmax=1)
form_jax = form_from_eqstate(structure, eqstate)

form_jax.to_json(os.path.join(HERE, "data/stadium_jax.json"))

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------

if OPTIMIZE_JAX:

    edges_opt = []
    for edge in topology.auxiliary_trail_edges():
        index = structure.edge_index[edge]
        if not topology.edge_attribute(edge, "is_constrained"):
            continue
        edges_opt.append(index)

    # edges_opt = tuple(sorted(edges_opt))

    # define loss function
    @jit
    def loss_fn(diff_model, static_model, structure, y):
        model = eqx.combine(diff_model, static_model)
        eqstate = model(structure, tmax=1)
        pred_y = eqstate.forces[edges_opt, :]
        return jnp.sum((pred_y - y) ** 2)

    # define targets
    y = 0.0

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.forces), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    # create bounds
    bounds_low_compas = []
    for edge, force in zip(structure.edges, model.forces):
        edge = tuple(edge)

        if not topology.is_deviation_edge(edge):
            bl = force
        else:
            index = structure.edge_index[edge]
            bl = topology.edge_attribute(edge, "bound_low")
            bl = force - fabs(bl)
        bounds_low_compas.append(bl)
    bounds_low_compas = jnp.asarray(bounds_low_compas)

    bounds_up_compas = []
    for edge, force in zip(structure.edges, model.forces):
        edge = tuple(edge)

        if not topology.is_deviation_edge(edge):
            bu = force
        else:
            index = structure.edge_index[edge]
            bu = topology.edge_attribute(edge, "bound_up")
            bu = force + fabs(bu)
        bounds_up_compas.append(bu)
    bounds_up_compas = jnp.asarray(bounds_up_compas)

    bound_low = eqx.tree_at(lambda tree: (tree.forces),
                            diff_model,
                            replace=(bounds_low_compas))
    bound_up = eqx.tree_at(lambda tree: (tree.forces),
                           diff_model,
                           replace=(bounds_up_compas))

    bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model, structure, y)
    print(f"{loss=}")

    # # solve optimization problem with scipy
    print("\n***Optimizing with scipy***")
    # optimizer = jaxopt.ScipyMinimize
    optimizer = jaxopt.ScipyBoundedMinimize

    opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=300)

    # opt_result = opt.run(diff_model, static_model, structure, y)
    start = time()
    opt_result = opt.run(diff_model, bounds, static_model, structure, y)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result
    print(opt_state_star)

    # evaluate loss function at optimum point
    loss = loss_fn(diff_model_star, static_model, structure, y)
    print(f"{loss=}")

    # generate optimized compas cem form diagram
    model_star = eqx.combine(diff_model_star, static_model)
    eqstate_star = model_star(structure)
    form_jax_opt = form_from_eqstate(structure, eqstate_star)

    form_jax_opt.to_json(os.path.join(HERE, "data/stadium_jax_opt.json"))

# ------------------------------------------------------------------------------
# Launch viewer
# ------------------------------------------------------------------------------

if VIEW:
    shift_vector = [150.0, 0.0, 0.0]
    form = form.transformed(Translation.from_vector(scale_vector(shift_vector, 0.)))
    forms = [form_jax]

    i = 1.
    if OPTIMIZE:
        form_opt = form_opt.transformed(Translation.from_vector(scale_vector(shift_vector, i * 2.)))
        forms.append(form_opt)
        i += 1.

    if OPTIMIZE_JAX:
        form_jax_opt = form_jax_opt.transformed(Translation.from_vector(scale_vector(shift_vector, i * 2.)))
        forms.append(form_jax_opt)
        i += 1.

    viewer = Viewer(width=1600, height=900, show_grid=False)
    viewer.view.color = (0.5, 0.5, 0.5, 1)  # change background to black

# ------------------------------------------------------------------------------
# Visualize topology diagram
# ------------------------------------------------------------------------------

    viewer.add(topology,
               edgewidth=0.03,
               show_loads=False)

# ------------------------------------------------------------------------------
# Visualize translated form diagram
# ------------------------------------------------------------------------------

    for form in forms:
        viewer.add(form,
                   edgewidth=(0.3, 0.5),
                   show_loads=False,
                   show_edgetext=False,
                   residualscale=4.0,
                   edgetext="force",
                   )

# ------------------------------------------------------------------------------
# Show scene
# -------------------------------------------------------------------------------

    viewer.show()
