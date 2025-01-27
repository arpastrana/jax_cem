import os

from math import fabs

from time import perf_counter

from compas.geometry import Point
from compas.geometry import Translation
from compas.geometry import scale_vector

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import TrailEdgeForceConstraint

from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.viewers import Viewer

import jaxopt

from jax import jit
from jax import grad

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_cem.datastructures import EquilibriumStructure
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.parameters import ParameterState

VIEW = True
OPTIMIZE = True
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
                         algorithm="LBFGS",
                         tmax=1,
                         iters=500,
                         eps=1e-6,
                         verbose=True)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology0)
parameters = ParameterState.from_topology_diagram(topology0)
model = EquilibriumModel(tmax=1)
eqstate = model(parameters, structure)

form_jax = FormDiagram.from_equilibrium_state(eqstate, structure)
# form_jax.to_json(os.path.join(HERE, "data/stadium_jax.json"))

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
    def loss_fn(diff_params, static_params, structure, y):
        parameters = eqx.combine(diff_params, static_params)
        eqstate = model(parameters, structure)
        pred_y = eqstate.forces[edges_opt, :]
        return jnp.sum((pred_y - y) ** 2)

    # define targets
    y = 0.0

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, parameters)
    filter_spec = eqx.tree_at(lambda tree: (tree.forces), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_params, static_params = eqx.partition(parameters, filter_spec)

    # create lower bounds
    bounds_low_compas = []
    for edge, force in zip(structure.edges, parameters.forces):
        edge = tuple(edge)

        if not topology.is_deviation_edge(edge):
            bl = force
        else:
            index = structure.edge_index[edge]
            bl = topology.edge_attribute(edge, "bound_low")
            bl = force - fabs(bl)
        bounds_low_compas.append(bl)
    bounds_low_compas = jnp.asarray(bounds_low_compas)

    # create upper bounds
    bounds_up_compas = []
    for edge, force in zip(structure.edges, parameters.forces):
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
                            diff_params,
                            replace=(bounds_low_compas))
    bound_up = eqx.tree_at(lambda tree: (tree.forces),
                           diff_params,
                           replace=(bounds_up_compas))

    bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_params, static_params, structure, y)
    print(f"{loss=}")

    # # solve optimization problem with scipy
    print("\n***Optimizing with scipy***")
    # optimizer = jaxopt.ScipyMinimize
    optimizer = jaxopt.ScipyBoundedMinimize

    opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=300)

    # opt_result = opt.run(diff_model, static_model, structure, y)
    start = perf_counter()
    opt_result = opt.run(diff_params, bounds, static_params, structure, y)
    end = perf_counter()
    diff_params_star, opt_state_star = opt_result

    # Summary
    print(f"Optimization time: {end - start:.2f} s")
    print(f"Success? {opt_state_star.success}")
    print(f"Iterations: {opt_state_star.iter_num}")

    # evaluate loss function at optimum point
    loss = loss_fn(diff_params_star, static_params, structure, y)
    print(f"{loss=}")

    # generate optimized compas cem form diagram
    parameters_star = eqx.combine(diff_params_star, static_params)
    eqstate_star = model(parameters_star, structure)
    form_jax_opt = FormDiagram.from_equilibrium_state(eqstate_star, structure)

    # form_jax_opt.to_json(os.path.join(HERE, "data/stadium_jax_opt.json"))

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
