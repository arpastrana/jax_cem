import os

from time import time

from compas.geometry import Point
from compas.geometry import Translation

from compas_cem.diagrams import TopologyDiagram

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import PointConstraint

from compas_cem.optimization import TrailEdgeParameter
from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.plotters import Plotter

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


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN = os.path.abspath(os.path.join(HERE, "data/03_bridge_2d.json"))

# ------------------------------------------------------------------------------
# Load topology diagram from JSON
# ------------------------------------------------------------------------------

topology = TopologyDiagram.from_json(IN)

# ------------------------------------------------------------------------------
# Add supports
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(1))
topology.add_support(NodeSupport(5))

# ------------------------------------------------------------------------------
# Apply loads
# ------------------------------------------------------------------------------

topology.add_load(NodeLoad(2, [0.0, -1.0, 0.0]))
topology.add_load(NodeLoad(6, [0.0, -1.0, 0.0]))

# ------------------------------------------------------------------------------
# Generate trails
# ------------------------------------------------------------------------------

topology.build_trails()

# ------------------------------------------------------------------------------
# Delete indirect deviation edges
# ------------------------------------------------------------------------------

deletable = list(topology.indirect_deviation_edges())
for u, v in deletable:
    topology.delete_edge(u, v)

# ------------------------------------------------------------------------------
# Generate trails
# ------------------------------------------------------------------------------

# topology.build_trails()
topology0 = topology.copy()

# ------------------------------------------------------------------------------
# Form-finding
# ------------------------------------------------------------------------------

form = static_equilibrium(topology)

# ------------------------------------------------------------------------------
# Initialize optimizer
# ------------------------------------------------------------------------------

opt = Optimizer()

# ------------------------------------------------------------------------------
# Define constraints
# ------------------------------------------------------------------------------

nodes_opt = [1, 5]
target_points = [(-20.67, 42.7, 0.0), (15.7, 28.84, 0.0)]
for node, target_point in zip(nodes_opt, target_points):
    opt.add_constraint(PointConstraint(node, target_point))

# ------------------------------------------------------------------------------
# Define optimization parameters
# ------------------------------------------------------------------------------

for edge in topology.trail_edges():
    opt.add_parameter(TrailEdgeParameter(edge, bound_low=15.0, bound_up=5.0))

for edge in topology.deviation_edges():
    opt.add_parameter(DeviationEdgeParameter(edge, bound_low=10.0, bound_up=10.0))

for key, parameter in opt.parameters.items():
    print(key, parameter, parameter.bound_low(topology), parameter.bound_up(topology))

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# optimize
form_opt = opt.solve(topology=topology, algorithm="LBFGS", iters=100, eps=1e-6, verbose=True)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology0)
model = EquilibriumModel.from_topology_diagram(topology0)
eqstate = model(structure)
print(model.lengths)
print(model.forces)
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------


# define loss function
@jit
def loss_fn(diff_model, static_model, structure, y):
    model = eqx.combine(diff_model, static_model)
    eqstate = model(structure, tmax=1)
    pred_y = eqstate.xyz[nodes_opt, :]
    return jnp.sum((y - pred_y) ** 2)


# define targets
y = np.asarray(target_points)

# set tree filtering specification
filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(lambda tree: (tree.lengths, tree.forces), filter_spec, replace=(True, True))

# split model into differentiable and static submodels
diff_model, static_model = eqx.partition(model, filter_spec)

# # create bounds
# bound_low = eqx.tree_at(lambda tree: (tree.lengths, tree.forces),
#                         diff_model,
#                         replace=(-3. * np.ones_like(model.forces), -5. * np.ones_like(model.lengths)))
# print(bound_low.lengths)
# print(bound_low.forces)
# bound_up = eqx.tree_at(lambda tree: (tree.lengths, tree.forces),
#                        diff_model,
#                        replace=(17. * np.ones_like(model.forces), 15. * np.ones_like(model.lengths)))
# print(bound_up.lengths)
# print(bound_up.forces)

# bounds = (bound_low, bound_up)

# evaluate loss function at the start
loss = loss_fn(diff_model, static_model, structure, y)
print(f"{loss=}")

# solve optimization problem with scipy
print("\n***Optimizing with scipy***")
optimizer = jaxopt.ScipyMinimize
# optimizer = jaxopt.ScipyBoundedMinimize

opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=100)

opt_result = opt.run(diff_model, static_model, structure, y)
# opt_result = opt.run(diff_model, bounds, static_model, structure, y)
diff_model_star, opt_state_star = opt_result

# evaluate loss function at optimum point
loss = loss_fn(diff_model_star, static_model, structure, y)
print(f"{loss=}")
print(opt_state_star)

# generate optimized compas cem form diagram
model_star = eqx.combine(diff_model_star, static_model)
eqstate_star = model_star(structure)
form_jax_opt = form_from_eqstate(structure, eqstate_star)

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

plotter = Plotter(figsize=(16, 9))
nodesize = 4.0
loadscale = 6.0

# ------------------------------------------------------------------------------
# Plot topology diagram
# ------------------------------------------------------------------------------

plotter.add(topology, nodesize=nodesize)

# ------------------------------------------------------------------------------
# Plot translated form diagram
# ------------------------------------------------------------------------------

T = Translation.from_vector([40.0, 0.0, 0.0])

# add target points
for target_point in target_points:
    x, y, z = target_point
    plotter.add(Point(x, y, z).transformed(T), size=5.0, facecolor=(1.0, 0.6, 0.0))

plotter.add(form.transformed(T), loadscale=loadscale, nodesize=nodesize)

plotter.add(form_jax.transformed(T), loadscale=loadscale, nodesize=nodesize)

# add target points
for target_point in target_points:
    x, y, z = target_point
    plotter.add(Point(x, y, z).transformed(T), size=5.0, facecolor=(1.0, 0.6, 0.0))

# ------------------------------------------------------------------------------
# Plot translated optimized form diagram
# ------------------------------------------------------------------------------

T = Translation.from_vector([90.0, 0.0, 0.0])

# add target points
for target_point in target_points:
    x, y, z = target_point
    plotter.add(Point(x, y, z).transformed(T), size=5.0, facecolor=(1.0, 0.6, 0.0))

# compas diagram
plotter.add(form_opt, loadscale=loadscale, reactionscale=5.0, nodesize=nodesize, show_nodetext=True)
# jax diagram
plotter.add(form_jax_opt.transformed(T), loadscale=loadscale, reactionscale=5.0, nodesize=nodesize, show_nodetext=True)

# ------------------------------------------------------------------------------
# Plot scene
# -------------------------------------------------------------------------------

plotter.zoom_extents()
plotter.show()
