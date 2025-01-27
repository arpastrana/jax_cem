import os

from time import time

from compas.geometry import Point
from compas.geometry import Translation

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import PointConstraint

from compas_cem.optimization import TrailEdgeParameter
from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.plotters import Plotter

import jaxopt

from jax import jit
from jax import value_and_grad

import jax.numpy as jnp
import numpy as np

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState

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

# deletable = list(topology.indirect_deviation_edges())
# for u, v in deletable:
#     topology.delete_edge(u, v)

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
parameters = ParameterState.from_topology_diagram(topology0)
model = EquilibriumModel(tmax=100, verbose=True)
eqstate = model(parameters, structure)

form_jax = FormDiagram.from_equilibrium_state(eqstate, structure)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------


def combine_parameters(theta):
    """
    Combine a tuple of parameters into a parameter state.
    """
    lengths, forces = theta

    return ParameterState(
        lengths=lengths,
        forces=forces,
        xyz=parameters.xyz,
        loads=parameters.loads,
        planes=parameters.planes)

# define loss function
@jit
@value_and_grad
def loss_fn(theta):
    """
    Compute the loss function for the optimization problem.
    """
    parameters = combine_parameters(theta)
    eqstate = model(parameters, structure)
    xyz = eqstate.xyz[nodes_opt, :]

    return jnp.sum(jnp.square(xyz - xyz_target))

# define targets
xyz_target = np.asarray(target_points)

# set initial optimization parameters
theta_init = (parameters.lengths, parameters.forces)

# create bounds
bound_low = (-10.0 * jnp.ones_like(parameters.forces), -10.0 * jnp.ones_like(parameters.lengths))
bound_up = (15.0 * jnp.ones_like(parameters.forces), 15.0 * jnp.ones_like(parameters.lengths))
bounds = (bound_low, bound_up)

# evaluate loss function at the start
time_start = time()
loss, grad = loss_fn(theta_init)
time_end = time()
print(f"Warm-up function evaluation time: {time_end - time_start:.2f} s")
print(f"{loss=}")

# Solve optimization problem with scipy
print("\n***Optimizing with scipy***")
optimizer = jaxopt.ScipyBoundedMinimize
optimizer = jaxopt.ScipyMinimize

opt = optimizer(
    fun=loss_fn,
    method="L-BFGS-B",
    tol=1e-6,
    maxiter=100,
    value_and_grad=True
    )

time_start = time()
opt_result = opt.run(theta_init)  # , bounds)
time_end = time()
theta_star, opt_state_star = opt_result

# Summary
print(f"Optimization time: {time_end - time_start:.2f} s")
print(f"Success? {opt_state_star.success}")
print(f"Iterations: {opt_state_star.iter_num}")

# Evaluate loss function at optimum point
loss, grad = loss_fn(theta_star)
grad_norm = jnp.linalg.norm(jnp.concatenate(grad))
print(f"{loss=}")
print(f"{grad_norm=}")


# Generate optimized compas cem form diagram
parameters_star = combine_parameters(theta_star)
eqstate_star = model(parameters_star, structure)
form_jax_opt = FormDiagram.from_equilibrium_state(eqstate_star, structure)

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
# plotter.add(form_opt.transformed(T), loadscale=loadscale, reactionscale=5.0, nodesize=nodesize, show_nodetext=True)

# jax diagram
plotter.add(form_jax_opt.transformed(T), loadscale=loadscale, reactionscale=5.0, nodesize=nodesize, show_nodetext=True)

# ------------------------------------------------------------------------------
# Plot scene
# -------------------------------------------------------------------------------

plotter.zoom_extents()
plotter.show()
