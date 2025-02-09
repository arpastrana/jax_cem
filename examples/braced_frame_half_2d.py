import numpy as np

from compas.geometry import Translation

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium_numpy

from compas_cem.plotters import Plotter

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState

from jax.tree_util import tree_map

import jax.numpy as jnp

import jaxopt

from time import perf_counter

from jax import value_and_grad
from jax import jit

# -------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------

HEIGHT = 5.0
WIDTH = 5.0
PX = 1.0

TMAX = 100
ETA = 1e-6

TOL = 1e-9
MAXITER = 100

ALPHA_XYZ = 1e-12
ALPHA_RESIDUAL_DIRECTION = 1.0
ALPHA_RESIDUAL = 10.0
ALPHA_LOADPATH = 1e-2

# ------------------------------------------------------------------------------
# Topology Diagram
# ------------------------------------------------------------------------------

points = [
    (0, [0.0, 0.0, 0.0]),
    (1, [0.0, HEIGHT, 0.0]),
    (2, [WIDTH / 2.0, HEIGHT / 2.0, 0.0])
    ]

trail_edges = [
    # ((0, 1), HEIGHT),
    ]

deviation_edges = [
    ((0, 1), 1.0),
    ((1, 2), -1.0),
    ((2, 0), 1.0)
    ]

# ------------------------------------------------------------------------------
# Topology Diagram
# ------------------------------------------------------------------------------

topology = TopologyDiagram()

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

for key, point in points:
    topology.add_node(Node(key, point))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

for (u, v), length in trail_edges:
    topology.add_edge(TrailEdge(u, v, length=length))

# ------------------------------------------------------------------------------
# Add Deviation Edges
# ------------------------------------------------------------------------------

for (u, v), force in deviation_edges:
    topology.add_edge(DeviationEdge(u, v, force=force))

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

# topology.add_support(NodeSupport(0))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

load = [PX, 0.0, 0.0]
topology.add_load(NodeLoad(1, load))

# ------------------------------------------------------------------------------
# Equilibrium of forces
# ------------------------------------------------------------------------------

topology.build_trails(auxiliary_trails=True)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with JAX CEM
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
parameters = ParameterState.from_topology_diagram(topology)
model = EquilibriumModel(tmax=TMAX, eta=ETA, verbose=True)

eqstate = model(parameters, structure)

form_jax = FormDiagram.from_equilibrium_state(eqstate, structure)

# ------------------------------------------------------------------------------
# Indexing
# ------------------------------------------------------------------------------

node_index = structure.node_index
node_index_xyz = node_index[0]
node_index_xyz_origin = node_index[2]
node_index_xyz_support = node_index[5]

node_index_residual = node_index[4]

edge_index = structure.edge_index
edge_index_length = edge_index[(0, 1)]
edge_indices_force = [
    edge_index[(0, 1)],
    edge_index[(1, 2)],
    edge_index[(2, 0)]
    ]

edge_indices_force = jnp.array(edge_indices_force, dtype=jnp.int64)

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

xyz_target = jnp.asarray(topology.node_coordinates(0))
vector_target = jnp.asarray([0.0, 1.0, 0.0])

# ------------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------------

def combine_parameters(theta):
    """
    Combine the parameters into a single array.
    """
    forces = theta[:3]
    forces = parameters.forces.at[edge_indices_force, :].set(forces[:, None])

    y = theta[-1]
    xyz = parameters.xyz.at[node_index_xyz_origin, 1].set(y)

    return ParameterState(
        xyz=xyz,
        loads=parameters.loads,
        forces=forces,
        lengths=parameters.lengths,
        planes=parameters.planes
        )


def goal_xyz(eq_state, xyz_target):
    xyz = eq_state.xyz[node_index_xyz, :]
    return ALPHA_XYZ * jnp.sum(jnp.square(xyz - xyz_target))


def goal_residual_direction(eq_state, vec_target):
    xyz_origin = eq_state.xyz[node_index_xyz_origin]
    xyz_support = eq_state.xyz[node_index_xyz_support]

    vector = xyz_origin - xyz_support
    length = jnp.linalg.norm(vector)
    vector_unit = vector / length

    return ALPHA_RESIDUAL_DIRECTION * jnp.sum(jnp.square(vector_unit - vec_target))


def goal_residual(eq_state):
    residuals = eq_state.reactions[node_index_residual]

    return ALPHA_RESIDUAL * jnp.sum(jnp.square(residuals))

def goal_loadpath(eq_state):

    forces = eq_state.forces[:-1, :]
    lengths = eq_state.lengths[:-1, :]
    energies = jnp.abs(forces) * lengths

    return ALPHA_LOADPATH * jnp.sum(energies)


@jit
@value_and_grad
def loss_fn(theta):
    """
    Compute the loss function for the braced frame.
    """
    params = combine_parameters(theta)
    eq_state = model(params, structure)

    loss_xyz = goal_xyz(eq_state, xyz_target)
    loss_residual_direction = goal_residual_direction(eq_state, vector_target)
    loss_residual = goal_residual(eq_state)
    loss_loadpath = goal_loadpath(eq_state)

    return loss_xyz + loss_residual_direction + loss_residual + loss_loadpath


# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

theta_init = [
    1.5,
    -1.5,
    1.5,
    topology.node_attribute(2, "y"),
    ]

theta_init = jnp.asarray(theta_init, dtype=jnp.float64)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# evaluate loss funclearction at the start
time_start = perf_counter()
loss, grad = loss_fn(theta_init)
time_end = perf_counter()

# Report loss values per component
parameters_init = combine_parameters(theta_init)
eqstate_init = model(parameters_init, structure)

loss_xyz = goal_xyz(eqstate_init, xyz_target)
loss_residual_direction = goal_residual_direction(eqstate_init, vector_target)
loss_residual = goal_residual(eqstate_init)
loss_loadpath = goal_loadpath(eqstate_init)

print(f"Warm-up function evaluation time: {time_end - time_start:.2f} s")
print(f"Theta init: {theta_init}")
print(f"\nLoss: {loss}")
print(f"\tLoss xyz: {loss_xyz / ALPHA_XYZ}")
print(f"\tLoss residual direction: {loss_residual_direction / ALPHA_RESIDUAL_DIRECTION}")
print(f"\tLoss residual: {loss_residual / ALPHA_RESIDUAL}")
print(f"\tLoss loadpath: {loss_loadpath / ALPHA_LOADPATH}")
print(f"Gradient norm: {jnp.linalg.norm(grad)}")

# Solve optimization problem with scipy
print("\n***Optimizing with scipy***")
optimizer = jaxopt.ScipyMinimize

opt = optimizer(
    fun=loss_fn,
    method="L-BFGS-B",
    tol=TOL,
    maxiter=MAXITER,
    value_and_grad=True
    )

time_start = perf_counter()
opt_result = opt.run(theta_init)  # , bounds)
time_end = perf_counter()
theta_star, opt_state_star = opt_result

# Summary
print(f"Optimization time: {time_end - time_start:.2f} s")
print(f"Success? {opt_state_star.success}")
print(f"Iterations: {opt_state_star.iter_num}")

# Evaluate loss function at optimum point
loss, grad = loss_fn(theta_star)
grad_norm = jnp.linalg.norm(grad)

# Report loss values per component
parameters_star = combine_parameters(theta_star)
eqstate_star = model(parameters_star, structure)

loss_xyz = goal_xyz(eqstate_star, xyz_target)
loss_residual_direction = goal_residual_direction(eqstate_star, vector_target)
loss_residual = goal_residual(eqstate_star)
loss_loadpath = goal_loadpath(eqstate_star)

print(f"\nLoss: {loss}")
print(f"\tLoss xyz: {loss_xyz / ALPHA_XYZ}")
print(f"\tLoss residual direction: {loss_residual_direction / ALPHA_RESIDUAL_DIRECTION}")
print(f"\tLoss residual: {loss_residual / ALPHA_RESIDUAL}")
print(f"\tLoss loadpath: {loss_loadpath / ALPHA_LOADPATH}")
print(f"Gradient norm: {grad_norm}")
print(f"Theta star: {theta_star}")

# Generate optimized compas cem form diagram

form_jax_opt = FormDiagram.from_equilibrium_state(eqstate_star, structure)

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

plotter = Plotter()

# ------------------------------------------------------------------------------
# Plot topology diagram
# ------------------------------------------------------------------------------

plotter.add(topology, nodesize=0.2, show_nodetext=True)

# ------------------------------------------------------------------------------
# Plot translated form diagram
# ------------------------------------------------------------------------------

plotter.add(form_jax.transformed(Translation.from_vector([WIDTH, 0.0, 0.0])),
            nodesize=0.2,
            loadscale=0.5,
            reactionscale=0.5,
            edgetext="force",
            show_edgetext=False)

plotter.add(form_jax_opt.transformed(Translation.from_vector([2 * WIDTH, 0.0, 0.0])),
            nodesize=0.2,
            loadscale=0.5,
            reactionscale=0.5,
            edgetext="force",
            show_edgetext=False)
# ------------------------------------------------------------------------------
# Plot scene
# -------------------------------------------------------------------------------

plotter.zoom_extents()
plotter.show()
