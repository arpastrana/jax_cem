from compas.geometry import length_vector
from compas.geometry import Translation

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge

from compas_cem.loads import NodeLoad

from compas_cem.plotters import Plotter

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState

import jax.numpy as jnp

from scipy.optimize import minimize

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

OPTIMIZE = True
TOL = 1e-9
MAXITER = 100

PARAMETRIZE_Y = True

ALPHA_RESIDUAL = 10000.0
ALPHA_LOADPATH = 1.0

# ------------------------------------------------------------------------------
# Topology Diagram
# ------------------------------------------------------------------------------

points = [
    (0, [0.0, 0.0, 0.0]),
    (1, [0.0, HEIGHT, 0.0]),
    (2, [WIDTH / 2.0, 0.5 * HEIGHT, 0.0]),
    (3, [WIDTH, 0.0, 0.0]),
    (4, [WIDTH, HEIGHT, 0.0])
    ]

trail_edges = [
    # ((0, 1), HEIGHT),
    ]

deviation_edges = [
    ((0, 1), 1.0),
    ((1, 2), -1.0),
    ((2, 0), 1.0),
    ((3, 4), -1.0),
    ((4, 2), 1.0),
    ((2, 3), -1.0),
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
topology.add_load(NodeLoad(4, load))

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

if OPTIMIZE:
    nodes_residual = [6, 7, 9]
    node_y = 2
    nodes_y = [node_y]
    node_index_y = structure.node_index[node_y]
    node_indices_residual = [structure.node_index[node] for node in nodes_residual]
    node_indices_residual = jnp.array(node_indices_residual, dtype=jnp.int64)

    edge_indices_force = [structure.edge_index[edge] for edge in topology.deviation_edges()]
    edge_indices_force = jnp.array(edge_indices_force, dtype=jnp.int64)

# ------------------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------------------

    def combine_parameters(theta):
        """
        Combine the parameters into a single array.
        """
        if PARAMETRIZE_Y:
            forces = theta[:-1]
        else:
            forces = theta

        forces = parameters.forces.at[edge_indices_force, :].set(forces[:, None])

        if PARAMETRIZE_Y:
            y = theta[-1]
            xyz = parameters.xyz.at[node_index_y, 1].set(y)
        else:
            xyz = parameters.xyz

        return ParameterState(
            xyz=xyz,
            loads=parameters.loads,
            forces=forces,
            lengths=parameters.lengths,
            planes=parameters.planes
            )


    def goal_residual(eq_state):
        residuals = eq_state.reactions[node_indices_residual]

        return ALPHA_RESIDUAL * jnp.sum(jnp.square(residuals))

    def goal_loadpath(eq_state):
        forces = eq_state.forces[edge_indices_force, :]
        lengths = eq_state.lengths[edge_indices_force, :]
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

        loss_residual = goal_residual(eq_state)
        loss_loadpath = goal_loadpath(eq_state)

        return loss_residual + loss_loadpath


# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

    theta_init = [topology.edge_force(edge) for edge in topology.deviation_edges()]
    if PARAMETRIZE_Y:
        theta_init += [topology.node_attribute(2, "y")]
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

    loss_residual = goal_residual(eqstate_init)
    loss_loadpath = goal_loadpath(eqstate_init)

    print(f"Warm-up function evaluation time: {time_end - time_start:.2f} s")
    print(f"Theta init: {theta_init}")
    print(f"\nLoss: {loss}")
    print(f"\tLoss residual: {loss_residual / ALPHA_RESIDUAL}")
    print(f"\tLoss loadpath: {loss_loadpath / ALPHA_LOADPATH}")
    print(f"Gradient norm: {jnp.linalg.norm(grad)}")

    # Solve optimization problem with scipy
    print("\n***Optimizing with scipy***")
    time_start = perf_counter()
    opt_result = minimize(
        fun=loss_fn,
        x0=theta_init,
        jac=True,
        method="L-BFGS-B",
        tol=TOL,
        options={"maxiter": MAXITER}
    )
    time_end = perf_counter()
    theta_star = opt_result.x

    # Summary
    print(f"Optimization time: {time_end - time_start:.2f} s")
    print(f"Success? {opt_result.success}")
    print(f"Iterations: {opt_result.nit}")

    # Evaluate loss function at optimum point
    loss, grad = loss_fn(theta_star)

    # Report loss values per component
    parameters_star = combine_parameters(theta_star)
    eqstate_star = model(parameters_star, structure)

    loss_residual = goal_residual(eqstate_star)
    loss_loadpath = goal_loadpath(eqstate_star)

    print(f"\nLoss: {loss}")
    print(f"\tLoss residual: {loss_residual / ALPHA_RESIDUAL}")
    print(f"\tLoss loadpath: {loss_loadpath / ALPHA_LOADPATH}")
    print(f"Gradient norm: {jnp.linalg.norm(grad)}")
    print(f"Theta star: {theta_star}")

    # Generate optimized compas cem form diagram
    form_jax_opt = FormDiagram.from_equilibrium_state(eqstate_star, structure)

    print(f"\nResiduals:")
    for node in nodes_residual:
        print(f"\tNode {node} residual: {length_vector(form_jax_opt.reaction_force(node))}")

    print(f"\nHeights:")
    for node in nodes_y:
        print(f"\tNode {node} ratio: {form_jax_opt.node_attribute(node, 'y') / HEIGHT:.3f}")

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

edges_2_show = [edge for edge in topology.edges() if not topology.is_auxiliary_trail_edge(edge)]
plotter.add(form_jax.transformed(Translation.from_vector([1.5 * WIDTH, 0.0, 0.0])),
            edges=edges_2_show,
            nodesize=0.2,
            loadscale=0.5,
            reactionscale=0.5,
            edgetext="force",
            show_edgetext=False)

if OPTIMIZE:
    plotter.add(form_jax_opt.transformed(Translation.from_vector([1.5 * 2 * WIDTH, 0.0, 0.0])),
                edges=edges_2_show,
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
