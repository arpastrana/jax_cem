from time import perf_counter
from math import pi
from math import cos
from math import sin

import numpy as np

from compas.geometry import Translation

from compas.utilities import pairwise

from compas_cem.diagrams import TopologyDiagram
from compas_cem.diagrams import FormDiagram

from compas_cem.elements import Node
from compas_cem.elements import DeviationEdge

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import TrailEdgeForceConstraint
from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.plotters import Plotter

import jax
import jaxopt

from jax import jit
from jax import value_and_grad

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState

# ------------------------------------------------------------------------------
# Create a topology diagram
# ------------------------------------------------------------------------------

# geometry parameters
diameter = 1.0
num_sides = 16  # only even numbers
appendix_length = 0.10
tension_force = 1.0
compression_force = -0.5
bound = 2.0
grad_method = "AD"

# test number of subdivisions is even
assert num_sides % 2 == 0

# create a topology diagram
topology = TopologyDiagram()

# create nodes, removing last
thetas = np.linspace(0.0, 2 * pi, num_sides + 1)[:-1]
radius = diameter / 2.0

for i, theta in enumerate(thetas):
    x = radius * cos(theta)
    y = radius * sin(theta)

    # nodes in the wheel
    topology.add_node(Node(i, [x, y, 0.0]))

# deviation edges in the perimeter of the wheel tension
for u, v in pairwise(list(range(num_sides)) + [0]):
    topology.add_edge(DeviationEdge(u, v, force=tension_force))

# internal deviation edges are in compression
half_num_sides = num_sides / 2.0

for u in range(int(half_num_sides)):
    v = int(u + half_num_sides)
    topology.add_edge(DeviationEdge(u, v, force=compression_force))

# ------------------------------------------------------------------------------
# Generate trails and auto generate auxiliary trails
# ------------------------------------------------------------------------------

topology.auxiliary_trail_length = appendix_length * -1.0
topology.build_trails(auxiliary_trails=True)
topology0 = topology.copy()

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium
# ------------------------------------------------------------------------------

form = static_equilibrium(topology, eta=1e-6, tmax=100)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# create optimizer
opt = Optimizer()

# add constraint: force in axiliary trail edges should be zero
for edge in topology.auxiliary_trail_edges():
    opt.add_constraint(TrailEdgeForceConstraint(edge, force=0.0))

# add optimization parameters
# the forces in all the deviation edges are allowed to change
for edge in topology.deviation_edges():
    opt.add_parameter(DeviationEdgeParameter(edge, bound, bound))

# optimize
form_opt = opt.solve(topology, algorithm="LBFGS", grad=grad_method, verbose=True)

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology0)
parameters = ParameterState.from_topology_diagram(topology0)
model = EquilibriumModel(tmax=1)
eqstate = model(parameters, structure)

form_jax = FormDiagram.from_equilibrium_state(eqstate, structure)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------

# find auxiliary edges
aux_edges = [structure.edge_index[edge] for edge in topology.auxiliary_trail_edges()]

# define loss function
@jit
@value_and_grad
def loss_fn(diff_params, static_params, structure, y):
    params = eqx.combine(diff_params, static_params)
    eqstate = model(params, structure)
    pred_y = eqstate.forces[aux_edges, :]
    return jnp.sum((y - pred_y) ** 2)

# define targets
y = 0.0

# set tree filtering specification
filter_spec = jtu.tree_map(lambda _: False, parameters)
filter_spec = eqx.tree_at(lambda tree: tree.forces, filter_spec, replace=True)

# split model into differentiable and static submodels
diff_params, static_params = eqx.partition(parameters, filter_spec)

# evaluate loss function at the start
time_start = perf_counter()
loss, grad = loss_fn(diff_params, static_params, structure, y)
time_end = perf_counter()
print(f"Loss function evaluation time: {time_end - time_start:.2f} s")
print(f"{loss=}")

# solve optimization problem with scipy
print("\n***Optimizing with scipy***")

optimizer = jaxopt.ScipyMinimize
opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=100, value_and_grad=True)

time_start = perf_counter()
opt_result = opt.run(diff_params, static_params, structure, y)
time_end = perf_counter()
print(f"Optimization time: {time_end - time_start:.2f} s")

diff_params_star, opt_state_star = opt_result

# evaluate loss function at optimum point
loss, grad = loss_fn(diff_params_star, static_params, structure, y)
print(f"{loss=}")

# generate optimized compas cem form diagram
parameters_star = eqx.combine(diff_params_star, static_params)
eqstate_star = model(parameters_star, structure)
form_jax_opt = FormDiagram.from_equilibrium_state(eqstate_star, structure)

# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

ns = 0.45
shift = diameter * 1.4
plotter = Plotter(figsize=(16.0, 9.0))

# plot topology diagram
plotter.add(topology, nodesize=ns)

# plot translated form diagram
T = Translation.from_vector([shift, 0.0, 0.0])
plotter.add(form.transformed(T), nodesize=ns)

# plot translated optimized form diagram
T = Translation.from_vector([shift * 2.0, 0.0, 0.0])
plotter.add(form_opt.transformed(T), nodesize=ns)

T = Translation.from_vector([shift * 3.0, 0.0, 0.0])
plotter.add(form_jax_opt.transformed(T), nodesize=ns, show_nodetext=True)

# show scene
plotter.zoom_extents(padding=-0.3)
plotter.show()
