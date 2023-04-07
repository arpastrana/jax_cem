from time import time
from math import pi
from math import cos
from math import sin

import numpy as np

from compas.geometry import Translation

from compas.utilities import pairwise

from compas_cem.diagrams import TopologyDiagram

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
from jax import grad

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate


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
model = EquilibriumModel.from_topology_diagram(topology0)
eqstate = model(structure)
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------

# find auxiliary edges
aux_edges = [structure.edge_index[edge] for edge in topology.auxiliary_trail_edges()]


# define loss function
@jit
def loss_fn(diff_model, static_model, structure, y):
    model = eqx.combine(diff_model, static_model)
    eqstate = model(structure)
    pred_y = eqstate.forces[aux_edges, :]
    return jnp.sum((y - pred_y) ** 2)


# define targets
y = 0.0

# set tree filtering specification
filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(lambda tree: tree.forces, filter_spec, replace=True)

# split model into differentiable and static submodels
diff_model, static_model = eqx.partition(model, filter_spec)

# evaluate loss function at the start
loss = loss_fn(diff_model, static_model, structure, y)
print(f"{loss=}")

# solve optimization problem with scipy
print("\n***Optimizing with scipy***")

optimizer = jaxopt.ScipyMinimize
opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=100)

opt_result = opt.run(diff_model, static_model, structure, y)
diff_model_star, opt_state_star = opt_result

# evaluate loss function at optimum point
loss = loss_fn(diff_model_star, static_model, structure, y)
print(f"{loss=}")

# generate optimized compas cem form diagram
model_star = eqx.combine(diff_model_star, static_model)
eqstate_star = model_star(structure)
form_jax_opt = form_from_eqstate(structure, eqstate_star)

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
plotter.add(form_jax.transformed(T), nodesize=ns)

# plot translated optimized form diagram
T = Translation.from_vector([shift * 2.0, 0.0, 0.0])
# plotter.add(form_opt.transformed(T), nodesize=ns)
plotter.add(form_jax_opt.transformed(T), nodesize=ns, show_nodetext=True)

# show scene
plotter.zoom_extents(padding=-0.3)
plotter.show()
