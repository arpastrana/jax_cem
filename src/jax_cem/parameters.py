from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from compas.utilities import pairwise

from compas_cem.diagrams import TopologyDiagram


# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

class ParameterState(NamedTuple):
    """
    The parameters of an equilibrium model.
    """
    xyz: jax.Array  # N x 3
    loads: jax.Array  # N x 3
    forces: jax.Array  # M x 1
    # TODO: find a way to treat edge lengths and planes edgewise, not nodewise
    lengths: jax.Array  # N x 1
    planes: jax.Array  # N x 6

    @classmethod
    def from_topology_diagram(cls, topology: TopologyDiagram):
        """
        Create a parameter state from a COMPAS CEM topology diagram.
        """
        return parameters_from_topology(cls, topology)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def parameters_from_topology(
        cls: type[ParameterState],
        topology: TopologyDiagram
        ) -> type[ParameterState]:
    """
    Create a parameter state from a COMPAS CEM topology diagram.

    Parameters
    ----------
    topology : `compas_cem.diagrams.TopologyDiagram`
        A valid topology diagram.
    """
    nodes = sorted(list(topology.nodes()))
    edges = list(topology.edges())

    loads = jnp.asarray([topology.node_load(node) for node in nodes])
    xyz = jnp.asarray([topology.node_coordinates(node) for node in nodes])

    forces = jnp.asarray([topology.edge_force(edge) for edge in edges])
    forces = jnp.reshape(forces, (-1, 1))

    # TODO: find a way to treat edge lengths or planes per edge, not per node
    lengths = np.zeros((topology.number_of_nodes(), 1))
    planes = np.zeros((topology.number_of_nodes(), 6))

    for trail in topology.trails():
        for u, v in pairwise(trail):
            edge = (u, v)
            if edge not in edges:
                edge = (v, u)
            plane = topology.edge_plane(edge)
            if plane is not None:
                origin, normal = plane
                plane = []
                plane.extend(origin)
                plane.extend(normal)
                planes[u, :] = plane
            else:
                length = topology.edge_length_2(edge)
                if not length:
                    raise ValueError(f"No length defined on trail edge {edge}")
                lengths[u, :] = length

    planes = jnp.asarray(planes)
    lengths = jnp.asarray(lengths)

    return cls(xyz=xyz, loads=loads, lengths=lengths, planes=planes, forces=forces)
