import jax

import numpy as np
import jax.numpy as jnp

from jax import vmap

import equinox as eqx

from compas.utilities import pairwise

from jax_cem.equilibrium import EquilibriumState
from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized


__all__ = ["EquilibriumModel"]


class EquilibriumModel(eqx.Module):
    """
    An equilibrium model that implements the combinatorial equilibrium modeling (CEM) framework.

    Parameters
    ----------
    xyz: `jnp.array`
        The XYZ coordinates of the start nodes of the trails in a topology diagram.
    loads: `jnp.array`
        The XYZ components of the point loads applied to the nodes of a topology diagram.
    forces: `jnp.array`
        The signed magnitude of the deviation edges of a topology diagram.
    planes: `jnp.array`
        The projection planes of trail edges of a topology diagram.
    """

    xyz: jax.Array  # N x 3 or A x 3?
    loads: jax.Array  # N x 3
    lengths: jax.Array  # N x 6?
    planes: jax.Array  # N x 6?
    forces: jax.Array  # M x 1  or C x 1?

    def __init__(self, xyz, loads, lengths, planes, forces):
        self.xyz = xyz
        self.loads = loads
        self.lengths = lengths
        self.planes = planes
        self.forces = forces

    @classmethod
    def from_topology_diagram(cls, topology):
        """
        Create an equilibrium model from a COMPAS CEM topology diagram.

        Parameters
        ----------
        topology : `compas_cem.diagrams.TopologyDiagram`
            A valid topology diagram.
        """
        return model_from_topology(cls, topology)

    def __call__(self, topology, *args, **kwargs):
        """
        Compute an equilibrium state on a structure given a topology diagram.

        The computation follows the combinatorial equilibrium modeling (CEM) form-finding algorithm.

        Parameters
        ----------
        topology : `jax_cem.equilibrium.EquilibriumStructure`
            A structure.

        Returns
        -------
        eqstate: `jax_cem.equilibrium.EquilibriumState`
            An equilibrium state.

        Assumptions
        -----------
        - No indirect deviation edges exist in the structure.
        - No shape-dependent loads are applied to the structure.
        """
        # for t in range(kmax)...
        xyz = jnp.zeros((topology.number_of_nodes() + 1, 3))
        xyz_seq = self.xyz[topology.origin_nodes, :]
        residuals_seq = jnp.zeros((topology.number_of_trails(), 3))

        residuals_seqs = []
        lengths_seqs = []

        for i, sequence in enumerate(topology.sequences):

            # update position matrix
            xyz = xyz.at[sequence, :].set(xyz_seq)

            # sequence equilibrium
            state_seq = self.sequence_equilibrium(topology, sequence, xyz, residuals_seq)
            xyz_seq, residuals_seq, lengths_seq = state_seq

            # store
            residuals_seqs.append(residuals_seq)
            lengths_seqs.append(lengths_seq)

        # node coordinates
        xyz = xyz[:-1]

        # reaction forces
        reactions = jnp.zeros((topology.number_of_nodes(), 3))
        reactions = reactions.at[sequence, :].set(residuals_seq)

        # edge forces
        forces = jnp.where(self.forces != 0.0, self.forces, 0.0)
        forces = self.edges_force(topology, residuals_seqs[:-1], lengths_seqs[:-1], forces)

        # edge lengths
        lengths = self.edges_length(topology, xyz)

        return EquilibriumState(xyz=xyz, reactions=reactions, lengths=lengths, loads=self.loads, forces=forces)


    # ------------------------------------------------------------------------------
    # Sequence equilibrium
    # ------------------------------------------------------------------------------

    def sequence_equilibrium(self, topology, sequence, xyz, residuals_seq):
        """
        Compute static equilibrium on all the nodes of a sequence.
        """
        # node positions
        xyz_seq = xyz[sequence, :]

        # padding mask
        is_sequence_padded = np.reshape(sequence, (-1, 1)) < 0

        # node residuals
        residuals_new = self.nodes_equilibrium(topology, sequence, xyz[:-1], residuals_seq)
        residuals_seq = jnp.where(is_sequence_padded, residuals_seq, residuals_new)

        # trail edge lengths
        lengths_plane = self.nodes_length_plane(topology, sequence, xyz_seq, residuals_seq)
        lengths_signed = self.lengths[sequence].ravel()
        lengths_seq = jnp.where(lengths_signed != 0.0, lengths_signed, lengths_plane)

        # next node position
        xyz_seq_new = self.nodes_position(xyz_seq, residuals_seq, lengths_seq)
        xyz_seq = jnp.where(is_sequence_padded, xyz_seq, xyz_seq_new)

        return xyz_seq, residuals_seq, lengths_seq

    # ------------------------------------------------------------------------------
    # Node equilibrium
    # ------------------------------------------------------------------------------

    def nodes_equilibrium(self, topology, sequence, xyz, residuals):
        """
        Calculate static equilibrium at one node of a topology diagram. Vectorized.
        """
        node_equilibrium_vmap = vmap(self.node_equilibrium, in_axes=(None, 0, 0, None))
        vectors = self.edges_vector(xyz, topology.connectivity)
        vectors = vmap(vector_normalized)(vectors)

        return node_equilibrium_vmap(topology, sequence, residuals, vectors)

    def node_equilibrium(self, topology, index, residual, vectors):
        """
        Calculate static equilibrium at one node of a topology diagram.
        """
        load = self.loads[index, :]
        incidence = topology.incidence[:, index] * topology.deviation_edges
        forces = jnp.ravel(self.forces) * incidence

        deviation = self.deviation_vector(forces, vectors)

        return self.residual_vector(residual, deviation, load)

    # ------------------------------------------------------------------------------
    # Node position
    # ------------------------------------------------------------------------------

    def nodes_position(self, xyz_seq, residuals, lengths):
        """
        Calculate the position of the next sequence of nodes of a topology diagram.
        """
        return vmap(self.position_vector)(xyz_seq, residuals, lengths)

    def node_position(self, xyz, residual, length):
        """
        Calculate the position of the next node on a trail of a topology diagram.
        """
        return self.position_vector(xyz, residual, length)

    # ------------------------------------------------------------------------------
    # Node lengths
    # ------------------------------------------------------------------------------

    def nodes_length_plane(self, topology, sequence, xyz_seq, residuals):
        """
        Calculate the outgoing edge lengths in a sequence of a topology diagram. Vectorized.
        """
        node_length_plane_vmap = vmap(self.node_length_plane, in_axes=(None, 0, 0, 0))

        return node_length_plane_vmap(topology, sequence, xyz_seq, residuals)

    def node_length_plane(self, topology, index, xyz, residual):
        """
        Compute the outgoing length from a node.

        Notes
        -----
        It assumes that the residual vector and plane normal vector are not parallel.
        """
        plane = self.planes[index, :]
        origin = plane[:3]
        normal = plane[3:]

        # return zero cos nop if plane normal is zero
        # may raise nans, use double where trick
        is_zero_normal = jnp.allclose(normal, 0.0)
        normal = jnp.where(is_zero_normal, jnp.ones_like(normal), normal)
        cos_nop = jnp.where(is_zero_normal, 0.0, normal @ (origin - xyz))

        # return zero length if residual is zero
        # may raise nans, use double where trick
        is_zero_res = jnp.allclose(residual, 0.0)
        residual = jnp.where(is_zero_res, jnp.ones_like(residual), residual)
        length = jnp.where(is_zero_res, 0.0, cos_nop / (normal @ vector_normalized(residual)))

        return length

    # ------------------------------------------------------------------------------
    # Edge lengths
    # ------------------------------------------------------------------------------

    def edges_length(self, topology, xyz):
        """
        The length of the edges in a topology diagram.
        """
        vectors = self.edges_vector(xyz, topology.connectivity)

        return vector_length(vectors)

    # ------------------------------------------------------------------------------
    # Edge forces
    # ------------------------------------------------------------------------------

    def edges_force(self, topology, residuals, lengths, forces):
        """
        The forces in the edges in a topology diagram.
        """
        trail_forces = self.trails_force(residuals, lengths)

        sequences_edges = topology.sequences_edges

        return forces.at[sequences_edges.ravel(), :].set(trail_forces)

    def trails_force(self, residuals, lengths):
        """
        The force in the trail edges of a topology diagram.
        """
        residuals = jnp.concatenate(residuals)
        lengths = jnp.reshape(jnp.concatenate(lengths), (-1, 1))
        forces = self.trail_force(residuals)

        return jnp.copysign(forces, lengths)

    # ------------------------------------------------------------------------------
    # Static operations
    # ------------------------------------------------------------------------------

    @staticmethod
    def deviation_vector(forces, vectors):
        """
        Calculate the resultant deviation vector incoming to a node.
        """
        return forces.T @ vectors

    @staticmethod
    def trail_length(trail_lengths, incidence):
        """
        Get the length of the next trail edge outgoing from a node.
        """
        return incidence * trail_lengths  # (num_edges, num_nodes)

    @staticmethod
    def trail_force(residual):
        """
        The force passing through a trail edge.
        """
        return vector_length(residual)

    @staticmethod
    def residual_vector(residual, deviation, load):
        """
        The updated residual vector at a node.
        """
        return residual - deviation - load

    @staticmethod
    def position_vector(position, residual, trail_length):
        """
        The position of the next node on a trail.
        """
        return position + trail_length * vector_normalized(residual)

    @staticmethod
    def edges_vector(xyz, connectivity):
        """
        The edge vectors of the graph.
        """
        return connectivity @ xyz


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def model_from_topology(cls, topology):
    """
    Create an equilibrium model from a COMPAS CEM topology diagram.

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

    edges = list(topology.edges())
    for trail in topology.trails():
        for u, v in pairwise(trail):
            edge = (u, v)
            if edge not in edges:
                edge = (v, u)
            plane = topology.edge_plane(edge)
            if plane is not None:
                origin, normal = plane
                planes[u, :] = origin + normal
            else:
                length = topology.edge_length_2(edge)
                if not length:
                    raise ValueError(f"No length defined on edge {edge}")
                lengths[u, :] = length

    planes = jnp.asarray(planes)
    lengths = jnp.asarray(lengths)

    return cls(xyz=xyz, loads=loads, lengths=lengths, planes=planes, forces=forces)


if __name__ == "__main__":
    pass
