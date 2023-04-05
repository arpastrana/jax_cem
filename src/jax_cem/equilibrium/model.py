import jax

import numpy as np
import jax.numpy as jnp

from jax import vmap

import equinox as eqx

from compas.utilities import pairwise

from jax_cem.equilibrium import EquilibriumState
from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized


class EquilibriumModel(eqx.Module):
    """
    An equilibrium model that implements the combinatorial equilibrium modeling (CEM) framework.

    Parameters
    ----------
    xyz: `jnp.array`
        The XYZ coordinates of the start nodes of the trails in a topology diagram.
    loads: `jnp.array`
        The XYZ components of the point loads applied to the nodes of a topology diagram.
    states: `jnp.array`
        The combinatorial force state of the edges of a topology diagram (tension or compression).
    forces: `jnp.array`
        The absolute magnitude of the deviation edges of a topology diagram.
    planes: `jnp.array`
        The projection planes of trail edges of a topology diagram.
    """
    xyz: jnp.array  # N x 3 or A x 3?
    loads : jnp.array  # N x 3
    states : jnp.array  # M x 1 (static)
    forces : jnp.array  # M x 1  or C x 1?
    planes  : jnp.array  # (static! for the time being!)

    @classmethod
    def from_topology_diagram(cls, topology):
        """
        Create an equilibrium model from a COMPAS CEM topology diagram.
        """
        loads = jnp.asarray([topology.node_load(node) for node in topology.nodes()])
        xyz = jnp.asarray([topology.node_coordinates(node) for node in topology.nodes()])
        forces = jnp.reshape(jnp.asarray([topology.edge_force(edge) for edge in topology.edges()]), (-1, 1))
        lengths = jnp.reshape(jnp.asarray([topology.edge_length_2(edge) for edge in topology.edges()]), (-1, 1))

        # TODO: find a way to treat edge lengths per edge, not per node
        lengths = np.zeros((topology.number_of_nodes(), 1))
        edges = list(topology.edges())
        for trail in topology.trails():
            for u, v in pairwise(trail):
                edge = (u, v)
                if edge not in edges:
                    edge = (v, u)
                lengths[u, :] = topology.edge_length_2(edge)
        lengths = jnp.asarray(lengths)

        return cls(xyz, lengths, forces, loads)

    def __call__(self, topology):
        """
        Compute an equilibrium state.

        The computation follows the combinatorial equilibrium modeling (CEM) form-finding algorithm.

        Returns
        -------
        eqstate: `jax_cem.equilibrium.EquilibriumState`
            An equilibrium state.

        Assumptions
        -----------
        - No indirect deviation edges exist.
        - No shape-dependent loads exist.
        """
        xyz = jnp.zeros((topology.number_of_nodes(), 3))
        # forces = jnp.zeros((topology.number_of_edges(), 1))
        residuals = jnp.zeros((topology.number_of_trails(), 3))

        # for t in range(kmax)...
        sequence_start= topology.sequences[0, :]
        xyz_seq = self.xyz[sequence_start, :]

        residuals_sequence = []
        # residuals_sequence = jnp.zeros((topology.number_of_sequences() - 1, topology.number_of_trails()))

        for i, sequence in enumerate(topology.sequences):

            # update position matrix
            xyz = xyz.at[sequence, :].set(xyz_seq)

            residuals = self.nodes_equilibrium(topology,
                                               sequence,
                                               xyz,
                                               residuals)

            residuals_sequence.append(residuals)

            # trail edge lengths
            lengths = self.nodes_length(topology, sequence, xyz_seq, residuals)

            # next position
            xyz_seq = self.nodes_position(xyz_seq, residuals, lengths)

        # trail edges force
        # (M x 1) ordering based on raveled sequences
        # trail_forces = self.trail_forces(residuals)
        # trail_sequence = topology.sequence_trail_index(sequence)
        # forces = forces.at[trail_sequence, :].set(trail_forces)

        # edge forces
        # TODO: combine trail and deviation edge forces
        # forces = self.edges_forces()

        # edge lengths
        lengths = self.edges_length(topology, xyz)

        return EquilibriumState(xyz=xyz,
                                reaction_forces=residuals,
                                lengths=lengths,
                                forces=forces)

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
    # Node equilibrium
    # ------------------------------------------------------------------------------

    def nodes_equilibrium(self, topology, sequence, xyz, residuals):
        """
        Calculate static equilibrium at one node of a topology diagram. Vectorized.
        """
        vectors = self.edges_vector(xyz, topology.connectivity)
        vectors = vmap(vector_normalized)(vectors)

        node_equilibrium_vmap = vmap(self.node_equilibrium, in_axes=(None, 0, None, 0, None, None))

        return node_equilibrium_vmap(topology, sequence, xyz, residuals, vectors)

    def node_equilibrium(self, topology, index, xyz, residual, vectors):
        """
        Calculate static equilibrium at one node of a topology diagram.
        """
        # data prep work
        position = xyz[index, :]
        load = self.loads[index, :]
        length = self.lengths[index, :]

        incidence = topology.incidence_signed[:, index]
        incidence = np.reshape(incidence, (-1, 1))  # NOTE: can we not reshape?

        deviation = self.deviation_vector(self.forces, vectors, incidence)
        deviation = deviation.flatten()  # NOTE: can we not flatten?

        return self.residual_vector(residual, deviation, load)

    # ------------------------------------------------------------------------------
    # Node lengths
    # ------------------------------------------------------------------------------

    def nodes_length(self, topology, sequence, xyz_seq, residuals):
        """
        Calculate the outgoing edge lengths in a sequence of a topology diagram. Vectorized.
        """
        node_length_vmap = vmap(self.node_length, in_axes=(None, 0, 0, 0))

        return node_length_vmap(topology, sequence, xyz_seq, residuals)

    def node_length(self, topology, index, xyz, residual):
        """
        Compute the outgoing length from a node.

        Notes
        -----
        It assumes that the residual vector and plane normal vector are not parallel.
        """
        incidence_node = topology.incidence[:, index]
        incidence_node = np.abs(np.reshape(incidence, (-1, 1)))  # NOTE: can we not reshape?
        mask = incidence_node * topology.trail_edges * topology.mask_trailedges_out

        plane = self.planes @ mask
        origin = plane[:, 3]
        normal = plane[3, :]

        cos_nr = normal @ vector_normalized(residual)
        op = origin - point
        cos_nop = normal @ op

        return cos_nop / cos_nr

    # ------------------------------------------------------------------------------
    # Edge lengths
    # ------------------------------------------------------------------------------

    def edges_length(self, topology, xyz):
        """
        The length of the edges in a topology diagram.
        """
        vectors = self.edges_vector(xyz, topology.connectivity)
        return vector_length(edge_vectors)

    # ------------------------------------------------------------------------------
    # Edge forces
    # ------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------
    # Static operations
    # ------------------------------------------------------------------------------

    @staticmethod
    def deviation_vector(deviation_forces, vectors, incidence):
        """
        Calculate the resultant deviation vector incoming to a node.
        """
        incident_forces = incidence * deviation_forces * states # (num edges, num nodes seq)
        return incident_forces.T @ vectors

    @staticmethod
    def trail_length(trail_lengths, incidence):
        """
        Get the length of the next trail edge outgoing from a node.
        """
        incident_length = incidence * trail_lengths  # (num_edges, num_nodes)

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
    def trails_force(residual):
        """
        The force passing through a trail edge.
        """
        return vector_length(residual)

    @staticmethod
    def edges_vector(xyz, connectivity):
        """
        The edge vectors of the graph.
        """
        return connectivity @ xyz


if __name__ == "__main__":
    # eqstate = model(topology)
    pass
