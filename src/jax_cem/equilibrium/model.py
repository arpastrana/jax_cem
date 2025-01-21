import jax

import numpy as np
import jax.numpy as jnp

from jax import vmap

from jax.lax import scan

import equinox as eqx
from equinox.internal import while_loop

from compas.utilities import pairwise

from jax_cem.equilibrium import EquilibriumState
from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized


class EquilibriumModel:
    """
    An equilibrium model that implements the combinatorial equilibrium modeling (CEM) framework.
    """
    def __init__(self):
        pass

    def __call__(self, structure, tmax=10, eta=1e-6):
        """
        Computes an equilibrium state on a structure.

        Parameters
        ----------
        structure : `jax_cem.equilibrium.EquilibriumStructure`
            A structure.

        Returns
        -------
        eq_state: `jax_cem.equilibrium.EquilibriumState`
            An equilibrium state.

        Assumptions
        -----------
        - No shape dependent loads exist in the structure.
        """
        xyz = jnp.zeros((structure.number_of_nodes() + 1, 3))  # NOTE: add dummy last row
        xyz, residuals, lengths = self.equilibrium(structure, xyz)

        if tmax > 1:
            xyz, residuals, lengths = self.equilibrium_iterative(structure, xyz, tmax, eta)

        return self.equilibrium_state(structure, xyz, residuals, lengths)

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(self, structure, xyz):
        """
        Calculate static equilibrium on a structure.
        """
        return self.sequences_equilibrium(structure, xyz, indirect=False)

    def equilibrium_iterative(self, structure, xyz, tmax, eta, scale=1e6):
        """
        Calculate static equilibrium on a structure iteratively.
        """
        def distance(xyz, xyz_last):
            return jnp.sum(jnp.linalg.norm(xyz_last[:-1] - xyz[:-1], axis=1))

        def cond_fn(val):
            xyz, xyz_last = val
            # calculate residual distance
            residual = distance(xyz, xyz_last)
            # if residual distance larger than threshold, continue iterating
            return residual > eta

        def body_fn(val):
            xyz_last, _ = val
            xyz, _, _ = self.sequences_equilibrium(structure, xyz_last, indirect=True)
            return xyz, xyz_last

        init_val = xyz * scale, xyz
        xyz_last, _ = while_loop(cond_fn, body_fn, init_val, max_steps=tmax, kind="checkpointed")

        return self.sequences_equilibrium(structure, xyz_last, indirect=True)

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(self, structure, xyz, residuals, lengths):
        """
        Put together an equilibrium state object.
        """
        # node positions
        xyz = xyz[:-1]  # NOTE: remove dummy last row created in __call__()

        # reaction forces
        reactions = jnp.zeros((structure.number_of_nodes(), 3))
        reactions = reactions.at[structure.support_nodes, :].set(residuals[-1, :])

        # edge forces
        forces = self.edges_force(structure, residuals[:-1, :], lengths[:-1, :], self.forces)

        # edge lengths
        lengths = self.edges_length(structure, xyz)

        return EquilibriumState(xyz=xyz, reactions=reactions, lengths=lengths, loads=self.loads, forces=forces)

    # ------------------------------------------------------------------------------
    # Sequence equilibrium
    # ------------------------------------------------------------------------------

    def sequences_equilibrium(self, structure, xyz, indirect):
        """
        Calculate static equilibrium on a structure.
        """

        def init_state():
            """
            Create an initial scan state
            """
            xyz_seq = self.xyz[structure.origin_nodes, :]
            residuals_seq = jnp.zeros((structure.number_of_trails(), 3))

            return xyz, xyz_seq, residuals_seq

        def sequence_equilibrium(state, sequence):
            """
            Compute static equilibrium on a sequence of nodes in a scan-compatible way.
            """
            _xyz, xyz_seq, residuals_seq = state
            _xyz = _xyz.at[sequence, :].set(xyz_seq)
            state_seq = self.sequence_equilibrium(structure, sequence, _xyz, xyz_seq, residuals_seq, indirect)
            xyz_seq, residuals_seq, lengths_seq = state_seq
            state = _xyz, xyz_seq, residuals_seq

            return state, (residuals_seq, lengths_seq)

        # create initial scan state
        state = init_state()

        # compute static equilibrium in the structure by scanning a function over all sequences
        state, (residuals_seqs, lengths_seqs) = scan(sequence_equilibrium, state, structure.sequences)
        xyz, _, _ = state

        return xyz, residuals_seqs, lengths_seqs

    def sequence_equilibrium(self, structure, sequence, xyz, xyz_seq, residuals_seq, indirect):
        """
        Compute static equilibrium on all the nodes of a sequence.
        """
        # padding mask
        is_sequence_padded = jnp.reshape(sequence, (-1, 1)) < 0

        # node residuals
        residuals_new = self.nodes_equilibrium(structure, sequence, xyz[:-1], residuals_seq, indirect)
        residuals_seq = jnp.where(is_sequence_padded, residuals_seq, residuals_new)

        # trail edge lengths
        # NOTE: Probably inefficient to pre-compute both versions of length
        # Perhaps moving length functions to arguments of jnp.where would skip that evaluation?
        lengths_plane = self.nodes_length_plane(structure, sequence, xyz_seq, residuals_seq)
        lengths_signed = self.lengths[sequence].ravel()
        lengths_seq = jnp.where(lengths_signed != 0.0, lengths_signed, lengths_plane)

        # next node position
        xyz_seq_new = self.nodes_position(xyz_seq, residuals_seq, lengths_seq)
        xyz_seq = jnp.where(is_sequence_padded, xyz_seq, xyz_seq_new)

        return xyz_seq, residuals_seq, lengths_seq

    # ------------------------------------------------------------------------------
    # Node equilibrium
    # ------------------------------------------------------------------------------

    def nodes_equilibrium(self, structure, sequence, xyz, residuals, indirect):
        """
        Calculate static equilibrium at one node of a structure. Vectorized.
        """
        node_equilibrium_vmap = vmap(self.node_equilibrium, in_axes=(None, 0, 0, None, None))
        vectors = self.edges_vector(xyz, structure.connectivity)
        vectors = vmap(vector_normalized)(vectors)

        return node_equilibrium_vmap(structure, sequence, residuals, vectors, indirect)

    def node_equilibrium(self, structure, index, residual, vectors, indirect):
        """
        Calculate static equilibrium at one node of a structure.
        """
        load = self.loads[index, :]
        incidence = structure.incidence[:, index] * structure.deviation_edges

        forces = jnp.ravel(self.forces) * incidence
        if not indirect:
            forces = forces * structure.indirect_edges

        deviation = self.deviation_vector(forces, vectors)

        return self.residual_vector(residual, deviation, load)

    # ------------------------------------------------------------------------------
    # Node position
    # ------------------------------------------------------------------------------

    def nodes_position(self, xyz_seq, residuals, lengths):
        """
        Calculate the position of the next sequence of nodes of a structure.
        """
        return vmap(self.position_vector)(xyz_seq, residuals, lengths)

    def node_position(self, xyz, residual, length):
        """
        Calculate the position of the next node on a trail of a structure.
        """
        return self.position_vector(xyz, residual, length)

    # ------------------------------------------------------------------------------
    # Node lengths
    # ------------------------------------------------------------------------------

    def nodes_length_plane(self, structure, sequence, xyz_seq, residuals):
        """
        Calculate the outgoing edge lengths in a sequence of a structure. Vectorized.
        """
        node_length_plane_vmap = vmap(self.node_length_plane, in_axes=(None, 0, 0, 0))

        return node_length_plane_vmap(structure, sequence, xyz_seq, residuals)

    def node_length_plane(self, structure, index, xyz, residual):
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

    def edges_length(self, structure, xyz):
        """
        The length of the edges of a structure.
        """
        vectors = self.edges_vector(xyz, structure.connectivity)

        return vector_length(vectors)

    # ------------------------------------------------------------------------------
    # Edge forces
    # ------------------------------------------------------------------------------

    def edges_force(self, structure, residuals, lengths, forces):
        """
        The forces in the edges of a structure.
        """
        trail_forces = self.trails_force(residuals, lengths)

        indices = structure.sequences_edges_indices
        trail_forces = trail_forces[indices]

        sequences_edges_flat = jnp.ravel(structure.sequences_edges)
        trail_indices = sequences_edges_flat[indices]
        forces_new = forces.at[trail_indices, :].set(trail_forces)

        return forces_new

    def trails_force(self, residuals, lengths):
        """
        The force in the trail edges of a structure.
        """
        residuals = jnp.concatenate(residuals)
        forces = self.trail_force(residuals)

        lengths = jnp.reshape(jnp.concatenate(lengths), (-1, 1))

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
