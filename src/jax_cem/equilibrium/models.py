from typing import Tuple

import jax
import jax.numpy as jnp

from jax import vmap
from jax.lax import scan

from equinox.internal import while_loop

from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized

from jax_cem.datastructures import Structure
from jax_cem.parameters import ParameterState
from jax_cem.equilibrium import EquilibriumState
from jax_cem.equilibrium import EquilibriumSequenceState


class EquilibriumModel:
    """
    An equilibrium model that implements the combinatorial equilibrium modeling (CEM) framework.
    """
    def __init__(
            self,
            tmax: int = 10,
            eta: float = 1.0e-6,
            scale: float = 1.0e6,
            verbose: bool = False
            ):
        self.tmax = tmax
        self.eta = eta
        self.scale = scale
        self.verbose = verbose

    def __call__(
            self,
            params: ParameterState,
            structure: Structure
            ) -> EquilibriumState:
        """
        Computes an equilibrium state on a structure.

        Parameters
        ----------
        parameters : `jax_cem.parameters.ParameterState`
            The parameters of the equilibrium model.
        structure : `jax_cem.datastructures.Structure`
            A structure.

        Returns
        -------
        eq_state: `jax_cem.equilibrium.EquilibriumState`
            An equilibrium state.

        Assumptions
        -----------
        - No shape dependent loads exist in the structure.
        """
        # NOTE: Add dummy last row to consider shifted sequences
        xyz = jnp.zeros((structure.number_of_nodes() + 1, 3))
        data = self.equilibrium(params, structure, xyz)

        if self.tmax > 1:
            xyz, *_ = data
            data = self.equilibrium_iterative(
                params,
                structure,
                xyz,
                self.tmax,
                self.eta,
                self.scale
                )

        return self.equilibrium_state(params, structure, data)

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(
            self,
            params: ParameterState,
            structure: Structure,
            data: Tuple[jax.Array, jax.Array, jax.Array]
            ) -> EquilibriumState:
        """
        Assemble an equilibrium state.
        """
        # Unpack data
        xyz, residuals, lengths = data

        # Node positions
        # NOTE: We remove the dummy last row created in __call__()
        xyz = xyz[:-1]

        # Reaction forces
        reactions = jnp.zeros((structure.number_of_nodes(), 3))
        reactions = reactions.at[structure.support_nodes, :].set(residuals[-1, :])

        # Edge forces
        forces = self.edges_force(structure, residuals[:-1, :], lengths[:-1, :], params.forces)

        # Edge lengths
        lengths = self.edges_length(structure, xyz)

        # Create equilibrium state
        state = EquilibriumState(
            xyz=xyz,
            reactions=reactions,
            lengths=lengths,
            loads=params.loads,
            forces=forces
            )

        return state

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(
            self,
            params: ParameterState,
            structure: Structure,
            xyz: jax.Array
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Calculate static equilibrium on a structure.
        """
        return self.sequences_equilibrium(params, structure, xyz, use_indirect=False)

    def equilibrium_iterative(
            self,
            params: ParameterState,
            structure: Structure,
            xyz: jax.Array,
            tmax: int,
            eta: float,
            scale: float
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Calculate static equilibrium on a structure iteratively.
        """
        def distance(xyz, xyz_last):
            return jnp.sum(jnp.linalg.norm(xyz_last[:-1] - xyz[:-1], axis=1))

        def cond_fn(val):
            xyz, xyz_last = val
            # Calculate residual distance
            residual = distance(xyz, xyz_last)
            # If residual distance larger than threshold, continue iterating
            return residual > eta

        def body_fn(val):
            xyz_last, _ = val
            xyz, _, _ = self.sequences_equilibrium(params, structure, xyz_last, use_indirect=True)
            return xyz, xyz_last

        # Initialize iteration
        init_val = xyz * scale, xyz

        # Iterate
        xyz_last, _ = while_loop(cond_fn, body_fn, init_val, max_steps=tmax, kind="checkpointed")

        return self.sequences_equilibrium(params, structure, xyz_last, use_indirect=True)

    # ------------------------------------------------------------------------------
    # Sequence equilibrium
    # ------------------------------------------------------------------------------

    def sequences_equilibrium(
            self,
            params: ParameterState,
            structure: Structure,
            xyz: jax.Array,
            use_indirect: bool
            ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Calculates equilibrium on all the sequences of a structure.
        """
        def calculate_sequence_state_start():
            """
            Creates an initial scan state.
            """
            xyz_seq = params.xyz[structure.origin_nodes, :]
            residuals_seq = jnp.zeros((structure.number_of_trails(), 3))

            return xyz, xyz_seq, residuals_seq

        def calculate_sequence_state(state, sequence):
            """
            Compute static equilibrium on a sequence of nodes in a scan-compatible way.
            """
            _xyz, xyz_seq, residuals_seq = state
            _xyz = _xyz.at[sequence, :].set(xyz_seq)

            state_seq = self.sequence_equilibrium(
                params,
                structure,
                sequence,
                _xyz,
                xyz_seq,
                residuals_seq,
                use_indirect
                )

            state_out = (_xyz, state_seq.xyz, state_seq.residuals)
            carry_out = (state_seq.residuals, state_seq.lengths)

            return state_out, carry_out

        # Create initial scan state
        state_start = calculate_sequence_state_start()

        # Compute static equilibrium in the structure by scanning a function over all sequences
        state_end, (residuals_seqs, lengths_seqs) = scan(
            calculate_sequence_state,
            state_start,
            structure.sequences
            )

        xyz, *_ = state_end

        return xyz, residuals_seqs, lengths_seqs

    def sequence_equilibrium(
            self,
            params: ParameterState,
            structure: Structure,
            sequence: jax.Array,
            xyz: jax.Array,
            xyz_seq: jax.Array,
            residuals_seq: jax.Array,
            use_indirect: bool
            ) -> EquilibriumSequenceState:
        """
        Compute static equilibrium on all the nodes of a sequence.
        """
        # Padding mask
        is_sequence_padded = jnp.reshape(sequence, (-1, 1)) < 0

        # Node residuals
        residuals_new = self.nodes_equilibrium(params, structure, sequence, xyz[:-1], residuals_seq, use_indirect)
        residuals_seq = jnp.where(is_sequence_padded, residuals_seq, residuals_new)

        # Trail edge lengths
        # NOTE: Probably inefficient to pre-compute both versions of length
        # Perhaps moving length functions to arguments of jnp.where would skip that evaluation?
        lengths_plane = self.nodes_length_plane(params, sequence, xyz_seq, residuals_seq)
        lengths_signed = params.lengths[sequence].ravel()
        lengths_seq = jnp.where(lengths_signed != 0.0, lengths_signed, lengths_plane)

        # Position of the next node
        xyz_seq_new = self.nodes_position(xyz_seq, residuals_seq, lengths_seq)
        xyz_seq = jnp.where(is_sequence_padded, xyz_seq, xyz_seq_new)

        # Create sequence state
        state = EquilibriumSequenceState(
            xyz=xyz_seq,
            residuals=residuals_seq,
            lengths=lengths_seq
            )

        return state

    # ------------------------------------------------------------------------------
    # Node equilibrium
    # ------------------------------------------------------------------------------

    def nodes_equilibrium(
            self,
            params: ParameterState,
            structure: Structure,
            sequence: jax.Array,
            xyz: jax.Array,
            residuals: jax.Array,
            use_indirect: bool
            ) -> jax.Array:
        """
        Calculate static equilibrium at one node of a structure. Vectorized.
        """
        node_equilibrium_vmap = vmap(self.node_equilibrium, in_axes=(None, None, 0, 0, None, None))
        vectors = edges_vector(xyz, structure.connectivity)
        vectors = vmap(vector_normalized)(vectors)

        return node_equilibrium_vmap(params, structure, sequence, residuals, vectors, use_indirect)

    def node_equilibrium(
            self,
            params: ParameterState,
            structure: Structure,
            index: int,
            residual: jax.Array,
            vectors: jax.Array,
            use_indirect: bool
            ) -> jax.Array:
        """
        Calculate static equilibrium at one node of a structure.
        """
        load = params.loads[index, :]
        incidence = structure.incidence[:, index] * structure.deviation_edges

        forces = jnp.ravel(params.forces) * incidence
        if not use_indirect:
            forces = forces * structure.indirect_edges

        deviation = deviation_vector(forces, vectors)

        return residual_vector(residual, deviation, load)

    # ------------------------------------------------------------------------------
    # Node position
    # ------------------------------------------------------------------------------

    def nodes_position(
            self,
            xyz_seq: jax.Array,
            residuals: jax.Array,
            lengths: jax.Array
            ) -> jax.Array:
        """
        Calculate the position of the next sequence of nodes of a structure.
        """
        return vmap(self.node_position)(xyz_seq, residuals, lengths)

    def node_position(
            self,
            xyz: jax.Array,
            residual: jax.Array,
            length: jax.Array
            ) -> jax.Array:
        """
        Calculate the position of the next node on a trail of a structure.
        """
        return position_vector(xyz, residual, length)

    # ------------------------------------------------------------------------------
    # Node lengths
    # ------------------------------------------------------------------------------

    def nodes_length_plane(
            self,
            params: ParameterState,
            sequence: jax.Array,
            xyz_seq: jax.Array,
            residuals: jax.Array
            ) -> jax.Array:
        """
        Calculate the outgoing edge lengths in a sequence. Vectorized.
        """
        node_length_plane_vmap = vmap(self.node_length_plane, in_axes=(None, 0, 0, 0))

        return node_length_plane_vmap(params, sequence, xyz_seq, residuals)

    def node_length_plane(
            self,
            params: ParameterState,
            index: int,
            xyz: jax.Array,
            residual: jax.Array
            ) -> jax.Array:
        """
        Compute the outgoing length from a node.

        Notes
        -----
        It assumes that the residual vector and plane normal vector are not parallel.
        """
        plane = params.planes[index, :]
        origin = plane[:3]
        normal = plane[3:]

        # Return zero cos nop if plane normal is zero
        # May raise NaNs, use double where trick
        is_zero_normal = jnp.allclose(normal, 0.0)
        normal = jnp.where(is_zero_normal, jnp.ones_like(normal), normal)
        cos_nop = jnp.where(is_zero_normal, 0.0, normal @ (origin - xyz))

        # Return zero length if residual is zero
        # May raise NaNs, use double where trick
        is_zero_res = jnp.allclose(residual, 0.0)
        residual = jnp.where(is_zero_res, jnp.ones_like(residual), residual)

        # Safeguard against residual pointing perpendularly to projection plane
        cos_nres = normal @ vector_normalized(residual)
        is_perp_res = jnp.allclose(cos_nres, 0.0)
        cos_nres_safe = jnp.where(is_perp_res, 1.0, cos_nres)
        length = jnp.where(is_zero_res, 0.0, cos_nop / cos_nres_safe)

        return length

    # ------------------------------------------------------------------------------
    # Edge lengths
    # ------------------------------------------------------------------------------

    def edges_length(
            self,
            structure: Structure,
            xyz: jax.Array
            ) -> jax.Array:
        """
        The length of the edges of a structure.
        """
        vectors = edges_vector(xyz, structure.connectivity)

        return vector_length(vectors)

    # ------------------------------------------------------------------------------
    # Edge forces
    # ------------------------------------------------------------------------------

    def edges_force(
            self,
            structure: Structure,
            residuals: jax.Array,
            lengths: jax.Array,
            forces: jax.Array
            ) -> jax.Array:
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

    def trails_force(
            self,
            residuals: jax.Array,
            lengths: jax.Array
            ) -> jax.Array:
        """
        The force in the trail edges of a structure.
        """
        residuals = jnp.concatenate(residuals)
        forces = trail_force(residuals)

        lengths = jnp.reshape(jnp.concatenate(lengths), (-1, 1))

        return jnp.copysign(forces, lengths)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def deviation_vector(
        forces: jax.Array,
        vectors: jax.Array
        ) -> jax.Array:
    """
    Calculate the resultant deviation vector incoming to a node.
    """
    return forces.T @ vectors


def trail_length(
        trail_lengths: jax.Array,
        incidence: jax.Array
        ) -> jax.Array:
    """
    Get the length of the next trail edge outgoing from a node.
    """
    return incidence * trail_lengths  # (num_edges, num_nodes)


def trail_force(residual: jax.Array) -> jax.Array:
    """
    The force passing through a trail edge.
    """
    return vector_length(residual)


def residual_vector(
        residual: jax.Array,
        deviation: jax.Array,
        load: jax.Array
        ) -> jax.Array:
    """
    The updated residual vector at a node.
    """
    return residual - deviation - load


def position_vector(
        position: jax.Array,
        residual: jax.Array,
        trail_length: jax.Array
        ) -> jax.Array:
    """
    The position of the next node on a trail.
    """
    return position + trail_length * vector_normalized(residual)


def edges_vector(
        xyz: jax.Array,
        connectivity: jax.Array
        ) -> jax.Array:
    """
    The edge vectors of the graph.
    """
    return connectivity @ xyz
