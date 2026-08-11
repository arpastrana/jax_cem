import jax.numpy as jnp
from equinox.internal import while_loop
from jax import vmap
from jax.lax import scan
from jax.ops import segment_sum
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_cem.datastructures import EquilibriumStructure
from jax_cem.datastructures import is_edge_deviation_direct
from jax_cem.equilibrium import EquilibriumSequenceState
from jax_cem.equilibrium import EquilibriumState
from jax_cem.equilibrium import EquilibriumTrailsState
from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized
from jax_cem.parameters import ParameterState

# ------------------------------------------------------------------------------
#  The combinatorial equilibrium model
# ------------------------------------------------------------------------------


class EquilibriumModel:
    """
    An equilibrium model that implements the combinatorial equilibrium modeling
    (CEM) framework.
    """

    def __init__(
        self,
        tmax: int = 10,
        eta: float = 1.0e-6,
        scale: float = 1.0e6,
        verbose: bool = False,
    ):
        self.tmax = tmax
        self.eta = eta
        self.scale = scale
        self.verbose = verbose

    def __call__(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
    ) -> EquilibriumState:
        """
        Computes an equilibrium state on a structure.

        Parameters
        ----------
        params :
            The parameters of the equilibrium model.
        structure :
            A structure.

        Returns
        -------
        eq_state :
            An equilibrium state.

        Notes
        -----
        The node positions carry a dummy last row, which a shifted sequence
        indexes with ``-1`` while it waits for its trail to start.

        Assumptions
        -----------
        - No shape dependent loads exist in the structure.
        """
        xyz = jnp.zeros((structure.num_nodes + 1, 3))
        trails_state = self.equilibrium(params, structure, xyz)

        if self.tmax > 1:
            trails_state = self.equilibrium_iterative(
                params,
                structure,
                trails_state.xyz,
                self.tmax,
                self.eta,
                self.scale,
            )

        return self.equilibrium_state(params, structure, trails_state)

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        trails_state: EquilibriumTrailsState,
    ) -> EquilibriumState:
        """
        Assemble an equilibrium state.

        Notes
        -----
        The trails state counts the sequences a structure spans, so the last
        entry of its trailing axes falls outside the trail edges between them.
        """
        # Node positions
        # NOTE: We remove the dummy last row created in __call__()
        xyz = trails_state.xyz[:-1]

        # Edge forces, which read the trail residual of every sequence
        forces = self.edges_force(
            structure,
            trails_state.residuals_trail[:-1, :],
            trails_state.lengths[:-1, :],
            params.forces,
        )

        # Edge vectors and lengths, and the residual at every node
        vectors = edges_vector(xyz, structure.edges)
        lengths = self.edges_length(vectors)
        residuals = self.nodes_residual(structure, forces, vectors, params.loads)

        # Create equilibrium state
        state = EquilibriumState(
            xyz=xyz,
            forces=forces,
            lengths=lengths,
            vectors=vectors,
            loads=params.loads,
            residuals=residuals,
        )

        return state

    # ------------------------------------------------------------------------------
    #  Equilibrium modes
    # ------------------------------------------------------------------------------

    def equilibrium(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        xyz: Float[Array, "nodes_padded 3"],
    ) -> EquilibriumTrailsState:
        """
        Calculate static equilibrium on a structure.
        """
        return self.sequences_equilibrium(params, structure, xyz, use_indirect=False)

    def equilibrium_iterative(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        xyz: Float[Array, "nodes_padded 3"],
        tmax: int,
        eta: float,
        scale: float,
    ) -> EquilibriumTrailsState:
        """
        Calculate static equilibrium on a structure iteratively.
        """

        def distance(xyz, xyz_last):
            return jnp.sum(jnp.linalg.norm(xyz_last[:-1] - xyz[:-1], axis=1))

        def cond_fn(val):
            xyz, xyz_last = val
            # Keep iterating while the nodes are still moving
            delta = distance(xyz, xyz_last)

            return delta > eta

        def body_fn(val):
            xyz_last, _ = val
            trails_state = self.sequences_equilibrium(
                params,
                structure,
                xyz_last,
                use_indirect=True,
            )
            return trails_state.xyz, xyz_last

        # Initialize iteration
        init_val = xyz * scale, xyz

        # Iterate
        xyz_last, _ = while_loop(
            cond_fn,
            body_fn,
            init_val,
            max_steps=tmax,
            kind="checkpointed",
        )

        return self.sequences_equilibrium(
            params,
            structure,
            xyz_last,
            use_indirect=True,
        )

    # ------------------------------------------------------------------------------
    # Sequence equilibrium
    # ------------------------------------------------------------------------------

    def sequences_equilibrium(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        xyz: Float[Array, "nodes_padded 3"],
        use_indirect: bool,
    ) -> EquilibriumTrailsState:
        """
        Calculates equilibrium on all the sequences of a structure.

        Notes
        -----
        The first pass leaves out the deviation edges that span two sequences,
        which only the iterative passes resolve. The flag settles that here, once,
        rather than at every node of every sequence.
        """
        forces = params.forces
        if not use_indirect:
            forces = jnp.where(is_edge_deviation_direct(structure), forces, 0.0)

        def calculate_sequence_state_start():
            """
            Creates an initial scan state.
            """
            residuals_trail = jnp.zeros((structure.num_trails, 3))

            return xyz, params.xyz_origin, residuals_trail

        def calculate_sequence_state(state, sequence_slots):
            """
            Compute static equilibrium on a sequence of nodes in a scan-compatible way.
            """
            sequence, edges_seq = sequence_slots
            _xyz, xyz_seq, residuals_trail = state
            _xyz = _xyz.at[sequence, :].set(xyz_seq)

            state_seq = self.sequence_equilibrium(
                params,
                structure,
                sequence,
                edges_seq,
                _xyz,
                xyz_seq,
                residuals_trail,
                forces,
            )

            state_out = (_xyz, state_seq.xyz, state_seq.residuals_trail)
            carry_out = (state_seq.residuals_trail, state_seq.lengths)

            return state_out, carry_out

        # Create initial scan state
        state_start = calculate_sequence_state_start()

        # Compute static equilibrium by scanning a function over all sequences
        state_end, (residuals_trail, lengths_seqs) = scan(
            calculate_sequence_state,
            state_start,
            (structure.sequences.nodes, structure.sequences.edges),
        )

        xyz, *_ = state_end

        return EquilibriumTrailsState(
            xyz=xyz,
            residuals_trail=residuals_trail,
            lengths=lengths_seqs,
        )

    def sequence_equilibrium(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        sequence: Int[Array, "trails"],
        edges_seq: Int[Array, "trails"],
        xyz: Float[Array, "nodes_padded 3"],
        xyz_seq: Float[Array, "trails 3"],
        residuals_trail: Float[Array, "trails 3"],
        forces: Float[Array, "edges_deviation"],
    ) -> EquilibriumSequenceState:
        """
        Compute static equilibrium on all the nodes of a sequence.

        Notes
        -----
        A slot that no trail edge leaves reads the length and the plane of the
        last edge, which the padding mask and the slot the forces are gathered
        from then drop, in the same way a padded node key reads the last node.
        """
        # Padding mask
        is_sequence_padded = jnp.reshape(sequence, (-1, 1)) < 0

        # Trail residuals
        residuals_new = self.nodes_equilibrium(
            params,
            structure,
            sequence,
            xyz[:-1],
            residuals_trail,
            forces,
        )
        residuals_trail = jnp.where(is_sequence_padded, residuals_trail, residuals_new)

        # Trail edge lengths
        # NOTE: Probably inefficient to pre-compute both versions of length.
        # Passing the length functions to jnp.where may skip one evaluation.
        lengths_plane = self.nodes_length_plane(
            params.planes[edges_seq, :],
            xyz_seq,
            residuals_trail,
        )
        lengths_signed = params.lengths[edges_seq]
        lengths_seq = jnp.where(lengths_signed != 0.0, lengths_signed, lengths_plane)

        # Position of the next node
        xyz_seq_new = self.nodes_position(xyz_seq, residuals_trail, lengths_seq)
        xyz_seq = jnp.where(is_sequence_padded, xyz_seq, xyz_seq_new)

        # Create sequence state
        state = EquilibriumSequenceState(
            xyz=xyz_seq,
            residuals_trail=residuals_trail,
            lengths=lengths_seq,
        )

        return state

    # ------------------------------------------------------------------------------
    # Node equilibrium
    # ------------------------------------------------------------------------------

    def nodes_equilibrium(
        self,
        params: ParameterState,
        structure: EquilibriumStructure,
        sequence: Int[Array, "trails"],
        xyz: Float[Array, "nodes 3"],
        residuals_trail: Float[Array, "trails 3"],
        forces: Float[Array, "edges_deviation"],
    ) -> Float[Array, "trails 3"]:
        """
        Calculate static equilibrium at the nodes of a sequence. Vectorized.

        Notes
        -----
        The deviation force is accumulated at every node of the structure and the
        sequence gathers the nodes it holds, which costs one pass over the
        deviation edges instead of one pass per node. A padded sequence entry
        gathers the last node, which the caller masks out.
        """
        deviations = self.nodes_deviation(structure, xyz, forces)

        return residual_trail_next(
            residuals_trail,
            deviations[sequence, :],
            params.loads[sequence, :],
        )

    def nodes_deviation(
        self,
        structure: EquilibriumStructure,
        xyz: Float[Array, "nodes 3"],
        forces: Float[Array, "edges_deviation"],
    ) -> Float[Array, "nodes 3"]:
        """
        The resultant deviation force at every node of a structure.

        Notes
        -----
        Only deviation edges reach the equilibrium of a node, and they are held in
        their own array, so the trail block is never touched.
        """
        edges = structure.edges_deviation
        vectors = vmap(vector_normalized)(edges_vector(xyz, edges))

        return nodes_resultant(forces, vectors, edges, structure.num_nodes)

    # ------------------------------------------------------------------------------
    # Node position
    # ------------------------------------------------------------------------------

    def nodes_position(
        self,
        xyz_seq: Float[Array, "trails 3"],
        residuals_trail: Float[Array, "trails 3"],
        lengths: Float[Array, "trails"],
    ) -> Float[Array, "trails 3"]:
        """
        Calculate the position of the next sequence of nodes of a structure.
        """
        return vmap(self.node_position)(xyz_seq, residuals_trail, lengths)

    def node_position(
        self,
        xyz: Float[Array, "3"],
        residual_trail: Float[Array, "3"],
        length: Float[Array, ""],
    ) -> Float[Array, "3"]:
        """
        Calculate the position of the next node on a trail of a structure.
        """
        return position_vector(xyz, residual_trail, length)

    # ------------------------------------------------------------------------------
    # Node lengths
    # ------------------------------------------------------------------------------

    def nodes_length_plane(
        self,
        planes_seq: Float[Array, "trails 6"],
        xyz_seq: Float[Array, "trails 3"],
        residuals_trail: Float[Array, "trails 3"],
    ) -> Float[Array, "trails"]:
        """
        Calculate the outgoing edge lengths in a sequence. Vectorized.
        """
        node_length_plane_vmap = vmap(self.node_length_plane)

        return node_length_plane_vmap(planes_seq, xyz_seq, residuals_trail)

    def node_length_plane(
        self,
        plane: Float[Array, "6"],
        xyz: Float[Array, "3"],
        residual_trail: Float[Array, "3"],
    ) -> Float[Array, ""]:
        """
        Compute the length of a trail edge from the plane that drives it.

        Notes
        -----
        The plane arrives as data, so the arithmetic reads no entity and the keying
        stays with the caller.

        It assumes that the trail residual and plane normal vector are not parallel.
        """
        origin = plane[:3]
        normal = plane[3:]

        # Return zero cos nop if plane normal is zero
        # May raise NaNs, use double where trick
        is_zero_normal = jnp.allclose(normal, 0.0)
        normal = jnp.where(is_zero_normal, jnp.ones_like(normal), normal)
        cos_nop = jnp.where(is_zero_normal, 0.0, normal @ (origin - xyz))

        # Return zero length if the trail residual is zero
        # May raise NaNs, use double where trick
        is_zero_res = jnp.allclose(residual_trail, 0.0)
        ones = jnp.ones_like(residual_trail)
        residual_trail = jnp.where(is_zero_res, ones, residual_trail)

        # Safeguard against the trail residual pointing perpendicularly to the plane
        cos_nres = normal @ vector_normalized(residual_trail)
        is_perp_res = jnp.allclose(cos_nres, 0.0)
        cos_nres_safe = jnp.where(is_perp_res, 1.0, cos_nres)
        length = jnp.where(is_zero_res, 0.0, cos_nop / cos_nres_safe)

        return length

    # ------------------------------------------------------------------------------
    # Edge lengths
    # ------------------------------------------------------------------------------

    def edges_length(
        self,
        vectors: Float[Array, "edges 3"],
    ) -> Float[Array, "edges"]:
        """
        The length of the edges of a structure.
        """
        return vmap(vector_length)(vectors)

    # ------------------------------------------------------------------------------
    # Node residuals
    # ------------------------------------------------------------------------------

    def nodes_residual(
        self,
        structure: EquilibriumStructure,
        forces: Float[Array, "edges"],
        vectors: Float[Array, "edges 3"],
        loads: Float[Array, "nodes 3"],
    ) -> Float[Array, "nodes 3"]:
        """
        The residual force at every node of a structure.

        Notes
        -----
        The residual is assembled from the edge forces and the loads rather than
        read off the sweep, so a free node the sweep left out of equilibrium
        reports a residual instead of a zero. It vanishes at a free node only once
        the indirect deviation edges have converged, and what remains at a support
        is the negation of its reaction.
        """
        units = vmap(vector_normalized)(vectors)
        edges = structure.edges
        resultants = nodes_resultant(forces, units, edges, structure.num_nodes)

        return loads + resultants

    # ------------------------------------------------------------------------------
    # Edge forces
    # ------------------------------------------------------------------------------

    def edges_force(
        self,
        structure: EquilibriumStructure,
        residuals_trail: Float[Array, "sequences_edges trails 3"],
        lengths: Float[Array, "sequences_edges trails"],
        forces: Float[Array, "edges_deviation"],
    ) -> Float[Array, "edges"]:
        """
        The forces in the edges of a structure.

        Notes
        -----
        The layout carries one slot per trail per sequence pair, and a trail that
        does not span a pair leaves its slot empty, so the trail forces come out
        in the order of the layout rather than of the edges. The slot each edge
        occupies reorders them in one gather, which drops the padding with them.

        The deviation block is a parameter, and the two concatenate in the edge
        order of the structure.
        """
        trail_forces = self.trails_force(residuals_trail, lengths)
        forces_trail = trail_forces[structure.sequences.edges_slot]

        return jnp.concatenate((forces_trail, forces))

    def trails_force(
        self,
        residuals_trail: Float[Array, "sequences_edges trails 3"],
        lengths: Float[Array, "sequences_edges trails"],
    ) -> Float[Array, "sequences_edges*trails"]:
        """
        The force in the trail edges of a structure.

        Notes
        -----
        The force takes the sign of the length of the trail edge it passes
        through, which is negative in compression.
        """
        residuals_trail = jnp.concatenate(residuals_trail)
        forces = vmap(trail_force)(residuals_trail)

        lengths = jnp.concatenate(lengths)

        return jnp.copysign(forces, lengths)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def nodes_resultant(
    forces: Float[Array, "edges_any"],
    vectors: Float[Array, "edges_any 3"],
    edges: Int[Array, "edges_any 2"],
    num_nodes: int,
) -> Float[Array, "nodes 3"]:
    """
    The resultant force at every node of a graph, from a set of edges.

    Parameters
    ----------
    forces :
        The force in every edge, positive in tension.
    vectors :
        The unit vector of every edge, from its tail to its head.
    edges :
        The node key pair of every edge.
    num_nodes :
        The number of nodes to accumulate the edges into.

    Returns
    -------
    resultants :
        The resultant force the edges exert at every node.

    Notes
    -----
    An edge vector runs from the tail of an edge to its head, so the force it
    carries pulls the tail forward along it and the head back against it. The two
    scatters are that pair of signs, and both push a node toward the far end of
    the edge.

    The edge set is a parameter, so the same accumulation serves the deviation
    edges the sweep resolves and the whole edge set a nodal residual reads.

    The scatter reduces in an order a dot product need not follow, so the result
    matches a dense product to rounding rather than to the bit. The package
    enables double precision, which leaves that difference orders below the
    tolerance the regression baselines are compared at.
    """
    resultants = forces[:, None] * vectors

    return segment_sum(resultants, edges[:, 0], num_nodes) - segment_sum(
        resultants,
        edges[:, 1],
        num_nodes,
    )


def trail_force(residual_trail: Float[Array, "3"]) -> Float[Array, ""]:
    """
    The force passing through a trail edge.
    """
    return vector_length(residual_trail)


def residual_trail_next(
    residual_trail: Float[Array, "trails 3"],
    deviation: Float[Array, "trails 3"],
    load: Float[Array, "trails 3"],
) -> Float[Array, "trails 3"]:
    """
    The trail residual a node passes on to its outgoing trail edge.

    Notes
    -----
    What the incoming trail edge delivers, less what the deviation edges and the
    load take, is what the outgoing trail edge is left to carry.

    The arithmetic broadcasts, so the function is not bound to the trail axis it
    is called on and carries no entity in its name.
    """
    return residual_trail - deviation - load


def position_vector(
    position: Float[Array, "3"],
    residual_trail: Float[Array, "3"],
    trail_length: Float[Array, ""],
) -> Float[Array, "3"]:
    """
    The position of the next node on a trail.
    """
    return position + trail_length * vector_normalized(residual_trail)


def edges_vector(
    xyz: Float[Array, "nodes 3"],
    edges: Int[Array, "edges 2"],
) -> Float[Array, "edges 3"]:
    """
    The vector of every edge of a graph, from its tail node to its head node.
    """
    return xyz[edges[:, 1], :] - xyz[edges[:, 0], :]
