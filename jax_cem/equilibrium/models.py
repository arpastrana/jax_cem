import equinox as eqx
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import vmap
from jax.lax import scan
from jax.ops import segment_sum
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_cem.datastructures import Sequence
from jax_cem.datastructures import Structure
from jax_cem.datastructures import gather_padded
from jax_cem.datastructures import indices_beyond
from jax_cem.datastructures import is_edge_deviation_direct
from jax_cem.equilibrium import EquilibriumSequenceState
from jax_cem.equilibrium import EquilibriumState
from jax_cem.equilibrium import EquilibriumTrailsState
from jax_cem.geometry import is_plane_absent
from jax_cem.geometry import vector_length
from jax_cem.geometry import vector_normalized
from jax_cem.parameters import Parameters

__all__ = [
    "EquilibriumModel",
]

# ------------------------------------------------------------------------------
#  The combinatorial equilibrium model
# ------------------------------------------------------------------------------


class EquilibriumModel:
    """
    An equilibrium model that implements the combinatorial equilibrium modeling
    (CEM) framework.

    Notes
    -----
    The class holds the settings of the solver and the control flow they drive:
    the sweep over the sequences, the fixed point that resolves the indirect
    deviation edges, and the state a call returns. The quantities those steps
    compute are functions of this module, since none of them reads a setting.
    """

    def __init__(
        self,
        tmax: int = 10,
        eta: float = 1.0e-6,
        scale: float = 1.0e6,
    ):
        self.tmax = tmax
        self.eta = eta
        self.scale = scale

    def __call__(
        self,
        params: Parameters,
        structure: Structure,
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
        The node positions carry one row per node. A sequence a shifted trail has
        not reached writes nothing, since its slot addresses no node.

        The first pass leaves the deviation edges that span two sequences out,
        which is what the iterative passes then resolve.

        Assumptions
        -----------
        - No shape dependent loads exist in the structure.
        """
        xyz = jnp.zeros((structure.num_nodes, 3))
        trails_state = self.trails_equilibrium(
            params,
            structure,
            xyz,
            use_indirect=False,
        )

        if self.tmax > 1:
            t_xyz = trails_state.xyz
            trails_state = self.equilibrium_iterative(params, structure, t_xyz)

        return self.equilibrium_state(params, structure, trails_state)

    # ------------------------------------------------------------------------------
    #  Equilibrium state
    # ------------------------------------------------------------------------------

    def equilibrium_state(
        self,
        params: Parameters,
        structure: Structure,
        trails_state: EquilibriumTrailsState,
    ) -> EquilibriumState:
        """
        Assemble an equilibrium state.

        Notes
        -----
        The trails state spans one slot per trail per sequence, which is the grid
        the layout addresses, so it is read whole rather than trimmed to the
        sequences that a trail edge joins.
        """
        xyz = trails_state.xyz

        # Edge forces, which read the trail residual of every sequence
        forces = edges_force(
            structure,
            trails_state.residuals_trail,
            trails_state.lengths,
            params.forces,
        )

        # Edge vectors and lengths, and the residual at every node
        vectors = edges_vector(xyz, structure.edges)
        lengths = edges_length(vectors)
        residuals = nodes_residual(structure, forces, vectors, params.loads)

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

    def equilibrium_iterative(
        self,
        params: Parameters,
        structure: Structure,
        xyz: Float[Array, "nodes 3"],
    ) -> EquilibriumTrailsState:
        """
        Calculate static equilibrium on a structure iteratively.

        Notes
        -----
        The iteration runs on the settings of the model, which it reads off it
        rather than receiving them back.
        """

        def sweep(xyz_in):
            return self.trails_equilibrium(params, structure, xyz_in, use_indirect=True)

        def distance(current, previous):
            return jnp.sum(jnp.linalg.norm(previous - current, axis=1))

        def cond_fn(val):
            current, previous = val
            # Keep iterating while the nodes are still moving
            return distance(current, previous) > self.eta

        def body_fn(val):
            current, _ = val

            return sweep(current).xyz, current

        # The sweep reads the positions only through normalized edge vectors, so
        # scaling the first iterate leaves it alone and starts the loop moving
        init_val = xyz * self.scale, xyz

        current, _ = while_loop(
            cond_fn,
            body_fn,
            init_val,
            max_steps=self.tmax,
            kind="checkpointed",
        )

        return sweep(current)

    # ------------------------------------------------------------------------------
    # Trail equilibrium
    # ------------------------------------------------------------------------------

    def trails_equilibrium(
        self,
        params: Parameters,
        structure: Structure,
        xyz: Float[Array, "nodes 3"],
        use_indirect: bool,
    ) -> EquilibriumTrailsState:
        """
        Calculate equilibrium along the trails of a structure, one sequence at a time.

        Notes
        -----
        The first pass leaves out the deviation edges that span two sequences,
        which only the iterative passes resolve. The pass masks them out of the
        forces it carries, once, rather than at every node of every sequence.
        """
        if use_indirect:
            params_pass = params
        else:
            is_direct = is_edge_deviation_direct(structure)
            forces_direct = jnp.where(is_direct, params.forces, 0.0)
            params_pass = eqx.tree_at(
                lambda tree: tree.forces,
                params,
                replace=forces_direct,
            )

        def calculate_sequence_state(state, sequence):
            """
            Compute static equilibrium on a sequence of nodes in a scan-compatible way.
            """
            xyz_nodes, state_seq = state

            # a padded slot addresses no node, and the scatter drops it
            nodes = indices_beyond(sequence.nodes, structure.num_nodes)
            xyz_nodes = xyz_nodes.at[nodes, :].set(state_seq.xyz, mode="drop")

            state_seq = self.sequence_equilibrium(
                params_pass,
                structure,
                sequence,
                xyz_nodes,
                state_seq,
            )

            carry_out = (state_seq.residuals_trail, state_seq.lengths)

            return (xyz_nodes, state_seq), carry_out

        # The sweep starts every trail at its origin, carrying no residual yet
        state_seq_start = EquilibriumSequenceState(
            xyz=params.xyz_origin,
            residuals_trail=jnp.zeros((structure.num_trails, 3)),
            lengths=jnp.zeros(structure.num_trails),
        )

        # Compute static equilibrium by scanning a function over all sequences
        state_end, (residuals_trail, lengths_seqs) = scan(
            calculate_sequence_state,
            (xyz, state_seq_start),
            structure.sequences,
        )

        xyz_end, _ = state_end

        trails_state = EquilibriumTrailsState(
            xyz=xyz_end,
            residuals_trail=residuals_trail,
            lengths=lengths_seqs,
        )

        return trails_state

    def sequence_equilibrium(
        self,
        params: Parameters,
        structure: Structure,
        sequence: Sequence,
        xyz: Float[Array, "nodes 3"],
        state_seq: EquilibriumSequenceState,
    ) -> EquilibriumSequenceState:
        """
        Compute static equilibrium on all the nodes of a sequence.

        Notes
        -----
        A slot that no trail edge leaves takes a zero length and a zero plane,
        which leaves the node where it is. A slot no trail occupies takes no
        deviation force and no load, so its residual is the one it came in with.
        Both follow from the slot addressing nothing rather than from a mask.

        The state it takes and the state it returns are the same container, which
        is what a scan carries from one sequence to the next. The lengths it
        receives belong to the sequence before and are not read.
        """
        xyz_seq = state_seq.xyz
        residuals_trail = state_seq.residuals_trail

        # Padding mask
        is_sequence_padded = sequence.nodes[:, None] < 0

        # Trail residuals
        residuals_new = nodes_equilibrium(
            params,
            structure,
            sequence.nodes,
            xyz,
            residuals_trail,
        )
        residuals_trail = jnp.where(is_sequence_padded, residuals_trail, residuals_new)

        # Trail edge lengths
        # NOTE: Probably inefficient to pre-compute both versions of length.
        # Passing the length functions to jnp.where may skip one evaluation.
        planes_seq = gather_padded(params.planes, sequence.edges)
        lengths_plane = nodes_length_plane(planes_seq, xyz_seq, residuals_trail)
        lengths_signed = gather_padded(params.lengths, sequence.edges)
        is_plane_missing = is_plane_absent(planes_seq)
        lengths_seq = jnp.where(is_plane_missing, lengths_signed, lengths_plane)

        # Position of the next node
        xyz_seq_new = nodes_position(xyz_seq, residuals_trail, lengths_seq)
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
    params: Parameters,
    structure: Structure,
    nodes_seq: Int[Array, "trails"],
    xyz: Float[Array, "nodes 3"],
    residuals_trail: Float[Array, "trails 3"],
) -> Float[Array, "trails 3"]:
    """
    Calculate static equilibrium at the nodes of a sequence. Vectorized.

    Notes
    -----
    The deviation force is accumulated at every node of the structure and the
    sequence gathers the nodes it holds, which costs one pass over the deviation
    edges instead of one pass per node. A padded slot addresses no node and
    gathers zero, so it carries its residual through unchanged.
    """
    deviation_forces = nodes_deviation_force(structure, xyz, params.forces)
    deviation_forces_seq = gather_padded(deviation_forces, nodes_seq)
    loads_seq = gather_padded(params.loads, nodes_seq)

    return residual_trail_next(residuals_trail, deviation_forces_seq, loads_seq)


def nodes_deviation_force(
    structure: Structure,
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
    xyz_seq: Float[Array, "trails 3"],
    residuals_trail: Float[Array, "trails 3"],
    lengths: Float[Array, "trails"],
) -> Float[Array, "trails 3"]:
    """
    Calculate the position of the next sequence of nodes of a structure.
    """
    return vmap(position_vector)(xyz_seq, residuals_trail, lengths)


# ------------------------------------------------------------------------------
# Node lengths
# ------------------------------------------------------------------------------


def nodes_length_plane(
    planes_seq: Float[Array, "trails 6"],
    xyz_seq: Float[Array, "trails 3"],
    residuals_trail: Float[Array, "trails 3"],
) -> Float[Array, "trails"]:
    """
    Calculate the outgoing edge lengths in a sequence. Vectorized.
    """
    return vmap(node_length_plane)(planes_seq, xyz_seq, residuals_trail)


def node_length_plane(
    plane: Float[Array, "6"],
    xyz: Float[Array, "3"],
    residual_trail: Float[Array, "3"],
) -> Float[Array, ""]:
    """
    Compute the length of a trail edge from the plane that drives it.

    Notes
    -----
    The length is how far the node sits from the plane along its normal, divided
    by how much a unit step along the trail residual closes that gap. The scale of
    the normal cancels between the two, so it need not arrive normalized.

    The plane arrives as data, so the arithmetic reads no entity and the keying
    stays with the caller.

    A degenerate input takes a length of zero rather than a nan, except a residual
    that runs parallel to the plane and so never meets it, which takes the offset.
    """
    origin = plane[:3]
    normal = plane[3:]

    # a zero normal points nowhere to project onto, so substitute before dividing
    is_normal_zero = is_plane_absent(plane)
    normal_safe = jnp.where(is_normal_zero, jnp.ones_like(normal), normal)
    offset_plane = jnp.where(is_normal_zero, 0.0, normal_safe @ (origin - xyz))

    # a zero residual has no direction to travel along, and normalizing it is nan
    is_residual_zero = jnp.allclose(residual_trail, 0.0)
    ones = jnp.ones_like(residual_trail)
    residual_safe = jnp.where(is_residual_zero, ones, residual_trail)

    # a residual parallel to the plane closes no gap, so the guard divides by one
    advance_normal = normal_safe @ vector_normalized(residual_safe)
    is_residual_parallel = jnp.allclose(advance_normal, 0.0)
    advance_safe = jnp.where(is_residual_parallel, 1.0, advance_normal)

    length = jnp.where(is_residual_zero, 0.0, offset_plane / advance_safe)

    return length


# ------------------------------------------------------------------------------
# Edge lengths
# ------------------------------------------------------------------------------


def edges_length(vectors: Float[Array, "edges 3"]) -> Float[Array, "edges"]:
    """
    The length of the edges of a structure.
    """
    return vmap(vector_length)(vectors)


# ------------------------------------------------------------------------------
# Node residuals
# ------------------------------------------------------------------------------


def nodes_residual(
    structure: Structure,
    forces: Float[Array, "edges"],
    vectors: Float[Array, "edges 3"],
    loads: Float[Array, "nodes 3"],
) -> Float[Array, "nodes 3"]:
    """
    The residual force at every node of a structure.

    Notes
    -----
    The residual is assembled from the edge forces and the loads rather than read
    off the sweep, so a free node the sweep left out of equilibrium reports a
    residual instead of a zero. It vanishes at a free node only once the indirect
    deviation edges have converged, and what remains at a support is the negation
    of its reaction.
    """
    units = vmap(vector_normalized)(vectors)
    edges = structure.edges
    resultants = nodes_resultant(forces, units, edges, structure.num_nodes)

    return loads + resultants


# ------------------------------------------------------------------------------
# Edge forces
# ------------------------------------------------------------------------------


def edges_force(
    structure: Structure,
    residuals_trail: Float[Array, "sequences trails 3"],
    lengths: Float[Array, "sequences trails"],
    forces: Float[Array, "edges_deviation"],
) -> Float[Array, "edges"]:
    """
    The forces in the edges of a structure.

    Notes
    -----
    The layout carries one slot per trail per sequence pair, and a trail that does
    not span a pair leaves its slot empty, so the trail forces come out in the
    order of the layout rather than of the edges. The slot each edge occupies
    reorders them in one gather, which drops the padding with them.

    A slot is one flat number over the grid, and the counts that decode it into a
    sequence and a trail are the structure's own, not the extent of the array being
    read, so the two cannot disagree about what a slot means.

    The deviation block is a parameter, and the two concatenate in the edge order
    of the structure.
    """
    trail_forces = trails_force(residuals_trail, lengths)

    grid_shape = (structure.num_sequences, structure.num_trails)
    slots = jnp.unravel_index(structure.edges_sequence, grid_shape)
    forces_trail = trail_forces[slots]

    return jnp.concatenate((forces_trail, forces))


def trails_force(
    residuals_trail: Float[Array, "sequences trails 3"],
    lengths: Float[Array, "sequences trails"],
) -> Float[Array, "sequences trails"]:
    """
    The force in the trail edges of a structure, one per slot of the layout.

    Notes
    -----
    The force takes the sign of the length of the trail edge it passes through,
    which is negative in compression.

    The two axes of the layout are kept rather than collapsed into one, so the
    slots stay addressable as the grid coordinates they are. The inner map runs
    over the trails of a sequence and the outer one over the sequences.
    """
    forces = vmap(vmap(trail_force))(residuals_trail)

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

    nodes_tail, nodes_head = edges[:, 0], edges[:, 1]
    at_tail = segment_sum(resultants, nodes_tail, num_nodes)
    at_head = segment_sum(resultants, nodes_head, num_nodes)

    return at_tail - at_head


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
