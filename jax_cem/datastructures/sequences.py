"""
The sequences an equilibrium steps through, and what sharing one settles.
"""

from typing import TYPE_CHECKING
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Int

from jax_cem.datastructures.indexing import indices_beyond

if TYPE_CHECKING:
    from jax_cem.datastructures.structures import Structure

__all__ = [
    "Sequence",
    "is_edge_deviation_direct",
    "sequences_from_trails",
]


class Sequence(NamedTuple):
    """
    One sequence of a structure, which an equilibrium steps through in one step.

    Attributes
    ----------
    nodes :
        The node key of every trail in the sequence, where ``-1`` marks a trail
        that does not reach it.
    trail_edge_index :
        The trail edge outgoing from every one of those nodes, where ``-1`` marks
        a node that none leaves.

    Notes
    -----
    Both fields are keyed by slot and hold the entity they name: a slot is one node
    of one trail, and the trail edge that leaves it. `Trails` carries a field of the
    same name keyed the other way round, by the trail edge, holding the slot it
    occupies; the shapes tell them apart, `"trails"` here against `"edges_trail"`
    there, and so does the rank once these are stacked into a grid.

    A trail edge joins the node a trail holds in one sequence to the node it holds
    in the next, so a trail of ``n`` nodes has ``n - 1`` edges and its last slot
    holds a node that no edge leaves. That slot is padded here and not in `nodes`,
    which is why a mask read off one does not speak for the other.

    The shapes describe one sequence, which is the view a scan presents to the
    step it drives. `Trails.sequences` stacks every sequence into the same
    container, which is what the scan is handed.
    """

    nodes: Int[Array, "trails"]
    trail_edge_index: Int[Array, "trails"]


def is_edge_deviation_direct(
    structure: "Structure",
) -> Bool[Array, "edges_deviation"]:
    """
    Mask the deviation edges of a structure whose two nodes share a sequence.

    Parameters
    ----------
    structure :
        A structure.

    Returns
    -------
    is_direct :
        Whether each deviation edge stays within one sequence.

    Notes
    -----
    A deviation edge that spans two sequences is indirect, and only the iterative
    equilibrium resolves it. Which edges those are follows from the layout, so it
    is computed rather than stored, and it cannot fall out of step with a trail
    that shifts.
    """
    sequences = structure.trails.sequences.nodes
    rows = jnp.broadcast_to(
        jnp.arange(structure.num_sequences)[:, None],
        sequences.shape,
    )

    nodes = indices_beyond(sequences, structure.num_nodes)
    sequence_of = jnp.full(structure.num_nodes, -1, dtype=int)
    sequence_of = sequence_of.at[nodes].set(rows, mode="drop")

    nodes_u, nodes_v = structure.edges_deviation[:, 0], structure.edges_deviation[:, 1]

    return sequence_of[nodes_u] == sequence_of[nodes_v]


def sequences_from_trails(
    trail_nodes: list[tuple[int, ...]],
    shifts: Int[np.ndarray, "trails"] | None = None,
) -> tuple[Int[np.ndarray, "sequences trails"], Int[np.ndarray, "trails"]]:
    """
    Lay out trails into the sequences that the equilibrium scan steps through.

    Parameters
    ----------
    trail_nodes :
        One tuple of node keys per trail, running from origin to support.
    shifts :
        The sequence each trail starts at. Defaults to every trail starting at
        the first sequence.

    Returns
    -------
    sequences :
        The node key at each sequence of each trail, where ``-1`` marks a
        sequence that a shifted or shorter trail does not reach.
    origin_nodes :
        The first node of each trail, column-aligned with the sequences.

    Notes
    -----
    One sequence holds the nodes that the algorithm puts in equilibrium in the
    same step, one per trail. Trails of unequal length, and trails that start
    late because they are shifted, leave the padding value behind.
    """
    if shifts is None:
        shifts = np.zeros(len(trail_nodes), dtype=int)
    if len(shifts) != len(trail_nodes):
        raise ValueError(
            f"Got {len(shifts)} shifts for {len(trail_nodes)} trails; they must match",
        )
    if np.any(np.asarray(shifts) < 0):
        raise ValueError("A trail cannot start before the first sequence")

    num_sequences = max(
        shift + len(trail) for shift, trail in zip(shifts, trail_nodes, strict=True)
    )

    sequences = np.full((num_sequences, len(trail_nodes)), -1, dtype=int)
    for index, (shift, trail) in enumerate(zip(shifts, trail_nodes, strict=True)):
        for offset, node in enumerate(trail):
            sequences[shift + offset][index] = node

    origin_nodes = np.asarray([trail[0] for trail in trail_nodes], dtype=int)

    return sequences, origin_nodes
