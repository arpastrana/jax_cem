"""
The sequences the equilibrium scan steps through, and what their layout settles.
"""

from typing import TYPE_CHECKING
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Int

if TYPE_CHECKING:
    from jax_cem.datastructures.structures import EquilibriumStructure

__all__ = [
    "Sequences",
    "build_sequences",
    "is_edge_deviation_direct",
    "sequences_from_trails",
]


class Sequences(NamedTuple):
    """
    The layout of the trails of a structure into the sequences it steps through.

    Attributes
    ----------
    nodes :
        The node key at each sequence of each trail, where ``-1`` marks a
        sequence that a shifted or shorter trail does not reach.
    edges :
        The trail edge outgoing from the node at each sequence of each trail,
        where ``-1`` marks a slot that no trail edge leaves.
    edges_slot :
        The slot of the layout that each trail edge occupies, counted along the
        flattened sequence pairs.

    Notes
    -----
    A shift changes all of this at once, so it is held together rather than as
    loose fields of a structure, which no longer have to be replaced in step.

    A trail edge joins the node a trail holds in one sequence to the node it
    holds in the next, so the pairs of consecutive sequences carry one slot per
    trail, and a trail that does not span a pair leaves its slot empty. The
    occupied slots are in bijection with the trail edges, which is what lets a
    quantity computed over the layout be read back in edge order by one gather.

    `edges` and `edges_slot` are the two directions of that bijection. The first
    is row-aligned with `nodes`, so the scan reads the node of a slot and the edge
    that leaves it together, and its last row is empty because the last sequence a
    trail reaches holds its support.
    """

    nodes: Int[Array, "sequences trails"]
    edges: Int[Array, "sequences trails"]
    edges_slot: Int[Array, "edges_trail"]

    @property
    def origin_nodes(self) -> Int[Array, "trails"]:
        """
        The first node of each trail, column-aligned with the sequences.
        """
        first = jnp.argmax(self.nodes >= 0, axis=0)

        return self.nodes[first, jnp.arange(self.nodes.shape[-1])]


def is_edge_deviation_direct(
    structure: "EquilibriumStructure",
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
    sequences = structure.sequences.nodes
    rows = jnp.broadcast_to(
        jnp.arange(structure.num_sequences)[:, None],
        sequences.shape,
    )

    # a padded sequence entry is -1, which lands on the extra last slot
    sequence_of = jnp.full(structure.num_nodes + 1, -1, dtype=int)
    sequence_of = sequence_of.at[sequences].set(rows)[:-1]

    nodes_u, nodes_v = structure.edges_deviation[:, 0], structure.edges_deviation[:, 1]

    return sequence_of[nodes_u] == sequence_of[nodes_v]


def build_sequences(
    trails: list[tuple[int, ...]],
    edges: Int[np.ndarray, "edges 2"],
    shifts: Int[np.ndarray, "trails"] | None = None,
) -> Sequences:
    """
    Lay out a set of trails into the sequences an equilibrium steps through.

    Parameters
    ----------
    trails :
        One tuple of node keys per trail, running from origin to support.
    edges :
        The node key pair of each edge, trail edges first.
    shifts :
        The sequence each trail starts at. Defaults to every trail starting at
        the first sequence.

    Returns
    -------
    sequences :
        The layout of the trails.

    Notes
    -----
    The slot each trail edge occupies is inverted from the map that reads an edge
    off a slot, so that the caller gathers in edge order rather than scattering
    into it. Both directions of the map are kept, since the equilibrium reads a
    length off a slot and its forces back off an edge.

    The map is padded with an empty row to the height of the sequences, so that
    the scan steps over the nodes and the edges of a sequence together.

    The index data is computed with NumPy and converted once here, so what the
    structure stores is device-resident.
    """
    nodes, _ = sequences_from_trails(trails, shifts)

    edge_index = {}
    for index, (u, v) in enumerate(edges):
        edge_index[(int(u), int(v))] = index

    edge_of_slot = []
    for pair in zip(nodes[:-1], nodes[1:], strict=True):
        for edge in zip(*pair, strict=True):
            u, v = int(edge[0]), int(edge[1])
            edge_of_slot.append(edge_index.get((u, v), edge_index.get((v, u), -1)))
    edge_of_slot = np.asarray(edge_of_slot, dtype=int)

    slots = np.flatnonzero(edge_of_slot >= 0)
    edges_slot = np.full(slots.size, -1, dtype=int)
    edges_slot[edge_of_slot[slots]] = slots

    # an occupied slot per trail edge, so the inversion leaves nothing behind
    if np.any(edges_slot < 0):
        raise ValueError("Every trail edge must occupy exactly one sequence slot")

    grid = edge_of_slot.reshape(nodes.shape[0] - 1, -1)
    empty = np.full((1, nodes.shape[-1]), -1, dtype=int)
    edges_sequences = np.concatenate((grid, empty))

    return Sequences(
        nodes=jnp.asarray(nodes),
        edges=jnp.asarray(edges_sequences),
        edges_slot=jnp.asarray(edges_slot),
    )


def sequences_from_trails(
    trails: list[tuple[int, ...]],
    shifts: Int[np.ndarray, "trails"] | None = None,
) -> tuple[Int[np.ndarray, "sequences trails"], Int[np.ndarray, "trails"]]:
    """
    Lay out trails into the sequences that the equilibrium scan steps through.

    Parameters
    ----------
    trails :
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
        shifts = np.zeros(len(trails), dtype=int)
    if len(shifts) != len(trails):
        raise ValueError(
            f"Got {len(shifts)} shifts for {len(trails)} trails; they must match",
        )
    if np.any(np.asarray(shifts) < 0):
        raise ValueError("A trail cannot start before the first sequence")

    num_sequences = max(
        shift + len(trail) for shift, trail in zip(shifts, trails, strict=True)
    )

    sequences = np.full((num_sequences, len(trails)), -1, dtype=int)
    for index, (shift, trail) in enumerate(zip(shifts, trails, strict=True)):
        for offset, node in enumerate(trail):
            sequences[shift + offset][index] = node

    origin_nodes = np.asarray([trail[0] for trail in trails], dtype=int)

    return sequences, origin_nodes
