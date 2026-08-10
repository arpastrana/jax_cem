"""
The trail search that orders the nodes of a structure into sequences.
"""

from typing import TYPE_CHECKING
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

if TYPE_CHECKING:
    from jax_cem.datastructures.structures import EquilibriumStructure

__all__ = [
    "align_trails",
    "sequences_from_trails",
    "trails_from_edges",
]


class SequenceData(NamedTuple):
    """
    Everything about a structure that the layout of its trails settles.
    """

    sequences: Int[np.ndarray, "sequences trails"]
    origin_nodes: Int[np.ndarray, "trails"]
    sequences_edges: Int[np.ndarray, "sequences_edges trails"]
    sequences_edges_indices: Int[np.ndarray, "sequences_edges_flat"]
    edges_deviation_direct: Float[Array, "edges"]


def align_trails(structure: "EquilibriumStructure") -> "EquilibriumStructure":
    """
    Start every trail at the sequence of the node it deviates to.

    Parameters
    ----------
    structure :
        A structure.

    Returns
    -------
    structure :
        A new structure whose trails are aligned across their deviation edges.

    Notes
    -----
    An origin node joined by a deviation edge to a node further along another
    trail carries that edge as an indirect one, which only the iterative
    equilibrium resolves. Aligning the two trails makes the edge direct, so the
    first equilibrium pass already accounts for it.

    The alignment reads the trails as the search found them and ignores any
    shift already applied, so aligning an aligned structure changes nothing.

    One pass makes an origin node's deviation edges direct, not every edge in the
    structure: shifting a trail can turn one of its other edges indirect. A
    structure whose deviation edges are all direct already is returned unchanged.
    """
    trails = trails_of(structure)
    edges = np.asarray(structure.edges)
    shifts = _shifts_from_indirect_edges(trails, np.asarray(structure.edges_deviation))

    data = sequence_data(
        trails,
        edges,
        len(structure.nodes),
        len(structure.edges_trail),
        shifts,
    )

    return eqx.tree_at(
        lambda s: (
            s.sequences,
            s.origin_nodes,
            s.sequences_edges,
            s.sequences_edges_indices,
            s.edges_deviation_direct,
        ),
        structure,
        replace=tuple(data),
    )


def trails_of(structure: "EquilibriumStructure") -> list[tuple[int, ...]]:
    """
    Read the trails of a structure off its sequences.

    Parameters
    ----------
    structure :
        A structure.

    Returns
    -------
    trails :
        One tuple of node keys per trail, running from origin to support.

    Notes
    -----
    A shift moves a trail down the sequences without reordering it, so dropping
    the padding recovers the trail the search found.
    """
    sequences = np.asarray(structure.sequences)

    return [tuple(int(node) for node in column[column >= 0]) for column in sequences.T]


def sequence_data(
    trails: list[tuple[int, ...]],
    edges: Int[np.ndarray, "edges 2"],
    num_nodes: int,
    num_edges_trail: int,
    shifts: Int[np.ndarray, "trails"] | None = None,
) -> SequenceData:
    """
    Derive everything the layout of a set of trails settles.

    Parameters
    ----------
    trails :
        One tuple of node keys per trail, running from origin to support.
    edges :
        The node key pair of each edge, trail edges first.
    num_nodes :
        The number of nodes in the structure.
    num_edges_trail :
        The number of trail edges, which is where the deviation block starts.
    shifts :
        The sequence each trail starts at. Defaults to every trail starting at
        the first sequence.

    Returns
    -------
    data :
        The sequences, the origin nodes, the sequence-to-edge maps, and the mask
        of deviation edges whose two nodes share a sequence.

    Notes
    -----
    Grouped into one function because a shift changes all of it at once, so the
    constructor and any transform that shifts a trail cannot derive one part
    without the rest.
    """
    sequences, origin_nodes = sequences_from_trails(trails, shifts)

    edge_index = {}
    for index, (u, v) in enumerate(edges):
        edge_index[(int(u), int(v))] = index

    sequences_edges = []
    for pair in zip(sequences[:-1], sequences[1:], strict=True):
        row = []
        for edge in zip(*pair, strict=True):
            u, v = int(edge[0]), int(edge[1])
            row.append(edge_index.get((u, v), edge_index.get((v, u), -1)))
        sequences_edges.append(row)
    sequences_edges = np.asarray(sequences_edges, dtype=int).reshape(
        max(len(sequences) - 1, 0),
        len(trails),
    )

    indices = np.flatnonzero(sequences_edges.ravel() >= 0)

    sequence_of = np.full(num_nodes, -1, dtype=int)
    for index, sequence in enumerate(sequences):
        for node in sequence:
            if node >= 0:
                sequence_of[node] = index

    is_deviation = np.zeros(len(edges), dtype=bool)
    is_deviation[num_edges_trail:] = True
    is_direct = sequence_of[edges[:, 0]] == sequence_of[edges[:, 1]]

    return SequenceData(
        sequences=sequences,
        origin_nodes=origin_nodes,
        sequences_edges=sequences_edges,
        sequences_edges_indices=indices,
        edges_deviation_direct=jnp.asarray(
            np.logical_and(is_deviation, is_direct).astype(float),
        ),
    )


def trails_from_edges(
    nodes: Int[np.ndarray, "nodes"],
    supports: Int[np.ndarray, "nodes_fixed"],
    edges_trail: Int[np.ndarray, "edges_trail 2"],
) -> list[tuple[int, ...]]:
    """
    Order the nodes of a structure into trails.

    Parameters
    ----------
    nodes :
        The node keys.
    supports :
        The keys of the supported nodes, one per trail.
    edges_trail :
        The node key pair of each trail edge.

    Returns
    -------
    trails :
        One tuple of node keys per trail, running from origin node to support
        node.

    Notes
    -----
    A trail is the path of trail edges that connects an origin node to a support
    node. The search walks away from each support until no unvisited trail edge
    remains, which makes the last node it reaches the origin node.

    Every node must lie on a trail. A node reachable only through deviation
    edges needs an auxiliary trail, which is a change of topology and therefore
    the caller's responsibility, not the search's.
    """
    if len(supports) == 0:
        raise ValueError("No supports assigned, so no trail can be built")
    if len(edges_trail) == 0:
        raise ValueError("No trail edges defined")

    neighbors = _trail_neighbors(edges_trail)

    trails = []
    visited_all = set()
    for support in supports:
        trail = []
        visited = set()
        node = int(support)

        while True:
            candidates = [n for n in neighbors.get(node, ()) if n not in visited]
            trail.append(node)
            visited.add(node)

            if not candidates:
                break
            if len(candidates) > 1:
                raise ValueError(
                    f"Node {node} continues into {len(candidates)} unvisited trail "
                    f"edges, so its trail is ambiguous",
                )
            node = candidates[0]

        trail.reverse()
        trails.append(tuple(trail))
        visited_all.update(visited)

    unassigned = set(int(node) for node in nodes) - visited_all
    if unassigned:
        raise ValueError(
            f"Nodes {sorted(unassigned)} do not lie on a trail. They need an "
            f"auxiliary trail to a support",
        )

    return trails


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


def _shifts_from_indirect_edges(
    trails: list[tuple[int, ...]],
    edges_deviation: Int[np.ndarray, "edges_deviation 2"],
) -> Int[np.ndarray, "trails"]:
    """
    Shift every trail to the sequence of the node it deviates to.

    Parameters
    ----------
    trails :
        One tuple of node keys per trail, running from origin to support.
    edges_deviation :
        The node key pair of each deviation edge.

    Returns
    -------
    shifts :
        The sequence each trail starts at, column-aligned with the trails.

    Notes
    -----
    An origin node joined by a deviation edge to a node further along another
    trail would otherwise carry that edge as an indirect one. Starting its trail
    at the other node's sequence makes the edge direct, which keeps it in the
    first, non-iterative equilibrium pass.

    One pass is enough whenever the trails form no cycle of such edges. A later
    trail may shift after an earlier one read its sequence, so the result
    depends on the order of the trails.
    """
    sequence_of = {}
    for trail in trails:
        for offset, node in enumerate(trail):
            sequence_of[node] = offset

    shifts = np.zeros(len(trails), dtype=int)
    for index, trail in enumerate(trails):
        origin = trail[0]
        for u, v in edges_deviation:
            u, v = int(u), int(v)
            if origin not in (u, v):
                continue
            other = v if origin == u else u
            if sequence_of.get(other, 0) != sequence_of[origin]:
                shifts[index] = sequence_of[other]
                sequence_of.update(
                    {node: shifts[index] + off for off, node in enumerate(trail)},
                )

    return shifts


def _trail_neighbors(
    edges_trail: Int[np.ndarray, "edges_trail 2"],
) -> dict[int, set[int]]:
    """
    Map each node to the nodes it shares a trail edge with.
    """
    neighbors: dict[int, set[int]] = {}
    for u, v in edges_trail:
        u, v = int(u), int(v)
        if u == v:
            raise ValueError(f"Trail edge ({u}, {v}) is a self-loop")
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    return neighbors
