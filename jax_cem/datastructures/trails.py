"""
The trail search that orders the nodes of a structure, and the shift that aligns
one trail with another.
"""

from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jaxtyping import Int

from jax_cem.datastructures.sequences import build_sequences

if TYPE_CHECKING:
    from jax_cem.datastructures.structures import EquilibriumStructure

__all__ = [
    "align_trails",
    "build_trails",
]


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

    sequences = build_sequences(trails, edges, shifts)

    return eqx.tree_at(lambda s: s.sequences, structure, replace=sequences)


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
    sequences = np.asarray(structure.sequences.nodes)

    return [tuple(int(node) for node in column[column >= 0]) for column in sequences.T]


def build_trails(
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
