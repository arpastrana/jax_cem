"""
The trails of a structure: the search that finds them, the layout that lays them
out into sequences, and the shift that aligns one with another.
"""

from typing import TYPE_CHECKING
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Int

from jax_cem.datastructures.sequences import Sequence
from jax_cem.datastructures.sequences import sequences_from_trails

if TYPE_CHECKING:
    from jax_cem.datastructures.structures import Structure

__all__ = [
    "Trails",
    "align_trails",
    "build_trails",
    "search_trails",
]


class Trails(NamedTuple):
    """
    The trails of a structure, laid out into the sequences it steps through.

    Attributes
    ----------
    sequences :
        Every sequence of the structure, stacked along a leading sequence axis.
    edges_sequence :
        The slot of the layout that each trail edge occupies, counted row by row
        over the grid that `Sequence.edges` stacks.
    nodes_sequence :
        The sequence that each node of the structure is put in equilibrium at.

    Notes
    -----
    The two maps out of the layout are named for the direction they run in, keyed
    entity first: `edges_sequence` is keyed by the trail edge and holds a slot,
    shaped `"edges_trail"`, where `Sequence.edges` is keyed by a slot and holds a
    trail edge, shaped `"trails"`. They used to share the one name and be told apart
    by shape alone, which put the distinction in this docstring rather than in what
    a reader of either one types.

    They cannot be confused silently either. Stacked, the map from slots is a grid
    where the map from edges is flat, so a swap fails on rank, and the two can never
    even hold the same number of entries: a trail of ``n`` nodes fills ``n`` slots
    and owns ``n - 1`` edges, so the grid always has more slots than the structure
    has trail edges.

    `edges_sequence` never carries the padding value. The map from slots does, at
    the last slot of every trail, which no edge leaves. That is the asymmetry the
    shared name hid: the map from slots must admit absence, and the map from edges
    is total, since every trail edge occupies exactly one slot. `build_trails`
    rejects a layout where it does not.

    `nodes_sequence` is the plain inverse of `sequences.nodes`, keyed by the node
    where that is keyed by the slot, and total for the same reason the map from
    edges is, since every node lies on exactly one trail and so occupies exactly one
    slot. A shift rewrites both along with the layout they read, which is why they
    are held here rather than on the structure.

    The pairs of consecutive sequences carry one slot per trail, and a trail that
    does not span a pair leaves its slot empty. The occupied slots are in
    bijection with the trail edges, which is what lets a quantity computed over
    the layout be read back in edge order by one gather.
    """

    sequences: Sequence
    edges_sequence: Int[Array, "edges_trail"]
    nodes_sequence: Int[Array, "nodes"]


def align_trails(structure: "Structure") -> "Structure":
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

    The layout and the origin nodes are replaced together. A shift leaves the
    origins alone, so the two agree, and replacing both is what keeps them
    agreeing rather than what a reader has to take on trust.
    """
    trail_nodes = read_trail_nodes(structure)
    shifts = _shifts_from_indirect_edges(
        trail_nodes,
        np.asarray(structure.edges_deviation),
    )

    trails, origin_nodes = build_trails(
        trail_nodes,
        np.asarray(structure.edges_trail),
        structure.num_nodes,
        shifts,
    )
    replaced = (trails, origin_nodes)
    aligned = eqx.tree_at(
        lambda tree: (tree.trails, tree.origin_nodes),
        structure,
        replace=replaced,
    )

    return aligned


def read_trail_nodes(structure: "Structure") -> list[tuple[int, ...]]:
    """
    Read the nodes of every trail of a structure off its layout.

    Parameters
    ----------
    structure :
        A structure.

    Returns
    -------
    trail_nodes :
        One tuple of node keys per trail, running from origin to support.

    Notes
    -----
    A shift moves a trail down the sequences without reordering it, so dropping
    the padding recovers the trail the search found.
    """
    sequences = np.asarray(structure.sequence_nodes)
    trail_nodes = [
        tuple(int(node) for node in column[column >= 0]) for column in sequences.T
    ]

    return trail_nodes


def search_trails(
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
    trail_nodes :
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

    trail_nodes = []
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
        trail_nodes.append(tuple(trail))
        visited_all.update(visited)

    unassigned = set(int(node) for node in nodes) - visited_all
    if unassigned:
        raise ValueError(
            f"Nodes {sorted(unassigned)} do not lie on a trail. They need an "
            f"auxiliary trail to a support",
        )

    return trail_nodes


def build_trails(
    trail_nodes: list[tuple[int, ...]],
    edges_trail: Int[np.ndarray, "edges_trail 2"],
    num_nodes: int,
    shifts: Int[np.ndarray, "trails"] | None = None,
) -> tuple[Trails, Int[Array, "trails"]]:
    """
    Lay out a set of trails into the sequences an equilibrium steps through.

    Parameters
    ----------
    trail_nodes :
        One tuple of node keys per trail, running from origin to support.
    edges_trail :
        The node key pair of each trail edge.
    num_nodes :
        The number of nodes the trails run over.
    shifts :
        The sequence each trail starts at. Defaults to every trail starting at
        the first sequence.

    Returns
    -------
    trails :
        The trails, laid out.
    origin_nodes :
        The first node of each trail, column-aligned with the sequences.

    Notes
    -----
    The slot each trail edge occupies is inverted from the map that reads an edge
    off a slot, so that the caller gathers in edge order rather than scattering
    into it. Both directions are kept, since the equilibrium reads a length off a
    slot and its forces back off an edge. The sequence of a node is inverted the
    same way, out of the layout that reads a node off a slot.

    The edges of a slot are padded with an empty row to the height of the
    sequences, so that the scan steps over the nodes and the edges of a sequence
    together.

    The origin nodes are read off the layout, at the first slot a column occupies,
    rather than walked out of the trails one at a time. A shift moves a trail down
    its column without reordering it, so that slot holds the node the trail starts
    at whatever the shift.

    The origins leave beside the trails rather than within them, since a shift
    rewrites the layout and leaves them alone, so they belong to the structure and
    not to the layout.

    The index data is computed with NumPy and converted once here, so what the
    structure stores is device-resident.
    """
    nodes = sequences_from_trails(trail_nodes, shifts)

    # the first occupied slot of a column holds the node its trail starts at
    origin_slots = np.argmax(nodes >= 0, axis=0)
    trail_columns = np.arange(nodes.shape[-1])
    origins = nodes[origin_slots, trail_columns]

    # only a trail edge can occupy a slot, so only trail edges are looked up
    edge_index = {}
    for index, (u, v) in enumerate(edges_trail):
        edge_index[(int(u), int(v))] = index

    # one row per pair of consecutive sequences, one slot per trail within it
    rows = []
    for pair in zip(nodes[:-1], nodes[1:], strict=True):
        row = []
        for edge in zip(*pair, strict=True):
            u, v = int(edge[0]), int(edge[1])
            row.append(edge_index.get((u, v), edge_index.get((v, u), -1)))
        rows.append(row)
    grid = np.asarray(rows, dtype=int)

    # sized by the edges rather than by the slots, so the check below is reached
    slots = np.flatnonzero(grid >= 0)
    edges_sequence = np.full(len(edges_trail), -1, dtype=int)
    edges_sequence[grid[grid >= 0]] = slots

    # an occupied slot per trail edge, so the inversion leaves nothing behind
    if np.any(edges_sequence < 0):
        missing = np.flatnonzero(edges_sequence < 0).tolist()
        raise ValueError(
            f"Trail edges {missing} occupy no sequence slot; every trail edge must "
            f"join a node of one sequence to the node of the next on its trail",
        )

    # every node lies on exactly one trail, so the layout holds each one once
    is_occupied = nodes >= 0
    rows_occupied, _ = np.nonzero(is_occupied)
    nodes_sequence = np.full(num_nodes, -1, dtype=int)
    nodes_sequence[nodes[is_occupied]] = rows_occupied

    # a slot per node, so the inversion leaves nothing behind
    if np.any(nodes_sequence < 0):
        absent = np.flatnonzero(nodes_sequence < 0).tolist()
        raise ValueError(
            f"Nodes {absent} occupy no sequence slot; every node must lie on a "
            f"trail to be put in equilibrium",
        )

    empty = np.full((1, nodes.shape[-1]), -1, dtype=int)

    sequences = Sequence(
        nodes=jnp.asarray(nodes),
        edges=jnp.asarray(np.concatenate((grid, empty))),
    )

    trails = Trails(
        sequences=sequences,
        edges_sequence=jnp.asarray(edges_sequence),
        nodes_sequence=jnp.asarray(nodes_sequence),
    )
    origin_nodes = jnp.asarray(origins)

    return trails, origin_nodes


def _shifts_from_indirect_edges(
    trail_nodes: list[tuple[int, ...]],
    edges_deviation: Int[np.ndarray, "edges_deviation 2"],
) -> Int[np.ndarray, "trails"]:
    """
    Shift every trail to the sequence of the node it deviates to.

    Parameters
    ----------
    trail_nodes :
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
    for trail in trail_nodes:
        for offset, node in enumerate(trail):
            sequence_of[node] = offset

    shifts = np.zeros(len(trail_nodes), dtype=int)
    for index, trail in enumerate(trail_nodes):
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
