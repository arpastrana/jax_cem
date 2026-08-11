"""
The sequences the equilibrium scan steps through, and what their layout settles.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

__all__ = [
    "SequenceData",
    "build_sequences",
    "sequences_from_trails",
]


class SequenceData(NamedTuple):
    """
    Everything about a structure that the layout of its trails settles.
    """

    sequences: Int[Array, "sequences trails"]
    origin_nodes: Int[Array, "trails"]
    sequences_edges: Int[Array, "sequences_edges trails"]
    sequences_edges_indices: Int[Array, "edges_trail"]
    edges_deviation_direct: Float[Array, "edges"]


def build_sequences(
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

    The index data is computed with NumPy and converted once here, so what the
    structure stores is device-resident.
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
        sequences=jnp.asarray(sequences),
        origin_nodes=jnp.asarray(origin_nodes),
        sequences_edges=jnp.asarray(sequences_edges),
        sequences_edges_indices=jnp.asarray(indices),
        edges_deviation_direct=jnp.asarray(
            np.logical_and(is_deviation, is_direct).astype(float),
        ),
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
