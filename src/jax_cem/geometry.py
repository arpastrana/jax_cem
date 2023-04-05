import jax.numpy as jnp


__all__ = ["vector_length",
           "vector_normalized"]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def vector_length(v, keepdims=True):
    """
    Calculate the length of a vector over its last dimension.
    """
    return jnp.linalg.norm(v, axis=-1, keepdims=keepdims)


def vector_normalized(u):
    """
    Scale a vector such that it has a unit length.
    """
    return u / vector_length(u)


def trail_length_from_plane_intersection_numpy(point, vector, plane, tol=1e-6):
    """
    Calculates the signed length of a trail edge from a vector-plane intersection.

    Parameters
    ----------
    point : ``list`` of ``float``
        The XYZ coordinates of the base position of the vector.
    direction : ``list`` of ``float``
        The XYZ coordinates of the vector.
    plane : ``Plane``
        A COMPAS plane defined by a base point and a normal vector.
    tol : ``float``, optional
        A tolerance to check if vector and the plane normal are parallel
        Defaults to ``1e-6``.

    Returns
    -------
    length : ``float``, ``None``
        The distance between ``pos`` and the resulting line-plane intersection.
        If not intersection is found, it returns ``None``.
    """
    origin, normal = plane
    cos_nv = np.dot(normal, normalize_vector_numpy(vector))

    if np.abs(cos_nv) < tol:
        return

    oa = origin - point
    cos_noa = np.dot(normal, oa)

    return cos_noa / cos_nv
