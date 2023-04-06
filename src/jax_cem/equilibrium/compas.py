from compas_cem.diagrams import FormDiagram
from compas_cem.elements import Node
from compas_cem.elements import Edge


__all__ = ["form_from_eqstate", "form_update"]


def form_from_eqstate(structure, eqstate):
    """
    Generate a form diagram from an equilibrium state calculated with JAX CEM.
    """
    form = FormDiagram()

    # add nodes
    for node in structure.nodes:
        form.add_node(Node(int(node)))

    # assign support nodes
    for node in structure.sequences[-1]:
        form.node_attribute(int(node), "type", "support")

    # add edges
    for u, v in structure.edges:
        form.add_edge(Edge(int(u), int(v), {}))

    # update form attributes
    form_update(form, structure, eqstate)

    return form


# ==========================================================================
# Helpers
# ==========================================================================


def form_update(form, structure, eqstate):
    """
    Update in-place the attributes of a form diagram with an equilibrium state.
    """
    xyz = eqstate.xyz.tolist()
    loads = eqstate.loads.tolist()
    lengths = eqstate.lengths.tolist()
    reactions = eqstate.reactions.tolist()
    forces = eqstate.forces.tolist()

    # update q values and lengths on edges
    for idx, edge in enumerate(structure.edges):
        form.edge_attribute(edge, name="force", value=forces[idx].pop())
        form.edge_attribute(edge, name="lengths", value=lengths[idx].pop())

    # update residuals on nodes
    for idx, node in enumerate(structure.nodes):
        form.node_attributes(node, "xyz", xyz[idx])
        form.node_attributes(node, ["rx", "ry", "rz"], reactions[idx])
        form.node_attributes(node, ["qx", "qy", "qz"], loads[idx])
