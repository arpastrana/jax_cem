"""
Check the shape annotations of the package while the suite runs.
"""

from jaxtyping import install_import_hook

# The hook must run before any test module imports jax_cem, which is why it sits
# in the root conftest rather than in the pytest addopts.
install_import_hook("jax_cem", "beartype.beartype")
