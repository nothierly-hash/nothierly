"""Static/non-secret configuration assets (YAML/TOML files, schemas, etc.).

Runtime, environment-driven settings live in :mod:`core.config`, not here.
This package holds configuration *data* files that are versioned alongside
the code (e.g. exchange metadata, strategy parameter defaults), as opposed
to secrets or per-deployment values which come from the environment/.env.
"""

from __future__ import annotations
