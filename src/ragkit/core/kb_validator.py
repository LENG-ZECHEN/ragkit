"""Knowledge-base name validation (ISS-003).

ES index names and filesystem paths are derived directly from `kb_name`
(see `core/graph/store.py` for paths, `core/_ragflow/rag/utils/es_conn.py`
for index names). Unvalidated user input is therefore a path-traversal
vector — names like ``"../../etc/passwd"`` would escape ``storage/graphs/``,
and names with uppercase / spaces / slashes would break Elasticsearch.

This module enforces a conservative allowlist that mirrors Elasticsearch's
own index-name rules.
"""

from __future__ import annotations

import re

# Allowlist:
#   - First char: lowercase letter or digit (ES index names can't start with _ or -)
#   - Body chars: lowercase letter / digit / underscore / hyphen
#   - Max 63 chars (well under ES's 255 limit; keeps filenames sane too)
_KB_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def validate_kb_name(name: str) -> None:
    """Raise ``ValueError`` if ``name`` isn't a safe knowledge-base name.

    Rejects path-traversal patterns (``../``, ``/``), uppercase letters
    (ES forbids), and other strings that would break ES index creation
    or could escape the ``storage/graphs/`` directory.

    Args:
        name: Candidate kb name from CLI / API input.

    Raises:
        ValueError: If ``name`` doesn't match the allowlist pattern.
    """
    if not isinstance(name, str) or not _KB_NAME_RE.match(name):
        raise ValueError(
            f"Invalid kb name {name!r}. "
            "Must match [a-z0-9][a-z0-9_-]{0,62} — lowercase letters, "
            "digits, '_', '-' only (no spaces, no '/', no uppercase, max 63 chars)."
        )
