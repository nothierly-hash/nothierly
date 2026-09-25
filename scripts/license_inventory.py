"""License inventory of the dependency closure, from the metadata of the installed distributions.

Reads the requirement manifests, walks ``Requires-Dist`` through the distributions installed in the running
environment, and classifies each one by the license it DECLARES: an SPDX expression, trove classifiers or the
``License`` field. Markers are evaluated for this machine and for Linux on Python 3.12 (what the Dockerfile
installs), so a Linux-only dependency shows up. Distributions in the closure that are not installed here cannot
be read and are listed separately.

Categories: ``copyleft`` (GPL, LGPL, AGPL), ``weak copyleft`` (MPL, EPL: file-level obligations),
``permissive`` and ``unresolved``. In an ``A OR B`` expression the holder may choose, so the loosest alternative
decides and the row is marked dual. ``A AND B`` applies both, so the strictest decides.

What this does not do, so an empty result is not read as a clearance: it trusts declared metadata (which is
sometimes wrong, for example a stray ``Other/Proprietary`` classifier); it does not scan source files or the
native libraries bundled inside wheels (``libzmq`` in ``pyzmq``, for one); and it is not legal advice. A
scanner such as ScanCode, and counsel, are the next step before a sale.

Usage::

    python scripts/license_inventory.py --output docs/DEPENDENCY_LICENSES.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

MANIFESTS: Final[tuple[str, ...]] = ("requirements.txt", "requirements-dev.txt", "retail_gateway/requirements.txt")
_ENVIRONMENTS: Final[tuple[dict[str, str], ...]] = (
    {},
    {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
     "python_version": "3.12", "python_full_version": "3.12.0", "platform_python_implementation": "CPython",
     "implementation_name": "cpython"},
)
_ORDER: Final[dict[str, int]] = {"copyleft": 3, "weak copyleft": 2, "unresolved": 1, "permissive": 0}
_STRONG = re.compile(r"\b[al]?gpl|gnu (lesser |affero )?general public", re.I)
_WEAK = re.compile(r"\bmpl\b|mpl-|mozilla|\bepl\b|epl-|eclipse", re.I)
_PERMISSIVE = re.compile(r"\bmit|\bbsd|apache|\bisc\b|\bpsf|python software|unlicense|public domain|zlib|cc0|0bsd|cnri|\btcl\b", re.I)

# Declared metadata that is wrong or missing, checked by hand against the shipped license file and the project page.
REVIEWED: Final[dict[str, str]] = {
    "ib-async": "BSD 2-Clause: the shipped LICENSE and the repository say so; the 'Other/Proprietary' classifier is a labelling error.",
    "v20": "MIT (copyright OANDA): the shipped LICENSE and the repository say so; the 'Other/Proprietary' classifier is a labelling error.",
    "streamlit-autorefresh": "MIT: metadata says UNKNOWN, but the shipped LICENSE is the MIT text (with the packaging template's copyright line) and the repository declares MIT.",
}


@dataclass(frozen=True)
class Row:
    name: str
    version: str
    declared: str
    category: str
    dual: bool
    origin: str  # "declared" (in a manifest) or "transitive"
    via: str


def _term_category(term: str) -> str | None:
    placeholder = re.fullmatch(r"\s*__(.+)__\s*", term)
    if placeholder:
        return placeholder.group(1)
    if _STRONG.search(term):
        return "copyleft"
    if _WEAK.search(term):
        return "weak copyleft"
    if _PERMISSIVE.search(term):
        return "permissive"
    return None


def _evaluate(expression: str) -> tuple[str, bool]:
    """``(category, dual)``. Parentheses first, then ``AND`` (strictest wins), then ``OR`` (loosest wins)."""
    while (group := re.search(r"\(([^()]*)\)", expression)):
        inner, _ = _evaluate(group.group(1))
        expression = f"{expression[:group.start()]} __{inner}__ {expression[group.end():]}"
    alternatives = [part for part in re.split(r"\s+OR\s+", expression) if part.strip()]
    per_alternative: list[str] = []
    for alternative in alternatives:
        found = [c for c in (_term_category(term) for term in re.split(r"\s+AND\s+|\s*;\s*", alternative)) if c is not None]
        if found:
            per_alternative.append(max(found, key=_ORDER.__getitem__))
    if not per_alternative:
        return "unresolved", False
    return min(per_alternative, key=_ORDER.__getitem__), len(alternatives) > 1


def categorize(declared: str) -> tuple[str, bool]:
    """``(category, dual)`` for a declared license string. ``A OR B`` lets the holder pick the loosest option."""
    return _evaluate(declared)


def declared_license(dist: metadata.Distribution) -> str:
    md = dist.metadata
    expression = (md.get("License-Expression") or "").strip()
    classifiers = "; ".join(c.split("::")[-1].strip() for c in md.get_all("Classifier", []) if c.startswith("License ::"))
    field = (md.get("License") or "").strip()
    if len(field) >= 120:  # some packages put the whole license text in this field: keep its title line only
        title = field.splitlines()[0].strip()
        field = title if len(title) < 80 else ""
    return expression or " ; ".join(part for part in (classifiers, field) if part and part.upper() != "UNKNOWN")


def manifest_requirements(path: Path) -> list[Requirement]:
    requirements: list[Requirement] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith(("-", "git+", "http")):
            continue
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement:
            continue
    return requirements


def _marker_applies(requirement: Requirement, extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    for environment in _ENVIRONMENTS:
        for extra in extras or {""}:
            try:
                if requirement.marker.evaluate({**environment, "extra": extra}):
                    return True
            except Exception:
                return True
    return False


def closure(roots: Iterable[Requirement]) -> tuple[dict[str, metadata.Distribution], dict[str, str], dict[str, str], set[str]]:
    """``(installed, absent, parents, declared)``: distributions found, names not installed, who pulled each, root names."""
    declared = set()
    queue: list[tuple[str, set[str]]] = []
    for requirement in roots:
        name = canonicalize_name(requirement.name)
        declared.add(name)
        queue.append((name, set(requirement.extras)))
    installed: dict[str, metadata.Distribution] = {}
    absent: dict[str, str] = {}
    parents: dict[str, str] = {}
    visited: set[tuple[str, frozenset[str]]] = set()
    while queue:
        name, extras = queue.pop()
        if (name, frozenset(extras)) in visited:
            continue
        visited.add((name, frozenset(extras)))
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            absent.setdefault(name, parents.get(name, ""))
            continue
        installed[name] = dist
        for line in dist.requires or []:
            try:
                child = Requirement(line)
            except InvalidRequirement:
                continue
            if _marker_applies(child, extras):
                child_name = canonicalize_name(child.name)
                parents.setdefault(child_name, name)
                queue.append((child_name, set(child.extras)))
    return installed, absent, parents, declared


def build_rows(installed: dict[str, metadata.Distribution], parents: dict[str, str], declared: set[str]) -> list[Row]:
    rows = []
    for name, dist in sorted(installed.items()):
        text = declared_license(dist)
        category, dual = categorize(text)
        rows.append(Row(name, dist.metadata["Version"], text or "(nothing declared)", category, dual,
                        "declared" if name in declared else "transitive", parents.get(name, "-")))
    return rows


def render(rows: Sequence[Row], absent: dict[str, str]) -> str:
    counts = {category: sum(1 for row in rows if row.category == category) for category in _ORDER}
    lines = [
        f"Closure read from installed metadata: {len(rows)} distributions ({sum(1 for r in rows if r.origin == 'declared')} declared in a manifest, "
        f"{sum(1 for r in rows if r.origin == 'transitive')} transitive). "
        + ", ".join(f"{counts[c]} {c}" for c in ("permissive", "weak copyleft", "copyleft", "unresolved")) + ".",
        "",
    ]
    flagged = [row for row in rows if row.category != "permissive"]
    if flagged:
        lines += ["| Distribution | Version | Origin | Declared license | Category | Note |", "|---|---|---|---|---|---|"]
        for row in sorted(flagged, key=lambda r: (-_ORDER[r.category], r.name)):
            note = REVIEWED.get(row.name, "dual: the holder chooses the loosest option" if row.dual else "")
            origin = row.origin if row.origin == "declared" else f"transitive, via {row.via}"
            lines.append(f"| {row.name} | {row.version} | {origin} | {row.declared} | {row.category} | {note} |")
        lines.append("")
    reviewed_but_permissive = [row for row in rows if row.name in REVIEWED and row.category == "permissive"]
    if reviewed_but_permissive:
        lines += ["Declared metadata that is misleading, reviewed by hand:", ""]
        lines += [f"- {row.name} {row.version}: {REVIEWED[row.name]}" for row in reviewed_but_permissive]
        lines.append("")
    if absent:
        lines += ["In the closure but not installed here (Linux-only, GPU or optional), so not read: "
                  + ", ".join(f"{name} (via {via or 'a manifest'})" for name, via in sorted(absent.items())) + ".", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="Write the Markdown here instead of printing it.")
    parser.add_argument("--manifest", action="append", type=Path, default=None, help="Requirement file (repeatable; default: the three in this repository).")
    args = parser.parse_args()
    manifests = args.manifest or [Path(m) for m in MANIFESTS]
    missing = [str(m) for m in manifests if not m.exists()]
    if missing:
        print(f"manifest not found: {', '.join(missing)}")
        return 2
    roots = [requirement for manifest in manifests for requirement in manifest_requirements(manifest)]
    installed, absent, parents, declared = closure(roots)
    text = render(build_rows(installed, parents, declared), absent)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
