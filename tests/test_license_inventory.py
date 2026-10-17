"""Tests for scripts/license_inventory.py: the classification of declared license strings, the closure over
requirement markers and extras (with fake distributions), and the report."""

from __future__ import annotations

import importlib.util
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest


def _load(name: str, filename: str) -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


li = _load("license_inventory_under_test", "license_inventory.py")


@pytest.mark.parametrize("declared, expected", [
    ("MIT", ("permissive", False)),
    ("BSD-3-Clause AND 0BSD AND MIT", ("permissive", False)),
    ("Apache-2.0 OR BSD-3-Clause", ("permissive", True)),
    ("GNU General Public License v3 (GPLv3) ; GPL", ("copyleft", False)),
    ("LGPL-3.0-or-later", ("copyleft", False)),
    ("MPL-2.0 AND MIT", ("weak copyleft", False)),
    ("EPL-2.0 OR GPL-2.0-or-later", ("weak copyleft", True)),  # the holder picks the EPL
    ("MPL-2.0 AND (Apache-2.0 OR MIT)", ("weak copyleft", False)),  # the AND still applies to the MPL part
    ("Apache-2.0 AND (MIT OR GPL-3.0-only)", ("permissive", False)),
    ("Other/Proprietary License ; MIT", ("permissive", False)),
    ("", ("unresolved", False)),
    ("Some Bespoke Terms", ("unresolved", False)),
])
def test_categorize(declared: str, expected: tuple[str, bool]) -> None:
    assert li.categorize(declared) == expected


def test_manifest_requirements_skip_comments_options_and_urls(tmp_path: Path) -> None:
    manifest = tmp_path / "requirements.txt"
    manifest.write_text(
        "# a comment\n--extra-index-url https://example.invalid/whl/cpu\n\nfoo>=1.0,<2.0  # trailing\n"
        "bar[extra]==2.1 ; sys_platform == 'linux'\ngit+https://example.invalid/x.git\nnot a requirement !!\n", encoding="utf-8",
    )

    requirements = li.manifest_requirements(manifest)

    assert [r.name for r in requirements] == ["foo", "bar"]
    assert requirements[1].extras == {"extra"} and requirements[1].marker is not None


class _Metadata(dict[str, str]):
    """The slice of ``email.message.Message`` the script uses: ``[]``, ``get`` and ``get_all``."""

    def get_all(self, key: str, default: Any = None) -> Any:
        return default


class _FakeDist:
    def __init__(self, version: str, license_text: str, requires: list[str]) -> None:
        self.metadata = _Metadata({"Version": version, "License-Expression": license_text})
        self.requires = requires


def _fake_environment(monkeypatch: pytest.MonkeyPatch, dists: dict[str, tuple[str, str, list[str]]]) -> None:
    def distribution(name: str) -> Any:
        if name not in dists:
            raise metadata.PackageNotFoundError(name)
        return _FakeDist(*dists[name])

    monkeypatch.setattr(li.metadata, "distribution", distribution)


def test_closure_follows_markers_for_this_machine_and_linux_and_skips_unrequested_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_environment(monkeypatch, {
        "root": ("1.0", "MIT", ["always", "linux-only; sys_platform == 'linux'", "extra-dep; extra == 'fancy'", "never; python_version < '3.0'"]),
        "always": ("1.0", "BSD-3-Clause", ["deep"]),
        "linux-only": ("1.0", "Apache-2.0", []),
        "deep": ("1.0", "GPL-3.0-only", []),
        "extra-dep": ("1.0", "MIT", []),
        "never": ("1.0", "MIT", []),
    })

    installed, absent, parents, declared = li.closure([li.Requirement("root")])

    assert set(installed) == {"root", "always", "linux-only", "deep"}  # 'extra-dep' needs the extra, 'never' has an impossible marker
    assert parents["deep"] == "always" and declared == {"root"} and absent == {}


def test_closure_with_a_requested_extra_pulls_the_extra_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_environment(monkeypatch, {"root": ("1.0", "MIT", ["extra-dep; extra == 'fancy'"]), "extra-dep": ("1.0", "MIT", [])})

    installed, _absent, _parents, _declared = li.closure([li.Requirement("root[fancy]")])

    assert set(installed) == {"root", "extra-dep"}


def test_a_dependency_that_is_not_installed_is_listed_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_environment(monkeypatch, {"root": ("1.0", "MIT", ["missing-one"])})

    installed, absent, _parents, _declared = li.closure([li.Requirement("root")])

    assert set(installed) == {"root"} and absent == {"missing-one": "root"}


def test_the_report_lists_only_what_needs_a_look_and_the_hand_reviewed_labelling_errors() -> None:
    rows = [
        li.Row("plain", "1.0", "MIT", "permissive", False, "declared", "-"),
        li.Row("gpl-thing", "2.0", "GPL-3.0-only", "copyleft", False, "transitive", "plain"),
        li.Row("dual-thing", "3.0", "EPL-2.0 OR GPL-2.0-or-later", "weak copyleft", True, "declared", "-"),
        li.Row("ib-async", "1.0", "BSD License ; Other/Proprietary License", "permissive", False, "declared", "-"),
        li.Row("streamlit-autorefresh", "1.0", "(nothing declared)", "unresolved", False, "declared", "-"),
    ]

    text = li.render(rows, {"nvidia-thing": "torch"})

    assert "5 distributions" in text and "2 permissive, 1 weak copyleft, 1 copyleft, 1 unresolved" in text
    assert "| gpl-thing | 2.0 | transitive, via plain |" in text
    assert text.index("gpl-thing") < text.index("dual-thing") < text.index("streamlit-autorefresh")  # strictest first
    assert "| plain |" not in text  # a plain permissive row is not repeated
    assert "- ib-async 1.0: BSD 2-Clause" in text
    assert "nvidia-thing (via torch)" in text


def test_a_license_text_pasted_into_the_license_field_is_read_from_its_title_line() -> None:
    pasted = (
        "MIT License", "", "Copyright (c) 2022 Someone", "",
        "Permission is hereby granted, free of charge, to any person obtaining a copy of this software.",
    )
    dist = _FakeDist("1.0", "", [])
    dist.metadata = _Metadata({"Version": "1.0", "License": "\n".join(pasted)})

    assert li.declared_license(dist) == "MIT License"
    assert li.categorize(li.declared_license(dist)) == ("permissive", False)
