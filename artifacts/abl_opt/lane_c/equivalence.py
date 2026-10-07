"""Lane C equivalence gate: HEAD implementation vs working-tree implementation.

Loads each lane-C owned module twice — once from `git show HEAD:<path>` into a
separate module object, once from the working tree — and compares their outputs
over the same 11,441-row corpus. This is a stronger check than a repr digest:
it asserts `base.f(args) == new.f(args)` for every row and every target, with
Python's own equality (so frozenset order is irrelevant and list order is not).

Usage:
    python artifacts/abl_opt/lane_c/equivalence.py [--base-ref HEAD] [--rows N]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("EUROMONITOR_PROJECT_ROOT", str(ROOT))

from micro_bench import build_targets, load_rows  # noqa: E402


def load_base_module(rel_path: str, base_ref: str):
    """Execute the HEAD revision of `rel_path` as its own module object."""
    source = subprocess.run(
        ["git", "show", f"{base_ref}:{rel_path}"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    name = "lane_c_base_" + rel_path.replace("/", "_").removesuffix(".py")
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / rel_path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def normalize_result(value):
    """Structural comparison for values whose class identity differs by design.

    The base revision and the working tree define their own `Candidate`
    dataclass, and a dataclass only compares equal to an instance of the SAME
    class, so a byte-identical result from the two module objects would look
    like a mismatch. Compare by fields instead.
    """
    if isinstance(value, (list, tuple)):
        return [normalize_result(item) for item in value]
    if hasattr(value, "start") and hasattr(value, "end") and hasattr(value, "label"):
        return ("Candidate", value.start, value.end, value.label,
                normalize_result(getattr(value, "normalized", None)),
                getattr(value, "source", None))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="HEAD")
    parser.add_argument("--rows", type=int, default=11441)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    rows = load_rows()
    if args.rows < len(rows):
        rows = rows[: args.rows]
    # Short aliases so the per-target argument lambdas read naturally.
    for row in rows:
        row["title"] = row["sku_name_eng"]
        row["description"] = row["description_short_eng"]
        row["breadcrumbs"] = row["breadcrumbs_eng"]

    # The base modules must be loaded BEFORE the callables in build_targets bind
    # the working-tree functions, and the working-tree modules are already in
    # sys.modules as `core.*` / `ner.*`.
    base = {
        rel: load_base_module(rel, args.base_ref)
        for rel in ("src/core/url_evidence.py", "src/core/date_evidence.py",
                    "src/core/critical_attributes.py", "src/core/sweetener_values.py",
                    "src/ner/ner_product_attributes.py")
    }
    import core.critical_attributes as new_ca
    import core.date_evidence as new_de
    import core.sweetener_values as new_sv
    import core.url_evidence as new_ue
    import ner.ner_product_attributes as new_ner

    base_ca = base["src/core/critical_attributes.py"]
    base_de = base["src/core/date_evidence.py"]
    base_sv = base["src/core/sweetener_values.py"]
    base_ue = base["src/core/url_evidence.py"]
    base_ner = base["src/ner/ner_product_attributes.py"]

    from core.text import normalized_attribute_text as fold

    columns = {
        "title": [row["sku_name_eng"] for row in rows],
        "attribute": [row["attribute"] for row in rows],
        "description": [row["description_short_eng"] for row in rows],
        "breadcrumbs": [row["breadcrumbs_eng"] for row in rows],
        "category": [row["category"] for row in rows],
        "url": [row["sku_url"] for row in rows],
        "image": [row["image_url"] for row in rows],
        "brand": [row["brand"] for row in rows],
    }

    checks = []

    def check(name, base_fn, new_fn, args_of_row):
        checks.append((name, base_fn, new_fn, args_of_row))

    check("url_text", base_ue.url_text, new_ue.url_text,
          lambda r: (r["sku_url"], r["image_url"]))
    check("extract_date_evidence", base_de.extract_date_evidence, new_de.extract_date_evidence,
          lambda r: (r["title"], r["attribute"], r["description"], r["breadcrumbs"], r["category"]))
    check("extract_critical_claims", base_ca.extract_critical_claims, new_ca.extract_critical_claims,
          lambda r: (r["title"], r["attribute"]))
    check("extract_description_claims", base_ca.extract_description_claims, new_ca.extract_description_claims,
          lambda r: (r["description"],))
    check("extract_flavor_tokens", base_ca.extract_flavor_tokens, new_ca.extract_flavor_tokens,
          lambda r: (r["title"],))
    check("extract_declared_flavor_tokens", base_ca.extract_declared_flavor_tokens,
          new_ca.extract_declared_flavor_tokens, lambda r: (r["title"], r["attribute"]))
    check("extract_made_from_tokens", base_ca.extract_made_from_tokens, new_ca.extract_made_from_tokens,
          lambda r: (r["title"], r["attribute"]))
    check("flavor_tokens_from_text", base_ca.flavor_tokens_from_text, new_ca.flavor_tokens_from_text,
          lambda r: (fold(r["title"]),))
    check("extract_sweetening_status", base_sv.extract_sweetening_status, new_sv.extract_sweetening_status,
          lambda r: (r["title"], r["attribute"], r["description"]))
    check("title_sweetener_types", base_sv.title_sweetener_types, new_sv.title_sweetener_types,
          lambda r: (r["title"], r["description"]))
    check("negated_sweetener_types", base_sv.negated_sweetener_types, new_sv.negated_sweetener_types,
          lambda r: (r["title"], r["attribute"], r["description"]))
    check("declared_sweeteners", base_sv.declared_sweeteners, new_sv.declared_sweeteners,
          lambda r: (r["attribute"],))
    check("parse_attribute_details", base_ner.parse_attribute_details, new_ner.parse_attribute_details,
          lambda r: (r["attribute"],))
    check("extract_title_attributes", base_ner.extract_title_attributes, new_ner.extract_title_attributes,
          lambda r: (r["title"],))
    check("_candidates_from_package_details", base_ner._candidates_from_package_details,
          new_ner._candidates_from_package_details, lambda r: (r["title"],))
    check("_candidates_from_measurements", base_ner._candidates_from_measurements,
          new_ner._candidates_from_measurements, lambda r: (r["title"],))
    check("find_brand_span", base_ner.find_brand_span, new_ner.find_brand_span,
          lambda r: (r["title"], r["brand"]))
    def unit_suffix_args(row):
        tokens = [item for item in new_ue.url_text(row["sku_url"]).split() if item[:1].isdigit()]
        return (tokens[0] if tokens else "", new_ue._spec())

    check("_is_noise", base_ue._is_noise, new_ue._is_noise,
          lambda r: tuple(new_ue.url_text(r["sku_url"]).split()))
    check("_has_unit_suffix", base_ue._has_unit_suffix, new_ue._has_unit_suffix, unit_suffix_args)
    check("source_consistency_flags", base_ca.source_consistency_flags, new_ca.source_consistency_flags,
          lambda r: (r["attribute"], r["title"], base_ca.extract_critical_claims(r["title"], r["attribute"])["sweetener"]))

    failures = 0
    compared = 0
    for name, base_fn, new_fn, args_of_row in checks:
        if args.only and args.only not in name:
            continue
        mismatches = []
        for index, row in enumerate(rows):
            call_args = args_of_row(row)
            try:
                expected = base_fn(*call_args)
            except Exception as error:  # noqa: BLE001 - base ref may raise
                expected = ("<raised>", type(error).__name__, str(error))
            try:
                actual = new_fn(*call_args)
            except Exception as error:  # noqa: BLE001
                actual = ("<raised>", type(error).__name__, str(error))
            compared += 1
            if normalize_result(actual) != normalize_result(expected):
                mismatches.append((index, call_args, expected, actual))
        if mismatches:
            failures += 1
            print(f"MISMATCH {name}: {len(mismatches)} rows; first 3:")
            for index, call_args, expected, actual in mismatches[:3]:
                print(f"   row {index} args={call_args!r}\n     base={expected!r}\n     new ={actual!r}")
        else:
            print(f"ok       {name} ({len(rows)} rows x args)")
    print(f"\n{compared} comparisons, {failures} mismatching targets")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
