"""Build the laya IDENTITY metrics payload: data/laya/metrics_pairs.csv.

The owner order ("add accuracy + f1, change the dataset to pairs we already
know are the same — give it all the pairs we know are the same"): the laya
identity decision stops flying on sampled states and flies on the
GROUND-TRUTH pairs — every one of the 576 track_setup listing pairs
(564 confirmed-same + 12 confirmed-different, owner approved), in
`data/track_setup/listing_pairs.csv` row order (DETERMINISTIC, no shuffle).

Composition
-----------
Each pair's identity state is composed from the TWO endpoints' standardized
attribute strings (`data/track_setup/eligible_catalog.csv`, the `attribute`
column of the catalog the pair skus live in) and joins them into ONE state
string shaped exactly like the final_validation.csv `attribute_pairs`
convention: the six frozen identity-slice fields
(volume/pack/package_type/sweetener/flavor/carbonation — the
training.build_final_validation SLICE_FIELDS), each rendered
`field: v1=<side literal> v2=<side literal>`, side 1 then side 2, joined
with "; " — the same side-by-side v1_/v2_ pair convention the frozen P0
validation CSV freezes into its v1_*/v2_* columns, as ONE joined string
(the state column the `identity` decision binding resolves).

Side literal shape mirrors final_validation.csv per field:
  * the field ABSENT from a side's standardized attribute string -> ''
    (UNMEASURED, never invented: the shared_graph_data convention — a
    silently written '[]' would look like a measured zero);
  * the field present on a side -> a list literal of bare tokens
    (never quoted), one '[token]', many '[a, b]'; per-field token mapping
    from the standardized attribute keys:
      volume        <- 'Volume'      (plain numerics floatified like the
                                      canonical volume_set spells them)
      pack          <- nothing       (the catalog vocabulary carries no
                                      pack-count field: pack stays
                                      unmeasured on both sides)
      package_type  <- 'Pack Type'
      sweetener     <- 'Sweetener'
      flavor        <- 'Flavour' / 'Flavor'
      carbonation   <- 'Carbonization' / 'Carbonation'
    tokens lowercased, comma-split, internal spaces -> '_', SORTED (the
    canonical extractor's stable order). 'sparkling' is carried verbatim:
    this side's source is the catalog standardization, not the P0
    canonical-records extractor.

Columns: the metrics CSV carries the SAME columns data/final_validation.csv
carries, PLUS the `attribute_pairs` state column the `identity` decision
binding resolves (cli.laya_lane.DECISION_BINDINGS["identity"]:
state_column='attribute_pairs', wanted_columns=('gtin1', 'gtin2',
'true_label') — a subset of the final_validation header). Populated:
  * gtin1/gtin2           — the endpoints' catalog gtin (raw spelling);
  * gtin1_norm/gtin2_norm — normalize_gtin (training.folds, THE single
    source: drop the float '.0' artifact, keep digits, left-zero-pad to 14);
  * true_label            — the listing_pairs ground-truth label (0/1);
  * v1_*/v2_*             — the composed per-side slice literals above;
  * fold/fold_2/component_id/component_id_2/straddles_fold/endpoint_in_train
    — carried with '' values: the ground-truth pairs file carries NO graph
    fold/component state (its provenance is the graph-track setup, not the
    P0 merged graph) and values are NEVER invented; the accuracy/F1 harvest
    measures over ALL rows and reads none of them.

Fail loud: any sku lookup missing exits BEFORE writing; the
final_validation header drifting from the emitted shape exits too; the
emitted file is then re-measured through the lane's own staging
measurement (cli.laya_lane stage shape: header + rows + sha256) and must
satisfy the `identity` binding's wanted columns + state column.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from core.common import TRAIN_ROOT
from training.folds import normalize_gtin

PAIRS_PATH = TRAIN_ROOT / "data/track_setup/listing_pairs.csv"
CATALOG_PATH = TRAIN_ROOT / "data/track_setup/eligible_catalog.csv"
FINAL_VALIDATION_PATH = TRAIN_ROOT / "data/final_validation.csv"
OUTPUT = TRAIN_ROOT / "data/laya/metrics_pairs.csv"

# The six frozen identity-slice fields (training.build_final_validation
# SLICE_FIELDS) — the fields the final_validation.csv `attribute_pairs`
# shape is composed from, in that composition order.
SLICE_FIELDS: tuple[str, ...] = (
    "volume", "pack", "package_type", "sweetener", "flavor", "carbonation",
)

# The final_validation.csv header this builder mirrors (the output adds
# one more column: the `attribute_pairs` state column). Verified against
# the file itself at build time — drift fails loud, never silently forks
# the shape.
FINAL_VALIDATION_COLUMNS: tuple[str, ...] = (
    "gtin1", "gtin2", "gtin1_norm", "gtin2_norm", "true_label",
    "fold", "fold_2", "component_id", "component_id_2", "straddles_fold",
    "endpoint_in_train",
    "v1_volume", "v2_volume", "v1_pack", "v2_pack",
    "v1_package_type", "v2_package_type", "v1_sweetener", "v2_sweetener",
    "v1_flavor", "v2_flavor", "v1_carbonation", "v2_carbonation",
)
STATE_COLUMN = "attribute_pairs"


def _tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _scrub(token: str) -> str:
    return token.strip().lower().replace(" ", "_")


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_side(attribute: str) -> dict[str, set[str]]:
    """Standardized attribute string -> {field: {token, ...}} over the six
    slice fields. Keys are matched lowercase-exact on the catalog's own
    standardized vocabulary (the mapping pin above); a standardized key
    the mapping does not carry is carried forward UNMAPPED, never coerced
    into a wrong slot."""
    fields: dict[str, set[str]] = {}
    for part in str(attribute).split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower()
        if not value.strip():
            continue
        if key == "volume":
            fields.setdefault("volume", set()).update(
                repr(float(token)) if _is_number(token) else _scrub(token)
                for token in _tokens(value))
        elif key == "pack type":
            fields.setdefault("package_type", set()).update(
                _scrub(piece)
                for token in _tokens(value) for piece in token.split("/"))
        elif key in ("sweetener", "flavour", "flavor", "carbonation",
                     "carbonization"):
            field = {"pack type": "package_type", "sweetener": "sweetener",
                     "flavour": "flavor", "flavor": "flavor",
                     "carbonization": "carbonation",
                     "carbonation": "carbonation"}[key]
            fields.setdefault(field, set()).update(
                _scrub(token) for token in _tokens(value))
    return fields


def _render(field: str, tokens: set[str]) -> str:
    """Token set -> the final_validation list-literal shape ('[]' /
    '[a]' / '[a, b]', bare tokens never quoted, sorted)."""
    if not tokens:
        return "[]"
    values = sorted(tokens, key=lambda token: _sort_key(field, token))
    return "[" + ", ".join(values) + "]"


def _sort_key(field: str, token: str):
    try:
        return (0, float(token), "")
    except ValueError:
        return (1, 0.0, token)


def compose_side(attribute: str) -> dict[str, str]:
    """One side's attribute string -> {field: side literal}: the six
    slice fields carry the final_validation list-literal shape; a field
    the side's attribute string does not measure is '' (unmeasured)."""
    fields = parse_side(attribute)
    state: dict[str, str] = {}
    for field in SLICE_FIELDS:
        if field in fields:
            state[field] = _render(field, fields[field])
        else:
            state[field] = ""
    return state


def compose_state(side_one: dict[str, str], side_two: dict[str, str]) -> str:
    """The ONE joined identity state string, in the final_validation
    `attribute_pairs` shape: the six slice fields, v1 then v2 per field."""
    return "; ".join(
        f"{field}: v1={side_one.get(field, '')} v2={side_two.get(field, '')}"
        for field in SLICE_FIELDS
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames)
        return header, list(reader)


def build(pairs_path: Path = PAIRS_PATH, catalog_path: Path = CATALOG_PATH,
          final_validation_path: Path = FINAL_VALIDATION_PATH,
          output: Path = OUTPUT) -> dict:
    """Resolve + compose EVERY ground-truth pair, in pairs-file row order.

    Returns the census dict (rows, label distribution, sha256, missing).
    """
    pairs_header, pairs = _read_csv(pairs_path)
    if pairs_header != ["sku_id1", "sku_id2", "label", "split"]:
        raise RuntimeError(f"listing_pairs header drifted: {pairs_header}")
    catalog_header, catalog = _read_csv(catalog_path)
    required_catalog_keys = ("sku_id", "gtin", "attribute")
    if not set(required_catalog_keys) <= set(catalog_header):
        # additive catalog columns are fine; the three keys the state
        # composition reads may never be missing
        raise RuntimeError(
            f"eligible_catalog is missing required columns "
            f"{[key for key in required_catalog_keys
                if key not in catalog_header]}: {catalog_header}")
    by_sku: dict[str, dict] = {}
    for row in catalog:
        if row["sku_id"] in by_sku:
            raise RuntimeError(
                f"eligible_catalog carries a duplicate sku_id: {row['sku_id']!r}")
        by_sku[row["sku_id"]] = row

    missing = sorted(
        {sku for pair in pairs
         for sku in (pair["sku_id1"], pair["sku_id2"]) if sku not in by_sku})
    if missing:
        raise RuntimeError(
            f"{len(missing)} pair sku_id(s) resolve to no eligible_catalog "
            f"row: {missing[:10]}")
    if any(int(pair["label"]) not in (0, 1) for pair in pairs):
        raise RuntimeError(
            "listing_pairs carries labels outside {0, 1}: "
            f"{sorted({pair['label'] for pair in pairs})}")

    if final_validation_path.is_file():
        fv_header, _ = _read_csv(final_validation_path)
        if fv_header != list(FINAL_VALIDATION_COLUMNS):
            raise RuntimeError(
                "final_validation.csv header drifted from the mirrored "
                f"shape {list(FINAL_VALIDATION_COLUMNS)}: {fv_header}")

    columns = list(FINAL_VALIDATION_COLUMNS) + [STATE_COLUMN]
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for pair in pairs:
        row_one = by_sku[pair["sku_id1"]]
        row_two = by_sku[pair["sku_id2"]]
        side_one = compose_side(row_one["attribute"])
        side_two = compose_side(row_two["attribute"])
        record = {
            "gtin1": row_one["gtin"], "gtin2": row_two["gtin"],
            "gtin1_norm": normalize_gtin(row_one["gtin"]),
            "gtin2_norm": normalize_gtin(row_two["gtin"]),
            "true_label": str(int(pair["label"])),
            # No graph fold/component state exists for the ground-truth
            # pairs: unmeasured and carried EMPTY, never invented.
            "fold": "", "fold_2": "", "component_id": "",
            "component_id_2": "", "straddles_fold": "",
            "endpoint_in_train": "",
            "attribute_pairs": compose_state(side_one, side_two),
        }
        for field in SLICE_FIELDS:
            record[f"v1_{field}"] = side_one[field]
            record[f"v2_{field}"] = side_two[field]
        records.append(record)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    # ── re-measure through the lane's own staging measurement ──────────
    # The emitted file must satisfy the IDENTITY binding exactly as a
    # state stage would measure it (probe, not re-implementation).
    from cli.laya_lane import DECISION_BINDINGS, _measure_csv

    binding = DECISION_BINDINGS["identity"]
    measured = _measure_csv(output, binding["wanted_columns"])
    if binding["state_column"] not in measured["columns"]:
        raise RuntimeError(
            f"{output.name} misses the identity state column "
            f"{binding['state_column']!r}")
    if measured["columns"] not in (
            columns, list(FINAL_VALIDATION_COLUMNS) + [STATE_COLUMN]):
        raise RuntimeError(
            f"{output.name} header drifted: {measured['columns']}")

    labels = Counter(record["true_label"] for record in records)
    census = {
        "rows": len(records),
        "label_distribution": {label: labels[label]
                               for label in sorted(labels)},
        "split_counts": dict(Counter(pair["split"] for pair in pairs)),
        "missing_sku_lookups": len(missing),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output": str(output),
        "columns": columns,
    }
    return census


def shape_proof(final_validation_path: Path = FINAL_VALIDATION_PATH,
                pairs_path: Path = PAIRS_PATH,
                catalog_path: Path = CATALOG_PATH,
                proof_samples: int = 3) -> list[dict]:
    """Quote composed samples vs final_validation rows (shape proof).

    A pair's composed state is proven shape-compatible against a
    final_validation row when the two endpoints' normalized gtins match a
    final_validation pair: the composed `attribute_pairs` value and the
    row's v1_*/v2_* slice columns are printed side by side — same side
    convention (v1 first, v2 second), same six slice fields.
    """
    pairs_header, pairs = _read_csv(pairs_path)
    _, catalog = _read_csv(catalog_path)
    by_sku = {row["sku_id"]: row for row in catalog}
    gtin_of_sku = {sku: row["gtin"] for sku, row in by_sku.items()}
    norm_of_sku = {sku: normalize_gtin(row["gtin"])
                   for sku, row in by_sku.items()}

    samples: list[dict] = []
    if not final_validation_path.is_file():
        return samples

    def fmt(value: object) -> str:
        return str(value) if str(value) != "" else "[]"

    fv_header, fv_rows = _read_csv(final_validation_path)
    by_norm_pair: dict[tuple[str, str], dict] = {}
    for row in fv_rows:
        key = (row["gtin1_norm"], row["gtin2_norm"])
        if key not in by_norm_pair:
            by_norm_pair[key] = row
    for pair in pairs:
        if len(samples) >= proof_samples:
            break
        key = (norm_of_sku[pair["sku_id1"]], norm_of_sku[pair["sku_id2"]])
        fv_row = by_norm_pair.get(key)
        if fv_row is None:
            continue
        side_one = compose_side(by_sku[pair["sku_id1"]]["attribute"])
        side_two = compose_side(by_sku[pair["sku_id2"]]["attribute"])
        samples.append({
            "listing_pair": (pair["sku_id1"], pair["sku_id2"]),
            "true_label": int(pair["label"]),
            "final_validation_true_label": int(fv_row["true_label"]),
            "attribute_pairs": compose_state(side_one, side_two),
            "final_validation_state": {
                field: (fmt(fv_row[f"v1_{field}"]),
                        fmt(fv_row[f"v2_{field}"]))
                for field in SLICE_FIELDS
            },
        })
    print("[laya-metrics-pairs] shape proof (composed attribute_pairs "
          "vs final_validation v1_*/v2_* slice rows):");
    for sample in samples:
        print(json.dumps(sample, indent=2))
    return samples


def main() -> None:
    if not PAIRS_PATH.is_file() or not CATALOG_PATH.is_file():
        raise FileNotFoundError(
            "the ground-truth inputs are missing: "
            f"{PAIRS_PATH} / {CATALOG_PATH}")
    census = build()
    shape_proof()
    import pandas as pd

    frame = pd.read_csv(OUTPUT, dtype=str, keep_default_na=False)
    labels = Counter(frame["true_label"])
    print(
        f"[laya-metrics-pairs] census: rows={len(frame)} "
        f"label_distribution=" + json.dumps(
            {label: labels[label] for label in sorted(labels)}) +
        f" missing_sku_lookups={census['missing_sku_lookups']} "
        f"sha256={census['sha256']} -> {OUTPUT}"
    )


if __name__ == "__main__":
    main()
