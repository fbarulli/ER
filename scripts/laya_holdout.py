"""scripts/laya_holdout.py — the honest laya evaluation holdout.

Assembles a **component-disjoint, difficulty-tagged** holdout from the real
labelled artifacts — never the pipeline-minted negatives (those live in the
posed corpus for *training* only). Sources:

  * data/final_validation.csv  — the P0 population (component_id, true_label,
    endpoint_in_train, straddles_fold): the ONLY fully product-disjoint truth.
  * data/track_setup/listing_pairs.csv — the real cross-retailer pairs; the
    `test` split is carried for a larger same/ different read.
  * data/gate_results.csv — the gate's HARD verdicts (`proceed`, `fallback`):
    label-less, scored for gate-agreement + difficulty attribution only.

Components are the connected components of the positive-pair graph (union-find
over ``training.folds.normalize_gtin``), the same entity key the pipeline uses,
so a product scored here cannot have been trained on. Output is deterministic
(sorted) and paired with a receipt census.

Output: data/laya/holdout.csv + data/laya/holdout.receipt.json.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from training.folds import normalize_gtin

ROOT = Path(__file__).resolve().parent.parent
LISTING_PATH = ROOT / "data/track_setup/listing_pairs.csv"
CATALOG_PATH = ROOT / "data/track_setup/eligible_catalog.csv"
P0_PATH = ROOT / "data/final_validation.csv"
GATE_PATH = ROOT / "data/gate_results.csv"
LABELED_PATH = ROOT / "data/labeled_pairs.csv"
OUTPUT_PATH = ROOT / "data/laya/holdout.csv"
RECEIPT_PATH = ROOT / "data/laya/holdout.receipt.json"

#: The columns every holdout row carries (stable, consumed by the eval).
COLUMNS = ("source", "stratum", "split", "label", "label_source", "component",
           "gtin1", "gtin2", "similarity", "gate_reason", "endpoint_in_train")


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class _UnionFind:
    """Deterministic union-find keyed by canonical GTIN."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:  # path compression
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep the smaller key as the root so ids are stable for a given graph.
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def component_index(listing: list[dict], catalog: list[dict],
                    p0: list[dict], labeled: list[dict]) -> _UnionFind:
    """Union-find over the positive-pair graph (the pipeline's entity key)."""
    uf = _UnionFind()
    sku_gtin = {row["sku_id"]: normalize_gtin(row["gtin"]) for row in catalog}
    for row in listing:
        if str(row.get("label", "")).strip() == "1":
            a = sku_gtin.get(row.get("sku_id1"))
            b = sku_gtin.get(row.get("sku_id2"))
            if a and b:
                uf.union(a, b)
    for row in p0:
        if str(row.get("true_label", "")).strip() in ("1", "true", "True"):
            a, b = normalize_gtin(row.get("gtin1")), normalize_gtin(row.get("gtin2"))
            if a and b:
                uf.union(a, b)
    for row in labeled:
        if str(row.get("true_label", "")).strip() in ("1", "true", "True"):
            a, b = normalize_gtin(row.get("gtin1")), normalize_gtin(row.get("gtin2"))
            if a and b:
                uf.union(a, b)
    return uf


def _component(uf: _UnionFind, gtin: str) -> str:
    key = normalize_gtin(gtin)
    return uf.find(key) if key else ""


def build_holdout(*, listing_path: Path = LISTING_PATH,
                  catalog_path: Path = CATALOG_PATH, p0_path: Path = P0_PATH,
                  gate_path: Path = GATE_PATH,
                  labeled_path: Path = LABELED_PATH) -> tuple[list[dict], dict]:
    """Assemble the component-disjoint, difficulty-tagged holdout rows."""
    listing = _read_csv(listing_path)
    catalog = _read_csv(catalog_path)
    p0 = _read_csv(p0_path)
    gate = _read_csv(gate_path)
    labeled = _read_csv(labeled_path) if Path(labeled_path).is_file() else []

    uf = component_index(listing, catalog, p0, labeled)
    sku_gtin = {row["sku_id"]: normalize_gtin(row["gtin"]) for row in catalog}
    rows: list[dict] = []

    # ── real listing pairs (carry their component-aware split) ─────────────
    for row in listing:
        g1 = sku_gtin.get(row.get("sku_id1"), "")
        g2 = sku_gtin.get(row.get("sku_id2"), "")
        label = str(row.get("label", "")).strip()
        rows.append({
            "source": "listing_pairs", "stratum": "real_listing",
            "split": str(row.get("split", "")).strip(),
            "label": label, "label_source": "listing",
            "component": _component(uf, g1), "gtin1": g1, "gtin2": g2,
            "similarity": "", "gate_reason": "", "endpoint_in_train": "",
        })

    # ── P0: the fully product-disjoint truth ───────────────────────────────
    for row in p0:
        rows.append({
            "source": "final_validation",
            "stratum": ("p0_disjoint"
                        if str(row.get("endpoint_in_train", "")).strip()
                        not in ("True", "true", "1") else "p0_overlap"),
            "split": "p0",
            "label": str(row.get("true_label", "")).strip(),
            "label_source": "final_validation",
            "component": str(row.get("component_id", "")).strip(),
            "gtin1": normalize_gtin(row.get("gtin1")),
            "gtin2": normalize_gtin(row.get("gtin2")),
            "similarity": "", "gate_reason": "",
            "endpoint_in_train": str(row.get("endpoint_in_train", "")).strip(),
        })

    # ── gate HARD verdicts: label-less difficulty strata ───────────────────
    for row in gate:
        decision = str(row.get("gate_decision", "")).strip()
        if decision == "hard_no":
            continue  # the easy, pipeline-verified-different mass: not truth
        if decision not in ("proceed", "fallback"):
            continue
        g1 = normalize_gtin(row.get("gtin1"))
        g2 = normalize_gtin(row.get("gtin2"))
        rows.append({
            "source": "gate_results",
            "stratum": f"gate_{decision}",
            "split": "gate", "label": "", "label_source": "gate_verdict",
            "component": _component(uf, g1),
            "gtin1": g1, "gtin2": g2,
            "similarity": str(row.get("similarity", "")).strip(),
            "gate_reason": str(row.get("gate_reason", "")).strip(),
            "endpoint_in_train": "",
        })

    rows.sort(key=lambda r: (r["source"], r["stratum"], r["component"],
                             r["gtin1"], r["gtin2"]))
    receipt = {
        "rows": len(rows),
        "components": len({r["component"] for r in rows if r["component"]}),
        "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        "by_stratum": dict(sorted(Counter(r["stratum"] for r in rows).items())),
        "labelled_rows": sum(1 for r in rows if r["label"] in ("0", "1")),
        "positives": sum(1 for r in rows if r["label"] == "1"),
        "negatives": sum(1 for r in rows if r["label"] == "0"),
        "note": ("component-disjoint real holdout; gate hard_no excluded (not "
                 "truth); gate proceed/fallback are label-less difficulty tags"),
    }
    return rows, receipt


def write_holdout(rows: list[dict], receipt: dict, *,
                  output: Path = OUTPUT_PATH,
                  receipt_path: Path = RECEIPT_PATH) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n",
                            encoding="utf-8")
    return {"rows": str(output), "receipt": str(receipt_path)}


def main() -> None:
    rows, receipt = build_holdout()
    paths = write_holdout(rows, receipt)
    print(json.dumps({**paths, **receipt}, indent=2), flush=True)


if __name__ == "__main__":
    main()
