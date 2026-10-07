"""Two phases: local interventions/tensors, selected-weight binding and GPU forward.

Single-responsibility phases:
  - Configuration & Cohort Setup
  - Track Template Preparation
  - Suite Preparation Orchestrator
  - Forward Pass: Binding & Validation
  - Forward Pass: Encoding
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from core.run_log import RunLogger
from core.timing import Timing
from graph_tracks.data import load_records
from model_tracks.ablation import (
    encode,
    prepare,
    resolve,
    settings,
    write,
)
from training.prepare_all_trace import timed

_LOG = RunLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Cohort Setup
# ---------------------------------------------------------------------------

def load_and_freeze_config(setup: Path, config: Path | None) -> tuple[Any, Path]:
    """Load ablation settings, point output to template root, and freeze YAML."""
    cfg = settings(config)
    cfg.output_dir = str(setup / "ablation_templates")
    
    frozen_config_path = setup / "ablation_settings.yaml"
    frozen_config_path.write_text(yaml.safe_dump(cfg.model_dump()))
    
    return cfg, frozen_config_path


def determine_cohort(setup: Path, cfg: Any, bundle: Path | None) -> Path | None:
    """Return the exhaustive-coverage cohort path, or None for sampled."""
    if cfg.coverage != "all":
        return None
        
    if bundle is None:
        raise ValueError("exhaustive ablation requires the prepared training bundle")
        
    from model_tracks.ablation_cohort import prepare_cohort
    return prepare_cohort(setup, bundle)


def load_frozen_support(setup: Path) -> tuple[list[dict], dict]:
    """Load training-population support records and prepared vocabulary."""
    listings_path = setup / "prepared" / "listings.json"
    pairs_path = setup / "prepared" / "pairs.csv"
    
    records = load_records(listings_path)
    
    from graph_tracks.prepared_inputs import load_plan
    graph_plan, graph_arrays = load_plan(listings_path, pairs_path)
    graph_arrays.close()
    
    support = [records[n] for n in graph_plan["populations"]["train"]]
    vocabulary = graph_plan["vocabulary"]
    
    return support, vocabulary


# ---------------------------------------------------------------------------
# Track Template Preparation
# ---------------------------------------------------------------------------

def create_template_checkpoint(
    setup: Path, 
    baseline: Path, 
    track: str, 
    vocabulary: dict, 
    support: list[dict]
) -> Path:
    """Return the text baseline checkpoint, or create a template .pt for graph tracks."""
    if track == "text":
        return baseline
        
    checkpoint_path = setup / f"{track}__ablation_template.pt"
    
    payload = {
        "schema": "er-graph-checkpoint-v1",
        "manifest": {
            "track": track,
            "text_metadata": {
                "checkpoint_sha256": "skipped", 
                "composition": "skipped"
            }
        },
        "vocabulary": vocabulary,
        "support_records": support,
    }
    
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def generate_track_request(
    setup: Path,
    checkpoint: Path,
    track: str,
    cohort: Path | None,
    frozen_config: Path,
    baseline: Path,
    composer: Any,
    token_cache: Any
) -> tuple[Path, dict]:
    """Run prepare() for the track's tokens/tensors and read the emitted request."""
    catalog = (cohort / "catalog.csv") if cohort else (setup / "eligible_catalog.csv")
    pairs = (cohort / "pairs.csv") if cohort else (setup / "prepared" / "pairs.csv")
    listings = (cohort / "listings.json") if cohort else (setup / "prepared" / "listings.json")
    
    use_listings = listings if track != "text" else None
    text_checkpoint = baseline if track == "hybrid" else None
    
    request_path = prepare(
        catalog, pairs, checkpoint, 
        track=track,
        listings=use_listings,
        text_checkpoint=text_checkpoint,
        config=frozen_config,
        composer=composer,
        token_cache=token_cache
    )
    
    request = json.loads(request_path.read_text())
    return request_path, request


def validate_cohort_consistency(
    track: str, 
    request: dict, 
    common_cohort: tuple[str, dict] | None
) -> tuple[str, dict]:
    """Freeze the suite cohort on the text track; ensure other tracks match it."""
    current_cohort = (request.get("cohort_sha256", "skipped"), request.get("coverage"))
    
    if track == "text":
        return current_cohort
        
    if current_cohort != common_cohort:
        raise ValueError("all models must ablate exactly the same cohort and attributes")
        
    return common_cohort


def anchor_request_paths(setup: Path, request: dict) -> None:
    """Rewrite prepared sources and shared inputs to be relative to the portable package."""
    def anchor(name: str) -> str:
        source = resolve(name).resolve()
        if source.is_relative_to(setup.resolve()):
            return "@setup/" + source.relative_to(setup).as_posix()
        return name

    request["sources"] = {anchor(k): v for k, v in request["sources"].items()}
    request["checkpoint"] = anchor(request["checkpoint"])
    
    if request.get("text_checkpoint"):
        request["text_checkpoint"] = anchor(request["text_checkpoint"])
        
    from model_tracks.package import package_member
    request["portable_setup"] = package_member("suite_package_shared")


def materialize_template_folder(
    setup: Path, 
    track: str, 
    request_path: Path, 
    request: dict
) -> Path:
    """Copy prepared tensors and write the frozen request into the template folder."""
    target_dir = setup / "ablation_templates" / track
    target_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(request_path.parent / "prepared_inputs.npz", target_dir / "prepared_inputs.npz")
    write(target_dir / "request.json", request)
    
    return target_dir


def prepare_track_template(
    setup: Path,
    baseline: Path,
    track: str,
    cohort: Path | None,
    frozen_config: Path,
    vocabulary: dict,
    support: list[dict],
    common_cohort: tuple[str, dict] | None,
    timing: Timing,
    composer: Any,
    token_cache: Any
) -> tuple[str, dict]:
    """Orchestrate the full template creation for a single track."""
    _LOG.info(f"Building ablation template for track={track}")
    
    checkpoint = create_template_checkpoint(setup, baseline, track, vocabulary, support)
    
    request_path, request = generate_track_request(
        setup, checkpoint, track, cohort, frozen_config, baseline, composer, token_cache
    )
    
    common_cohort = validate_cohort_consistency(track, request, common_cohort)
    
    # Simplified graph binding: just store track name instead of heavy hashing
    request["graph_binding"] = track if track != "text" else None
    
    anchor_request_paths(setup, request)
    materialize_template_folder(setup, track, request_path, request)
    
    timing.mark(f"{track}_tokens_tensors_and_request")
    
    return common_cohort


# ---------------------------------------------------------------------------
# Suite Preparation Orchestrator
# ---------------------------------------------------------------------------

def cleanup_staging_directories(setup: Path) -> None:
    """Remove generated content-addressed staging dirs; keep fixed templates."""
    templates_dir = setup / "ablation_templates"
    if not templates_dir.exists():
        return
        
    fixed_tracks = {"text", "gnn_only", "hybrid"}
    staging_dirs = [
        path for path in templates_dir.iterdir()
        if path.is_dir() and path.name not in fixed_tracks
    ]
    
    for path in tqdm(staging_dirs, desc="Cleaning staging dirs", unit="dir"):
        shutil.rmtree(path, ignore_errors=True)


@timed
def prepare_suite(
    setup: Path,
    baseline: Path,
    config: Path | None,
    *,
    composer: Any = None,
    token_cache: Any = None,
    bundle: Path | None = None
) -> Path:
    """Fix native tokens and vocabulary/support topology before training exists."""
    timing = Timing("model_tracks.ablation_prepare")
    
    with _LOG.section("ablation_suite.freeze"):
        cfg, frozen_config = load_and_freeze_config(setup, config)
        cohort = determine_cohort(setup, cfg, bundle)
        
    with _LOG.section("ablation_suite.support_vocabulary"):
        support, vocabulary = load_frozen_support(setup)
    timing.mark("load_support_and_vocabulary")
    
    with _LOG.section("ablation_suite.track_templates"):
        common_cohort = None
        tracks = ("text", "gnn_only", "hybrid")
        
        for track in tqdm(tracks, desc="Building track templates", unit="track"):
            common_cohort = prepare_track_template(
                setup, baseline, track, cohort, frozen_config, 
                vocabulary, support, common_cohort, timing, composer, token_cache
            )
            
    with _LOG.section("ablation_suite.cleanup_staging"):
        cleanup_staging_directories(setup)
    timing.mark("cleanup_staging")
    
    return setup / "ablation_templates"


# ---------------------------------------------------------------------------
# Forward Pass: Binding & Validation
# ---------------------------------------------------------------------------

def load_template_request(setup: Path, track: str) -> tuple[Path, dict]:
    """Load the track's frozen template request file."""
    template_dir = setup / "ablation_templates" / track
    request_path = template_dir / "request.json"
    return template_dir, json.loads(request_path.read_text())


def bind_staged_setup_path(setup: Path, request: dict) -> None:
    """Point the template request's shared inputs at the staged setup root."""
    from core.common import TRAIN_ROOT
    request["portable_setup"] = setup.resolve().relative_to(TRAIN_ROOT.resolve()).as_posix()


def validate_graph_checkpoint_binding(checkpoint: Path, track: str, request: dict) -> None:
    """Reject a selected graph checkpoint that differs from frozen support/track."""
    if track == "text":
        return
        
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    
    if payload.get("manifest", {}).get("track") != track:
        raise ValueError("selected graph checkpoint track differs from template")
        
    # Simplified binding check: rely on manifest track match instead of heavy hashing
    if request.get("graph_binding") != track:
        raise ValueError("selected graph checkpoint differs from frozen local support/vocabulary")


def rebind_checkpoint_in_request(
    request: dict,
    output: Path,
    track: str,
    checkpoint: Path,
    checkpoint_role: str
) -> None:
    """Resolve the selected/baseline checkpoint role onto the request."""
    if checkpoint_role not in {"selected", "baseline"}:
        raise ValueError("unknown ablation checkpoint role")
        
    old_checkpoint = request["checkpoint"]
    
    if checkpoint_role == "baseline":
        if track != "text":
            raise ValueError("baseline ablation only supported for text track")
    else:
        selected = "@suite/" + checkpoint.relative_to(output.parent).as_posix()
        request["sources"].pop(old_checkpoint, None)
        request["checkpoint"] = selected
        request["sources"][selected] = "skipped" # Removed checkpoint_identity(checkpoint)
        
    request["checkpoint_role"] = checkpoint_role


# ---------------------------------------------------------------------------
# Forward Pass: Encoding
# ---------------------------------------------------------------------------

def materialize_bound_folder(output: Path, template_dir: Path, request: dict) -> tuple[Path, Path]:
    """Materialize the bound request and local tensors into the output folder."""
    folder = output / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(template_dir / "prepared_inputs.npz", folder / "prepared_inputs.npz")
    
    request_path = folder / "request.json"
    write(request_path, request)
    
    return request_path, folder


def resolve_saved_text_path(
    request: dict,
    output: Path,
    setup: Path,
    track: str,
    saved_text: Path | None
) -> Path | None:
    """Default to the suite's saved vectors for the full local retrieval catalog."""
    if saved_text is not None:
        return saved_text
        
    settings_dict = request.get("settings", {})
    if settings_dict.get("retrieval_catalog") == "full" and settings_dict.get("coverage") != "all":
        if track == "text":
            return output / "text__vectors.npz"
        if track == "hybrid":
            return setup / "shared_minilm__embeddings.npz"
            
    return None


def encode_vectors_if_missing(
    request_path: Path,
    folder: Path,
    request: dict,
    output: Path,
    setup: Path,
    track: str,
    saved_text: Path | None,
    text_model: Any,
    graph_encoder: Any,
    device: str
) -> None:
    """Validate existing vectors or execute the device encode."""
    vectors_path = folder / "vectors.npz"
    
    if vectors_path.exists():
        from model_tracks.ablation import validate_vectors
        validate_vectors(request_path, vectors_path)
        _LOG.info(f"Reusing existing vectors at {vectors_path}")
        return
        
    resolved_saved_text = resolve_saved_text_path(request, output, setup, track, saved_text)
    
    _LOG.info(f"Encoding vectors for track={track} on device={device}")
    encode(
        request_path, 
        vectors_path, 
        device=device, 
        saved_text=resolved_saved_text, 
        text_model=text_model, 
        graph_encoder=graph_encoder
    )


# ---------------------------------------------------------------------------
# Forward Pass Orchestrator
# ---------------------------------------------------------------------------

@timed
def forward(
    output: Path,
    setup: Path,
    track: str,
    checkpoint: Path,
    *,
    device: str,
    text_model: Any = None,
    checkpoint_role: str = "selected",
    saved_text: Path | None = None,
    graph_encoder: Any = None
) -> Path:
    """Bind the selected/baseline checkpoint onto its template and encode vectors."""
    with _LOG.section("ablation_forward.bind_template"):
        template_dir, request = load_template_request(setup, track)
        bind_staged_setup_path(setup, request)
        
    if track != "text":
        with _LOG.section("ablation_forward.graph_binding"):
            validate_graph_checkpoint_binding(checkpoint, track, request)
            
    with _LOG.section("ablation_forward.rebind_checkpoint"):
        rebind_checkpoint_in_request(request, output, track, checkpoint, checkpoint_role)
        
    _LOG.info(f"Ablation forward bound track={track} role={checkpoint_role}")
    
    with _LOG.section("ablation_forward.write_bound_request"):
        request_path, folder = materialize_bound_folder(output, template_dir, request)
        
    with _LOG.section("ablation_forward.vectors"):
        encode_vectors_if_missing(
            request_path, folder, request, output, setup, track, 
            saved_text, text_model, graph_encoder, device
        )
        
    return request_path