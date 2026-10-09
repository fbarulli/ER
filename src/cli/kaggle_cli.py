"""Argument parsing and dispatch for the Kaggle lane."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class KaggleCLI:
    """The shared er-kaggle, module, and backend command surface."""

    @staticmethod
    def run() -> None:
        from cli import kaggle_lane as lane

        parser = argparse.ArgumentParser(description=lane.__doc__)
        parser.add_argument("--what", choices=["package", "upload", "download", "submission",
                            "credentials", "bundle-kernel", "bundle-fetch", "kernel-status",
                            "train-kernel", "embed-kernel", "embed-objective", "finalize-kernel",
                            "kernel-logs", "fetch-results", "stop", "supervise", "autowatch",
                            "kernel-stream", "chain"],
                            default="package")
        parser.add_argument("--dataset-csv", type=Path, default=None,
                            help="cohort export to package (default: the SSOT "
                                 "dataset binding)")
        parser.add_argument("--config-json", type=Path, default=None,
                            help="optional dataset_metadata.json content override")
        parser.add_argument("--submission-input", type=Path, default=None)
        parser.add_argument("--submission-output", type=Path, default=None)
        # One explicit flag flips the lane from local dry-run to the live
        # kaggle subprocess. Default stays dry-run: this box has no kaggle
        # credentials and nothing here may silently reach the network.
        parser.add_argument("--execute", action="store_true",
                            help="actually invoke the kaggle CLI (requires "
                                 "credentials + configured kaggle.slug)")
        parser.add_argument("--no-wait", action="store_true",
                            help="stop: issue the cancel/replace and return "
                                 "immediately (verdict 'requested'); never block "
                                 "in the bounded verify poll")
        parser.add_argument("--kernel", choices=["cpu", "gpu", "embed", "finalize"], default="cpu",
                            help="which configured kernel slug kernel-status "
                                 "resolves (default: cpu)")
        parser.add_argument("--kind", choices=["bundle", "train", "embed", "finalize"], default=None,
                            help="fetch-results/supervise: which kernel output to "
                                 "fetch and verify (default: bundle)")
        parser.add_argument("--slug", default=None,
                            help="kernel-logs: explicit owner/slug (default: "
                                 "resolved from --kernel)")
        parser.add_argument("--follow", action="store_true",
                            help="kernel-logs: poll until a terminal status")
        parser.add_argument("--checkpoint", default=None,
                            help="embed-kernel: git-shipped checkpoint path "
                                 "(default: config kaggle.checkpoint)")
        parser.add_argument("--run-tag", default=None,
                            help="run tag for GPU kernels (default: from "
                                 "config kaggle.run_tag_prefix + UTC stamp)")
        parser.add_argument("--cohort", choices=lane._spec().cohort_tags, default=None,
                            help="bundle-kernel/bundle-fetch: which root-level "
                                 "cohort export the CPU kernel remaps onto "
                                 "dataset.csv (default: full)")
        parser.add_argument("--key-env", default=None,
                            help="environment variable holding the Kaggle API "
                                 "token (default: the configured "
                                 "credentials.keys.kaggle_api_key)")
        parser.add_argument("--revision", default=None,
                            help="pin the bundle kernel to this git revision "
                                 "(default: the current HEAD)")
        parser.add_argument("--with-embed", action="store_true",
                            help="chain: continue into embed-kernel after the "
                                 "train watcher reports completion")
        parser.add_argument("--with-finalize", action="store_true",
                            help="chain: finish with the remote CPU finalize job "
                                 "(model_tracks.bundle_steps role=result) so the "
                                 "sealed result bundle is built on a Kaggle VM, "
                                 "never on the operator box")
        args = parser.parse_args()
        if args.what == "credentials":
            print(json.dumps(lane.write_credentials(key_env=args.key_env, execute=args.execute),
                             indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to write the "
                      "credential file", flush=True)
            return
        cohort_resolved = args.cohort or lane._spec().default_cohort
        if args.what == "chain":
            plan = lane.run_chain(cohort=cohort_resolved, with_embed=args.with_embed,
                             with_finalize=args.with_finalize,
                             execute=args.execute)
            print(json.dumps(plan, indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to run "
                      "the chain end-to-end", flush=True)
            return
        if args.what == "embed-objective":
            verdict = lane.embed_objective(execute=args.execute)
            print(json.dumps(verdict, indent=2), flush=True)
            if verdict.get("available") is False:
                # Explicit absence, never a silent skip: the objective is
                # declared unavailable and the caller learns the fix.
                print(lane._stamp(), f"[kaggle-lane] embed objective UNAVAILABLE: "
                      f"{verdict.get('reason')}", flush=True)
                raise SystemExit(2)
            return
        if args.what == "bundle-kernel":
            receipt = lane.stage_bundle_kernel(revision=args.revision, cohort=cohort_resolved)
            print(lane._stamp(), f"[kaggle-lane] staged bundle kernel ({cohort_resolved}): "
                  f"{json.dumps(receipt, indent=2)}", flush=True)
            if args.execute:
                print(json.dumps(lane.push_bundle_kernel(Path(receipt["staged"])), indent=2),
                      flush=True)
            else:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to push the kernel",
                      flush=True)
            return
        if args.what == "bundle-fetch":
            plan = lane.fetch_kernel_output(kind="bundle", execute=args.execute,
                                       cohort=cohort_resolved if args.cohort else None)
            print(json.dumps(plan, indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to download the "
                      "kernel output", flush=True)
            return
        if args.what == "train-kernel" or args.what == "embed-kernel":
            kind = "train" if args.what == "train-kernel" else "embed"
            receipt = lane.stage_gpu_kernel(
                kind=kind,
                slug=args.slug,
                revision=args.revision,
                run_tag=args.run_tag,
                checkpoint=args.checkpoint,
            )
            print(lane._stamp(), f"[kaggle-lane] staged {kind} kernel: {json.dumps(receipt, indent=2)}",
                  flush=True)
            if args.execute:
                plan = lane.push_kernel(Path(receipt["staged"]))
                print(json.dumps(plan, indent=2), flush=True)
            else:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to push the kernel",
                      flush=True)
            return
        if args.what == "finalize-kernel":
            receipt = lane.stage_finalize_kernel(
                revision=args.revision,
                run_tag=args.run_tag,
                slug=args.slug,
            )
            print(lane._stamp(), f"[kaggle-lane] staged finalize kernel: "
                  f"{json.dumps(receipt, indent=2)}", flush=True)
            if args.execute:
                plan = lane.push_kernel(Path(receipt["staged"]))
                print(json.dumps(plan, indent=2), flush=True)
            else:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to push the kernel",
                      flush=True)
            return
        if args.what == "kernel-logs":
            spec = lane._spec()
            identity = lane.kernel_identity(args.kernel, spec)
            resolved = args.slug or identity.slug(spec)
            print(json.dumps(lane.kernel_logs(slug=resolved, follow=args.follow,
                                         execute=args.execute), indent=2), flush=True)
            return
        if args.what == "fetch-results":
            kind = args.kind or "train"
            print(json.dumps(lane.fetch_kernel_output(kind=kind, execute=args.execute, slug=args.slug),
                             indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to download and "
                      "verify the result archive", flush=True)
            return
        if args.what == "autowatch":
            plan = lane.autowatch_kernel(which=args.kernel, execute=args.execute, slug=args.slug)
            print(json.dumps(plan, indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to "
                      "watch, download and release", flush=True)
            return
        if args.what == "supervise":
            kinds = [args.kind] if args.kind else ["bundle", "train", "embed"]
            plan = lane.supervise_kernels(kinds=kinds, execute=args.execute)
            print(json.dumps(plan, indent=2), flush=True)
            if not args.execute:
                print(lane._stamp(), "[kaggle-lane] dry-run only; pass --execute to poll and fetch",
                      flush=True)
            return
        if args.what == "kernel-stream":
            spec = lane._spec()
            identity = lane.kernel_identity(args.kernel, spec)
            resolved = args.slug or identity.slug(spec)
            print(json.dumps(lane.stream_kernel_logs(resolved), indent=2), flush=True)
            return
        if args.what == "kernel-status":
            print(json.dumps(lane.kernel_status(which=args.kernel, slug=args.slug), indent=2), flush=True)
            return
        if args.what == "stop":
            spec = lane._spec()
            identity = lane.kernel_identity(args.kernel, spec)
            resolved = args.slug or identity.slug(spec)
            print(json.dumps(lane.stop_kernel(slug=resolved, which=args.kernel,
                                         execute=args.execute,
                                         wait=not args.no_wait), indent=2), flush=True)
            return
        spec = lane._spec()
        if args.what == "submission":
            if args.submission_input is None or args.submission_output is None:
                parser.error("--what submission needs --submission-input and --submission-output")
            lane.package_submission(args.submission_input, args.submission_output)
            return
        from core.common import F

        dataset_csv = args.dataset_csv or Path(F["dataset"])
        if args.what == "package":
            config = (json.loads(args.config_json.read_text(encoding="utf-8"))
                      if args.config_json else None)
            package = lane.package_export(dataset_csv, config=config)
            print(
                lane._stamp(),
                f"[kaggle-lane] packaged {package.export_path} "
                f"rows={package.census.rows} sha256={package.census.sha256[:12]} "
                f"-> {package.archive_path}",
                flush=True,
            )
            return
        package = lane.package_export(dataset_csv)
        if args.what == "upload":
            plan = lane.upload_dataset(package, execute=args.execute)
        else:
            plan = lane.download_dataset(package, execute=args.execute)
        print(lane._stamp(), f"[kaggle-lane] {args.what}: {json.dumps(plan, indent=2)}", flush=True)
        if not args.execute:
            print(
                lane._stamp(),
                "[kaggle-lane] dry-run only; pass --execute (with credentials "
                "and kaggle.slug configured) to touch the network",
                flush=True,
            )

