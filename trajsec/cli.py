"""Command line for Experiment 2.

    python -m trajsec.cli reproduce [--seed N] [--skip-redteam]
    python -m trajsec.cli scan
    python -m trajsec.cli verify-audit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from canonical import sha256_hex

from agentlab.store import TrajectoryStore as AgentTrajectoryStore

from trajsec.adversary import (
    access_matrix,
    cascading_failure,
    matrix_markdown,
    paired_suite,
    personas,
    idempotence_check,
    plaintext_absent,
    token_scope_property,
)
from trajsec.audit import AuditLog, tamper_suite
from trajsec.config import ARTIFACTS_DIR, Settings, load_settings
from trajsec.corpus import build_corpus
from trajsec.detect import rules_fingerprint, scan_document
from trajsec.evaluate import evaluate
from trajsec.model import ACTIONS, PROFILES
from trajsec.policy import matrix_fingerprint
from trajsec.store import ProtectedStore
from trajsec.utility import utility_report
from trajsec.vault import KeyContext, assert_key_not_in_store, load_or_create_key


def _load_documents(settings: Settings) -> list[dict[str, Any]]:
    store = AgentTrajectoryStore(settings.runs_dir)
    docs = [t.model_dump(mode="json") for t in store.iter_trajectories()]
    if not docs:
        raise SystemExit(
            f"no trajectories under {settings.runs_dir}. Run `python run_experiment.py --e1` first."
        )
    return docs


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# --- commands --------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    settings = load_settings()
    documents = _load_documents(settings)
    totals = {"findings": 0, "active": 0, "suppressed": 0}
    for doc in documents:
        finding_set = scan_document(doc, settings)
        totals["findings"] += len(finding_set.findings)
        totals["active"] += len(finding_set.active)
        totals["suppressed"] += len(finding_set.suppressed)
    print(json.dumps(totals, indent=2))
    return 0


def cmd_verify_audit(args: argparse.Namespace) -> int:
    settings = load_settings()
    log = AuditLog(settings.artifacts_dir / "audit.jsonl", store_salt="trajsec-store-salt")
    ok, bad, reason = log.verify()
    print(json.dumps({"entries": len(log.entries), "chain_ok": ok, "first_bad": bad, "reason": reason}, indent=2))
    return 0 if ok else 1


def cmd_reproduce(args: argparse.Namespace) -> int:
    settings = load_settings()
    artifacts = settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    _banner("Experiment 2 - trajectory security")
    print(f"thresholds : {settings.thresholds_version}")
    print(f"runs dir   : {settings.runs_dir}")
    print(f"artifacts  : {artifacts}")

    documents = _load_documents(settings)
    print(f"trajectories: {len(documents)}")

    key_context = load_or_create_key(artifacts)
    print(f"key custody: {key_context.custody} ({key_context.describe()['key_fingerprint']})")

    # --- corpus and evaluation --------------------------------------------
    _banner("Corpus and evaluation")
    seeds = (args.seed,) if args.seed else settings.seeds
    corpus = build_corpus(documents, settings, seeds=seeds)
    per_seed = {seed: len(entries) for seed, entries in corpus.items()}
    injected = sum(len(e.manifest.items) for e in corpus[seeds[0]])
    print(f"seeds      : {list(corpus)}  ({per_seed})")
    print(f"items/seed : {injected}")

    report = evaluate(
        corpus, settings, original_documents=documents, bootstrap_iterations=args.bootstrap
    )
    _print_headline(report)

    # --- sanitization and utility -----------------------------------------
    _banner("Sanitization and utility")
    from trajsec.sanitize import apply, plan

    store_root = artifacts / "protected"
    if store_root.exists() and args.clean:
        shutil.rmtree(store_root)
    audit = AuditLog(artifacts / "audit.jsonl", store_salt="trajsec-store-salt")

    protected_store = ProtectedStore(store_root, key_context, audit, settings)
    profiles = sorted(PROFILES)
    utility: dict[str, Any] = {}
    sample_run = None

    for doc in documents[: args.sample_runs]:
        try:
            record = protected_store.ingest(doc, owner_principal="owner-1", profile="redacted")
        except Exception as exc:
            print(f"  ingest refused {str(doc.get('run_id'))[:12]}: {exc}")
            continue
        if sample_run is None:
            sample_run = record
        print(
            f"  {record.run_id[:12]} class={record.meta.classification:12s} "
            f"types={len(record.meta.finding_types)}"
        )

    if sample_run is not None:
        for profile_name in ("redacted", "scrubbed"):
            sanitize_plan = plan(
                sample_run.raw, sample_run.finding_set, PROFILES[profile_name], key_context
            )
            from trajsec.vault import Vault

            vault = Vault(store_root / f"vault_{profile_name}.db") if PROFILES[profile_name].vault else None
            sanitized, san_report, replacement_map, produced = apply(
                sample_run.raw, sanitize_plan, key_context,
                policy_version=settings.policy_version, vault=vault,
            )
            if vault is not None:
                vault.close()
            entry = utility_report(
                sample_run.raw, sanitized, san_report, replacement_map, sanitize_plan
            )
            entry["tokens"] = san_report.tokens
            entry["decision_summary"] = sanitize_plan.summary()
            entry["idempotence"] = idempotence_check(
                sample_run.raw, sanitized, settings, produced
            )
            utility[profile_name] = entry
            (artifacts / f"protected_{profile_name}.json").write_text(
                json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  {profile_name:10s} tokens={san_report.tokens:4d} "
                f"footprint={san_report.destruction_footprint:.6f} "
                f"audit_agreement={entry['audit_answer_agreement']['agreement']:.3f} "
                f"path_preservation={entry['path_preservation']['preservation_rate']:.3f} "
                f"idempotent={entry['idempotence']['idempotent']}"
            )

    # --- access control ----------------------------------------------------
    _banner("Access control")
    people = personas()
    matrix_rows: list[dict[str, Any]] = []
    suite_result = None
    if sample_run is not None:
        protected_store.publish(sample_run.run_id)
        matrix_rows = access_matrix(protected_store, sample_run.run_id, people, list(ACTIONS))
        (artifacts / "access_matrix.md").write_text(
            "# Access matrix (generated by calling the policy engine)\n\n"
            + matrix_markdown(matrix_rows)
            + "\n",
            encoding="utf-8",
        )
        matrix_digest = sha256_hex(matrix_fingerprint(matrix_rows))[:16]
        allowed = sum(1 for r in matrix_rows if r["allowed"])
        print(f"matrix     : {len(matrix_rows)} cells, {allowed} allow, {len(matrix_rows) - allowed} deny")
        print(f"fingerprint: {matrix_digest}")

        if not args.skip_redteam:
            suite_result = paired_suite(protected_store, sample_run.run_id, people)
            print(
                f"redteam    : {suite_result.passed}/{len(suite_result.cases)} cases passed"
                f"  unpaired denials: {suite_result.pairs_balanced() or 'none'}"
            )
            for case in suite_result.cases:
                if not case.passed:
                    print(f"    FAIL {case.case_id}: expected {case.expectation}, got {case.outcome}")

    # --- cascade -----------------------------------------------------------
    _banner("Detection failure -> authorization failure")
    flat_corpus = [entry for entries in corpus.values() for entry in entries]
    cascade_full = cascading_failure(flat_corpus, key_context, settings, label="full stack")
    # The mechanism is demonstrated by ablating a layer and watching the
    # cascade appear where it was absent, rather than by asserting it.
    cascade_no_l3 = cascading_failure(
        flat_corpus, key_context, settings, layers=["L4_deny", "L2", "L1", "L4_allow"],
        label="without L3",
    )
    for result in (cascade_full, cascade_no_l3):
        print(
            f"  {result['label']:14s} records={result['records']:3d} "
            f"P(under-classified)={result['p_record_under_classified']:.3f} "
            f"P(exported in clear)={result['p_record_exported_in_clear']:.3f}"
        )
    cascade = {"full_stack": cascade_full, "without_L3": cascade_no_l3}
    for evasion, bucket in sorted(cascade_full["by_evasion"].items()):
        print(
            f"    {evasion:10s} p_missed={bucket['p_missed']:.3f} "
            f"p_exported_in_clear={bucket['p_exported_in_clear']:.3f}"
        )

    # --- tamper suite ------------------------------------------------------
    _banner("Audit log tamper suite")
    seed_entries = audit.entries[: min(8, len(audit.entries))]
    tamper = (
        tamper_suite(seed_entries, artifacts / "tamper", store_salt=audit.store_salt)
        if seed_entries
        else {"operations": [], "detection_rate": 0.0, "detected": 0, "total": 0, "claim": "no entries"}
    )
    for row in tamper["operations"]:
        mark = "detected" if row["detected"] else "NOT detected"
        print(f"  {row['operation']:20s} {mark}")
    print(f"  detection rate: {tamper['detected']}/{tamper['total']} = {tamper['detection_rate']}")

    audit.anchor()
    chain_ok, bad_seq, reason = audit.verify()
    print(f"  audit chain: {'ok' if chain_ok else f'BROKEN at {bad_seq}: {reason}'} ({len(audit.entries)} entries)")

    # --- token scope property ---------------------------------------------
    scope = token_scope_property(
        key_context,
        value="s3cr3t-Pa55w0rd",
        run_a="run-a",
        run_b="run-b",
    )

    # --- key material never persisted -------------------------------------
    artifact_bytes: dict[str, bytes] = {}
    for path in sorted(artifacts.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
            artifact_bytes[str(path.relative_to(artifacts))] = path.read_bytes()
    key_check = {"clean": True}
    try:
        for blob in artifact_bytes.values():
            assert_key_not_in_store(key_context, blob)
    except AssertionError as exc:
        key_check = {"clean": False, "error": str(exc)}

    # --- write the report bundle ------------------------------------------
    _banner("Artifacts")
    bundle = {
        "settings": {
            "thresholds_version": settings.thresholds_version,
            "policy_version": settings.policy_version,
            "rules_fingerprint": rules_fingerprint(),
            "seeds": list(corpus),
        },
        "evaluation": report,
        "utility": utility,
        "access_matrix_fingerprint": sha256_hex(matrix_fingerprint(matrix_rows))[:16],
        "redteam": suite_result.as_dict() if suite_result else None,
        "cascading_failure": cascade,
        "tamper_suite": tamper,
        "token_scope_property": scope,
        "key_custody": key_context.describe(),
        "key_material_absent_from_artifacts": key_check,
        "sandbox_declaration": {
            "all_secrets_synthetic": True,
            "real_credentials_recorded": False,
            "note": "every credential in the corpus and the workspace is synthetic; the live model API key is never recorded, only its fingerprint",
        },
    }
    (artifacts / "eval_report.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    manifest = {
        "schema_version": "1.0",
        "inputs": {
            "trajectories": len(documents),
            "trajectory_digests": {
                str(doc.get("run_id")): sha256_hex(json.dumps(doc, sort_keys=True))[:16]
                for doc in documents
            },
        },
        "rules_fingerprint": rules_fingerprint(),
        "thresholds_version": settings.thresholds_version,
        "policy_version": settings.policy_version,
        "seeds": list(corpus),
        "artifacts": {name: sha256_hex(blob) for name, blob in artifact_bytes.items()},
        "sandbox_declaration": bundle["sandbox_declaration"],
    }
    (artifacts / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  eval_report.json  ({(artifacts / 'eval_report.json').stat().st_size:,} bytes)")
    print(f"  access_matrix.md")
    print(f"  MANIFEST.json     ({len(manifest['artifacts'])} artifact digests)")
    print(f"  key material absent from every artifact: {key_check['clean']}")

    return 0


def _print_headline(report: dict[str, Any]) -> None:
    h = report["headline"]
    print(f"  primary seed            : {report['primary_seed']}")
    print(f"  calibration / test runs : {len(report['splits']['calibration'])} / {len(report['splits']['test'])}")
    print(f"  item_recall_span        : {h['item_recall_span_excluding_expected_misses']}  (CI {h['ci']['item_recall_span'][1:3]})")
    print(f"  item_recall_value       : {h['item_recall_value_excluding_expected_misses']}  (CI {h['ci']['item_recall_value'][1:3]})")
    print(f"  precision               : {h['precision']}  (CI {h['ci']['precision'][1:3]})")
    print(f"  recall (span-level)     : {h['recall']}  (CI {h['ci']['recall'][1:3]})")
    print(f"  expected-miss items     : {h['expected_miss_items']} of {h['items_total']}")
    print()
    print("  ablation:")
    for row in report["ablation"]:
        print(
            f"    {row['config']:18s} span={row['item_recall_span']:.3f} "
            f"val={row['item_recall_value']:.3f} P={row['precision']:.3f} "
            f"R={row['recall']:.3f} F1={row['f1']:.3f} "
            f"dR={row['delta_recall']} dP={row['delta_precision']}"
        )
    natural = report["natural_false_positives"]
    print()
    print(
        f"  natural-text findings   : {natural['findings_on_natural_text']}, "
        f"false positives {natural['adjudicated_false_positive']} "
        f"(rate {natural['fp_rate_per_finding']}, Wilson {natural['fp_rate_wilson_95']})"
    )


# --- entry -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trajsec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    reproduce = sub.add_parser("reproduce", help="run the full Experiment 2 pipeline")
    reproduce.add_argument("--seed", type=int, default=None, help="single corpus seed")
    reproduce.add_argument("--bootstrap", type=int, default=2000, help="bootstrap iterations")
    reproduce.add_argument("--sample-runs", type=int, default=11, help="how many runs to ingest")
    reproduce.add_argument("--skip-redteam", action="store_true")
    reproduce.add_argument("--clean", action="store_true", help="wipe the protected store first")
    reproduce.set_defaults(func=cmd_reproduce)

    sub.add_parser("scan", help="scan stored trajectories and print counts").set_defaults(func=cmd_scan)
    sub.add_parser("verify-audit", help="verify the audit chain").set_defaults(func=cmd_verify_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
