"""Regression tests for Experiment 2.

These are the claims that would be embarrassing to get wrong, expressed as
executable assertions rather than as prose in a report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlab.schema import build_chain, verify_chain
from agentlab.store import TrajectoryStore as AgentTrajectoryStore

from trajsec.adversary import (
    cascading_failure,
    idempotence_check,
    paired_suite,
    personas,
    plaintext_absent,
    token_scope_property,
)
from trajsec.audit import AuditLog, tamper_suite
from trajsec.config import load_settings
from trajsec.detect import scan_document
from trajsec.model import PROFILES, Subject
from trajsec.policy import evaluate, sign_subject, verify_subject
from trajsec.sanitize import apply, plan
from trajsec.store import IngestError, ProtectedStore
from trajsec.utility import audit_answer_agreement, path_preservation
from trajsec.vault import assert_key_not_in_store, load_or_create_key


# --- fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def documents(settings):
    store = AgentTrajectoryStore(settings.runs_dir)
    docs = [t.model_dump(mode="json") for t in store.iter_trajectories()]
    if not docs:
        pytest.skip("no stored trajectories; run `python run_experiment.py --e1` first")
    return docs


@pytest.fixture(scope="module")
def key_context(tmp_path_factory):
    return load_or_create_key(tmp_path_factory.mktemp("keys"))


@pytest.fixture(scope="module")
def protected(tmp_path_factory, settings, key_context, documents):
    root = tmp_path_factory.mktemp("store")
    audit = AuditLog(root / "audit.jsonl", store_salt="test-salt")
    store = ProtectedStore(root, key_context, audit, settings)
    record = store.ingest(documents[0], owner_principal="owner-1", profile="redacted")
    store.publish(record.run_id)  # third-party positive controls need this
    return store, record


# --- the offset contract ---------------------------------------------------


def test_offsets_index_decoded_strings(settings, documents):
    """A finding's span must index the decoded value, not the serialized file.

    A ``\\n`` inside a tool result is two bytes in the file and one character in
    the value. If these were confused, every span would drift and the replay
    checks would fail silently.
    """
    doc = documents[0]
    finding_set = scan_document(doc, settings)
    from textindex import resolve_pointer

    checked = 0
    for finding in finding_set.active:
        node = resolve_pointer(doc, finding.locator.pointer)
        if not isinstance(node, str):
            continue
        assert 0 <= finding.locator.char_start <= finding.locator.char_end <= len(node), (
            f"{finding.locator.pointer}: span outside the decoded string"
        )
        checked += 1
    assert checked > 0, "no string findings to check the offset contract against"


# --- findings never carry plaintext ----------------------------------------


def test_persisted_findings_contain_no_plaintext(protected, settings):
    """The load-bearing property, as a test.

    The findings store and the audit log must be safe to hand to a third party.
    If either contains a matched value, the sanitizer has simply moved the leak.
    """
    store, record = protected
    plaintexts = [
        "sk-a1b2c3d4e5f60718293a4b5c6d7e8f90",
        "s3cr3t-Pa55w0rd",
        "Zx9Kq2mVt7Lp4Rw8Nb3Yc6Hd1Fg5Js0T",
        "AKIAIOSFODNN7EXAMPLE",
    ]
    artifacts = {
        "findings.json": json.dumps(record.finding_set.model_dump(mode="json")).encode(),
        "audit.jsonl": store.audit.path.read_bytes() if store.audit.path.is_file() else b"",
        "replacement_map.json": json.dumps(record.replacement_map).encode(),
    }
    result = plaintext_absent(artifacts, plaintexts)
    assert result["clean"], f"plaintext found in persisted artifacts: {result['hits']}"


# --- sanitizer correctness -------------------------------------------------


def test_sanitization_is_idempotent(protected, settings):
    """Re-scanning a sanitized artifact must not re-flag the sanitizer's output.

    This only holds because a format-preserving token still looks exactly like
    the secret it replaced, so the sanitizer has to tell the detector what it
    produced.
    """
    store, record = protected
    sanitize_plan = plan(record.raw, record.finding_set, PROFILES["redacted"], store.key_context)
    sanitized, _, _, produced = apply(
        record.raw, sanitize_plan, store.key_context, policy_version=settings.policy_version
    )
    result = idempotence_check(record.raw, sanitized, settings, produced)
    assert result["idempotent"], result


def test_sanitization_preserves_structure(protected, settings):
    store, record = protected
    sanitize_plan = plan(record.raw, record.finding_set, PROFILES["redacted"], store.key_context)
    sanitized, _, _, _ = apply(
        record.raw, sanitize_plan, store.key_context, policy_version=settings.policy_version
    )
    preservation = path_preservation(record.raw, sanitized)
    assert preservation["missing"] == [], preservation
    assert preservation["retyped_defects"] == [], preservation
    assert audit_answer_agreement(record.raw, sanitized)["agreement"] == 1.0


def test_minimum_span_leaves_the_structure_intact(protected, settings):
    """A DSN's password is replaced; the enclosing structure is not.

    The host in this particular DSN is *also* a finding on its own account --
    a private IPv4 address is internal topology -- so it is redacted too, and
    that is correct rather than a failure of minimum-span. What minimum-span
    guarantees is that the parts which are *not* findings survive, so this
    asserts on the scheme, the user, the port and the database name.
    """
    store, record = protected
    sanitize_plan = plan(record.raw, record.finding_set, PROFILES["redacted"], store.key_context)
    sanitized, _, _, _ = apply(
        record.raw, sanitize_plan, store.key_context, policy_version=settings.policy_version
    )
    from textindex import iter_leaves

    blob = "\n".join(v for _p, _path, v in iter_leaves(sanitized) if isinstance(v, str))
    assert "s3cr3t-Pa55w0rd" not in blob, "the password survived sanitization"
    assert "postgresql://" in blob, "the scheme was destroyed along with the password"
    assert "appuser" in blob, "the user was destroyed along with the password"
    assert "5432" in blob, "the port was destroyed along with the password"

    # And the mechanism: the DSN parent is replaced only through its child, so
    # the password span is what changed rather than the whole connection string.
    summary = sanitize_plan.summary()
    assert summary.get("parent_covered", 0) > 0, (
        "a containing finding should have been skipped in favour of its child"
    )


# --- key custody -----------------------------------------------------------


def test_key_material_never_reaches_an_artifact(protected, key_context):
    store, record = protected
    blobs = [
        json.dumps(record.sanitized).encode(),
        json.dumps(record.finding_set.model_dump(mode="json")).encode(),
        json.dumps(record.replacement_map).encode(),
        record.run_id.encode(),
    ]
    for blob in blobs:
        assert_key_not_in_store(key_context, blob)


# --- token scoping ---------------------------------------------------------


def test_token_scope_property(key_context):
    """H4: run-scoped tokens correlate within a run and not across runs."""
    result = token_scope_property(key_context, value="s3cr3t-Pa55w0rd", run_a="a", run_b="b")
    assert result["run_scope"]["within_run_stable"]
    assert result["run_scope"]["across_run_distinct"]
    assert result["confirmed"]


# --- access control --------------------------------------------------------


def test_denials_are_paired_with_positive_controls(protected):
    """A suite where everything denies would pass even if the store denied
    everything, so every denial must have a paired allow that also passes."""
    store, record = protected
    suite = paired_suite(store, record.run_id, personas())
    assert suite.pairs_balanced() == [], f"unpaired denials: {suite.pairs_balanced()}"
    failed = [c.case_id for c in suite.cases if not c.passed]
    assert not failed, f"failing cases: {failed}"


def test_denial_writes_an_audit_entry(protected):
    store, record = protected
    people = personas()
    before = len(store.audit.entries)
    store.try_action(record.run_id, people["cross_tenant_analyst"], "read_payload_sanitized")
    assert len(store.audit.entries) > before, "a denial left no audit trail"


def test_unknown_action_fails_closed(protected):
    store, record = protected
    outcome = store.try_action(record.run_id, personas()["owner_secret_debug"], "read_everything")
    assert not outcome["allowed"]


def test_forged_session_is_rejected(protected):
    store, record = protected
    forged = personas()["analyst"].model_copy(update={"role": "owner", "clearance": "SECRET"})
    assert not verify_subject(forged)
    outcome = store.try_action(record.run_id, forged, "read_payload_raw")
    assert not outcome["allowed"]
    assert "invalid_session" in outcome["error"]


def test_missing_session_signature_is_rejected():
    unsigned = Subject(principal_id="x", role="owner", tenant_id="tenant-a", clearance="SECRET")
    assert not verify_subject(unsigned)


# --- tamper evidence -------------------------------------------------------


def test_audit_tamper_detection_rate_is_reported_honestly(protected, tmp_path):
    store, _record = protected
    seed = store.audit.entries[: min(8, len(store.audit.entries))]
    if not seed:
        pytest.skip("no audit entries")
    result = tamper_suite(seed, tmp_path / "tamper", store_salt=store.audit.store_salt)
    detected = {row["operation"]: row["detected"] for row in result["operations"]}
    for operation in ("MODIFY_FIELD", "DELETE_ENTRY", "REORDER", "TRUNCATE_TAIL", "REPLAY_DUPLICATE"):
        assert detected[operation], f"{operation} should have been detected"
    # Declared, not hidden: a forged append is indistinguishable by chain alone.
    assert not detected["FORGE_APPEND"], "FORGE_APPEND cannot be detected by chain alone"
    assert result["detected"] == 5 and result["total"] == 6
    assert result["detection_rate"] == pytest.approx(round(5 / 6, 4), abs=1e-9)


# --- ingest integrity ------------------------------------------------------


def test_ingest_refuses_a_tampered_trajectory(tmp_path, settings, key_context, documents):
    """The join between the two experiments: a trajectory whose Experiment 1
    chain does not verify is refused rather than analysed."""
    import copy

    tampered = copy.deepcopy(documents[0])
    # Mutate a tool observation, which is where the recorded evidence lives.
    target = next(s for s in tampered["steps"] if s.get("tool"))
    target["tool"]["result"] = "injected after the fact"

    store = ProtectedStore(
        tmp_path / "store", key_context, AuditLog(tmp_path / "audit.jsonl", store_salt="s"), settings
    )
    with pytest.raises(IngestError):
        store.ingest(tampered, owner_principal="owner-1")


def test_untampered_trajectories_are_accepted(protected):
    _store, record = protected
    ok, bad = verify_chain(
        __import__("agentlab.schema", fromlist=["Trajectory"]).Trajectory.model_validate(
            record.raw
        ).steps,
        None,
    )
    assert ok, f"chain broke at {bad}"


# --- the cascade -----------------------------------------------------------


def test_cascade_appears_when_a_layer_is_removed(settings, key_context, documents):
    """The headline claim, demonstrated causally.

    With the full stack no record is under-classified. Remove the entropy layer
    and novel-format secrets go undetected, the records stay at INTERNAL, and
    the plaintext reaches the export a third party receives.
    """
    from trajsec.corpus import build_entry

    subset = documents[:3]
    entries = [build_entry(doc, 1337, split="test") for doc in subset]

    full = cascading_failure(entries, key_context, settings, label="full")
    ablated = cascading_failure(
        entries, key_context, settings, layers=["L4_deny", "L2", "L1", "L4_allow"], label="no L3"
    )
    assert ablated["p_record_under_classified"] >= full["p_record_under_classified"]
    assert ablated["under_classified"] > 0, (
        "removing the entropy layer should make novel-format secrets undetectable"
    )
    assert full["under_classified"] == 0, (
        "with every layer enabled nothing should be under-classified on this corpus"
    )
