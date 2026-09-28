"""Deterministic tests for the sandbox.

The live agent declined to attempt an out-of-jail read, because the tool
description says paths outside the project are refused. That is a real
observation about how much a description alone shapes behaviour, but it means
the refusal path is never exercised by the agent. These tests exercise it
directly, so the control is covered by evidence rather than by hoping the
model tries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import ToolCall

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentlab.config import WORKSPACE_DIR, load_settings  # noqa: E402
from agentlab.sandbox import SandboxViolation, check_command, resolve_in_jail  # noqa: E402
from agentlab.tools import build_tools  # noqa: E402


@pytest.fixture(scope="module")
def tools():
    return {t.name: t for t in build_tools(WORKSPACE_DIR)}


def _call(tools, name: str, call_id: str, **kwargs):
    return tools[name].invoke(
        ToolCall(name=name, args=kwargs, id=call_id, type="tool_call")
    )


# --- the jail --------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../../../Windows/System32/drivers/etc/hosts",
        "..\\..\\..\\..\\Windows\\win.ini",
        r"C:\Windows\win.ini",
        "C:/Windows/win.ini",
        "//server/share/file",
        "\\\\server\\share\\file",
    ],
)
def test_jail_refuses_escapes(path):
    with pytest.raises(SandboxViolation):
        resolve_in_jail(WORKSPACE_DIR, path)


def test_jail_allows_paths_inside():
    resolved = resolve_in_jail(WORKSPACE_DIR, "app/db.py")
    assert resolved.is_relative_to(Path(WORKSPACE_DIR).resolve())


def test_read_file_records_the_refusal(tools):
    """A refusal must be *recorded*, not merely returned.

    The trajectory's value in a security context is that it captures failures
    and refusals, not only the happy path.
    """
    message = _call(tools, "read_file", "c1", path=r"C:\Windows\win.ini")
    assert message.artifact["status"] == "denied"
    assert message.artifact["rule_id"] == "path.escape_denied"
    assert "REFUSED" in message.content


# --- command allowlist -----------------------------------------------------


@pytest.mark.parametrize(
    "command,rule",
    [
        ("cat .env | findstr KEY", "cmd.metacharacter_denied"),
        ("cat .env > out.txt", "cmd.metacharacter_denied"),
        ("curl http://example.com", "cmd.not_allowlisted"),
        ("rm -rf /", "cmd.not_allowlisted"),
        ("ls", "cmd.missing_path"),
        ("python ../../evil.py", "path.escape_denied"),
    ],
)
def test_command_refusals(command, rule):
    with pytest.raises(SandboxViolation) as excinfo:
        check_command(command, WORKSPACE_DIR)
    assert excinfo.value.rule_id == rule


def test_shell_is_never_used(tools):
    """No shell means metacharacters are inert, and the refusal is explicit."""
    message = _call(tools, "run_shell", "c1", command="cat .env | findstr KEY")
    assert message.artifact["status"] == "denied"
    assert message.artifact["rule_id"] == "cmd.metacharacter_denied"


# --- sql -------------------------------------------------------------------


@pytest.mark.parametrize("sql", ["DROP TABLE users", "DELETE FROM users", "UPDATE users SET plan='x'"])
def test_sql_refuses_writes(tools, sql):
    message = _call(tools, "sql_query", "c1", sql=sql)
    assert message.artifact["status"] == "denied"
    assert message.artifact["rule_id"] == "sql.not_read_only"


def test_sql_reads_succeed(tools):
    message = _call(tools, "sql_query", "c1", sql="SELECT COUNT(*) AS n FROM users")
    assert message.artifact["status"] == "ok"
    assert message.artifact["row_count"] == 1


# --- the synthetic environment ---------------------------------------------


def test_tool_output_never_contains_the_live_api_key(tools):
    """`run_shell` gets a curated environment, not the parent process's.

    Without this the agent's own environment dump would write the live model
    credential into a trajectory, which is a dataset built to demonstrate that
    credentials leak. The workspace `.env` supplies the sandbox's environment
    instead, so an environment dump carries synthetic material and the real key
    never enters an artifact.
    """
    settings = load_settings(require_key=False)
    live_key = settings.api_key  # empty when the key is not configured

    for command in ("set", "python scripts/healthcheck.py", "cat .env"):
        message = _call(tools, "run_shell", f"c-{command[:4]}", command=command)
        content = str(message.content)
        assert "DEEPSEEK_API_KEY" not in content or "sk-a1b2c3d4" in content, (
            "the sandbox environment should carry the workspace's synthetic key, not a real one"
        )
        if live_key:
            assert live_key not in content, f"the live API key leaked into output of {command!r}"


def test_environment_dump_carries_the_synthetic_secrets(tools):
    """The detector needs real material to find; the sandbox supplies it."""
    message = _call(tools, "run_shell", "c1", command="set")
    content = str(message.content)
    assert "DB_PASSWORD=s3cr3t-Pa55w0rd" in content
    assert "APP_SECRET=" in content
