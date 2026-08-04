"""Runtime profile scoping for skills and subprocess env."""

import json
import os
from pathlib import Path

from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache
from agent.runtime_profile_scope import (
    clear_runtime_profile_scope,
    set_runtime_profile_scope,
)
from tools.environments.local import _make_run_env
from tools.skills_tool import skill_view


def _write_skill(root: Path, name: str, description: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nUse {name}.",
        encoding="utf-8",
    )
    return skill_dir


def test_runtime_profile_skill_scope_replaces_user_skills(tmp_path):
    hermes_home = Path(os.environ["HERMES_HOME"])
    _write_skill(hermes_home / "skills", "dba-diagnose-mysql", "DBA diagnosis")
    profile_skill = _write_skill(
        tmp_path / "profiles" / "global-entry" / "skills",
        "kbcloud-platform-skill",
        "Platform catalog",
    )
    clear_skills_system_prompt_cache(clear_snapshot=True)

    tokens = set_runtime_profile_scope(skill_dirs=[profile_skill])
    try:
        prompt = build_skills_system_prompt()
        assert "kbcloud-platform-skill" in prompt
        assert "dba-diagnose-mysql" not in prompt

        visible = json.loads(skill_view("kbcloud-platform-skill", preprocess=False))
        hidden = json.loads(skill_view("dba-diagnose-mysql", preprocess=False))
        assert visible["success"] is True
        assert hidden["success"] is False
    finally:
        clear_runtime_profile_scope(tokens)
        clear_skills_system_prompt_cache(clear_snapshot=True)


def test_runtime_profile_env_blocklist_hides_kubeconfig(monkeypatch):
    monkeypatch.setenv("KUBECONFIG", "/secret/kubeconfig")
    tokens = set_runtime_profile_scope(env_blocklist=["KUBECONFIG"])
    try:
        run_env = _make_run_env({})
    finally:
        clear_runtime_profile_scope(tokens)

    assert "KUBECONFIG" not in run_env
