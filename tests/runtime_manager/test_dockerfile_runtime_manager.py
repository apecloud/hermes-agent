from pathlib import Path


def test_runtime_manager_image_copies_session_state_modules():
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile = repo_root / "Dockerfile.runtime-manager"
    text = dockerfile.read_text()

    state_modules = sorted(path.name for path in repo_root.glob("hermes_state*.py"))
    assert state_modules

    if " hermes_state*.py " in text:
        return

    missing = [module for module in state_modules if f" {module} " not in text]
    assert not missing
