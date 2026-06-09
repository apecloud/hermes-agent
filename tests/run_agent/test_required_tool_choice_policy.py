from types import SimpleNamespace

from agent.conversation_loop import _maybe_apply_required_tool_choice


def _agent(policy="require_until_first_tool"):
    return SimpleNamespace(tool_choice_policy=policy)


def _kwargs():
    return {
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": "检查一下当前集群的整体健康状态"}],
        "tools": [{"type": "function", "function": {"name": "terminal", "parameters": {}}}],
    }


def test_requires_tool_choice_until_first_tool_result():
    api_kwargs = _kwargs()
    _maybe_apply_required_tool_choice(
        _agent(),
        api_kwargs,
        [{"role": "user", "content": "检查一下当前集群的整体健康状态"}],
        0,
    )

    assert api_kwargs["tool_choice"] == "required"


def test_required_tool_choice_policy_releases_after_tool_result():
    api_kwargs = _kwargs()
    _maybe_apply_required_tool_choice(
        _agent(),
        api_kwargs,
        [
            {"role": "user", "content": "检查一下当前集群的整体健康状态"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "kubectl output"},
        ],
        0,
    )

    assert "tool_choice" not in api_kwargs


def test_required_tool_choice_policy_does_not_override_explicit_choice():
    api_kwargs = _kwargs()
    api_kwargs["tool_choice"] = "auto"
    _maybe_apply_required_tool_choice(
        _agent(),
        api_kwargs,
        [{"role": "user", "content": "检查一下当前集群的整体健康状态"}],
        0,
    )

    assert api_kwargs["tool_choice"] == "auto"
