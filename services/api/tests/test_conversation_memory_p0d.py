from app.agent.conversation_memory import build_conversation_memory


def _messages(count):
    return [
        {
            "id": f"m-{index}",
            "role": "user" if index % 2 == 0 else "assistant",
            "content": (
                "必须保留检索项目经历作为主线。"
                if index == 0
                else f"conversation message {index}"
            ),
        }
        for index in range(count)
    ]


def test_long_conversation_uses_stable_summary_and_recent_window():
    state, recent = build_conversation_memory(
        _messages(16),
        {},
        ["document:one"],
        [
            {
                "resource_id": "document:one",
                "label": "notes.pdf",
                "kind": "other",
                "readable": True,
                "content_hash": "hash-v1",
            }
        ],
    )
    assert len(recent) == 12
    assert recent[0]["id"] == "m-4"
    assert state["summarized_message_count"] == 4
    assert "必须保留检索项目经历" in state["stable_summary"]
    assert "必须保留检索项目经历作为主线。" in state["decisions"]
    assert state["resource_index"]["document:one"]["status"] == "loaded"

    updated, updated_recent = build_conversation_memory(
        _messages(18),
        state,
        ["document:one", "draft:missing"],
        [
            {
                "resource_id": "document:one",
                "label": "notes.pdf",
                "kind": "other",
                "readable": True,
                "content_hash": "hash-v2",
            }
        ],
    )
    assert len(updated_recent) == 12
    assert updated["summarized_message_count"] == 6
    assert updated["stable_summary"].count("conversation message 4") == 1
    assert updated["resource_index"]["document:one"]["status"] == "changed"
    assert updated["resource_index"]["draft:missing"]["status"] == "missing"
    assert "resource_changed:document:one" in updated["resource_issues"]
    assert "resource_missing:draft:missing" in updated["resource_issues"]


def test_recent_window_has_bounded_content_size():
    messages = [
        {"id": f"m-{index}", "role": "user", "content": "x" * 10_000}
        for index in range(20)
    ]
    state, recent = build_conversation_memory(messages, {}, [], [])
    assert len(recent) <= 12
    assert sum(len(item["content"]) for item in recent) <= 32_000
    assert len(state["stable_summary"]) <= 8_000


def test_latest_user_instruction_resolves_stable_decision_conflict():
    messages = _messages(16)
    messages[-1] = {
        "id": "m-15",
        "role": "user",
        "content": "不要保留检索项目经历作为文书主线。",
    }
    state, _ = build_conversation_memory(messages, {}, [], [])
    assert state["decision_conflicts"] == [
        {
            "previous": "必须保留检索项目经历作为主线。",
            "latest": "不要保留检索项目经历作为文书主线。",
            "resolution": "latest_user_instruction_wins",
        }
    ]
