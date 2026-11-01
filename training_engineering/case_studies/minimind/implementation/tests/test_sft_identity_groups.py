import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("group_sft_identity_templates", ROOT / "scripts/group_sft_identity_templates.py")
grouping = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(grouping)


def occurrence(line=1, prompt="p", context="c", message="m", cue="identity_question_cue"):
    return {"candidate_line": line, "user_prompt_sha256": prompt, "context_sha256": context,
            "message_sha256": message, "prompt_cue": cue}


def test_same_reply_tracks_different_questions_reasoning_and_full_messages():
    groups = {}
    grouping.add_group(groups, "我是MiniMind。", occurrence(), "reason-a")
    grouping.add_group(groups, "我是MiniMind。", occurrence(2, "other", "other-context", "m2", "no_identity_question_cue"), "reason-b")
    key, value = next(iter(groups.items()))
    summary = grouping.group_summary(key, value)
    assert summary["occurrences"] == summary["record_count"] == 2
    assert summary["nearest_user_prompt_variants"] == summary["counterpart_variants"] == 2
    assert summary["full_message_variants"] == summary["context_variants"] == 2
    assert summary["prompt_cue_occurrences"] == {"identity_question_cue": 1, "no_identity_question_cue": 1}


def test_occurrences_are_not_record_counts_and_whitespace_is_not_normalized():
    groups = {}
    grouping.add_group(groups, "answer", occurrence(), "r")
    grouping.add_group(groups, "answer", occurrence(), "r")
    grouping.add_group(groups, "answer\n", occurrence(2), "r")
    key = grouping.digest_text("answer")
    assert grouping.group_summary(key, groups[key])["occurrences"] == 2
    assert grouping.group_summary(key, groups[key])["record_count"] == 1
    assert len(groups) == 2


def test_missing_null_empty_and_different_reasoning_are_distinct():
    variants = [grouping.reasoning_variant(item) for item in ({}, {"reasoning_content": None},
                {"reasoning_content": ""}, {"reasoning_content": "a"}, {"reasoning_content": "b"})]
    assert len(set(variants)) == 5


def test_user_cue_selects_next_assistant_but_not_after_another_user():
    messages = [{"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]
    assert grouping.assistant_indices(messages, [{"message_index": 0}, {"message_index": 1}]) == [1]
    assert grouping.assistant_indices(messages, [{"message_index": 2}]) == [3]
    assert grouping.assistant_indices([{"role": "user"}, {"role": "user"}, {"role": "assistant"}], [{"message_index": 0}]) == []


def test_empty_fields_are_not_ranked_as_identity_templates():
    groups = {}
    assert grouping.add_group(groups, "", occurrence(), "r") is False
    assert grouping.add_group(groups, None, occurrence(), "r") is False
    assert groups == {}


def test_contrast_examples_are_bounded_and_distinct_by_context():
    groups = {}
    for index in range(8):
        grouping.add_group(groups, "same", occurrence(index, context=str(index)), "r")
    grouping.add_group(groups, "same", occurrence(10, context="0"), "r")
    value = groups[grouping.digest_text("same")]
    assert len(grouping.group_summary(grouping.digest_text("same"), value)["examples"]) == 3
