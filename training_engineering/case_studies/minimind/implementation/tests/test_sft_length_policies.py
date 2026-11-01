import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_sft_length_policies.py"
spec = importlib.util.spec_from_file_location("analyze_sft_length_policies", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    spec.loader.exec_module(analysis)


@pytest.mark.parametrize("limit,retained,lost,state", [
    (3, 1, 4, "assistant_without_eos"),
    (6, 4, 1, "assistant_eos_without_newline"),
    (7, 5, 0, "not_truncated"),
])
def test_cut_distinguishes_missing_eos_and_missing_newline(limit, retained, lost, state):
    ids = [1, 10, 30, 31, 32, 2, 234]
    labels = [-100, -100, 30, 31, 32, 2, 234]
    effect = analysis.truncation_effect(ids, labels, limit, 2)
    assert effect["targets_before"] == 5
    assert effect["targets_retained"] == retained
    assert effect["targets_lost"] == lost
    assert effect["ending_state"] == state


def test_cut_in_context_can_remove_all_supervision_without_partial_assistant():
    ids = [1, 10, 11, 12, 30, 2, 234]
    labels = [-100, -100, -100, -100, 30, 2, 234]
    effect = analysis.truncation_effect(ids, labels, 3, 2)
    assert effect["ending_state"] == "outside_assistant_supervision"
    assert effect["zero_targets_after_cut"] is True
    assert effect["targets_lost"] == 3


def test_cut_can_drop_later_turn_but_keep_a_complete_earlier_reply():
    ids = [1, 30, 2, 234, 10, 11, 31, 2, 234]
    labels = [-100, 30, 2, 234, -100, -100, 31, 2, 234]
    effect = analysis.truncation_effect(ids, labels, 5, 2)
    assert effect["ending_state"] == "outside_assistant_supervision"
    assert effect["targets_retained"] == effect["targets_lost"] == 3
    assert effect["zero_targets_after_cut"] is False


def test_position_zero_is_not_counted_as_a_target():
    effect = analysis.truncation_effect([1, 2, 3], [1, -100, 3], 2, 2)
    assert effect["targets_before"] == effect["targets_lost"] == 1
    assert effect["targets_retained"] == 0


def test_whole_record_loss_ratios_have_distinct_denominators():
    counts = {"dropped_records": 1, "dropped_input_positions": 100,
              "dropped_targets_after_shift": 90}
    result = analysis.finish_exclusion(counts, total_records=10, total_targets=180)
    assert result["retained_records"] == 9
    assert result["dropped_record_fraction"] == 0.1
    assert result["dropped_target_fraction"] == 0.5


def test_invalid_length_or_alignment_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        analysis.truncation_effect([1, 2], [-100, 2], 1, 2)
    with pytest.raises(ValueError, match="align"):
        analysis.truncation_effect([1, 2], [-100], 2, 2)
