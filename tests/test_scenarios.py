"""The shipped demo must behave exactly as advertised."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import compare, scripted  # noqa: E402


def test_harness_meets_every_injected_scenario():
    for scenario in scripted.SCENARIOS:
        baseline, outcome, _, _ = compare.run_scenario(scenario)
        assert scenario["check"](outcome["text"]), scenario["id"]
        expected_status = (
            "failed" if scenario["kind"] == "explicit_failure" else None
        )
        if expected_status:
            assert outcome["status"] == expected_status, scenario["id"]
        else:
            assert outcome["status"] in ("success", "recovered"), scenario["id"]


def test_baseline_fails_the_injected_faults_it_should():
    by_id = {s["id"]: s for s in scripted.SCENARIOS}
    _, _, _, _ = compare.run_scenario(by_id["control_success"])  # smoke

    baseline, _, _, _ = compare.run_scenario(by_id["unknown_tool"])
    assert baseline["failure_reason"] == "unhandled_exception: KeyError"

    baseline, _, _, _ = compare.run_scenario(by_id["invalid_args"])
    assert baseline["failure_reason"] == "unhandled_exception: TypeError"

    baseline, _, _, _ = compare.run_scenario(by_id["empty_then_answer"])
    assert baseline["failure_reason"] == "blank_output"

    baseline, _, _, _ = compare.run_scenario(by_id["numeric_strings"])
    assert baseline["status"] == "completed"
    assert "15" not in baseline["text"]  # answered 78: silently wrong


def test_repeated_read_is_cached_but_identical_writes_are_not():
    by_id = {s["id"]: s for s in scripted.SCENARIOS}
    _, _, bcount, hcount = compare.run_scenario(by_id["repeated_read"])
    assert bcount["get_weather"] == 2 and hcount["get_weather"] == 1

    _, _, bcount, hcount = compare.run_scenario(by_id["write_twice"])
    assert bcount["save_note"] == 2 and hcount["save_note"] == 2


def test_compare_demo_exits_zero_and_prints_the_honest_summary(capsys):
    assert compare.main([]) == 0
    printed = capsys.readouterr().out
    assert "injected" in printed.lower()
    assert "expected outcome met" in printed
    assert "9/9" in printed
