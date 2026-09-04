"""
VOICEGUARD Attack Lab.

Automated, repeatable adversarial testing of the version manager's core
guarantee: a result tagged with an obsolete version must NEVER be
accepted as current.

This tests the logic directly (no voice/audio required), which is what
makes it fast and fully repeatable -- run it as many times as you want,
get the same guarantees every time. This is the "evidence and
reproducibility" artifact for RIME_EVIDENCE.md.

Separately, do a handful of manual live-audio runs (through agent.py
console) to confirm the full pipeline behaves the same way end-to-end --
log those manually in RIME_EVIDENCE.md alongside these automated results,
and say plainly which is which. Don't blur the two.

Run:
    python attack_lab.py
"""

import random
import time
from dataclasses import dataclass

from state_manager import VersionManager, StaleResultError, ToolExecutor


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str


def scenario_baseline_interruption(vm: VersionManager, tools: ToolExecutor) -> ScenarioResult:
    """Single correction: v1's result must be rejected, v2's accepted."""
    v1 = vm.new_version("check item A")
    v2 = vm.new_version("actually item B")
    r1 = tools.run(v1.id, "item A")
    r2 = tools.run(v2.id, "item B")

    v1_rejected = _is_rejected(vm, r1["version_id"])
    v2_accepted = _is_accepted(vm, r2["version_id"])

    passed = v1_rejected and v2_accepted
    return ScenarioResult(
        "baseline_interruption", passed,
        f"v1_rejected={v1_rejected}, v2_accepted={v2_accepted}",
    )


def scenario_double_correction(vm: VersionManager, tools: ToolExecutor) -> ScenarioResult:
    """Two corrections before any tool result returns: only v3 may be accepted."""
    v1 = vm.new_version("check item A, 20 units")
    v2 = vm.new_version("actually item B")
    v3 = vm.new_version("wait, item A, 50 units")

    r1 = tools.run(v1.id, "item A")
    r2 = tools.run(v2.id, "item B")
    r3 = tools.run(v3.id, "item A")

    passed = (
        _is_rejected(vm, r1["version_id"])
        and _is_rejected(vm, r2["version_id"])
        and _is_accepted(vm, r3["version_id"])
    )
    return ScenarioResult("double_correction", passed, "v1 & v2 rejected, v3 accepted")


def scenario_out_of_order_results(vm: VersionManager, tools: ToolExecutor) -> ScenarioResult:
    """Newer version's result arrives before the older one -- order must not matter."""
    v1 = vm.new_version("check item A")
    v2 = vm.new_version("actually item B")

    # Simulate v2's result arriving FIRST, then v1's stale one arriving after.
    r2 = tools.run(v2.id, "item B")
    r1 = tools.run(v1.id, "item A")

    passed = _is_accepted(vm, r2["version_id"]) and _is_rejected(vm, r1["version_id"])
    return ScenarioResult("out_of_order_results", passed, "v2 accepted despite arriving first, v1 still rejected")


def scenario_no_correction_control(vm: VersionManager, tools: ToolExecutor) -> ScenarioResult:
    """Control case: with no interruption, the single result must be accepted."""
    v1 = vm.new_version("check item C")
    r1 = tools.run(v1.id, "item C")
    passed = _is_accepted(vm, r1["version_id"])
    return ScenarioResult("no_correction_control", passed, "single request accepted with no interference")


def scenario_rapid_triple_correction(vm: VersionManager, tools: ToolExecutor) -> ScenarioResult:
    """Three corrections in quick succession -- only the last may survive."""
    versions = [
        vm.new_version("check item A"),
        vm.new_version("item B instead"),
        vm.new_version("no wait, item C"),
        vm.new_version("actually back to item A, 50 units"),
    ]
    results = [tools.run(v.id, "item") for v in versions]

    all_but_last_rejected = all(_is_rejected(vm, r["version_id"]) for r in results[:-1])
    last_accepted = _is_accepted(vm, results[-1]["version_id"])

    passed = all_but_last_rejected and last_accepted
    return ScenarioResult("rapid_triple_correction", passed, "only the final of 4 versions accepted")


def _is_rejected(vm: VersionManager, version_id: int) -> bool:
    try:
        vm.validate(version_id)
        return False
    except StaleResultError:
        return True


def _is_accepted(vm: VersionManager, version_id: int) -> bool:
    try:
        vm.validate(version_id)
        return True
    except StaleResultError:
        return False


SCENARIOS = [
    scenario_baseline_interruption,
    scenario_double_correction,
    scenario_out_of_order_results,
    scenario_no_correction_control,
    scenario_rapid_triple_correction,
]


def run_all(runs_per_scenario: int = 20) -> list[ScenarioResult]:
    all_results: list[ScenarioResult] = []
    for scenario_fn in SCENARIOS:
        passes = 0
        last_detail = ""
        for _ in range(runs_per_scenario):
            vm = VersionManager()
            tools = ToolExecutor()
            # Suppress per-event prints during the batch run for a clean scorecard.
            vm._event = lambda msg: None  # type: ignore[method-assign]
            result = scenario_fn(vm, tools)
            last_detail = result.detail
            if result.passed:
                passes += 1
        all_results.append(
            ScenarioResult(
                name=scenario_fn.__name__.replace("scenario_", ""),
                passed=(passes == runs_per_scenario),
                detail=f"{passes}/{runs_per_scenario} passed -- {last_detail}",
            )
        )
    return all_results


def print_scorecard(results: list[ScenarioResult]) -> None:
    print("=" * 64)
    print("VOICEGUARD ATTACK LAB -- automated logic-level scorecard")
    print("=" * 64)
    total_scenarios = len(results)
    passed_scenarios = sum(1 for r in results if r.passed)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}]  {r.name:<28} {r.detail}")
    print("-" * 64)
    print(f"  {passed_scenarios}/{total_scenarios} scenarios fully passed all runs")
    print("=" * 64)
    print(
        "\nNOTE: this tests the version-manager logic directly, not the live\n"
        "audio pipeline. Log a handful of manual end-to-end voice runs\n"
        "separately in RIME_EVIDENCE.md and label them as such."
    )


if __name__ == "__main__":
    results = run_all(runs_per_scenario=20)
    print_scorecard(results)