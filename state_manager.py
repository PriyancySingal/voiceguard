"""
VOICEGUARD core: version manager + state reconciler.

This is the heart of the project. Every user utterance creates a new
"version" of the conversation. Any tool call or piece of generated
speech is tagged with the version that requested it. If the user
speaks again before that work finishes, the version advances and
everything tagged with the OLD version becomes obsolete -- it must
never be spoken or applied to state.

This file has NO dependency on LiveKit, Rime, or any audio pipeline.
That's intentional: the correctness of "stale work gets rejected" is
a pure logic problem, and we want to prove it's right in isolation
before wiring it into anything real-time. Run this file directly to
see a scripted demo of exactly the stress scenario in the pitch doc.
"""

from dataclasses import dataclass, field
from enum import Enum
import itertools
import time


class VersionStatus(Enum):
    CURRENT = "current"
    OBSOLETE = "obsolete"


@dataclass
class ConversationVersion:
    id: int
    intent_text: str
    status: VersionStatus = VersionStatus.CURRENT
    created_at: float = field(default_factory=time.monotonic)


class StaleResultError(Exception):
    """Raised when something tries to speak/apply a result from an obsolete version."""
    pass


class VersionManager:
    """
    Owns the single source of truth for 'what does the user currently want'.

    Usage:
        vm = VersionManager()
        v1 = vm.new_version("check item A")      # -> ConversationVersion(id=1, ...)
        # ... tool call kicks off, tagged with v1.id ...
        v2 = vm.new_version("actually item B")    # v1 is now OBSOLETE automatically
        # when v1's tool result eventually arrives:
        vm.validate(v1.id)                        # raises StaleResultError
        vm.validate(v2.id)                        # OK, v2 is current
    """

    def __init__(self):
        self._counter = itertools.count(start=1)
        self._versions: dict[int, ConversationVersion] = {}
        self._current_id: int | None = None
        self._log: list[str] = []

    @property
    def current(self) -> ConversationVersion | None:
        if self._current_id is None:
            return None
        return self._versions[self._current_id]

    def new_version(self, intent_text: str) -> ConversationVersion:
        """Called every time the user says something new (initial request OR correction)."""
        # Mark the previous version obsolete, if one exists.
        if self._current_id is not None:
            prev = self._versions[self._current_id]
            prev.status = VersionStatus.OBSOLETE
            self._event(f"V{prev.id} ({prev.intent_text!r}) marked OBSOLETE")

        vid = next(self._counter)
        version = ConversationVersion(id=vid, intent_text=intent_text)
        self._versions[vid] = version
        self._current_id = vid
        self._event(f"V{vid} ({intent_text!r}) is now CURRENT")
        return version

    def validate(self, version_id: int) -> ConversationVersion:
        """
        Call this before speaking a result or applying a tool result to state.
        Raises StaleResultError if the version is obsolete -- callers MUST
        catch this and discard the result rather than acting on it.
        """
        version = self._versions[version_id]
        if version.status is VersionStatus.OBSOLETE:
            self._event(
                f"REJECTED: result for V{version_id} ({version.intent_text!r}) "
                f"is stale -- current is V{self._current_id}"
            )
            raise StaleResultError(
                f"Version {version_id} is obsolete; current version is {self._current_id}"
            )
        self._event(f"ACCEPTED: result for V{version_id} is current -- safe to speak/apply")
        return version

    def _event(self, message: str) -> None:
        self._log.append(message)
        print(f"[state]  {message}")

    @property
    def log(self) -> list[str]:
        return list(self._log)


class ToolExecutor:
    """
    Stand-in for a real tool call (e.g. an inventory lookup). In the real
    system this is async and network-bound; here we simulate a delay so we
    can deterministically test out-of-order / stale-result scenarios.
    """

    def run(self, version_id: int, query: str, delay_seconds: float = 0.0):
        # In the real implementation this is an async network call.
        # Here we just simulate the delay synchronously for the demo.
        time.sleep(delay_seconds)
        return {"version_id": version_id, "query": query, "result": f"20 units of {query}"}


def run_demo():
    """
    Reproduces the exact stress scenario from the pitch doc:

        "Check whether we have 20 units of Item A"      -> tool call starts (slow)
        "Actually, make that Item B"                      -> V1 obsolete
        "Wait -- no, I need Item A, but 50 units"          -> V2 obsolete, V3 current
        (V1's slow tool result finally arrives)             -> REJECTED, never spoken
    """
    print("=" * 60)
    print("VOICEGUARD stress scenario")
    print("=" * 60)

    vm = VersionManager()
    tools = ToolExecutor()

    v1 = vm.new_version("check item A, 20 units")
    # Simulate: tool call for v1 kicks off but is slow (e.g. 2s network delay).
    # We don't block on it yet -- in the real system this runs in the background.

    v2 = vm.new_version("actually item B")
    v3 = vm.new_version("wait, item A, 50 units")

    # Now V1's slow tool result finally arrives, long after the user moved on.
    stale_result = tools.run(v1.id, "item A", delay_seconds=0)  # delay=0 for demo speed

    print("-" * 60)
    print("V1's tool result has arrived. Checking whether it's safe to speak...")
    try:
        vm.validate(stale_result["version_id"])
        print(f"WOULD SPEAK: {stale_result['result']}  <-- BUG if we get here")
    except StaleResultError as e:
        print(f"Correctly discarded: {e}")

    # Now simulate V3's own tool result arriving and being spoken.
    current_result = tools.run(v3.id, "item A", delay_seconds=0)
    print("-" * 60)
    print("V3's tool result has arrived (this is the user's real latest request).")
    validated = vm.validate(current_result["version_id"])
    print(f"SAFE TO SPEAK: 50 units of item A (version {validated.id}: {validated.intent_text!r})")

    print("=" * 60)
    print("Full event log:")
    for line in vm.log:
        print(" ", line)


if __name__ == "__main__":
    run_demo()