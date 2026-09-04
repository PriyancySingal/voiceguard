"""
VOICEGUARD - Day 3: live wiring of the version manager.

What changed from Day 1:
- Every time the user finishes speaking, on_user_turn_completed() creates
  a new version. This automatically marks whatever was in flight as
  obsolete -- no separate "cancel" step needed.
- check_inventory() is a deliberately SLOW tool (3s simulated lookup) so
  there's a real window to interrupt it during testing. It captures which
  version was current when it started, and validates that version again
  right before returning a result. If the user has moved on in the
  meantime, it returns a CANCELLED result instead of stock numbers --
  the LLM is instructed never to speak numbers from a cancelled result.

Try the stress scenario live:
  1. python agent.py console
  2. Say: "check item A"          (tool starts, 3s delay)
  3. While it's still checking, say: "actually item B"
  4. Then quickly say: "wait, item A, 50 units"
  5. Watch the console: you should see the state_manager log REJECT the
     first (item A) result and ACCEPT only the final one.
"""

import asyncio
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    RunContext,
    function_tool,
    ChatContext,
    ChatMessage,
)
from livekit.plugins import deepgram, google, rime, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from state_manager import VersionManager, StaleResultError

load_dotenv(dotenv_path=".env.local")

# Fake inventory for the demo -- swap for a real DB/API later if you have time.
FAKE_INVENTORY = {"item a": 20, "item b": 35, "item c": 5}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are VOICEGUARD, a hands-free voice assistant for a "
                "warehouse worker checking inventory and placing orders. "
                "Keep responses short and spoken-friendly: one or two "
                "sentences, no lists, no markdown. Always use the "
                "check_inventory tool when the user asks about stock or "
                "wants to place an order -- never guess a number yourself. "
                "If a tool result says CANCELLED, do not state any number "
                "from it -- just briefly acknowledge and ask what they "
                "need now."
            )
        )
        self.vm = VersionManager()

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        # Fires every time the user finishes speaking, INCLUDING a
        # correction that interrupts something still in flight. This is
        # the single place versions advance and old work becomes obsolete.
        text = new_message.text_content or ""
        self.vm.new_version(text)

    @function_tool()
    async def check_inventory(
        self, context: RunContext, item: str, quantity: int
    ) -> str:
        """Check current stock for an item and whether the requested quantity is available.

        Args:
            item: The item name to check (e.g. "item a").
            quantity: The quantity the user wants.
        """
        # Snapshot which version was current when THIS call started.
        my_version = self.vm.current.id

        # Simulated slow backend lookup. This delay is deliberate -- it's
        # what gives you room to interrupt/correct yourself before it
        # resolves, so you can test the fencing logic live.
        await asyncio.sleep(3.0)

        try:
            self.vm.validate(my_version)
        except StaleResultError:
            # The user moved on before this lookup finished. Never return
            # stock numbers here -- the LLM is instructed not to speak them.
            return (
                "CANCELLED: the user has since changed their request. "
                "Do not state any number from this result. Briefly "
                "acknowledge and ask what they need now."
            )

        stock = FAKE_INVENTORY.get(item.lower().strip(), 0)
        status = "available" if stock >= quantity else "NOT fully available"
        return (
            f"Stock check for {item}: {stock} units in stock. "
            f"Requested {quantity} -- {status}."
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-3.6-flash"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        tts=rime.TTS(model="mistv2", speaker="astra"),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions=(
            "Greet the worker briefly and ask what they need -- an "
            "inventory check, an order, or a status update."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))