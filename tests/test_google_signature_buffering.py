"""P2: Gemini thought-signature buffering, empty-text bookmark fix, F1.

Covers the fork-side hardening for Gemini 3 thought signatures:

- A trailing empty-text signature chunk on a function-call-only response must
  anchor to the first function call — an empty text bookmark can never match
  in the adapter and previously dropped the signature (next request 400).
- Signature frames are buffered during streaming and flushed only on commit
  (function-call execution or normal completion), never mid-stream.
- Interruption/cancellation discards buffered signatures together with any
  unexecuted deferred function calls, so no stale signature can reference a
  call that never ran and no deferred tool fires after a barge-in (F1).
- Provider/API errors during streaming are logged at ERROR with a traceback.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock

from google.genai.types import (
    Candidate,
    Content,
    FunctionCall,
    GenerateContentResponse,
    Part,
)
from loguru import logger

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
)
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.llm import GoogleLLMService


def _chunk(*parts):
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=list(parts)))]
    )


def _fc(fc_id="fc-1", name="send_sms"):
    return Part(function_call=FunctionCall(id=fc_id, name=name, args={}))


def _sig(signature=b"sig"):
    # Gemini 3's trailing empty-text signature chunk.
    return Part(text="", thought_signature=signature)


class _Harness:
    """GoogleLLMService with the network stream and frame plumbing mocked."""

    def __init__(self, chunks):
        self.service = GoogleLLMService(api_key="test-key", model="gemini-3.7-flash")
        self.pushed = []  # (frame, direction) in push order
        self.run_fc_calls = 0
        self.order = []  # interleaved record of pushes and fc executions

        async def fake_stream(context):
            async def agen():
                for c in chunks:
                    yield c

            return agen()

        async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
            self.pushed.append((frame, direction))
            self.order.append(("push", type(frame).__name__))

        async def fake_run_fc(function_calls):
            self.run_fc_calls += 1
            self.order.append(("run_function_calls", len(function_calls)))

        self.service._stream_content_universal_context = fake_stream
        self.service.push_frame = fake_push
        self.service.run_function_calls = fake_run_fc

    def signature_frames(self):
        return [f for f, _ in self.pushed if isinstance(f, LLMMessagesAppendFrame)]

    def bookmarks(self):
        return [f.messages[0].message["bookmark"] for f in self.signature_frames()]

    async def run(self):
        await self.service._process_context(LLMContext())


class TestEmptyTextBookmark(unittest.TestCase):
    def test_fc_only_response_anchors_trailing_signature_to_function_call(self):
        h = _Harness([_chunk(_fc("fc-1")), _chunk(_sig(b"sig-1"))])
        asyncio.run(h.run())

        assert h.bookmarks() == [{"function_call": "fc-1"}]

    def test_text_response_keeps_text_bookmark(self):
        h = _Harness(
            [
                _chunk(Part(text="Sure, "), Part(text="one moment")),
                _chunk(_sig(b"sig-2")),
            ]
        )
        asyncio.run(h.run())

        assert h.bookmarks() == [{"text": "Sure, one moment"}]


class TestSignatureCommitPoints(unittest.TestCase):
    def test_immediate_fc_path_flushes_before_execution(self):
        h = _Harness([_chunk(_fc("fc-1")), _chunk(_sig())])
        asyncio.run(h.run())

        assert h.run_fc_calls == 1
        flush_idx = h.order.index(("push", "LLMMessagesAppendFrame"))
        exec_idx = h.order.index(("run_function_calls", 1))
        assert flush_idx < exec_idx

    def test_pure_text_response_flushes_before_end_frame(self):
        h = _Harness([_chunk(Part(text="Hello"), _sig())])
        asyncio.run(h.run())

        assert h.run_fc_calls == 0
        names = [name for _, name in h.order]
        assert names.index("LLMMessagesAppendFrame") < names.index(
            "LLMFullResponseEndFrame"
        )

    def test_deferred_signatures_flush_only_on_bot_stopped_speaking(self):
        h = _Harness(
            [
                _chunk(Part(text="Sending it now.")),
                _chunk(_fc("fc-9")),
                _chunk(_sig()),
            ]
        )
        asyncio.run(h.run())

        # Nothing flushed yet; the fc is deferred until TTS finishes.
        assert h.signature_frames() == []
        assert len(h.service._pending_function_calls) == 1
        assert len(h.service._pending_signature_frames) == 1

        asyncio.run(
            h.service.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
        )

        assert h.run_fc_calls == 1
        assert h.signature_frames() != []
        flush_idx = h.order.index(("push", "LLMMessagesAppendFrame"))
        exec_idx = h.order.index(("run_function_calls", 1))
        assert flush_idx < exec_idx
        assert h.service._pending_function_calls == []


class TestSignatureDiscard(unittest.TestCase):
    def _deferred_harness(self):
        h = _Harness(
            [
                _chunk(Part(text="Sending it now.")),
                _chunk(_fc("fc-9")),
                _chunk(_sig()),
            ]
        )
        asyncio.run(h.run())
        return h

    def test_interruption_discards_signatures_and_deferred_calls(self):
        h = self._deferred_harness()

        asyncio.run(
            h.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        )

        assert h.service._pending_function_calls == []
        assert h.service._pending_signature_frames == []
        assert h.signature_frames() == []
        assert h.run_fc_calls == 0

    def test_stream_error_discards_signatures_and_logs_traceback(self):
        service = GoogleLLMService(api_key="test-key", model="gemini-3.7-flash")
        pushed = []

        async def failing_stream(context):
            async def agen():
                yield _chunk(Part(text="Hi", thought_signature=b"sig-4"))
                raise ValueError("provider exploded")

            return agen()

        async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        service._stream_content_universal_context = failing_stream
        service.push_frame = fake_push

        records = []
        sink_id = logger.add(lambda m: records.append(str(m)), level="ERROR")
        try:
            asyncio.run(service._process_context(LLMContext()))
        finally:
            logger.remove(sink_id)

        assert service._pending_signature_frames == []
        assert not [f for f in pushed if isinstance(f, LLMMessagesAppendFrame)]
        assert any("error during Gemini streaming" in r for r in records)
        # The traceback must be part of the log record, not just the message.
        assert any("ValueError" in r and "provider exploded" in r for r in records)

    def test_new_context_discards_stale_buffer(self):
        h = _Harness([_chunk(Part(text="Fresh"), _sig(b"sig-new"))])
        # Simulate a leftover from an aborted previous response.
        h.service._pending_signature_frames.append(
            LLMMessagesAppendFrame(["stale"])
        )

        asyncio.run(h.run())

        # Only the new response's signature is flushed; the stale one is gone.
        assert h.bookmarks() == [{"text": "Fresh"}]

    def test_cancel_frame_clears_pending_and_closes_client(self):
        h = self._deferred_harness()
        h.service._close_client = AsyncMock()

        asyncio.run(h.service.cancel(CancelFrame()))

        assert h.service._pending_function_calls == []
        assert h.service._pending_signature_frames == []
        h.service._close_client.assert_awaited_once()

    def test_mid_stream_interruption_poisons_response(self):
        """A caller barging in WHILE the model is still streaming: the
        half-generated response never commits signatures and its function
        calls are dropped when the stream settles."""
        service = GoogleLLMService(api_key="test-key", model="gemini-3.7-flash")
        pushed = []
        run_fc_calls = 0
        first_chunk_seen = asyncio.Event()
        resume_stream = asyncio.Event()

        async def fake_stream(context):
            async def agen():
                yield _chunk(Part(text="Wait, "), _sig(b"early"))
                first_chunk_seen.set()
                await resume_stream.wait()
                yield _chunk(_fc("fc-late"))
                yield _chunk(_sig(b"late"))

            return agen()

        async def fake_push(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        async def fake_run_fc(function_calls):
            nonlocal run_fc_calls
            run_fc_calls += 1

        service._stream_content_universal_context = fake_stream
        service.push_frame = fake_push
        service.run_function_calls = fake_run_fc

        async def drive():
            task = asyncio.create_task(service._process_context(LLMContext()))
            await asyncio.wait_for(first_chunk_seen.wait(), timeout=1)
            # One signature is buffered, then the caller barges in mid-stream.
            assert len(service._pending_signature_frames) == 1
            await service.process_frame(
                InterruptionFrame(), FrameDirection.DOWNSTREAM
            )
            resume_stream.set()
            await task

        asyncio.run(drive())

        assert service._pending_signature_frames == []
        assert service._pending_function_calls == []
        assert run_fc_calls == 0
        assert not [f for f in pushed if isinstance(f, LLMMessagesAppendFrame)]
        assert any(isinstance(f, LLMFullResponseEndFrame) for f in pushed)
        assert service._in_stream is False
        assert service._response_interrupted is True

    def test_deferred_non_cancellable_transition_survives_interruption(self):
        """Edge transitions register with cancel_on_interruption=False: an
        interruption in the deferred window must NOT drop them (or their
        signatures) — they still execute when TTS stops."""
        h = _Harness(
            [
                _chunk(Part(text="Thanks, goodbye.")),
                _chunk(_fc("fc-end", name="end_call")),
                _chunk(_sig()),
            ]
        )

        async def end_call_handler(params):
            pass

        h.service.register_function(
            "end_call", end_call_handler, cancel_on_interruption=False
        )
        asyncio.run(h.run())

        asyncio.run(
            h.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        )

        # The transition and its signature survive the interruption.
        assert len(h.service._pending_function_calls) == 1
        assert len(h.service._pending_signature_frames) == 1
        assert h.run_fc_calls == 0

        asyncio.run(
            h.service.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
        )

        assert h.run_fc_calls == 1
        assert h.signature_frames() != []

    def test_mixed_deferred_calls_only_cancellable_dropped(self):
        h = _Harness(
            [
                _chunk(Part(text="Sending it, then done.")),
                _chunk(_fc("fc-sms", name="send_sms")),
                _chunk(_fc("fc-end", name="end_call")),
                _chunk(_sig()),
            ]
        )

        async def end_call_handler(params):
            pass

        h.service.register_function(
            "end_call", end_call_handler, cancel_on_interruption=False
        )
        asyncio.run(h.run())
        assert len(h.service._pending_function_calls) == 2

        asyncio.run(
            h.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        )

        # send_sms (default cancellable) is dropped; end_call survives, and
        # the buffer is kept because a survivor still needs its signature.
        survivors = h.service._pending_function_calls
        assert [fc.function_name for fc in survivors] == ["end_call"]
        assert len(h.service._pending_signature_frames) == 1


class TestMixedBatchSignatureOwnership(unittest.TestCase):
    """Codex P2 round 2: when a dropped cancellable call owns a signature,
    keeping the surviving calls would commit them unsigned (Gemini 3 wants
    the signature on the first FC part) and the next request would 400."""

    def _register_end_call(self, service):
        async def end_call_handler(params):
            pass

        service.register_function(
            "end_call", end_call_handler, cancel_on_interruption=False
        )

    def _next_request_messages(self, h, spoken_text, surviving_fc=None):
        """Build the universal context the next request would be built from:
        committed user/assistant text, the surviving tool call (if any), and
        whatever signature frames actually flushed."""
        messages = [
            {"role": "user", "content": "Yes, please"},
            {"role": "assistant", "content": spoken_text},
        ]
        if surviving_fc is not None:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": surviving_fc.tool_call_id,
                            "function": {
                                "name": surviving_fc.function_name,
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            )
        for frame in h.signature_frames():
            messages.extend(frame.messages)
        return messages

    def test_dropped_call_owning_signature_discards_entire_batch(self):
        """Direct first-FC signature: send_sms (first, cancellable) owns the
        signature; end_call survives cancellability — but it would be
        committed unsigned, so the whole batch must be discarded."""
        h = _Harness(
            [
                _chunk(
                    Part(text="Doing both now."),
                    Part(
                        function_call=FunctionCall(
                            id="fc-sms", name="send_sms", args={}
                        ),
                        thought_signature=b"sig-sms",
                    ),
                    _fc("fc-end", name="end_call"),
                )
            ]
        )
        self._register_end_call(h.service)
        asyncio.run(h.run())
        assert len(h.service._pending_function_calls) == 2
        assert h.service._pending_signature_frames[0].messages[0].message[
            "bookmark"
        ] == {"function_call": "fc-sms"}

        asyncio.run(
            h.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        )

        # The entire batch is discarded — including the non-cancellable
        # survivor — because its signature belonged to the dropped call.
        assert h.service._pending_function_calls == []
        assert h.service._pending_signature_frames == []

        asyncio.run(
            h.service.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
        )
        assert h.run_fc_calls == 0
        assert h.signature_frames() == []

        # Round-trip: the next request's context carries no function calls
        # and no orphan signatures, so it cannot 400.
        adapter = GeminiLLMAdapter()
        converted = adapter._from_universal_context_messages(
            self._next_request_messages(h, "Doing both now.")
        )
        fc_parts = [
            p
            for m in converted.messages
            for p in m.parts
            if getattr(p, "function_call", None)
        ]
        assert fc_parts == []

    def test_survivor_owning_signature_keeps_batch_and_valid_next_request(self):
        """Signature anchored to the surviving end_call: batch is kept, and
        the next request's context has every function-call part signed."""
        h = _Harness(
            [
                _chunk(
                    Part(text="Doing both now."),
                    _fc("fc-sms", name="send_sms"),
                    Part(
                        function_call=FunctionCall(
                            id="fc-end", name="end_call", args={}
                        ),
                        thought_signature=b"sig-end",
                    ),
                )
            ]
        )
        self._register_end_call(h.service)
        asyncio.run(h.run())

        asyncio.run(
            h.service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
        )

        survivors = h.service._pending_function_calls
        assert [fc.function_name for fc in survivors] == ["end_call"]
        assert len(h.service._pending_signature_frames) == 1

        asyncio.run(
            h.service.process_frame(
                BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM
            )
        )
        assert h.run_fc_calls == 1
        assert h.signature_frames() != []

        # Round-trip: every function-call part in the next request's context
        # carries its thought signature.
        adapter = GeminiLLMAdapter()
        converted = adapter._from_universal_context_messages(
            self._next_request_messages(h, "Doing both now.", survivors[0])
        )
        fc_parts = [
            p
            for m in converted.messages
            for p in m.parts
            if getattr(p, "function_call", None)
        ]
        assert len(fc_parts) == 1
        assert fc_parts[0].function_call.id == "fc-end"
        assert fc_parts[0].thought_signature == b"sig-end"


if __name__ == "__main__":
    unittest.main()
