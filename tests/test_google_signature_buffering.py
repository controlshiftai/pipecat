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
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
)
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


if __name__ == "__main__":
    unittest.main()
