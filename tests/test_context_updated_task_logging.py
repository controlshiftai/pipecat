"""Regression test: a failed on_context_updated task must be logged loudly.

Previously _context_updated_task_finished only discarded the task, so engine
set_node crashes (e.g. the Gemini LLMSpecificMessage TypeError in
update_llm_context) died silently — the post-transition generation never ran
and nothing in the logs explained why.
"""

import asyncio
import unittest

from loguru import logger

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
)


class TestContextUpdatedTaskLogging(unittest.TestCase):
    def _make_failed_task(self) -> asyncio.Task:
        async def make():
            async def failing():
                raise ValueError("boom")

            task = asyncio.get_running_loop().create_task(failing())
            try:
                await task
            except ValueError:
                pass
            return task

        return asyncio.run(make())

    def test_failed_task_is_logged_and_discarded(self):
        aggregator = LLMAssistantAggregator(LLMContext())
        task = self._make_failed_task()
        aggregator._context_updated_tasks.add(task)

        records = []
        sink_id = logger.add(lambda message: records.append(str(message)), level="ERROR")
        try:
            aggregator._context_updated_task_finished(task)
        finally:
            logger.remove(sink_id)

        self.assertNotIn(task, aggregator._context_updated_tasks)
        self.assertTrue(
            any("on_context_updated task failed" in record for record in records),
            f"expected failure log, got: {records}",
        )

    def test_successful_task_is_discarded_without_error_log(self):
        aggregator = LLMAssistantAggregator(LLMContext())

        async def make():
            task = asyncio.get_running_loop().create_task(asyncio.sleep(0))
            await task  # ensure it completes successfully before the loop closes
            return task

        task = asyncio.run(make())
        assert task.done() and not task.cancelled() and task.exception() is None
        aggregator._context_updated_tasks.add(task)

        records = []
        sink_id = logger.add(lambda message: records.append(str(message)), level="ERROR")
        try:
            aggregator._context_updated_task_finished(task)
        finally:
            logger.remove(sink_id)

        self.assertNotIn(task, aggregator._context_updated_tasks)
        self.assertFalse(
            any("on_context_updated task failed" in record for record in records),
            f"unexpected failure log: {records}",
        )

    def test_cancelled_task_is_discarded_without_error_log(self):
        aggregator = LLMAssistantAggregator(LLMContext())

        async def make():
            task = asyncio.get_running_loop().create_task(asyncio.sleep(10))
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return task

        task = asyncio.run(make())
        assert task.cancelled()
        aggregator._context_updated_tasks.add(task)

        records = []
        sink_id = logger.add(lambda message: records.append(str(message)), level="ERROR")
        try:
            aggregator._context_updated_task_finished(task)
        finally:
            logger.remove(sink_id)

        self.assertNotIn(task, aggregator._context_updated_tasks)
        self.assertFalse(
            any("on_context_updated task failed" in record for record in records),
            f"unexpected failure log for cancelled task: {records}",
        )


if __name__ == "__main__":
    unittest.main()
