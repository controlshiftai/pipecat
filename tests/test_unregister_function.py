"""Regression test: LLMService.unregister_function must be a safe no-op.

The workflow engine unregisters stale transition handlers on every set_node.
Functions registered without a start_callback have no _start_callbacks entry,
and unregistering a never-registered name must not crash either — previously
both raised KeyError on the real service (mocked-LLM unit tests hid this).
"""

import unittest

from pipecat.services.llm_service import LLMService


class DummyLLMService(LLMService):
    pass


class TestUnregisterFunction(unittest.TestCase):
    def _service(self) -> LLMService:
        return DummyLLMService()

    def test_unregister_function_without_start_callback(self):
        service = self._service()

        async def handler(params):
            pass

        service.register_function("edge", handler)  # no start_callback
        service.unregister_function("edge")  # must not raise KeyError
        self.assertFalse(service.has_function("edge"))

    def test_unregister_never_registered_function_is_noop(self):
        service = self._service()
        service.unregister_function("ghost")  # must not raise KeyError

    def test_unregister_removes_start_callback(self):
        service = self._service()

        async def handler(params):
            pass

        async def start_callback(function_name, llm, context):
            pass

        service.register_function("edge", handler, start_callback=start_callback)
        self.assertIn("edge", service._start_callbacks)

        service.unregister_function("edge")
        self.assertFalse(service.has_function("edge"))
        self.assertNotIn("edge", service._start_callbacks)

    def test_unregister_leaves_other_functions_intact(self):
        service = self._service()

        async def handler_a(params):
            pass

        async def handler_b(params):
            pass

        service.register_function("a", handler_a)
        service.register_function("b", handler_b)
        service.unregister_function("a")
        self.assertFalse(service.has_function("a"))
        self.assertTrue(service.has_function("b"))


if __name__ == "__main__":
    unittest.main()
