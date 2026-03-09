"""
API Tests
=========

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
from minigpt.api.schemas import CompletionRequest, CompletionResponse, CompletionChoice, UsageInfo


class TestSchemas:
    def test_completion_request(self):
        req = CompletionRequest(prompt="Hello world")
        assert req.max_tokens == 128
        assert req.temperature == 0.7

    def test_completion_request_custom(self):
        req = CompletionRequest(
            prompt="Test", max_tokens=256, temperature=0.5, top_k=40
        )
        assert req.max_tokens == 256
        assert req.top_k == 40

    def test_completion_response(self):
        resp = CompletionResponse(
            id="test-123",
            model="minigpt",
            choices=[CompletionChoice(text="output", finish_reason="stop", tokens_generated=5)],
            usage=UsageInfo(prompt_tokens=2, completion_tokens=5, total_tokens=7),
        )
        assert resp.choices[0].text == "output"
        assert resp.usage.total_tokens == 7
