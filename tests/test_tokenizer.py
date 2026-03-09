"""
Tokenizer Tests
===============

Author: Prajit Datta (https://github.com/prajitdatta)
"""

import pytest
from minigpt.data.tokenizer import BPETokenizer


class TestBPETokenizer:
    def test_byte_level_encoding(self):
        tok = BPETokenizer(vocab_size=256)
        text = "hello"
        ids = tok.encode(text)
        assert len(ids) == 5  # one per byte
        assert tok.decode(ids) == text

    def test_train_and_encode(self):
        tok = BPETokenizer(vocab_size=300)
        tok.train("hello world hello world hello world " * 50, verbose=False)
        ids = tok.encode("hello world")
        # After merges, should be fewer tokens than raw bytes
        assert len(ids) <= 11
        decoded = tok.decode(ids)
        assert decoded == "hello world"

    def test_save_load(self, tmp_path):
        tok = BPETokenizer(vocab_size=300)
        tok.train("the quick brown fox " * 100, verbose=False)

        path = str(tmp_path / "tok.json")
        tok.save(path)
        loaded = BPETokenizer.load(path)

        text = "the quick brown fox"
        assert tok.encode(text) == loaded.encode(text)

    def test_special_tokens(self):
        tok = BPETokenizer(vocab_size=256, special_tokens={"<|endoftext|>": 50256})
        ids = tok.encode("hello<|endoftext|>world")
        assert 50256 in ids

    def test_empty_string(self):
        tok = BPETokenizer(vocab_size=256)
        assert tok.encode("") == []
        assert tok.decode([]) == ""
