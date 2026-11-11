"""Regression tests for #7865: ``Memory.add()`` must report why a fact was not stored.

Before the fix, a fact dropped by the md5 hash short-circuit, a failed fallback
insert, a failed embed, or an extraction parse failure surfaced only as a
``logger.debug``/``logger.error`` line, leaving ``{"results": []}`` ambiguous
between "nothing to store" and "everything was dropped".

These tests drive the public ``add()`` return shape (index/vector-store/LLM
operations are all mocked; no external services are used) and fail on the
unmodified parent commit with ``KeyError: 'skipped'`` / missing skip reports.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem0.memory.main import AsyncMemory, Memory

_FACT = "likes tea"
_FACT_HASH = hashlib.md5(_FACT.encode()).hexdigest()


def _make_memory(mocker, cls: type = Memory):
    """Build a Memory/AsyncMemory with all factories and side effects mocked.

    Mirrors the mocking idiom of tests/memory/test_main.py: factory-created
    collaborators (LLM, embedder, vector store, entity store, SQLite) are
    replaced with MagicMocks, and the one-shot OSS notices are patched out so
    tests observe only the add() bookkeeping.
    """
    mock_embedder = mocker.MagicMock()
    embedder = mock_embedder.return_value
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    embedder.embed_batch.side_effect = lambda texts, action=None: [[0.1, 0.2, 0.3] for _ in texts]
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mock_embedder)

    mock_vector_store = mocker.MagicMock()
    mock_vector_store.return_value.search.return_value = []
    mocker.patch(
        "mem0.utils.factory.VectorStoreFactory.create",
        side_effect=[mock_vector_store.return_value, mocker.MagicMock()],
    )

    mock_llm = mocker.MagicMock()
    mocker.patch("mem0.utils.factory.LlmFactory.create", mock_llm)
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())

    memory = cls()
    memory.custom_instructions = None
    memory.api_version = "v1.1"
    memory.db.get_last_messages = MagicMock(return_value=[])
    memory.db.save_messages = MagicMock()

    mocker.patch("mem0.memory.main.capture_event")
    mocker.patch("mem0.memory.main.display_first_run_notice")
    mocker.patch("mem0.memory.main.display_temporal_usage_notice")
    mocker.patch("mem0.memory.main.display_scale_threshold_notice")
    mocker.patch("mem0.memory.main.detect_scale_threshold_from_add_result", return_value=None)
    mocker.patch("mem0.memory.main.detect_temporal_usage_from_metadata", return_value=None)

    return memory, mock_llm, embedder


def _extract(mock_llm, *texts):
    mock_llm.return_value.generate_response.return_value = (
        '{"memory": [' + ",".join('{"text": "%s"}' % t for t in texts) + "]}"
    )


class TestAddSkippedReportingSync:
    def test_duplicate_in_store_reported_with_existing_id(self, mocker):
        """A fact whose hash matches a stored memory is reported, not silently dropped."""
        memory, mock_llm, _ = _make_memory(mocker)
        memory.vector_store.search.return_value = [
            SimpleNamespace(id="existing-mem-1", payload={"data": _FACT, "hash": _FACT_HASH})
        ]
        _extract(mock_llm, _FACT)

        result = memory.add("I like tea", user_id="u1")

        assert result["results"] == []
        assert result["skipped"] == [
            {"memory": _FACT, "reason": "duplicate", "existing_id": "existing-mem-1", "error": None}
        ]

    def test_duplicate_within_batch_reported(self, mocker):
        """The second copy of a fact repeated inside one batch is reported once."""
        memory, mock_llm, _ = _make_memory(mocker)
        _extract(mock_llm, _FACT, _FACT)

        result = memory.add("I like tea", user_id="u1")

        assert [r["memory"] for r in result["results"]] == [_FACT]
        assert result["results"][0]["event"] == "ADD"
        # In-batch duplicate: nothing stored yet to point at, so existing_id is None.
        assert result["skipped"] == [{"memory": _FACT, "reason": "duplicate", "existing_id": None, "error": None}]

    def test_all_duplicates_no_longer_looks_like_empty_extraction(self, mocker):
        """The issue's concrete scenario: every fact deduped -> results [] must
        still carry the skip reports so callers can tell dedup worked."""
        memory, mock_llm, _ = _make_memory(mocker)
        memory.vector_store.search.return_value = [
            SimpleNamespace(id="existing-mem-1", payload={"data": _FACT, "hash": _FACT_HASH})
        ]
        _extract(mock_llm, _FACT)

        result = memory.add("I like tea", user_id="u1")

        assert result["results"] == []
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "duplicate"

    def test_failed_fallback_insert_reported(self, mocker):
        """A record the vector store rejected in the one-by-one fallback is
        reported with its error instead of vanishing after a logger.error."""
        memory, mock_llm, _ = _make_memory(mocker)
        _extract(mock_llm, "fact one", "fact two")

        def insert(vectors=None, ids=None, payloads=None, **kwargs):
            if len(ids) > 1:
                raise RuntimeError("batch insert down")
            if payloads[0]["data"] == "fact two":
                raise RuntimeError("record rejected")

        memory.vector_store.insert.side_effect = insert

        result = memory.add("msg", user_id="u1")

        assert [r["memory"] for r in result["results"]] == ["fact one"]
        assert result["skipped"] == [
            {"memory": "fact two", "reason": "insert_failed", "existing_id": None, "error": "record rejected"}
        ]

    def test_embed_failed_reported(self, mocker):
        """A fact whose embedding failed (and was logged as a warning) is reported."""
        memory, mock_llm, embedder = _make_memory(mocker)
        _extract(mock_llm, "fact one", "fact two")

        def embed(text, action=None):
            if text == "fact two":
                raise RuntimeError("embed rejected")
            return [0.1, 0.2, 0.3]

        embedder.embed_batch.side_effect = lambda texts, action=None: (_ for _ in ()).throw(
            RuntimeError("batch embed down")
        )
        embedder.embed.side_effect = embed

        result = memory.add("msg", user_id="u1")

        assert [r["memory"] for r in result["results"]] == ["fact one"]
        assert result["skipped"] == [
            {"memory": "fact two", "reason": "embed_failed", "existing_id": None, "error": "embed rejected"}
        ]

    def test_extraction_parse_failure_reported(self, mocker):
        """Unparseable extraction output is reported, so results [] no longer
        means the same as 'LLM extracted nothing'."""
        memory, mock_llm, _ = _make_memory(mocker)
        mock_llm.return_value.generate_response.return_value = "invalid json"

        result = memory.add("msg", user_id="u1")

        assert result["results"] == []
        assert len(result["skipped"]) == 1
        skip = result["skipped"][0]
        assert skip["reason"] == "extraction_parse_failed"
        assert skip["memory"] is None

    def test_no_skipped_key_when_all_facts_stored(self, mocker):
        """Backward-compat guard: the response shape is unchanged when nothing
        was dropped — no 'skipped' key appears."""
        memory, mock_llm, _ = _make_memory(mocker)
        _extract(mock_llm, _FACT)

        result = memory.add("I like tea", user_id="u1")

        assert list(result.keys()) == ["results"]
        assert result["results"] == [{"id": result["results"][0]["id"], "memory": _FACT, "event": "ADD"}]

    def test_no_skipped_key_when_llm_extracts_nothing(self, mocker):
        """A genuinely empty extraction must NOT produce skip reports — that is
        exactly the distinction the new field exists to preserve."""
        memory, mock_llm, _ = _make_memory(mocker)
        mock_llm.return_value.generate_response.return_value = '{"memory": []}'

        result = memory.add("msg", user_id="u1")

        assert result == {"results": []}


@pytest.mark.asyncio
class TestAddSkippedReportingAsync:
    async def test_async_duplicate_in_store_reported_with_existing_id(self, mocker):
        memory, mock_llm, _ = _make_memory(mocker, cls=AsyncMemory)
        memory.vector_store.search.return_value = [
            SimpleNamespace(id="existing-mem-1", payload={"data": _FACT, "hash": _FACT_HASH})
        ]
        _extract(mock_llm, _FACT)

        result = await memory.add("I like tea", user_id="u1")

        assert result["results"] == []
        assert result["skipped"] == [
            {"memory": _FACT, "reason": "duplicate", "existing_id": "existing-mem-1", "error": None}
        ]

    async def test_async_all_duplicates_reported(self, mocker):
        memory, mock_llm, _ = _make_memory(mocker, cls=AsyncMemory)
        memory.vector_store.search.return_value = [
            SimpleNamespace(id="existing-mem-1", payload={"data": _FACT, "hash": _FACT_HASH})
        ]
        _extract(mock_llm, _FACT)

        result = await memory.add("I like tea", user_id="u1")

        assert result["results"] == []
        assert len(result["skipped"]) == 1
        assert result["skipped"][0]["reason"] == "duplicate"

    async def test_async_failed_fallback_insert_reported(self, mocker):
        memory, mock_llm, _ = _make_memory(mocker, cls=AsyncMemory)
        _extract(mock_llm, "fact one", "fact two")

        def insert(vectors=None, ids=None, payloads=None, **kwargs):
            if len(ids) > 1:
                raise RuntimeError("batch insert down")
            if payloads[0]["data"] == "fact two":
                raise RuntimeError("record rejected")

        memory.vector_store.insert.side_effect = insert

        result = await memory.add("msg", user_id="u1")

        assert [r["memory"] for r in result["results"]] == ["fact one"]
        assert result["skipped"] == [
            {"memory": "fact two", "reason": "insert_failed", "existing_id": None, "error": "record rejected"}
        ]

    async def test_async_no_skipped_key_when_all_facts_stored(self, mocker):
        memory, mock_llm, _ = _make_memory(mocker, cls=AsyncMemory)
        _extract(mock_llm, _FACT)

        result = await memory.add("I like tea", user_id="u1")

        assert list(result.keys()) == ["results"]
