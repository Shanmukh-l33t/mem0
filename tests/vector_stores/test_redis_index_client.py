"""Regression tests for #7819: RedisDB must wire its Redis client into
redisvl's SearchIndex without calling ``SearchIndex.set_client()``.

redisvl 0.28 removed the deprecated connection APIs (``connect``,
``set_client``, ``connection_args``, ``redis_kwargs`` — redis/redis-vl-python
PR #727) and prescribes passing ``redis_client`` to the constructor instead.
``SearchIndex.from_dict`` has forwarded ``**kwargs`` to that constructor since
redisvl 0.1.0, so ``SearchIndex.from_dict(schema, redis_client=client)`` works
on both sides of the change.

mem0 called ``set_client()`` in ``__init__``, ``create_col`` and ``reset``, so
a fresh install using the docs line (``pip install redis redisvl``) failed with
``AttributeError: 'SearchIndex' object has no attribute 'set_client'``.

These tests build a *real* ``SearchIndex`` from the installed redisvl (no
mocked ``from_dict``), but they never talk to a server: ``redis.Redis.from_url``
is lazy and the index's ``create()``/``delete()``/``query()`` calls are patched
out, so no running Redis instance is required.
"""

from contextlib import contextmanager
from unittest.mock import patch

import mem0.vector_stores.redis as redis_module
from mem0.vector_stores.redis import RedisDB


@contextmanager
def _no_server():
    """Run the block against the real redisvl SearchIndex class while keeping
    the server-touching index methods from opening a connection."""
    with (
        patch.object(redis_module.SearchIndex, "create"),
        patch.object(redis_module.SearchIndex, "delete"),
        patch.object(redis_module.SearchIndex, "query", return_value=[]),
    ):
        yield


def test_init_wires_client_into_index_without_set_client():
    """__init__ must hand its Redis client to the index at construction time.

    On redisvl >= 0.28 the old ``self.index.set_client(self.client)`` raises
    ``AttributeError: 'SearchIndex' object has no attribute 'set_client'`` and
    RedisDB cannot even be constructed.
    """
    with _no_server():
        db = RedisDB("redis://localhost:6379", "demo", 8)

        assert db.index.client is db.client, "the SearchIndex must be bound to the client RedisDB created"


def test_create_col_wires_client_into_new_index():
    """create_col() builds a second index; it must bind the same client
    instead of calling the removed set_client()."""
    with _no_server():
        db = RedisDB("redis://localhost:6379", "demo", 8)

        index = db.create_col(name="col_7819", vector_size=8)

        assert index.client is db.client


def test_reset_wires_client_into_recreated_index():
    """reset() drops and rebuilds the index; the rebuilt index must be bound
    to the existing client rather than calling the removed set_client()."""
    with _no_server():
        db = RedisDB("redis://localhost:6379", "demo", 8)

        db.reset()

        assert db.index.client is db.client


def test_search_runs_against_the_bound_client():
    """search() must dispatch through the index RedisDB constructed — the path
    that used to depend on set_client() having been called."""
    with _no_server():
        db = RedisDB("redis://localhost:6379", "demo", 8)

        assert db.search("hello", [0.1, 0.2, 0.3, 0.4], top_k=5) == []
