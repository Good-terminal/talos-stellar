"""Tests for openclaw talos_skill — issue #887.

Covers:
- Structured {"error": ..., "status": ...} results on HTTP 4xx/5xx errors.
- talos_shutdown closes the shared client (no unclosed-client warnings).
- get_pending_jobs and update_status log non-200 responses instead of silently
  swallowing them.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """Return an httpx.MockTransport that replays *responses* in order."""
    iterator = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(iterator)

    return httpx.MockTransport(handler)


def _json_response(status: int, body: dict | list) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"Content-Type": "application/json"},
        content=json.dumps(body).encode(),
    )


def _make_client(transport: httpx.MockTransport):
    """Create a TalosClient wired to a mock transport."""
    from talos_skill.client import TalosClient  # local import to avoid side-effects at module load

    client = TalosClient(
        base_url="http://test.invalid",
        api_key="test-key",
        talos_id="talos_test",
    )
    # Replace the internal httpx client with one using the mock transport.
    client._http = httpx.AsyncClient(
        base_url="http://test.invalid",
        headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        transport=transport,
        timeout=5.0,
    )
    return client


# ---------------------------------------------------------------------------
# TalosClient unit tests
# ---------------------------------------------------------------------------

class TestTalosClientGetPendingJobs:
    @pytest.mark.asyncio
    async def test_returns_jobs_on_200(self):
        transport = _mock_transport([
            _json_response(200, [{"id": "job_1", "serviceName": "research"}]),
        ])
        client = _make_client(transport)
        jobs = await client.get_pending_jobs()
        assert jobs == [{"id": "job_1", "serviceName": "research"}]
        await client.close()

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_non_200(self, caplog):
        transport = _mock_transport([_json_response(401, {"error": "Unauthorized"})])
        client = _make_client(transport)

        with caplog.at_level(logging.WARNING, logger="talos_skill.client"):
            jobs = await client.get_pending_jobs()

        assert jobs == []
        assert any("401" in r.message for r in caplog.records), (
            "Expected a warning log containing the status code"
        )
        await client.close()

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_500(self, caplog):
        transport = _mock_transport([_json_response(500, {"error": "Internal Server Error"})])
        client = _make_client(transport)

        with caplog.at_level(logging.WARNING, logger="talos_skill.client"):
            jobs = await client.get_pending_jobs()

        assert jobs == []
        assert any("500" in r.message for r in caplog.records)
        await client.close()


class TestTalosClientUpdateStatus:
    @pytest.mark.asyncio
    async def test_succeeds_silently_on_200(self, caplog):
        transport = _mock_transport([_json_response(200, {"ok": True})])
        client = _make_client(transport)

        with caplog.at_level(logging.WARNING, logger="talos_skill.client"):
            await client.update_status(True)

        assert not caplog.records
        await client.close()

    @pytest.mark.asyncio
    async def test_logs_warning_on_non_200(self, caplog):
        transport = _mock_transport([_json_response(403, {"error": "Forbidden"})])
        client = _make_client(transport)

        with caplog.at_level(logging.WARNING, logger="talos_skill.client"):
            await client.update_status(True)

        assert any("403" in r.message for r in caplog.records), (
            "Expected a warning log for non-200 update_status response"
        )
        await client.close()


class TestTalosClientClose:
    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        transport = _mock_transport([])
        client = _make_client(transport)
        # Must not raise and must mark the underlying client as closed.
        await client.close()
        assert client._http.is_closed


# ---------------------------------------------------------------------------
# tools.py — structured error results
# ---------------------------------------------------------------------------

class TestToolsStructuredErrors:
    """Each tool function must return {"error": ..., "status": ...} on HTTP errors."""

    def _patch_client(self, monkeypatch, client):
        """Replace the module-level _client in tools with *client*."""
        import talos_skill.tools as tools_mod
        monkeypatch.setattr(tools_mod, "_client", client)

    @pytest.mark.asyncio
    async def test_talos_status_returns_structured_error_on_401(self, monkeypatch):
        transport = _mock_transport([_json_response(401, {"error": "Unauthorized"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_status
        result = await talos_status()

        assert result.get("status") == 401
        assert "error" in result
        assert "test-key" not in str(result), "API key must not appear in error result"
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_status_returns_structured_error_on_500(self, monkeypatch):
        transport = _mock_transport([_json_response(500, {"error": "Internal Server Error"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_status
        result = await talos_status()

        assert result.get("status") == 500
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_status_returns_structured_error_on_404(self, monkeypatch):
        transport = _mock_transport([_json_response(404, {"error": "Not Found"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_status
        result = await talos_status()

        assert result.get("status") == 404
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_discover_returns_structured_error_on_401(self, monkeypatch):
        transport = _mock_transport([_json_response(401, {"error": "Unauthorized"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_discover
        result = await talos_discover()

        assert result.get("status") == 401
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_register_returns_structured_error_on_422(self, monkeypatch):
        transport = _mock_transport([_json_response(422, {"error": "Unprocessable Entity"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_register
        result = await talos_register(
            name="TestAgent",
            category="Development",
            description="A test agent",
        )

        assert result.get("status") == 422
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_submit_result_returns_structured_error_on_404(self, monkeypatch):
        transport = _mock_transport([_json_response(404, {"error": "Job not found"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_submit_result
        result = await talos_submit_result(job_id="job_missing", result={"output": "done"})

        assert result.get("status") == 404
        assert "error" in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_report_returns_structured_error_on_500(self, monkeypatch):
        transport = _mock_transport([_json_response(500, {"error": "Internal Server Error"})])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_report
        result = await talos_report(action="activity", content="hello", channel="test")

        assert result.get("status") == 500
        assert "error" in result
        await client.close()

    # ── success paths ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_talos_status_success_path(self, monkeypatch):
        transport = _mock_transport([
            _json_response(200, {
                "name": "TestAgent",
                "status": "Active",
                "agentOnline": True,
                "revenues": [{"amount": "5.00"}],
                "commerceServices": [{}],
                "activities": [{}, {}],
                "approvals": [],
                "gtmBudget": 100,
            }),
        ])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_status
        result = await talos_status()

        assert result["name"] == "TestAgent"
        assert result["total_revenue_usdc"] == 5.0
        assert "error" not in result
        await client.close()

    @pytest.mark.asyncio
    async def test_talos_discover_success_path(self, monkeypatch):
        transport = _mock_transport([
            _json_response(200, [
                {"talosId": "t1", "serviceName": "research", "price": "2.00", "currency": "USDC"},
            ]),
        ])
        client = _make_client(transport)
        self._patch_client(monkeypatch, client)

        from talos_skill.tools import talos_discover
        result = await talos_discover()

        assert result["count"] == 1
        assert result["services"][0]["name"] == "research"
        await client.close()


# ---------------------------------------------------------------------------
# talos_shutdown
# ---------------------------------------------------------------------------

class TestTalosShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_client_and_resets_module_global(self, monkeypatch):
        import talos_skill.tools as tools_mod

        transport = _mock_transport([])
        client = _make_client(transport)
        monkeypatch.setattr(tools_mod, "_client", client)

        from talos_skill.tools import talos_shutdown
        await talos_shutdown()

        assert client._http.is_closed, "HTTP client must be closed after shutdown"
        assert tools_mod._client is None, "Module _client must be reset to None"

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch):
        """Calling talos_shutdown twice must not raise."""
        import talos_skill.tools as tools_mod

        transport = _mock_transport([])
        client = _make_client(transport)
        monkeypatch.setattr(tools_mod, "_client", client)

        from talos_skill.tools import talos_shutdown
        await talos_shutdown()
        await talos_shutdown()  # second call with _client=None

    @pytest.mark.asyncio
    async def test_register_calls_register_shutdown_if_supported(self):
        """register() should wire talos_shutdown if the host api has register_shutdown."""
        from talos_skill import register, talos_shutdown

        api = MagicMock()
        api.register_shutdown = MagicMock()
        register(api)

        api.register_shutdown.assert_called_once_with(talos_shutdown)

    @pytest.mark.asyncio
    async def test_register_skips_shutdown_if_not_supported(self):
        """register() must not raise when the host api lacks register_shutdown."""
        from talos_skill import register

        api = MagicMock(spec=["register_tool"])  # no register_shutdown attribute
        register(api)  # should not raise
