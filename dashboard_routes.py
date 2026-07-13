"""
Dashboard routes for Ombre Brain — ported from server.py custom routes.
Provides the admin dashboard web UI and its backing API endpoints.
Accessed via GatewayService (request.app.state.gateway_service) for all services.
"""

from __future__ import annotations

import hashlib
import hmac
import json as _json_lib
import logging
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from utils import (
    load_config,
    strip_wikilinks,
)

logger = logging.getLogger("ombre_brain.dashboard")

# ---------------------------------------------------------------------------
# Auth infrastructure
# ---------------------------------------------------------------------------

_dashboard_sessions: dict[str, float] = {}


def _dashboard_auth_file() -> str:
    buckets_dir = os.environ.get("OMBRE_BUCKETS_DIR", "buckets")
    state_dir = os.path.join(os.path.dirname(os.path.abspath(buckets_dir)), "state")
    return os.path.join(state_dir, ".dashboard_auth.json")


def _load_dashboard_password_hash() -> str | None:
    try:
        path = _dashboard_auth_file()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _json_lib.load(f)
            return data.get("password_hash")
    except Exception:
        logger.warning("Failed to load dashboard auth file", exc_info=True)
    return None


def _save_dashboard_password_hash(password: str) -> None:
    salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    path = _dashboard_auth_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json_lib.dump({"password_hash": f"{salt}:{digest}"}, f)


def _verify_dashboard_hash(password: str, stored: str) -> bool:
    if ":" not in stored:
        return False
    salt, digest = stored.split(":", 1)
    current = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, current)


def _dashboard_setup_needed() -> bool:
    if os.environ.get("OMBRE_DASHBOARD_PASSWORD", ""):
        return False
    return _load_dashboard_password_hash() is None


def _verify_dashboard_password(password: str) -> bool:
    env_password = os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")
    if env_password:
        return hmac.compare_digest(password, env_password)
    stored = _load_dashboard_password_hash()
    return bool(stored and _verify_dashboard_hash(password, stored))


def _create_dashboard_session() -> str:
    token = secrets.token_urlsafe(32)
    _dashboard_sessions[token] = time.time() + 86400 * 7
    return token


def _dashboard_authenticated(request: Request) -> bool:
    token = request.cookies.get("ombre_session")
    if not token:
        return False
    expiry = _dashboard_sessions.get(token)
    if expiry is None or time.time() > expiry:
        _dashboard_sessions.pop(token, None)
        return False
    return True


def _require_dashboard_auth(request: Request) -> Response | None:
    if _dashboard_authenticated(request):
        return None
    return JSONResponse(
        {"error": "unauthorized", "setup_needed": _dashboard_setup_needed()},
        status_code=401,
    )


def _dashboard_login_response() -> JSONResponse:
    token = _create_dashboard_session()
    response = JSONResponse({"ok": True})
    response.set_cookie(
        "ombre_session",
        token,
        httponly=True,
        samesite="lax",
        max_age=86400 * 7,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _int_between(value, default, lower, upper):
    try:
        num = int(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, num))


def _float_between(value, default, lower, upper):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(lower, min(upper, num))


def _mask_key(api_key: str) -> str:
    return f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")


def _ai_author_name() -> str:
    from identity import identity_names
    cfg = load_config()
    names = identity_names(cfg)
    return names.get("ai_name", "Haven")


def _dashboard_author_name() -> str:
    from identity import identity_names
    cfg = load_config()
    names = identity_names(cfg)
    return names.get("user_name", "用户")


def _bucket_dashboard_sort_key(b: dict) -> tuple:
    meta = b.get("metadata", {})
    pinned = 1 if meta.get("pinned") else 0
    anchor = 1 if meta.get("anchor") else 0
    score = _decay_score(meta)
    return (pinned, anchor, score)


def _decay_score(meta: dict) -> float:
    try:
        from decay_engine import DecayEngine
        config = load_config()
        engine = DecayEngine(config)
        return engine.calculate_score(meta)
    except Exception:
        return 0.0


def _bucket_light_payload(bucket: dict) -> dict:
    meta = bucket.get("metadata", {})
    return {
        "id": bucket["id"],
        "name": meta.get("name", bucket["id"]),
        "type": meta.get("type", "dynamic"),
        "domain": meta.get("domain", []),
        "tags": meta.get("tags", []),
        "facets": meta.get("facets", []),
        "resolved": meta.get("resolved", False),
        "pinned": meta.get("pinned", False),
        "anchor": meta.get("anchor", False),
        "digested": meta.get("digested", False),
        "protected": meta.get("protected", False),
        "self_anchor": False,
        "created": meta.get("created", ""),
        "last_active": meta.get("last_active", ""),
        "score": _decay_score(meta),
    }


def _bucket_light_sort_key(item: dict) -> tuple:
    pinned = 1 if item.get("pinned") else 0
    anchor = 1 if item.get("anchor") else 0
    score = item.get("score", 0)
    return (pinned, anchor, score)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def dashboard_root(request: Request) -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


async def dashboard_page(request: Request) -> HTMLResponse:
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


async def dashboard_assets(request: Request) -> Response:
    asset_path = str(request.path_params.get("path") or "").strip().replace("\\", "/")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard_assets"))
    target = os.path.abspath(os.path.join(base_dir, asset_path))
    if not target.startswith(base_dir + os.sep) or not os.path.isfile(target):
        return PlainTextResponse("dashboard asset not found", status_code=404)
    return FileResponse(target)


# --- Auth routes ---

async def auth_status(request: Request) -> JSONResponse:
    return JSONResponse({
        "authenticated": _dashboard_authenticated(request),
        "setup_needed": _dashboard_setup_needed(),
        "identity": {
            "ai_name": _ai_author_name(),
            "user_name": _dashboard_author_name(),
        },
    })


async def auth_setup(request: Request) -> JSONResponse:
    if not _dashboard_setup_needed():
        return JSONResponse({"error": "already configured"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "").strip()
    if len(password) < 6:
        return JSONResponse({"error": "password must be at least 6 characters"}, status_code=400)
    _save_dashboard_password_hash(password)
    return _dashboard_login_response()


async def auth_login(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    password = str(body.get("password") or "")
    if _verify_dashboard_password(password):
        return _dashboard_login_response()
    return JSONResponse({"error": "password rejected"}, status_code=401)


async def auth_logout(request: Request) -> JSONResponse:
    token = request.cookies.get("ombre_session")
    if token:
        _dashboard_sessions.pop(token, None)
    response = JSONResponse({"ok": True})
    response.delete_cookie("ombre_session")
    return response


# --- API routes ---

async def api_config_get(request: Request) -> JSONResponse:
    # Support both dashboard session auth and gateway Bearer token auth
    if _dashboard_authenticated(request):
        return await _dashboard_config_get(request)
    # Fall back to gateway Bearer auth
    gs = request.app.state.gateway_service
    return await gs.handle_config(request)


async def _dashboard_config_get(request: Request) -> JSONResponse:

    gs = request.app.state.gateway_service
    config = gs.config

    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {}) if isinstance(config.get("embedding", {}), dict) else {}
    rerank = config.get("reranker", {}) if isinstance(config.get("reranker", {}), dict) else {}
    gateway_cfg = config.get("gateway", {}) if isinstance(config.get("gateway", {}), dict) else {}
    recall_cfg = config.get("recall", {}) if isinstance(config.get("recall", {}), dict) else {}
    persona_cfg = config.get("persona", {}) if isinstance(config.get("persona", {}), dict) else {}
    dream_cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}

    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": _mask_key(dehy.get("api_key", "")),
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
            "base_url": emb.get("base_url", ""),
            "api_key_masked": _mask_key(emb.get("api_key", "")),
            "effective_base_url": gs.embedding_engine.base_url,
            "has_own_api_key": bool(emb.get("api_key", "")),
        },
        "reranker": {
            "enabled": bool(getattr(gs.reranker_engine, "enabled", False)),
            "model": getattr(gs.reranker_engine, "model", rerank.get("model", "")),
            "base_url": str(rerank.get("base_url") or getattr(gs.reranker_engine, "base_url", "")),
            "api_key_masked": _mask_key(rerank.get("api_key", "")),
        },
        "gateway": {
            "cooldown_hours": gateway_cfg.get("cooldown_hours", 6),
            "memory_sentinel_enabled": _bool_value(gateway_cfg.get("memory_sentinel_enabled"), True),
            "domain_sentinel_enabled": _bool_value(gateway_cfg.get("domain_sentinel_enabled"), True),
            "retrieval_mode": str(gateway_cfg.get("retrieval_mode", "graph")),
        },
        "recall": {
            "query_resurface_enabled": _bool_value(recall_cfg.get("query_resurface_enabled"), False),
        },
        "persona": {
            "enabled": bool(getattr(gs.persona_engine, "enabled", persona_cfg.get("enabled", True))),
            "model": getattr(gs.persona_engine, "model", persona_cfg.get("model", "")),
            "base_url": getattr(gs.persona_engine, "base_url", persona_cfg.get("base_url", "")),
            "api_key_masked": _mask_key(getattr(gs.persona_engine, "api_key", "") or persona_cfg.get("api_key", "")),
        },
        "dream": {
            "enabled": gs.dream_engine.enabled,
            "auto_enabled": getattr(gs.dream_engine, "auto_enabled", True),
            "model": getattr(gs.dream_engine, "model", ""),
            "base_url": getattr(gs.dream_engine, "base_url", ""),
            "api_key_masked": _mask_key(getattr(gs.dream_engine, "api_key", "")),
        },
    })


async def api_config_update(request: Request) -> JSONResponse:
    # Support both dashboard session auth and gateway Bearer token auth
    if _dashboard_authenticated(request):
        return await _dashboard_config_update(request)
    # Fall back to gateway Bearer auth
    gs = request.app.state.gateway_service
    return await gs.handle_config(request)


async def _dashboard_config_update(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    gs = request.app.state.gateway_service
    config = gs.config
    updated = []

    # Dehydration
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        gs.dehydrator.model = dehy.get("model", gs.dehydrator.model)
        gs.dehydrator.base_url = dehy.get("base_url", gs.dehydrator.base_url)
        gs.dehydrator.api_key = dehy.get("api_key", gs.dehydrator.api_key)
        gs.dehydrator.max_tokens = dehy.get("max_tokens", 1024)
        gs.dehydrator.temperature = dehy.get("temperature", 0.1)
        gs.dehydrator.api_available = bool(gs.dehydrator.api_key)

    # Embedding
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        for key in ("enabled", "model", "base_url"):
            if key in e:
                emb[key] = e[key]
                updated.append(f"embedding.{key}")
        if "api_key" in e and e["api_key"]:
            emb["api_key"] = e["api_key"]
            updated.append("embedding.api_key")

    # Gateway
    if "gateway" in body:
        g = body["gateway"]
        gw = config.setdefault("gateway", {})
        for key in ("cooldown_hours", "memory_sentinel_enabled", "domain_sentinel_enabled", "retrieval_mode"):
            if key in g:
                gw[key] = g[key]
                updated.append(f"gateway.{key}")

    # Persona
    if "persona" in body:
        p = body["persona"]
        pc = config.setdefault("persona", {})
        for key in ("enabled", "model", "base_url"):
            if key in p:
                pc[key] = p[key]
                updated.append(f"persona.{key}")
        if "api_key" in p and p["api_key"]:
            pc["api_key"] = p["api_key"]
            updated.append("persona.api_key")

    # Dream
    if "dream" in body:
        d = body["dream"]
        dc = config.setdefault("dream", {})
        for key in ("enabled", "auto_enabled", "model", "base_url", "temperature", "max_tokens"):
            if key in d:
                dc[key] = d[key]
                updated.append(f"dream.{key}")
        if "api_key" in d and d["api_key"]:
            dc["api_key"] = d["api_key"]
            updated.append("dream.api_key")

    # Save config to file
    config_path = os.environ.get("OMBRE_CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        updated.append("config_file_saved")
    except Exception as exc:
        logger.warning("Failed to save config: %s", exc)

    return JSONResponse({"updated": updated})


async def api_persona(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        session_id = (request.query_params.get("session_id") or "").strip() or None
        events_limit = _int_between(request.query_params.get("events_limit"), 20, 1, 100)
        sessions_limit = _int_between(request.query_params.get("sessions_limit"), 20, 1, 100)
        return JSONResponse(
            gs.persona_engine.get_dashboard_payload(
                session_id=session_id,
                events_limit=events_limit,
                sessions_limit=sessions_limit,
            )
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_dreams(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        limit = _int_between(request.query_params.get("limit", "30"), 30, 1, 100)
    except Exception:
        limit = 30
    try:
        return JSONResponse(gs.dream_engine.dashboard_payload(limit=max(1, min(100, limit))))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_buckets(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        all_buckets = await gs.bucket_mgr.list_all(include_archive=True)
        from memory_metadata import normalize_memory_metadata
        from self_anchor import is_self_anchor_bucket
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            metadata_view = normalize_memory_metadata(b)
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "facets": meta.get("facets", []),
                "metadata_view": metadata_view,
                **metadata_view,
                "source": meta.get("source", ""),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "importance": meta.get("importance", 5),
                "confidence": meta.get("confidence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "protected": meta.get("protected", False),
                "anchor": meta.get("anchor", False),
                "digested": meta.get("digested", False),
                "self_anchor": is_self_anchor_bucket(b),
                "profile_kind": meta.get("profile_kind", ""),
                "memory_subject": meta.get("memory_subject", ""),
                "memory_layer": meta.get("memory_layer", ""),
                "period": meta.get("period"),
                "date": meta.get("date"),
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 0),
                "comment_count": meta.get("comment_count", 0),
                "score": _decay_score(meta),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=_bucket_dashboard_sort_key, reverse=True)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_buckets_light(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        params = request.query_params
        include_archive = str(params.get("include_archive") or "").lower() in {"1", "true", "yes", "on"}
        limit = max(1, min(_int_between(params.get("limit"), 500, 1, 2000), 2000))
        offset = max(0, _int_between(params.get("offset"), 0, 0, 999999))
        all_buckets = await gs.bucket_mgr.list_all(include_archive=include_archive)
        items = [_bucket_light_payload(b) for b in all_buckets]
        items.sort(key=_bucket_light_sort_key, reverse=True)
        return JSONResponse({
            "buckets": items[offset:offset + limit],
            "count": len(items),
            "include_archive": include_archive,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_memories(request: Request) -> JSONResponse:
    """Create or update one memory bucket from the dashboard."""
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    bucket_id = str(body.get("id") or "").strip()
    content = str(body.get("content") or "").strip()
    edits = body.get("edits") if isinstance(body.get("edits"), dict) else None

    if not bucket_id:
        return JSONResponse({"error": "missing bucket id"}, status_code=400)

    try:
        if edits:
            result = await _apply_bucket_edits(bucket_id, edits, gs)
        elif content:
            # Create or overwrite
            meta_payload = {}
            for key in ("name", "domain", "tags", "facets", "type", "importance", "pinned",
                         "anchor", "resolved", "protected", "source", "valence", "arousal",
                         "confidence", "memory_layer", "profile_kind"):
                if key in body:
                    meta_payload[key] = body[key]

            existing = await gs.bucket_mgr.get(bucket_id)
            if existing:
                await gs.bucket_mgr.update(bucket_id, content=content, **meta_payload)
            else:
                await gs.bucket_mgr.create(
                    bucket_id=bucket_id,
                    content=content,
                    metadata=meta_payload,
                )
            result = {"id": bucket_id, "status": "saved"}
        else:
            return JSONResponse({"error": "no content or edits provided"}, status_code=400)

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def _apply_bucket_edits(bucket_id: str, edits: dict, gs) -> dict:
    bucket = await gs.bucket_mgr.get(bucket_id)
    if not bucket:
        raise ValueError("bucket not found")

    meta = bucket.get("metadata", {})
    update_kwargs = {}

    if "content" in edits:
        update_kwargs["content"] = edits["content"]
    for key in ("name", "domain", "tags", "facets", "type", "importance", "pinned",
                 "anchor", "resolved", "protected", "source", "memory_layer", "profile_kind"):
        if key in edits:
            update_kwargs[key] = edits[key]
    if "valence" in edits:
        update_kwargs["valence"] = float(edits["valence"])
    if "arousal" in edits:
        update_kwargs["arousal"] = float(edits["arousal"])
    if "confidence" in edits:
        update_kwargs["confidence"] = float(edits["confidence"])

    if update_kwargs:
        await gs.bucket_mgr.update(bucket_id, **update_kwargs)

    return {"id": bucket_id, "status": "updated"}


async def api_search(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        matches = await gs.bucket_mgr.search(query, limit=10)
        result = []
        for b in matches:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "resolved": meta.get("resolved", False),
                "pinned": meta.get("pinned", False),
                "anchor": meta.get("anchor", False),
                "last_active": meta.get("last_active", ""),
                "created": meta.get("created", ""),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_status(request: Request) -> JSONResponse:
    err = _require_dashboard_auth(request)
    if err:
        return err
    gs = request.app.state.gateway_service
    try:
        stats = await gs.bucket_mgr.get_stats()
        return JSONResponse({
            "decay_engine": "running",
            "buckets": {
                "permanent": stats.get("permanent_count", 0),
                "dynamic": stats.get("dynamic_count", 0),
                "archive": stats.get("archive_count", 0),
                "feel": stats.get("feel_count", 0),
                "total": (
                    stats.get("permanent_count", 0)
                    + stats.get("dynamic_count", 0)
                    + stats.get("archive_count", 0)
                    + stats.get("feel_count", 0)
                ),
            },
            "using_env_password": bool(os.environ.get("OMBRE_DASHBOARD_PASSWORD", "")),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

DASHBOARD_ROUTES = [
    Route("/", dashboard_root, methods=["GET"]),
    Route("/dashboard", dashboard_page, methods=["GET"]),
    Route("/dashboard-assets/{path:path}", dashboard_assets, methods=["GET"]),
    # Auth
    Route("/auth/status", auth_status, methods=["GET"]),
    Route("/auth/setup", auth_setup, methods=["POST"]),
    Route("/auth/login", auth_login, methods=["POST"]),
    Route("/auth/logout", auth_logout, methods=["POST"]),
    # API
    Route("/api/config", api_config_get, methods=["GET"]),
    Route("/api/config", api_config_update, methods=["POST"]),
    Route("/api/persona", api_persona, methods=["GET"]),
    Route("/api/dreams", api_dreams, methods=["GET"]),
    Route("/api/buckets", api_buckets, methods=["GET"]),
    Route("/api/buckets/light", api_buckets_light, methods=["GET"]),
    Route("/api/memories", api_memories, methods=["POST"]),
    Route("/api/search", api_search, methods=["GET"]),
    Route("/api/status", api_status, methods=["GET"]),
]
