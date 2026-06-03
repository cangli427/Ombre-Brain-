from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LAYER_CORE = "core_memory"
LAYER_ANCHOR = "long_term_anchor"
LAYER_DYNAMIC = "dynamic_memory"
LAYER_RELATIONSHIP_WEATHER = "relationship_weather"
LAYER_AFFECT_CONTEXT = "affect_context"
LAYER_FAVORITE = "favorite_memory"
LAYER_DREAM = "dream"
LAYER_SOURCE_RECORD = "source_record"
LAYER_ARCHIVE = "archive"

DIRECT_CONTENT = "content_only"
DIRECT_EXPLICIT = "explicit_only"
DIRECT_EXPLICIT_OR_CONTENT = "explicit_or_content"
DIRECT_RESONANCE = "resonance_only"
DIRECT_NEVER = "never"

RENDER_DIRECT_AUTO = "direct_auto"
RENDER_SUMMARY = "summary"
RENDER_STABLE = "stable_rule_or_original"
RENDER_WEATHER = "weather"
RENDER_AUXILIARY = "auxiliary_context"
RENDER_FAVORITE = "favorite_card"
RENDER_DREAM_ORIGINAL = "dream_original"
RENDER_SOURCE_ONLY = "source_only"

DIFFUSE_SOURCE = "source"
DIFFUSE_CAREFUL_SOURCE = "careful_source"
DIFFUSE_CHAIN_ONLY = "chain_only"
DIFFUSE_NEVER = "never"

CONTEXT_ONLY_SECTIONS = frozenset({"comment", "affect_anchor", "favorite_reason"})
RELATIONSHIP_WEATHER_TAGS = frozenset(
    {"relationship_weather", "daily_impression", "weekly_impression"}
)
RAW_SOURCE_TAGS = frozenset({"raw_source", "chat_log", "diary_source", "source_record"})
FAVORITE_TAG = "haven_favorite"
FAVORITE_PREFIX = "flavor_"


@dataclass(frozen=True)
class MemoryLayerPolicy:
    layer: str
    direct_seed_policy: str
    render_policy: str
    gateway_section: str
    cooldown_policy: str
    diffusion_policy: str
    preserves_original: bool

    @property
    def can_direct_seed(self) -> bool:
        return self.direct_seed_policy != DIRECT_NEVER

    @property
    def can_diffuse(self) -> bool:
        return self.diffusion_policy != DIFFUSE_NEVER


LAYER_POLICIES: dict[str, MemoryLayerPolicy] = {
    LAYER_CORE: MemoryLayerPolicy(
        layer=LAYER_CORE,
        direct_seed_policy=DIRECT_EXPLICIT_OR_CONTENT,
        render_policy=RENDER_STABLE,
        gateway_section="Core Memory",
        cooldown_policy="rare",
        diffusion_policy=DIFFUSE_CAREFUL_SOURCE,
        preserves_original=True,
    ),
    LAYER_ANCHOR: MemoryLayerPolicy(
        layer=LAYER_ANCHOR,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_DIRECT_AUTO,
        gateway_section="Recalled Memory",
        cooldown_policy="normal",
        diffusion_policy=DIFFUSE_SOURCE,
        preserves_original=True,
    ),
    LAYER_DYNAMIC: MemoryLayerPolicy(
        layer=LAYER_DYNAMIC,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_DIRECT_AUTO,
        gateway_section="Recalled Memory",
        cooldown_policy="normal",
        diffusion_policy=DIFFUSE_SOURCE,
        preserves_original=True,
    ),
    LAYER_RELATIONSHIP_WEATHER: MemoryLayerPolicy(
        layer=LAYER_RELATIONSHIP_WEATHER,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_WEATHER,
        gateway_section="Relationship Weather",
        cooldown_policy="interval_or_config",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_AFFECT_CONTEXT: MemoryLayerPolicy(
        layer=LAYER_AFFECT_CONTEXT,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_AUXILIARY,
        gateway_section="attached_to_reliable_memory",
        cooldown_policy="parent",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_FAVORITE: MemoryLayerPolicy(
        layer=LAYER_FAVORITE,
        direct_seed_policy=DIRECT_CONTENT,
        render_policy=RENDER_FAVORITE,
        gateway_section="Haven Favorite Memory",
        cooldown_policy="separate_budget",
        diffusion_policy=DIFFUSE_CAREFUL_SOURCE,
        preserves_original=True,
    ),
    LAYER_DREAM: MemoryLayerPolicy(
        layer=LAYER_DREAM,
        direct_seed_policy=DIRECT_RESONANCE,
        render_policy=RENDER_DREAM_ORIGINAL,
        gateway_section="Dream",
        cooldown_policy="dream_surface_rules",
        diffusion_policy=DIFFUSE_CHAIN_ONLY,
        preserves_original=True,
    ),
    LAYER_SOURCE_RECORD: MemoryLayerPolicy(
        layer=LAYER_SOURCE_RECORD,
        direct_seed_policy=DIRECT_NEVER,
        render_policy=RENDER_SOURCE_ONLY,
        gateway_section="none",
        cooldown_policy="not_injected",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
    LAYER_ARCHIVE: MemoryLayerPolicy(
        layer=LAYER_ARCHIVE,
        direct_seed_policy=DIRECT_EXPLICIT,
        render_policy=RENDER_SUMMARY,
        gateway_section="explicit_lookup_only",
        cooldown_policy="sinks",
        diffusion_policy=DIFFUSE_NEVER,
        preserves_original=True,
    ),
}


def policy_for_layer(layer: str) -> MemoryLayerPolicy:
    return LAYER_POLICIES.get(str(layer or ""), LAYER_POLICIES[LAYER_DYNAMIC])


def infer_bucket_layer(bucket: dict[str, Any] | None) -> str:
    bucket = bucket if isinstance(bucket, dict) else {}
    meta = _metadata(bucket)
    tags = _tags(meta)
    bucket_type = _lower(meta.get("type") or meta.get("bucket_type"))

    if _truthy(meta.get("archived")) or _truthy(meta.get("digested")) or _truthy(meta.get("resolved")):
        return LAYER_ARCHIVE
    if bucket_type == "archived":
        return LAYER_ARCHIVE
    if bucket_type in {"source", "raw", "chat_log", "diary_source"} or tags & RAW_SOURCE_TAGS:
        return LAYER_SOURCE_RECORD
    if bucket_type == "dream" or "dream" in tags or "night_dream" in tags:
        return LAYER_DREAM
    if bucket_type == "feel" and tags & RELATIONSHIP_WEATHER_TAGS:
        return LAYER_RELATIONSHIP_WEATHER
    if _truthy(meta.get("pinned")) or _truthy(meta.get("protected")) or bucket_type == "permanent":
        return LAYER_CORE
    if _truthy(meta.get("anchor")) or _truthy(meta.get("bucket_anchor")):
        return LAYER_ANCHOR
    if _has_favorite_tag(tags):
        return LAYER_FAVORITE
    if bucket_type == "feel":
        return LAYER_AFFECT_CONTEXT
    return LAYER_DYNAMIC


def policy_for_bucket(bucket: dict[str, Any] | None) -> MemoryLayerPolicy:
    return policy_for_layer(infer_bucket_layer(bucket))


def infer_moment_layer(moment: dict[str, Any] | None) -> str:
    moment = moment if isinstance(moment, dict) else {}
    section = _lower(moment.get("section"))
    if section in CONTEXT_ONLY_SECTIONS:
        return LAYER_AFFECT_CONTEXT
    return infer_bucket_layer({"metadata": _moment_metadata(moment), "id": moment.get("bucket_id")})


def policy_for_moment(moment: dict[str, Any] | None) -> MemoryLayerPolicy:
    return policy_for_layer(infer_moment_layer(moment))


def can_moment_be_direct_seed(moment: dict[str, Any] | None, *, explicit_lookup: bool = False) -> bool:
    policy = policy_for_moment(moment)
    if policy.direct_seed_policy == DIRECT_NEVER:
        return False
    if policy.direct_seed_policy == DIRECT_EXPLICIT:
        return bool(explicit_lookup)
    return True


def can_bucket_diffuse(bucket: dict[str, Any] | None) -> bool:
    return policy_for_bucket(bucket).can_diffuse


def is_context_only_section(section: object) -> bool:
    return _lower(section) in CONTEXT_ONLY_SECTIONS


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return meta


def _moment_metadata(moment: dict[str, Any]) -> dict[str, Any]:
    meta = _metadata(moment)
    mapped = dict(meta)
    if "bucket_type" in meta and "type" not in mapped:
        mapped["type"] = meta.get("bucket_type")
    if "bucket_anchor" in meta and "anchor" not in mapped:
        mapped["anchor"] = meta.get("bucket_anchor")
    if "bucket_pinned" in meta and "pinned" not in mapped:
        mapped["pinned"] = meta.get("bucket_pinned")
    if "bucket_protected" in meta and "protected" not in mapped:
        mapped["protected"] = meta.get("bucket_protected")
    if "bucket_favorite_tags" in meta and "tags" not in mapped:
        mapped["tags"] = meta.get("bucket_favorite_tags")
    if meta.get("bucket_favorite") and "tags" not in mapped:
        mapped["tags"] = [FAVORITE_TAG]
    return mapped


def _tags(meta: dict[str, Any]) -> set[str]:
    raw = meta.get("tags") or meta.get("bucket_tags") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, (list, tuple, set)):
        return set()
    return {_lower(tag) for tag in raw if str(tag or "").strip()}


def _has_favorite_tag(tags: set[str]) -> bool:
    return FAVORITE_TAG in tags or any(tag.startswith(FAVORITE_PREFIX) for tag in tags)


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lower(value: object) -> str:
    return str(value or "").strip().lower()
