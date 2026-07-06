from pathlib import Path

import pytest

from openaiproxy.models.models import EndpointConfig
from openaiproxy.services.config_service import ConfigService, ConfigValidationError
from openaiproxy.services.trace_service import assemble_stream_content
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


def _endpoint(name: str, **overrides) -> EndpointConfig:
    defaults = dict(
        name=name,
        base_url="http://localhost:9000/v1",
        api_key="sk-123",
        models=["gpt-5.1"],
        aliases={},
        log=True,
        substitute_role={},
        enabled=True,
    )
    defaults.update(overrides)
    return EndpointConfig(**defaults)


@pytest.fixture
def config_service(tmp_path: Path) -> ConfigService:
    repository = YAMLLLMProxyConfigRepository(tmp_path / "llmproxy.yaml")
    return ConfigService(repository)


def test_update_endpoints_persists(config_service: ConfigService):
    updated = config_service.update_endpoints([_endpoint("openai")])
    assert [ep.name for ep in updated.endpoints] == ["openai"]
    assert config_service.get_config().endpoints[0].api_key == "sk-123"


def test_update_keeps_existing_api_key_when_empty(config_service: ConfigService):
    config_service.update_endpoints([_endpoint("openai", api_key="sk-secret")])
    config_service.update_endpoints([_endpoint("openai", api_key="")])
    assert config_service.get_config().endpoints[0].api_key == "sk-secret"


def test_update_rejects_duplicate_names(config_service: ConfigService):
    with pytest.raises(ConfigValidationError):
        config_service.update_endpoints([_endpoint("a"), _endpoint("a")])


def test_update_rejects_missing_models(config_service: ConfigService):
    with pytest.raises(ConfigValidationError):
        config_service.update_endpoints([_endpoint("a", models=[])])


def test_resolve_route_prefers_exact_match_over_wildcard(config_service: ConfigService):
    config_service.update_endpoints(
        [
            _endpoint("fallback", models=["*"]),
            _endpoint("openai", models=["gpt-5.1"]),
        ]
    )
    resolved = config_service.resolve_route("gpt-5.1")
    assert resolved is not None
    assert resolved.name == "openai"

    wildcard = config_service.resolve_route("unknown-model")
    assert wildcard is not None
    assert wildcard.name == "fallback"


def test_resolve_route_skips_disabled(config_service: ConfigService):
    config_service.update_endpoints([_endpoint("openai", enabled=False)])
    assert config_service.resolve_route("gpt-5.1") is None


def test_assemble_stream_content():
    chunks = [
        'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\ndata: [DONE]\n\n',
    ]
    assert assemble_stream_content(chunks) == "Hello"


def test_assemble_stream_content_ignores_malformed():
    assert assemble_stream_content(["data: not-json\n\n", "garbage"]) == ""
