from pathlib import Path

from openaiproxy.models.models import EndpointConfig, LLMProxyConfig
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


def _make_config() -> LLMProxyConfig:
    return LLMProxyConfig(
        host="0.0.0.0",
        port=8000,
        logs_dir="./logs",
        trace_dir="./traces",
        endpoints=[
            EndpointConfig(
                name="default",
                base_url="http://localhost:9000/v1",
                api_key="sk-123",
                models=["*"],
                aliases={},
                log=True,
                substitute_role={},
                enabled=True,
            ),
            EndpointConfig(
                name="secondary",
                base_url="http://localhost:9001/v1",
                api_key="",
                models=["gpt-5.1", "qwen3"],
                aliases={"opus": "qwen3"},
                log=False,
                substitute_role={"developer": "system"},
                enabled=False,
            ),
        ],
    )


def test_save_and_load_round_trip(tmp_path: Path):
    config_path = tmp_path / "llmproxy.yaml"
    repository = YAMLLLMProxyConfigRepository(config_path)

    repository.save(_make_config())
    loaded = repository.load()

    assert loaded.host == "0.0.0.0"
    assert loaded.port == 8000
    assert len(loaded.endpoints) == 2

    default = loaded.endpoints[0]
    assert default.name == "default"
    assert default.models == ["*"]
    assert default.enabled is True
    assert default.log is True

    secondary = loaded.endpoints[1]
    assert secondary.name == "secondary"
    assert secondary.aliases == {"opus": "qwen3"}
    assert secondary.substitute_role == {"developer": "system"}
    assert secondary.enabled is False
    assert secondary.log is False
    assert secondary.api_key == ""


def test_load_missing_file_returns_empty_config(tmp_path: Path):
    repository = YAMLLLMProxyConfigRepository(tmp_path / "missing.yaml")
    config = repository.load()
    assert config.endpoints == []


def test_enabled_defaults_to_true(tmp_path: Path):
    config_path = tmp_path / "llmproxy.yaml"
    config_path.write_text(
        "llmproxy:\n"
        "  endpoints:\n"
        "    ep:\n"
        "      base_url: http://x/v1\n"
        "      models: ['*']\n"
    )
    repository = YAMLLLMProxyConfigRepository(config_path)
    config = repository.load()
    assert config.endpoints[0].enabled is True
