from __future__ import annotations

from openaiproxy.bootstrap import resolve_logs_trace_dirs
from openaiproxy.api.app import create_app
from openaiproxy.settings import Settings
from openaiproxy.yaml_repository.yaml_repository import YAMLLLMProxyConfigRepository


settings = Settings()
config_repository = YAMLLLMProxyConfigRepository(settings.llmproxy_config_path)
logs_dir, trace_dir = resolve_logs_trace_dirs(settings, config_repository)
logs_dir.mkdir(parents=True, exist_ok=True)

app = create_app(config_repository=config_repository, logs_dir=logs_dir, trace_dir=trace_dir)
