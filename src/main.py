from openaiproxy.api.app import create_app
from openaiproxy.bootstrap import resolve_listen_host_port, resolve_logs_trace_dirs
from openaiproxy.settings import Settings
from openaiproxy.yaml.yaml_repository import YAMLLLMProxyConfigRepository

settings = Settings()

config_repository = YAMLLLMProxyConfigRepository(settings.llmproxy_config_path)
logs_dir, trace_dir = resolve_logs_trace_dirs(settings, config_repository)
logs_dir.mkdir(parents=True, exist_ok=True)
trace_dir.mkdir(parents=True, exist_ok=True)

app = create_app(config_repository=config_repository, logs_dir=logs_dir, trace_dir=trace_dir)


if __name__ == "__main__":
    import uvicorn

    listen_host, listen_port = resolve_listen_host_port(settings, config_repository)
    uvicorn.run(
        app,
        host=listen_host,
        port=listen_port,
        reload=False
    )
