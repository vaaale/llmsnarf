from pathlib import Path

from pydantic import AliasChoices
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llmproxy_config_path: Path = Field(
        default=Path("./llmproxy.yaml"),
        validation_alias=AliasChoices("LLMPROXY_CONFIG_PATH"),
    )
    logs_dir: Path | None = Field(default=None, validation_alias=AliasChoices("LOGS_DIR"))
    trace_dir: Path | None = Field(default=None, validation_alias=AliasChoices("TRACE_DIR"))

    host: str | None = Field(default=None, validation_alias=AliasChoices("HOST", "LISTEN_HOST"))
    port: int | None = Field(default=None, validation_alias=AliasChoices("PORT", "LISTEN_PORT"))
