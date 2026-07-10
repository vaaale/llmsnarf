from pathlib import Path

from pydantic import AliasChoices
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_CONFIG_FILENAME = "llmsnarf.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llmsnarf_config: Path = Field(
        default=Path("./config"),
        validation_alias=AliasChoices("LLMSNARF_CONFIG"),
    )
    llmproxy_config_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LLMPROXY_CONFIG_PATH"),
    )
    logs_dir: Path | None = Field(default=None, validation_alias=AliasChoices("LOGS_DIR"))
    trace_dir: Path | None = Field(default=None, validation_alias=AliasChoices("TRACE_DIR"))
    frontend_dir: Path = Field(default=Path("./frontend"), validation_alias=AliasChoices("FRONTEND_DIR"))

    host: str | None = Field(default=None, validation_alias=AliasChoices("HOST", "LISTEN_HOST"))
    port: int | None = Field(default=None, validation_alias=AliasChoices("PORT", "LISTEN_PORT"))

    @property
    def config_file_path(self) -> Path:
        if self.llmproxy_config_path is not None:
            return self.llmproxy_config_path
        return self.llmsnarf_config / _CONFIG_FILENAME
