from __future__ import annotations

from abc import ABC, abstractmethod

from openaiproxy.models.models import LLMProxyConfig


class LLMProxyConfigRepository(ABC):
    @abstractmethod
    def load(self) -> LLMProxyConfig:
        raise NotImplementedError

    @abstractmethod
    def save(self, config: LLMProxyConfig) -> None:
        raise NotImplementedError
