from __future__ import annotations

import json as _json
import logging
import re as _re

import httpx

from openaiproxy.interface.repository import LLMProxyConfigRepository
from openaiproxy.models.models import WebFetchConfig, WebSearchConfig

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_FETCH_TOOL_NAME = "web_fetch"

WEB_SEARCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": WEB_SEARCH_TOOL_NAME,
        "description": (
            "Search the web using multiple search engines simultaneously. "
            "Returns ranked results with titles, snippets, and URLs. "
            "Optionally includes extracted page content for the top results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
}

WEB_FETCH_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": WEB_FETCH_TOOL_NAME,
        "description": (
            "Fetch and extract the full content of a single web page by URL. "
            "Use this to read a specific page in full — for example after finding a relevant URL via web_search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch.",
                }
            },
            "required": ["url"],
        },
    },
}


def _extract_sources(text: str, fmt: str) -> list[dict]:
    if fmt == "ndjson":
        sources: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if isinstance(obj, dict) and "url" in obj:
                    entry: dict = {"url": obj["url"]}
                    if obj.get("title"):
                        entry["title"] = obj["title"]
                    sources.append(entry)
            except Exception:
                pass
        return sources
    if fmt == "json":
        try:
            data = _json.loads(text)
            if isinstance(data, list):
                result: list[dict] = []
                for item in data:
                    if isinstance(item, dict) and "url" in item:
                        entry = {"url": item["url"]}
                        if item.get("title"):
                            entry["title"] = item["title"]
                        result.append(entry)
                return result
        except Exception:
            pass
    # markdown / text: extract [title](url) links, deduplicated
    seen: set[str] = set()
    sources = []
    for title, url in _re.findall(r'\[([^\]]*)\]\((https?://[^)]+)\)', text):
        if url not in seen:
            seen.add(url)
            entry = {"url": url}
            if title:
                entry["title"] = title
            sources.append(entry)
    return sources


class WebSearchService:
    def __init__(self, config_repository: LLMProxyConfigRepository, logger: logging.Logger):
        self._config_repository = config_repository
        self._logger = logger

    def get_search_config(self) -> WebSearchConfig:
        return self._config_repository.load().web_search

    def get_fetch_config(self) -> WebFetchConfig:
        return self._config_repository.load().web_fetch

    def get_config(self) -> WebSearchConfig:
        return self.get_search_config()

    async def search(self, query: str) -> tuple[str, list[dict]]:
        config = self.get_search_config()
        params: dict = {
            "text": query,
            "format": config.format,
            "limit": config.limit,
            "filter": config.filter,
            "extract": config.extract,
            "extract_mode": config.extract_mode,
            "mode": config.mode,
        }
        if config.engines:
            params["engines"] = ",".join(config.engines)
        url = f"{config.base_url}/mega/search"
        self._logger.info("web_search query=%r url=%s", query, url)
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.get(url, params=params)
        except Exception as exc:
            self._logger.warning("web_search failed query=%r error=%s", query, exc)
            return f"Web search failed: {exc}", []

        if response.status_code != 200:
            self._logger.warning(
                "web_search error query=%r status=%s body=%s", query, response.status_code, response.text[:500]
            )
            return f"Web search failed with status {response.status_code}: {response.text[:500]}", []

        content = response.text
        sources = _extract_sources(content, config.format)
        return content, sources

    async def fetch(self, url: str) -> str:
        config = self.get_fetch_config()
        params: dict = {
            "url": url,
            "format": config.format,
            "mode": config.mode,
        }
        endpoint_url = f"{config.base_url}/extract"
        self._logger.info("web_fetch url=%r endpoint=%s", url, endpoint_url)
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.get(endpoint_url, params=params)
        except Exception as exc:
            self._logger.warning("web_fetch failed url=%r error=%s", url, exc)
            return f"Web fetch failed: {exc}"

        if response.status_code != 200:
            self._logger.warning(
                "web_fetch error url=%r status=%s body=%s", url, response.status_code, response.text[:500]
            )
            return f"Web fetch failed with status {response.status_code}: {response.text[:500]}"

        return response.text
