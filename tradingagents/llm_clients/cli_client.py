"""Chat models served by a locally installed, subscription-authenticated CLI.

``codex-cli`` runs ``codex exec`` and ``claude-cli`` runs ``claude -p``; neither
needs an API key, since each CLI uses its own login. Every call is one
non-interactive CLI process in a fresh empty directory, with the conversation
written out as labelled text on stdin.

Neither CLI takes LangChain tools, so tool calling is emulated: the bound tools
are described in the prompt and the CLI is held to a JSON schema (``content``
plus ``tool_calls`` with the arguments as a JSON string) through its own
structured-output option, ``codex exec --output-schema`` or ``claude
--json-schema``. The reply comes back as an ``AIMessage`` with real
``tool_calls``, so the analysts' tool loop (``bind_tools`` + ``ToolNode``) and
``with_structured_output`` run unchanged.

The CLIs are coding agents with tools of their own. Each run is kept from
them as far as their flags allow (read-only sandbox, empty working directory,
no user config, web search and hooks off for codex; no tools, MCP servers or
setting sources for claude), and the prompt forbids them, but codex still
offers a read-only shell inside the empty directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from .base_client import BaseLLMClient
from .cli_process import CLIOutputLimitError, run_cli_subprocess
from .validators import validate_model

CLI_PROVIDERS = {"codex-cli": "codex", "claude-cli": "claude"}

_NO_OWN_TOOLS = (
    "You are running as a text-generation backend. Do not use any tools of your own "
    "(shell, files, web search, sub-agents): everything you need is in this message."
)

_TOOL_PROTOCOL = """[Tools]
You cannot run the tools below yourself. To use them, list the calls in "tool_calls" \
and the caller runs them and sends back their results in a later message.
Reply with a JSON object holding "content" and "tool_calls":
- To call tools: one entry per call, {{"name": <tool name>, "arguments_json": <the \
arguments as a JSON object string matching the tool's parameters>}}. Several \
independent calls may go in one reply. "content" may be empty.
- To finish: "tool_calls" is [] and "content" is your complete final answer.
{requirement}
Available tools:
{tools}"""

_TRANSIENT = ("429", "500", "502", "503", "529", "rate limit", "overloaded", "temporarily",
              "service unavailable", "connection reset", "timed out", "stream disconnected")

_SLOTS_GUARD = threading.Lock()
_SLOTS: dict[int, threading.BoundedSemaphore] = {}


def _slots(limit: int) -> threading.BoundedSemaphore:
    """The process-wide pool of concurrent CLI runs, shared by both model tiers."""
    with _SLOTS_GUARD:
        return _SLOTS.setdefault(limit, threading.BoundedSemaphore(limit))


class CLIReplyError(RuntimeError):
    """The CLI answered, but not in the shape the call asked for."""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(p for p in parts if p)


def messages_to_prompt(messages: list[BaseMessage]) -> str:
    """The conversation as labelled plain text, tool calls and results included."""
    parts = [f"[Operating note]\n{_NO_OWN_TOOLS}"]
    for message in messages:
        text = _text(message.content)
        if isinstance(message, ToolMessage):
            parts.append(f"[Tool result: {message.name or 'tool'} (call {message.tool_call_id})]\n{text}")
        elif isinstance(message, AIMessage):
            if text:
                parts.append(f"[Assistant]\n{text}")
            if message.tool_calls:
                calls = "\n".join(f"- {c['name']} {json.dumps(c['args'], default=str, ensure_ascii=False)}"
                                  f" (call {c['id']})" for c in message.tool_calls)
                parts.append(f"[Assistant tool calls]\n{calls}")
        elif message.type == "system":
            parts.append(f"[System instructions]\n{text}")
        else:
            parts.append(f"[User]\n{text}")
    return "\n\n".join(parts)


def _tool_names(tools: list[dict], tool_choice) -> tuple[list[str], bool]:
    """The callable tool names and whether a call is required, from ``tool_choice``."""
    names = [t["function"]["name"] for t in tools]
    if isinstance(tool_choice, dict):
        tool_choice = tool_choice.get("function", {}).get("name") or tool_choice.get("name")
    if tool_choice in (None, "auto"):
        return names, False
    if tool_choice in ("any", "required", True):
        return names, True
    if tool_choice in names:
        return [tool_choice], True
    raise ValueError(f"tool_choice {tool_choice!r} names no bound tool")


def tool_reply_schema(names: list[str]) -> dict:
    """The JSON schema the CLI's final message is held to while tools are bound."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "tool_calls"],
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments_json"],
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "arguments_json": {"type": "string"},
                    },
                },
            },
        },
    }


def _tool_section(tools: list[dict], names: list[str], required: bool) -> str:
    listed = [t["function"] for t in tools if t["function"]["name"] in names]
    described = "\n".join(
        f"- {f['name']}: {f.get('description', '').strip()}\n"
        f"  parameters: {json.dumps(f.get('parameters', {}), ensure_ascii=False)}"
        for f in listed
    )
    requirement = ("You must call at least one tool in this reply."
                   if required else "Call tools while you still need data; finish once you have enough.")
    return _TOOL_PROTOCOL.format(requirement=requirement, tools=described)


def parse_tool_reply(reply, names: list[str], required: bool) -> AIMessage:
    """An ``AIMessage`` with real ``tool_calls`` from the CLI's JSON reply."""
    try:
        data = json.loads(reply) if isinstance(reply, str) else reply
        raw_calls = data["tool_calls"]
        content = data.get("content") or ""
    except (TypeError, KeyError, json.JSONDecodeError) as exc:
        raise CLIReplyError(f"CLI reply is not the tool-call JSON object: {exc}") from exc
    calls = []
    for raw in raw_calls:
        name = raw.get("name") if isinstance(raw, dict) else None
        if name not in names:
            raise CLIReplyError(f"CLI called an unknown tool {name!r}")
        try:
            args = json.loads(raw.get("arguments_json") or "{}")
        except json.JSONDecodeError as exc:
            raise CLIReplyError(f"arguments for {name} are not JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise CLIReplyError(f"arguments for {name} are not a JSON object")
        calls.append({"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:24]}",
                      "type": "tool_call"})
    if required and not calls:
        raise CLIReplyError("CLI was required to call a tool and called none")
    return AIMessage(content=content, tool_calls=calls)


class CLIChatModel(BaseChatModel):
    """A LangChain chat model that runs one CLI process per call."""

    backend: Literal["codex", "claude"]
    model: str
    # Executable to run; None looks the backend's name up on PATH.
    command: str | None = None
    # codex: model_reasoning_effort; claude: --effort. None keeps the CLI default.
    effort: str | None = None
    timeout: float = Field(default=600, gt=0)
    # Retries after the first attempt, for transient failures, timeouts and malformed replies.
    max_retries: int = Field(default=2, ge=0)
    max_concurrency: int = Field(default=2, ge=1)
    backoff_seconds: float = Field(default=2.0, ge=0)
    max_output_bytes: int = Field(default=4_000_000, ge=1)

    @property
    def _llm_type(self) -> str:
        return f"{self.backend}-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "effort": self.effort}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        formatted = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, tools=None, tool_choice=None,
                  **kwargs) -> ChatResult:
        kwargs.pop("ls_structured_output_format", None)
        if stop:
            raise ValueError("CLI models do not support stop sequences")
        if kwargs:
            raise ValueError(f"Unsupported CLI call options: {sorted(kwargs)}")
        prompt = messages_to_prompt(messages)
        names, required, schema = [], False, None
        if tools and tool_choice != "none":
            names, required = _tool_names(tools, tool_choice)
            schema = tool_reply_schema(names)
            prompt = f"{prompt}\n\n{_tool_section(tools, names, required)}"
        message = self._call_with_retries(prompt, schema, names, required)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _call_with_retries(self, prompt, schema, names, required) -> AIMessage:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
            try:
                reply = self._run_once(prompt, schema)
                if schema is None:
                    return AIMessage(content=reply)
                return parse_tool_reply(reply, names, required)
            except (CLIReplyError, subprocess.TimeoutExpired) as exc:
                last_error = exc
            except RuntimeError as exc:
                # codex echoes the prompt on stderr; only the tail holds the error.
                if not any(token in str(exc)[-300:].lower() for token in _TRANSIENT):
                    raise
                last_error = exc
        raise RuntimeError(f"{self.backend} CLI failed after {self.max_retries + 1} attempts: "
                           f"{last_error}") from last_error

    def _run_once(self, prompt: str, schema: dict | None):
        """One CLI run: the final text, or the parsed JSON object when ``schema`` is set."""
        with _slots(self.max_concurrency), tempfile.TemporaryDirectory(prefix="tradingagents-cli-") as cwd:
            if self.backend == "codex":
                output_path = os.path.join(cwd, "final.txt")
                cmd = self._codex_command(cwd, output_path, schema)
            else:
                output_path = None
                cmd = self._claude_command(schema)
            try:
                result = run_cli_subprocess(cmd, input=prompt, timeout=self.timeout, cwd=cwd,
                                            max_output_bytes=self.max_output_bytes,
                                            watched_output_path=output_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"'{cmd[0]}' was not found; install the {self.backend} CLI and log in to it"
                ) from exc
            except CLIOutputLimitError as exc:
                raise CLIReplyError(str(exc)) from exc
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "<no output>").strip()
                raise RuntimeError(f"{self.backend} CLI exited {result.returncode}: {detail[-800:]}")
            if self.backend == "codex":
                return self._codex_reply(output_path, schema)
            return self._claude_reply(result.stdout, schema)

    def _codex_command(self, cwd: str, output_path: str, schema: dict | None) -> list[str]:
        cmd = [self.command or "codex", "exec", "--model", self.model,
               "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
               "--ignore-user-config", "--ignore-rules", "--disable", "hooks",
               "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
               "--color", "never", "--output-last-message", output_path]
        if self.effort:
            cmd += ["-c", f"model_reasoning_effort={json.dumps(self.effort)}"]
        if schema is not None:
            schema_path = os.path.join(cwd, "reply-schema.json")
            with open(schema_path, "w", encoding="utf-8") as handle:
                json.dump(schema, handle)
            cmd += ["--output-schema", schema_path]
        return [*cmd, "-"]

    def _codex_reply(self, output_path: str, schema: dict | None):
        try:
            with open(output_path, encoding="utf-8") as handle:
                text = handle.read().strip()
        except FileNotFoundError as exc:
            raise CLIReplyError("codex wrote no final message") from exc
        if not text:
            raise CLIReplyError("codex returned an empty final message")
        return text if schema is None else _json_object(text)

    def _claude_command(self, schema: dict | None) -> list[str]:
        cmd = [self.command or "claude", "-p", "--model", self.model, "--output-format", "json",
               "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--no-session-persistence", "--disable-slash-commands"]
        if self.effort:
            cmd += ["--effort", self.effort]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(schema)]
        return cmd

    def _claude_reply(self, stdout: str, schema: dict | None):
        envelope = _json_object(stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude CLI reported an error: {str(envelope.get('result'))[:800]}")
        if schema is None:
            text = (envelope.get("result") or "").strip()
            if not text:
                raise CLIReplyError("claude returned an empty result")
            return text
        structured = envelope.get("structured_output")
        return structured if isinstance(structured, dict) else _json_object(envelope.get("result") or "")


def _json_object(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLIReplyError(f"CLI output is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CLIReplyError("CLI output is not a JSON object")
    return data


_PASSTHROUGH_KWARGS = ("timeout", "max_retries", "callbacks", "effort", "max_concurrency")
# Forwarded to every provider by build_llm_kwargs, but neither CLI takes them.
_UNSUPPORTED_KWARGS = ("temperature", "max_tokens")


class CLIClient(BaseLLMClient):
    """Client for the ``codex-cli`` and ``claude-cli`` providers."""

    def __init__(self, model: str, base_url: str | None = None, provider: str = "codex-cli",
                 **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        if self.base_url:
            raise ValueError(f"{self.provider} takes no backend URL; unset TRADINGAGENTS_LLM_BACKEND_URL "
                             "and the tier backend URLs")
        self.warn_if_unknown_model()
        ignored = [key for key in _UNSUPPORTED_KWARGS if self.kwargs.get(key) is not None]
        if ignored:
            warnings.warn(f"{self.provider} ignores {', '.join(ignored)}", RuntimeWarning, stacklevel=2)
        options = {key: self.kwargs[key] for key in _PASSTHROUGH_KWARGS if self.kwargs.get(key) is not None}
        return CLIChatModel(backend=CLI_PROVIDERS[self.provider], model=self.model, **options)

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
