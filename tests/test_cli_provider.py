"""codex-cli / claude-cli providers: CLI invocation, emulated tool calling, retries.

A fake executable stands in for each CLI. It records its argv and stdin, then
plays the next queued reply: for codex it writes the reply to the
--output-last-message file, for claude it prints the JSON envelope.
"""

import json
import os
import shutil
import stat
import sys
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients import build_llm_kwargs, create_llm_client, create_tier_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.cli_client import (
    CLIChatModel,
    CLIClient,
    messages_to_prompt,
    parse_tool_reply,
    reply_language_matches,
)

FAKE_CLI = """#!{python}
import json, os, sys, time
state = os.environ["FAKE_CLI_DIR"]
calls = sorted(n for n in os.listdir(state) if n.startswith("call-"))
n = len(calls)
schema = None
if "--output-schema" in sys.argv:
    with open(sys.argv[sys.argv.index("--output-schema") + 1]) as f:
        schema = json.load(f)
with open(os.path.join(state, "call-%03d.json" % n), "w") as f:
    json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "schema": schema}}, f)
with open(os.path.join(state, "replies.json")) as f:
    replies = json.load(f)
reply = replies[min(n, len(replies) - 1)]
if reply.get("sleep"):
    time.sleep(reply["sleep"])
if reply.get("exit"):
    sys.stderr.write(reply.get("stderr", ""))
    sys.exit(reply["exit"])
if "-o" in sys.argv or "--output-last-message" in sys.argv:
    flag = "--output-last-message" if "--output-last-message" in sys.argv else "-o"
    with open(sys.argv[sys.argv.index(flag) + 1], "w") as f:
        f.write(reply["text"])
else:
    sys.stdout.write(reply["text"])
"""


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    """Install a fake CLI; returns (command path, set_replies, recorded calls)."""
    state = tmp_path / "state"
    state.mkdir()
    command = tmp_path / "fake-cli"
    command.write_text(FAKE_CLI.format(python=sys.executable))
    command.chmod(command.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_CLI_DIR", str(state))

    def set_replies(*replies):
        (state / "replies.json").write_text(json.dumps(list(replies)))

    def calls():
        return [json.loads(p.read_text()) for p in sorted(state.glob("call-*.json"))]

    return str(command), set_replies, calls


def codex(command, **kwargs):
    return CLIChatModel(backend="codex", model="gpt-6-luna", command=command,
                        backoff_seconds=0, **kwargs)


def claude(command, **kwargs):
    return CLIChatModel(backend="claude", model="sonnet", command=command,
                        backoff_seconds=0, **kwargs)


def tool_reply(content="", calls=()):
    return {"text": json.dumps({"content": content, "tool_calls": [
        {"name": name, "arguments_json": json.dumps(args)} for name, args in calls]})}


@tool
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Retrieve OHLCV price data for a ticker."""
    return f"{symbol},{start_date},{end_date},100.0"


@tool
def get_news(ticker: str) -> str:
    """Retrieve recent headlines for a ticker."""
    return f"{ticker} headline"


class Verdict(BaseModel):
    rating: str
    reason: str


# ---- codex invocation -------------------------------------------------------

def test_codex_plain_call_reads_the_final_message_file(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": "  the report  "})

    result = codex(command).invoke([SystemMessage("be brief"), HumanMessage("analyze NVDA")])

    assert result.content == "the report"
    assert result.tool_calls == []
    (call,) = calls()
    argv = call["argv"]
    assert argv[0] == "exec" and argv[-1] == "-"
    assert argv[argv.index("--model") + 1] == "gpt-6-luna"
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check"):
        assert flag in argv
    assert 'web_search="disabled"' in argv
    assert "--output-schema" not in argv
    assert "[System instructions]\nbe brief" in call["stdin"]
    assert "[User]\nanalyze NVDA" in call["stdin"]
    assert "Do not use any tools of your own" in call["stdin"]


def test_codex_effort_goes_in_as_a_config_override(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": "ok"})

    codex(command, effort="high").invoke("hi")

    argv = calls()[0]["argv"]
    assert 'model_reasoning_effort="high"' in argv


def test_codex_runs_in_an_empty_directory_that_is_removed(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": "ok"})

    codex(command).invoke("hi")

    output_file = calls()[0]["argv"][calls()[0]["argv"].index("--output-last-message") + 1]
    assert os.path.basename(os.path.dirname(output_file)).startswith("tradingagents-cli-")
    assert not os.path.exists(os.path.dirname(output_file))


# ---- emulated tool calling --------------------------------------------------

AFTER_A_TOOL_RESULT = [
    HumanMessage("analyze NVDA"),
    AIMessage(content="", tool_calls=[{"name": "get_news", "args": {"ticker": "NVDA"},
                                       "id": "call_1", "type": "tool_call"}]),
    ToolMessage(content="NVDA headline", name="get_news", tool_call_id="call_1"),
]


def test_bound_tools_come_back_as_real_tool_calls(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("get_stock_data", {"symbol": "NVDA", "start_date": "2026-09-01",
                                                      "end_date": "2026-10-01"}),
                                  ("get_news", {"ticker": "NVDA"})]))

    result = codex(command).bind_tools([get_stock_data, get_news]).invoke("analyze NVDA")

    assert [c["name"] for c in result.tool_calls] == ["get_stock_data", "get_news"]
    assert result.tool_calls[0]["args"] == {"symbol": "NVDA", "start_date": "2026-09-01",
                                            "end_date": "2026-10-01"}
    assert len({c["id"] for c in result.tool_calls}) == 2
    (call,) = calls()
    assert "--output-schema" in call["argv"]
    assert "- get_stock_data: Retrieve OHLCV price data" in call["stdin"]
    assert '"symbol"' in call["stdin"]


def test_the_reply_schema_lists_only_the_bound_tool_names(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content="final"))

    codex(command).bind_tools([get_stock_data, get_news]).invoke("hi")

    schema = calls()[0]["schema"]
    item = schema["properties"]["tool_calls"]["items"]
    assert item["properties"]["name"]["enum"] == ["get_stock_data", "get_news"]
    assert schema["required"] == ["content", "tool_calls"]


def test_a_final_answer_with_tools_bound_has_no_tool_calls(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content="final report"))

    result = codex(command).bind_tools([get_news]).invoke(AFTER_A_TOOL_RESULT)

    assert result.content == "final report"
    assert result.tool_calls == []
    assert len(calls()) == 1


def test_tool_calls_and_results_reach_the_cli_as_text():
    prompt = messages_to_prompt([
        HumanMessage("analyze NVDA"),
        AIMessage(content="", tool_calls=[{"name": "get_news", "args": {"ticker": "NVDA"},
                                           "id": "call_1", "type": "tool_call"}]),
        ToolMessage(content="NVDA headline", name="get_news", tool_call_id="call_1"),
    ])

    assert '[Assistant tool calls]\n- get_news {"ticker": "NVDA"} (call call_1)' in prompt
    assert "[Tool result: get_news (call call_1)]\nNVDA headline" in prompt


def test_the_analyst_tool_loop_runs_on_the_cli(fake_cli):
    """take_turn as each analyst runs it: a call, the tool's result, the report."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    from tradingagents.agents.analysts.turn import take_turn

    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("get_news", {"ticker": "NVDA"})]),
                tool_reply(content="NVDA report"))
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Tools: {tool_names}"), MessagesPlaceholder(variable_name="messages")]
    ).partial(tool_names="get_news")
    llm = codex(command)
    messages = [HumanMessage("NVDA")]

    first, report = take_turn(prompt, llm, [get_news], messages)
    assert report == "" and first.tool_calls[0]["name"] == "get_news"
    messages += [first, get_news.invoke(first.tool_calls[0])]
    second, report = take_turn(prompt, llm, [get_news], messages)

    assert report == "NVDA report"
    assert "[Tool result: get_news" in calls()[1]["stdin"]
    assert "NVDA headline" in calls()[1]["stdin"]


def test_structured_output_rides_on_a_forced_tool_call(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("Verdict", {"rating": "Buy", "reason": "momentum"})]))

    verdict = codex(command).with_structured_output(Verdict).invoke("decide")

    assert verdict == Verdict(rating="Buy", reason="momentum")
    assert "You must call at least one tool" in calls()[0]["stdin"]


def test_a_named_tool_choice_offers_only_that_tool(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("get_news", {"ticker": "NVDA"})]))

    codex(command).bind_tools([get_stock_data, get_news], tool_choice="get_news").invoke("hi")

    stdin = calls()[0]["stdin"]
    assert "- get_news:" in stdin and "- get_stock_data:" not in stdin


def test_parse_rejects_unknown_tools_bad_arguments_and_a_missing_required_call():
    from tradingagents.llm_clients.cli_client import CLIReplyError

    def reply(calls):
        return json.dumps({"content": "", "tool_calls": calls})

    with pytest.raises(CLIReplyError, match="unknown tool"):
        parse_tool_reply(reply([{"name": "rm", "arguments_json": "{}"}]), ["get_news"], False)
    with pytest.raises(CLIReplyError, match="not JSON"):
        parse_tool_reply(reply([{"name": "get_news", "arguments_json": "{oops"}]), ["get_news"], False)
    with pytest.raises(CLIReplyError, match="not a JSON object"):
        parse_tool_reply(reply([{"name": "get_news", "arguments_json": "[1]"}]), ["get_news"], False)
    with pytest.raises(CLIReplyError, match="called none"):
        parse_tool_reply(reply([]), ["get_news"], True)
    with pytest.raises(CLIReplyError, match="not the tool-call JSON"):
        parse_tool_reply("plain text", ["get_news"], False)
    with pytest.raises(CLIReplyError, match="no tool and gave no answer"):
        parse_tool_reply(json.dumps({"content": "  ", "tool_calls": []}), ["get_news"], False)


def test_an_empty_reply_with_tools_bound_is_retried_not_filed_as_a_report(fake_cli):
    """An analyst whose first turn came back empty filed an empty report and the run went on."""
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content=""), tool_reply(calls=[("get_news", {"ticker": "TSLA"})]))

    result = codex(command, max_retries=1).bind_tools([get_news]).invoke("analyze TSLA")

    assert result.tool_calls[0]["name"] == "get_news"
    assert len(calls()) == 2


# ---- retries ----------------------------------------------------------------

def test_a_malformed_reply_is_retried(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": "not json"}, tool_reply(content="final"))

    result = codex(command, max_retries=1).bind_tools([get_news]).invoke(AFTER_A_TOOL_RESULT)

    assert result.content == "final"
    assert len(calls()) == 2


def test_a_transient_failure_is_retried(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"exit": 1, "stderr": "error: 429 Too Many Requests"}, {"text": "ok"})

    assert codex(command, max_retries=1).invoke("hi").content == "ok"
    assert len(calls()) == 2


def test_a_permanent_failure_is_not_retried(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"exit": 2, "stderr": "error: not logged in"})

    with pytest.raises(RuntimeError, match="not logged in"):
        codex(command, max_retries=2).invoke("hi")
    assert len(calls()) == 1


def test_a_timeout_kills_the_run_and_the_retries_run_out(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"sleep": 30, "text": "late"})

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        codex(command, timeout=0.5, max_retries=1).invoke("hi")
    assert time.monotonic() - started < 10
    assert len(calls()) == 2


def test_a_missing_executable_says_which_cli_to_install(tmp_path):
    with pytest.raises(FileNotFoundError, match="install the codex CLI"):
        codex(str(tmp_path / "no-such-codex")).invoke("hi")


# ---- claude backend ---------------------------------------------------------

def test_claude_plain_call_reads_the_result_field(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": json.dumps({"is_error": False, "result": "the report"})})

    assert claude(command, effort="max").invoke("hi").content == "the report"
    argv = calls()[0]["argv"]
    assert argv[:3] == ["-p", "--model", "sonnet"]
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--effort") + 1] == "max"
    assert "--strict-mcp-config" in argv and "--json-schema" not in argv


def test_claude_tool_calls_come_from_structured_output(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": json.dumps({"is_error": False, "result": "...", "structured_output": {
        "content": "", "tool_calls": [{"name": "get_news", "arguments_json": '{"ticker": "NVDA"}'}]}})})

    result = claude(command).bind_tools([get_news]).invoke("hi")

    assert result.tool_calls[0]["args"] == {"ticker": "NVDA"}
    schema = json.loads(calls()[0]["argv"][calls()[0]["argv"].index("--json-schema") + 1])
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == ["get_news"]


def test_claude_error_envelope_raises(fake_cli):
    command, set_replies, _ = fake_cli
    set_replies({"text": json.dumps({"is_error": True, "result": "Not logged in"})})

    with pytest.raises(RuntimeError, match="Not logged in"):
        claude(command).invoke("hi")


# ---- provider wiring --------------------------------------------------------

def test_factory_builds_cli_models_without_an_api_key():
    llm = create_llm_client("codex-cli", "gpt-6-sol", timeout=900, effort="high").get_llm()

    assert isinstance(llm, CLIChatModel)
    assert (llm.backend, llm.model, llm.timeout, llm.effort) == ("codex", "gpt-6-sol", 900, "high")
    assert create_llm_client("claude-cli", "opus").get_llm().backend == "claude"
    assert get_api_key_env("codex-cli") is None and get_api_key_env("claude-cli") is None


def test_config_cli_settings_reach_each_tier():
    config = {"llm_provider": "codex-cli", "quick_think_llm": "gpt-6-luna",
              "deep_think_llm": "gpt-6-sol", "cli_effort": "medium", "cli_timeout": 1800,
              "cli_max_concurrency": 3, "llm_max_retries": 4}

    assert build_llm_kwargs(config) == {"effort": "medium", "timeout": 1800,
                                        "max_concurrency": 3, "max_retries": 4}
    deep = create_tier_client(config, "deep").get_llm()
    assert (deep.model, deep.timeout, deep.max_concurrency, deep.max_retries) == (
        "gpt-6-sol", 1800, 3, 4)


def test_cli_env_overrides(monkeypatch):
    from tradingagents.default_config import build_default_config

    monkeypatch.setenv("TRADINGAGENTS_CLI_TIMEOUT", "1800")
    monkeypatch.setenv("TRADINGAGENTS_CLI_EFFORT", "high")
    monkeypatch.setenv("TRADINGAGENTS_CLI_MAX_CONCURRENCY", "4")
    config = build_default_config()

    assert (config["cli_timeout"], config["cli_effort"], config["cli_max_concurrency"]) == (
        1800, "high", 4)


def test_a_backend_url_is_refused():
    with pytest.raises(ValueError, match="takes no backend URL"):
        CLIClient("gpt-6-luna", "http://localhost:1234", provider="codex-cli").get_llm()


def test_sampling_options_the_cli_cannot_take_are_warned_about():
    with pytest.warns(RuntimeWarning, match="ignores temperature, max_tokens"):
        create_llm_client("codex-cli", "gpt-6-luna", temperature=0.0, max_tokens=100).get_llm()


def test_the_provider_menu_offers_both_clis():
    from cli.prompts import _llm_provider_table

    keys = {key: url for _, key, url in _llm_provider_table()}
    assert keys["codex-cli"] is None and keys["claude-cli"] is None


# ---- the real CLI (opt in with -m integration) ------------------------------

@pytest.mark.integration
@pytest.mark.skipif(shutil.which("codex") is None, reason="codex CLI not installed")
def test_real_codex_calls_a_bound_tool():
    llm = CLIChatModel(backend="codex", model=os.environ.get("CODEX_TEST_MODEL", "gpt-6-luna"),
                       effort="low")

    result = llm.bind_tools([get_news]).invoke(
        "Analyze NVDA. You have no data yet: call get_news for it first.")

    assert result.tool_calls and result.tool_calls[0]["name"] == "get_news"
    assert result.tool_calls[0]["args"].get("ticker") == "NVDA"


# ---- reply language ---------------------------------------------------------


KOREAN = "엔비디아의 최근 분기 매출은 크게 늘었고 영업이익률도 높은 수준을 유지했습니다. " * 4
KAZAKH = "NVDA фундаментальды талдауы: түсім өсті, операциялық маржа жоғары деңгейде сақталды. " * 4
ASK_KOREAN = [SystemMessage("Analyze NVDA. Write your entire response in Korean, except the labelled lines."),
              HumanMessage("NVDA")]


def test_the_language_check_tells_scripts_apart():
    assert reply_language_matches(KOREAN, "Korean")
    assert not reply_language_matches(KAZAKH, "Korean")
    assert not reply_language_matches("The quarter was strong and margins held up well. " * 4, "Korean")
    assert reply_language_matches("El trimestre fue fuerte y los márgenes se mantuvieron. " * 4, "Spanish")
    assert not reply_language_matches(KOREAN, "Spanish")
    assert reply_language_matches("決算は好調で、利益率も高い水準を維持しました。" * 6, "Japanese")
    assert not reply_language_matches("季度业绩强劲，利润率保持在高位。" * 8, "Japanese")


def test_short_replies_and_unknown_languages_are_not_judged():
    assert reply_language_matches("Hold", "Korean")
    assert reply_language_matches(KAZAKH, "Vietnamese")


def test_a_reply_in_the_wrong_language_is_asked_for_again(fake_cli):
    """A run asked for Korean got one analyst's report in Kazakh."""
    command, set_replies, calls = fake_cli
    set_replies({"text": KAZAKH}, {"text": KOREAN})

    result = codex(command, max_retries=1).invoke(ASK_KOREAN)

    assert result.content == KOREAN.strip()
    assert len(calls()) == 2
    assert "Write the whole answer in Korean" not in calls()[0]["stdin"]
    assert "Write the whole answer in Korean" in calls()[1]["stdin"]


def test_a_final_report_with_tools_bound_is_checked_too(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content=KAZAKH), tool_reply(content=KOREAN))

    result = codex(command, max_retries=1).bind_tools([get_news]).invoke([*ASK_KOREAN, *AFTER_A_TOOL_RESULT[1:]])

    assert result.content == KOREAN
    assert len(calls()) == 2


def test_structured_output_prose_is_checked(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("Verdict", {"rating": "Buy", "reason": KAZAKH})]),
                tool_reply(calls=[("Verdict", {"rating": "Buy", "reason": KOREAN})]))

    verdict = codex(command, max_retries=1).with_structured_output(Verdict).invoke(ASK_KOREAN)

    assert verdict.reason == KOREAN
    assert len(calls()) == 2


def test_tool_calls_are_not_judged_by_language(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(calls=[("get_news", {"ticker": "NVDA"})]))

    codex(command, max_retries=1).bind_tools([get_news]).invoke(ASK_KOREAN)

    assert len(calls()) == 1


def test_without_a_language_instruction_any_language_passes(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies({"text": KAZAKH})

    codex(command, max_retries=1).invoke([HumanMessage("Reflect on the decision in English.")])

    assert len(calls()) == 1


def test_a_wrong_language_that_persists_is_kept_with_a_warning_not_fatal(fake_cli, caplog):
    """The figures in a mis-languaged report still hold; losing the run would cost more."""
    command, set_replies, calls = fake_cli
    set_replies({"text": KAZAKH})

    with caplog.at_level("WARNING"):
        result = codex(command, max_retries=1).invoke(ASK_KOREAN)

    assert result.content == KAZAKH.strip()
    assert len(calls()) == 2
    assert "not in Korean" in caplog.text


def test_the_check_reads_the_instruction_the_agents_actually_send(monkeypatch):
    """If the wording in agents.context drifts, the check would silently switch off."""
    from tradingagents.agents import context
    from tradingagents.llm_clients.cli_client import _LANGUAGE_ASKED

    monkeypatch.setattr("tradingagents.dataflows.config.get_config", lambda: {"output_language": "Korean"})

    assert _LANGUAGE_ASKED.search(context.get_language_instruction()).group(1).strip() == "Korean"



# ---- tool use ---------------------------------------------------------------

def test_with_tools_bound_the_note_points_at_them_not_at_the_message():
    """A news analyst read "everything you need is in this message", called no
    tool, and reported that its data tools were not connected."""
    bound = messages_to_prompt([HumanMessage("analyze GOOGL")], tools_bound=True)
    plain = messages_to_prompt([HumanMessage("analyze GOOGL")])

    assert "everything you need is in this message" not in bound
    assert "[Tools]" in bound and "tool_calls" in bound
    assert "everything you need is in this message" in plain


def test_a_first_answer_that_called_no_tool_is_asked_again(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content="The data tools are not connected, so no report."),
                tool_reply(calls=[("get_news", {"ticker": "GOOGL"})]))

    result = codex(command, max_retries=1).bind_tools([get_news]).invoke("analyze GOOGL")

    assert result.tool_calls[0]["name"] == "get_news"
    assert len(calls()) == 2
    assert "you have not called any yet" not in calls()[0]["stdin"]
    assert "you have not called any yet" in calls()[1]["stdin"]


def test_an_answer_after_a_tool_result_is_not_asked_again(fake_cli):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content="final report"))

    codex(command, max_retries=1).bind_tools([get_news]).invoke(AFTER_A_TOOL_RESULT)

    assert len(calls()) == 1


def test_a_first_answer_that_never_calls_a_tool_is_kept_with_a_warning(fake_cli, caplog):
    command, set_replies, calls = fake_cli
    set_replies(tool_reply(content="no data, so Hold"))

    with caplog.at_level("WARNING"):
        result = codex(command, max_retries=1).bind_tools([get_news]).invoke("analyze GOOGL")

    assert result.content == "no data, so Hold"
    assert len(calls()) == 2
    assert "called no tool" in caplog.text
