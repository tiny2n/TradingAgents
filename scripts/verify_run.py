"""Check a saved analysis against the data its agents were given.

Replays the tool calls of the ticker's latest run on the date (from its
``message_tool.log``), fetches the sentiment analyst's pre-fetched inputs the
same way the analyst does, then pulls every dollar amount, price and percentage
out of the saved reports and matches it against the numbers in that data.

What it prints is what needs a human look: figures that are not literally in
the data. Most are derived (a margin, a growth rate, entry price minus ATR) and
need recomputing; anything left after that is unsupported.

Limits: StockTwits and Reddit are fetched live, so they can differ from what the
run saw; the check matches the data the tools serve, not an outside source.

Run from the repository root:

    .venv/bin/python scripts/verify_run.py AAPL 2026-10-08
    .venv/bin/python scripts/verify_run.py AAPL 2026-10-08 --report results/reports/AAPL_20261009_141038
"""

from __future__ import annotations

import argparse
import collections
import glob
import inspect
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from tradingagents.agents import tools
from tradingagents.agents.post_screen import jev_screen
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors.reddit import fetch_reddit_posts, subreddits_for
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages
from tradingagents.default_config import DEFAULT_CONFIG

# "1,234억 5,678만 달러", "-3억 달러", "마이너스 10억 9,200만 달러": Korean amounts in USD.
_AMOUNT = re.compile(r"(마이너스\s*|-)?(?:(\d[\d,]*)조\s*)?(?:(\d[\d,]*)억)?\s*(?:(\d[\d,]*)만)?\s*달러")
# "$331.8B", "$1.2T", "$67B": a scaled dollar figure, compared in USD millions.
_SCALED = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s?([MBT])\b")
_SCALE = {"M": 1, "B": 1_000, "T": 1_000_000}
# "$522.61", "522.61달러": a price or per-share figure.
_PRICE = re.compile(r"\$\s?(\d[\d,]*\.\d+)(?!\d|\.\d|\s?[MBT]\b)|(\d[\d,]*\.\d+)\s?달러")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def _half_unit(figure: str) -> float:
    """Half the last printed digit: "331.8" is anything that rounds to it, within 0.05."""
    decimals = len(figure.split(".")[1]) if "." in figure else 0
    return 0.5 * 10 ** -decimals


def replay_tool_calls(ticker: str, day: str, results_dir: Path, out: Path) -> list[str]:
    """Rerun the latest run's tool calls; returns the names of calls that failed."""
    log = results_dir / ticker / day / "message_tool.log"
    lines = log.read_text().splitlines()
    # The log is appended to by every run on the date: take the last one.
    start = max(i for i, line in enumerate(lines) if f"Selected ticker: {ticker}" in line)
    calls = re.findall(r"\[Tool Call\] (\w+)\((.*)\)", "\n".join(lines[start:]))
    failed = []
    for i, (name, argstr) in enumerate(calls):
        func = getattr(tools, name).func
        params = inspect.signature(func).parameters
        kwargs = {}
        for part in filter(None, argstr.split(", ")):
            key, value = part.split("=", 1)
            kwargs[key] = int(value) if "int" in str(params[key].annotation) else value
        # The graph injects the instrument and the run date; tools name the instrument either way.
        for key in ("symbol", "ticker"):
            if key in params and key not in kwargs:
                kwargs[key] = ticker
        if "trade_date" in params:
            kwargs["trade_date"] = day
        try:
            result = str(func(**kwargs))
        except Exception as exc:  # recorded and reported, not fatal
            result = f"ERROR {exc!r}"
            failed.append(name)
        tag = name + "".join(f"_{kwargs[k]}".replace(" ", "_") for k in ("indicator", "topic", "freq") if k in kwargs)
        (out / f"{i:02d}_{tag}.txt").write_text(result)
    print(f"replayed {len(calls)} tool calls from {log}")
    return failed


def fetch_sentiment_inputs(ticker: str, day: str, out: Path) -> None:
    """What the sentiment analyst fetches before calling the model (it calls no tools)."""
    week_ago = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    screen = jev_screen(ticker)
    (out / "s1_news.txt").write_text(tools.get_news.func(ticker, week_ago, day))
    (out / "s2_stocktwits.txt").write_text(
        fetch_stocktwits_messages(ticker, limit=30, start_date=week_ago, end_date=day, screen=screen))
    (out / "s3_reddit.txt").write_text(
        fetch_reddit_posts(ticker, subreddits_for(ticker), start_date=week_ago, end_date=day, screen=screen))


def data_numbers(out: Path) -> set[float]:
    source = " ".join(path.read_text() for path in out.glob("*.txt"))
    # CSV rows join their cells with commas, so split on them; prose numbers may
    # carry thousands separators, so read those whole as well.
    numbers = {float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", source.replace(",", " "))}
    numbers |= {float(t.replace(",", "")) for t in re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", source)}
    return numbers


def check_reports(report_dir: Path, numbers: set[float]) -> None:
    def near(value: float, tol: float) -> bool:
        return any(abs(value - n) <= tol for n in numbers)

    counts: collections.Counter = collections.Counter()
    unmatched: dict[str, list[tuple[str, str, str]]] = collections.defaultdict(list)
    for path in sorted(report_dir.glob("*/*.md")):
        text = path.read_text()
        name = "/".join(path.parts[-2:])

        def context(match, text=text):
            return text[max(0, match.start() - 90):match.end() + 20].replace("\n", " ")

        for m in _AMOUNT.finditer(text):
            jo, eok, man = (float(x.replace(",", "")) if x else 0.0 for x in m.group(2, 3, 4))
            if not (jo or eok or man):
                continue
            millions = jo * 1e6 + eok * 100 + man * 0.01
            # Half the smallest unit the report used: 억 is 100 million, 만 is 0.01 million.
            tol = 0.5 if man else 50
            counts["amount"] += 1
            if not (near(millions, tol) or near(-millions, tol)):
                unmatched["amount"].append((name, m.group(0).strip(), context(m)))
        # Each figure matches anything that rounds to it at the precision it was printed with.
        for m in _SCALED.finditer(text):
            figure, scale = m.group(1).replace(",", ""), _SCALE[m.group(2)]
            counts["scaled $"] += 1
            if not near(float(figure) * scale, _half_unit(figure) * scale):
                unmatched["scaled $"].append((name, m.group(0), context(m)))
        for m in _PRICE.finditer(text):
            figure = (m.group(1) or m.group(2)).replace(",", "")
            counts["price"] += 1
            if not near(float(figure), _half_unit(figure)):
                unmatched["price"].append((name, m.group(0), context(m)))
        for m in _PERCENT.finditer(text):
            counts["percent"] += 1
            if not near(float(m.group(1)), _half_unit(m.group(1))):
                unmatched["percent"].append((name, m.group(0), context(m)))

    print(f"numbers checked in {report_dir}: {dict(counts)}")
    for kind in ("amount", "scaled $", "price", "percent"):
        seen = set()
        items = [item for item in unmatched[kind] if (item[0], item[1]) not in seen and not seen.add((item[0], item[1]))]
        print(f"\n### {kind}: {len(items)} not literally in the data (recompute or trace each)")
        for report, figure, ctx in items:
            print(f"  {report} | {figure} | {ctx}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ticker")
    parser.add_argument("date", help="the run's analysis date, YYYY-MM-DD")
    parser.add_argument("--report", type=Path, help="saved report folder; default: the ticker's latest")
    parser.add_argument("--out", type=Path, help="where to keep the refetched data; default: a temporary folder")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    set_config(DEFAULT_CONFIG)
    results_dir = Path(DEFAULT_CONFIG["results_dir"])
    report_dir = args.report or Path(max(glob.glob(str(results_dir / "reports" / f"{ticker}_*"))))
    out = args.out or Path(tempfile.mkdtemp(prefix=f"verify-{ticker}-"))
    out.mkdir(parents=True, exist_ok=True)

    failed = replay_tool_calls(ticker, args.date, results_dir, out)
    if failed:
        print(f"tool calls that failed to replay: {failed}")
    fetch_sentiment_inputs(ticker, args.date, out)
    print(f"refetched data kept in {out}")
    check_reports(report_dir, data_numbers(out))


if __name__ == "__main__":
    main()
