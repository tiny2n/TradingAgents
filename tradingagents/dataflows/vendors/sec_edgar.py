"""Company statements as they were filed, from SEC EDGAR.

Every other fundamentals vendor serves a period's current value and cuts the
statement at the fiscal period end. That is two claims a run should not make: a
period that has ended is not public until the company files, weeks later, and a
figure that was later restated is not what investors saw at the time.

EDGAR reports every fact with the date it was filed, so a run dated ``as_of_date``
serves exactly what was on file by then, restatements included at the vintage
that was current: Apple's 2008 total assets read 39.6B until the 2010 amendment
restated them to 36.2B.

Access needs no key or account, only a User-Agent identifying the caller, which
SEC requires and refuses requests without. US filers only: anything absent from
EDGAR's ticker map falls through to the next configured vendor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import requests

from tradingagents import __version__
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.files import replace_file

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# A filing history only changes when something new is filed, so one fetch per
# company per day serves every date a run asks about.
_CACHE_TTL_SECONDS = 24 * 60 * 60

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A 10-Q's cash flows are often filed only year to date. A quarterly table takes
# the quarter where filed, else the span to date, named on its row.
_YEAR_TO_DATE = ((150, 200, 6), (240, 290, 9))

# A quarter filed only inside a year to date is still asked for, and a model doing
# the subtraction itself got it wrong (8,282 - 2,493 written as 5,735), so the
# table serves it: the year to date less the span to date one quarter earlier,
# both on file by the run date, on a row labelled derived. Per-share figures are
# not derived, since the share count differs between the spans.
_DERIVED = -1
_QUARTER_APART = (75, 105)

_FREE_CASH_FLOW = "Free Cash Flow (OCF - CapEx)"

# A fiscal year is a period an annual report covers. A 10-Q balance has no span
# to reject, and some filers' 10-Qs report twelve-month totals that pass the span
# check, so either would read as a fiscal year. The value is still the latest
# filing of any form: a recast after a split or spin-off counts from its filing.
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")


def _user_agent() -> str:
    """Who SEC sees. No account or key exists; callers identify themselves.

    www.sec.gov, which serves the ticker map, refuses a User-Agent carrying no
    contact address: a client name alone or with a project URL gets 403, one
    with an address gets 200. So the default carries a placeholder address and
    the package version. Set SEC_EDGAR_USER_AGENT to your own name and address
    so SEC can reach you about your traffic rather than the project.
    """
    configured = os.getenv("SEC_EDGAR_USER_AGENT", "").strip()
    return configured or f"TradingAgents/{__version__} (contact@example.com)"




def _fetch_json(url: str) -> dict:
    """Read a public EDGAR document, respecting SEC's identification rule."""
    try:
        response = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # Every failure here is "this vendor cannot serve it now", so the router
        # moves on instead of seeing a transport exception it has no rule for.
        raise VendorUnavailableError(f"SEC EDGAR request failed ({status or type(exc).__name__})") from exc
    except ValueError as exc:
        raise VendorUnavailableError("SEC EDGAR returned an unreadable response") from exc


def _cached_json(url: str, name: str) -> dict:
    path = Path(get_config()["data_cache_dir"]) / "sec_edgar" / name
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_TTL_SECONDS:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            pass  # a truncated file is a miss, not a failure
    data = _fetch_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_file(path, lambda temp: Path(temp).write_text(json.dumps(data), encoding="utf-8"))
    return data


def cik_for(ticker: str) -> str | None:
    """The filer's CIK, or None when the ticker is not a US filer."""
    table = _cached_json(_TICKERS_URL, "company_tickers.json")
    wanted = ticker.strip().upper()
    for entry in table.values():
        if entry.get("ticker", "").upper() == wanted:
            return f"{int(entry['cik_str']):010d}"
    return None


def _span_index(fact: dict, spans: tuple[tuple[int, int], ...]) -> int | None:
    """Which of ``spans`` a duration fact covers (0 for an instant fact), or None."""
    if "start" not in fact:
        return 0
    days = (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days
    return next((i for i, (low, high) in enumerate(spans) if low <= days <= high), None)


def _as_of(facts: dict, tags: tuple[str, ...], as_of_date: str, spans: tuple[tuple[int, int], ...],
           forms: tuple[str, ...] = ()) -> tuple[dict, str]:
    """({(period end, span index): value}, unit) for the tags the filer reports, as known then.

    A period reported more than once takes its latest filing on or before the
    date, so an amendment counts from the day it was filed and not before. The
    unit comes from the filing: most lines are USD, earnings per share are
    USD/shares, and scaling those alike would print a real figure as zero. A
    duration fact (revenue, cash flow) must cover one of ``spans``; an instant
    fact (a balance) has no span and serves any.
    """
    values: dict[tuple[str, int], float] = {}
    chosen_unit = "USD"
    # Tags are tried in order and a period keeps the first one that reports it:
    # filers renamed lines over the years, so one tag covers only part of the
    # history. Values are never added across tags, which would double count.
    for tag in tags:
        for unit, unit_values in ((facts.get(tag) or {}).get("units", {})).items():
            latest: dict[tuple[str, int], dict] = {}
            covered: set[tuple[str, int]] = set()   # periods a filing of ``forms`` reports
            for fact in unit_values:
                index = _span_index(fact, spans)
                key = (fact["end"], index)
                if fact["filed"] > as_of_date or index is None or key in values:
                    continue
                if not forms or fact.get("form", "").startswith(forms):
                    covered.add(key)
                seen = latest.get(key)
                if seen is None or fact["filed"] >= seen["filed"]:
                    latest[key] = fact
            latest = {key: fact for key, fact in latest.items() if key in covered}
            if latest:
                chosen_unit = unit
                values.update({key: fact["val"] for key, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def _statement(kind: str, ticker: str, freq: str, as_of_date: str, title: str) -> str:
    as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")
    cik = cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")

    facts = _cached_json(_FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        raise NoMarketDataError(ticker, ticker, "US filer with no us-gaap facts")

    quarterly = freq.lower() == "quarterly"
    # A balance is a point in time and its row is named by the date alone; a
    # duration row names its span, so a quarter, a year to date and a fiscal year
    # read apart on the row itself.
    balance = kind == "balance_sheet"
    if quarterly:
        spans = (_SPANS["quarterly"], *((low, high) for low, high, _ in _YEAR_TO_DATE))
        names = {0: "" if balance else " (3 months)", _DERIVED: " (3 months derived)",
                 **{i: f" ({months} months YTD)" for i, (_, _, months) in enumerate(_YEAR_TO_DATE, 1)}}
    else:
        spans, names = (_SPANS["annual"],), {0: "" if balance else " (fiscal year)"}
    forms = () if quarterly else _ANNUAL_FORMS
    lines = {label: _as_of(us_gaap, tags, as_of_date, spans, forms) for label, tags in _STATEMENTS[kind]}
    # Each row takes the shortest span it reports for a period, and a column
    # holds one span of one period, so a row filed only to date keeps its figure
    # beside a row filed by quarter.
    chosen = {label: {} for label in lines}
    for label, (values, _) in lines.items():
        for end, index in values:
            chosen[label][end] = min(index, chosen[label].get(end, index))
    derived = _derived_quarters(lines, chosen) if quarterly and not balance else {}
    periods = sorted({(end, index) for spans_of in chosen.values() for end, index in spans_of.items()}
                     | {(end, _DERIVED) for by_end in derived.values() for end in by_end})
    if not periods:
        raise NoMarketDataError(ticker, ticker, f"no {freq} {title.lower()} filed by {as_of_date}")

    tagged = {label: line for label, line in lines.items() if line[0]}
    untagged = [label for label in lines if label not in tagged]
    header = [
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the column says otherwise",
        f"# SEC EDGAR facts filed on or before {as_of_date}, at the values filed then",
        "# One row per period, newest first; every figure is on the row that names its period.",
    ]
    if quarterly and not balance:
        header.append(
            '# "(3 months)" rows are single quarters. "(6 months YTD)" and "(9 months YTD)" rows are '
            "fiscal year-to-date totals, not quarters and not full fiscal years. A fiscal fourth "
            'quarter is filed only inside the full year: ask for freq="annual".'
        )
    if derived:
        header.append(
            '# "(3 months derived)" rows are quarters filed only inside a year to date: the '
            "year-to-date row of that date less the year-to-date or quarter row one quarter "
            "earlier, computed here from their printed figures, not filed."
        )
    if untagged:
        header.append(f"# Unavailable (not tagged by this filer): {', '.join(untagged)}")
    # A model subtracting capex from operating cash flow got the digits wrong and
    # the debate repeated its figure, so the difference is served. It is taken
    # from the two printed cells of one row, one filing's period and span, so it
    # checks by eye and never mixes a quarter with a year to date.
    free_cash_flow = "Operating Cash Flow" in tagged and "Capital Expenditure" in tagged
    columns = [label if unit == "USD" else f"{label} ({unit})" for label, (_, unit) in tagged.items()]
    if free_cash_flow:
        columns.append(_FREE_CASH_FLOW)
        header.append("# Free Cash Flow is Operating Cash Flow minus Capital Expenditure on the same "
                      "row, computed here from those two figures, not filed.")
    # One row per period rather than one column: across dozens of period columns
    # a reader counting cells along a line item quotes a neighbouring period.
    rows = [",".join(["period", *columns])]
    for end, index in reversed(periods):
        cells = {}
        for label, (values, unit) in tagged.items():
            if index == _DERIVED:
                value = derived.get(label, {}).get(end)
                cells[label] = "" if value is None else str(value)
                continue
            value = values.get((end, index)) if chosen[label].get(end) == index else None
            # Plain numbers: a thousands separator would split the CSV field.
            cells[label] = "" if value is None else _millions(value) if unit == "USD" else f"{value:.2f}"
        row = list(cells.values())
        if free_cash_flow:
            ocf, capex = cells["Operating Cash Flow"], cells["Capital Expenditure"]
            row.append(str(int(ocf) - int(capex)) if ocf and capex else "")
        rows.append(",".join([end + names[index], *row]))
    return "\n".join(header) + "\n\n" + "\n".join(rows) + "\n"


def _millions(value: float) -> str:
    return f"{value / 1e6:.0f}"


def _derived_quarters(lines: dict, chosen: dict) -> dict[str, dict[str, int]]:
    """{label: {period end: quarter}} for USD lines whose quarter is filed only to date."""
    derived: dict[str, dict[str, int]] = {}
    for label, (values, unit) in lines.items():
        if unit != "USD":
            continue
        for end, index in values:
            # A filed quarter, or a year to date shadowed by one, needs nothing derived.
            if index == 0 or chosen[label].get(end) != index:
                continue
            closing = date.fromisoformat(end)
            earlier = [e for e, i in values if i == index - 1
                       and _QUARTER_APART[0] <= (closing - date.fromisoformat(e)).days <= _QUARTER_APART[1]]
            if earlier:
                # From the printed figures, so the subtraction checks by eye.
                quarter = int(_millions(values[(end, index)])) - int(_millions(values[(max(earlier), index - 1)]))
                derived.setdefault(label, {})[end] = quarter
    return derived


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Balance sheet as filed on or before ``as_of_date``."""
    return _statement("balance_sheet", ticker, freq, as_of_date, "Balance Sheet")


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Income statement as filed on or before ``as_of_date``.

    A fourth quarter is never derived: filers report it only inside the annual
    figure, and subtracting three separately filed quarters would invent a number
    with no filing date behind it.
    """
    return _statement("income_statement", ticker, freq, as_of_date, "Income Statement")


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Cash flow statement as filed on or before ``as_of_date``."""
    return _statement("cashflow", ticker, freq, as_of_date, "Cash Flow Statement")
