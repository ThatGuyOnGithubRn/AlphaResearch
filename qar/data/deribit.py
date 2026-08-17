"""Deribit public API client, with on-disk caching.

Chosen over the equity venues for three reasons that matter to the research in
part 3: no API key, no rate-limit gate, and a genuinely deep option chain with a
mark IV per instrument. Crypto also trades continuously, so there are no session
breaks or holidays to special-case in a realised-variance calculation.

Everything fetched is cached to ``data/raw/`` as JSON. That is not an
optimisation — it is what makes a result reproducible. A backtest that silently
re-downloads a moving market is a backtest whose numbers cannot be checked
tomorrow, and the cached payloads are small enough to commit.

Only the standard library is used for HTTP, so the package's dependency
footprint stays at NumPy and SciPy.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DeribitClient",
    "OptionQuote",
    "DEFAULT_CACHE",
    "DeribitError",
]

BASE_URL = "https://www.deribit.com/api/v2/public/"
DEFAULT_CACHE = Path("data/raw")
_USER_AGENT = "qar-research/0.1 (+educational use)"


class DeribitError(RuntimeError):
    """The API was unreachable or returned an error payload."""


@dataclass(frozen=True)
class OptionQuote:
    """One live option quote, normalised out of Deribit's book summary.

    Deribit quotes option prices in units of the underlying (a BTC call costs
    some fraction of a BTC), so ``mark_price`` is multiplied by the underlying
    to reach a currency price. Forgetting that conversion is the single most
    common mistake when first pulling this data, and it silently produces
    implied vols that are wrong by orders of magnitude.
    """

    instrument: str
    underlying_price: float
    strike: float
    expiry_timestamp_ms: int
    kind: str                 # "call" or "put"
    mark_price_coin: float    # in units of the underlying
    mark_iv: float            # Deribit's own implied vol, in percent
    bid_coin: float | None
    ask_coin: float | None
    open_interest: float
    timestamp_ms: int

    @property
    def mark_price(self) -> float:
        """Mark price converted to currency units."""
        return self.mark_price_coin * self.underlying_price

    @property
    def time_to_expiry(self) -> float:
        """Years to expiry, on a 365-day basis."""
        seconds = (self.expiry_timestamp_ms - self.timestamp_ms) / 1000.0
        return max(seconds / (365.0 * 24 * 3600), 0.0)

    @property
    def implied_vol(self) -> float:
        """Deribit's mark IV as a decimal, for comparison against our own solve."""
        return self.mark_iv / 100.0

    @property
    def spread(self) -> float | None:
        """Bid-ask spread in currency units — the floor for any arbitrage tolerance."""
        if self.bid_coin is None or self.ask_coin is None:
            return None
        return (self.ask_coin - self.bid_coin) * self.underlying_price


class DeribitClient:
    """Read-only client for Deribit's public endpoints.

    Parameters
    ----------
    cache_dir:
        Where JSON payloads are stored. Set to ``None`` to disable caching,
        which is almost never what you want for research.
    offline:
        Serve only from cache and raise if a payload is missing. Use this in
        tests and when re-running an analysis you want to be bit-identical.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = DEFAULT_CACHE,
        offline: bool = False,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.offline = offline
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        key = urllib.parse.urlencode(sorted(params.items()))
        safe = "".join(c if c.isalnum() or c in "-_=&." else "_" for c in key)
        return self.cache_dir / f"{endpoint}__{safe}.json"

    def get(
        self, endpoint: str, params: dict[str, Any], refresh: bool = False
    ) -> Any:
        """Fetch an endpoint, preferring the cache unless ``refresh`` is set."""
        path = self._cache_path(endpoint, params)

        if path is not None and path.exists() and not refresh:
            return json.loads(path.read_text())["result"]

        if self.offline:
            raise DeribitError(
                f"offline mode and no cached payload at {path}. "
                "Run once with offline=False to populate the cache."
            )

        url = BASE_URL + endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeribitError(f"request to {endpoint} failed: {exc}") from exc

        if "result" not in payload:
            raise DeribitError(f"unexpected payload from {endpoint}: {payload}")

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=1))

        return payload["result"]

    # -- endpoints --------------------------------------------------------

    def index_price(self, index_name: str = "btc_usd", refresh: bool = False) -> float:
        """Current index (spot) price."""
        result = self.get("get_index_price", {"index_name": index_name}, refresh)
        return float(result["index_price"])

    def option_chain(
        self, currency: str = "BTC", refresh: bool = False
    ) -> list[OptionQuote]:
        """The full live option chain for a currency.

        Instruments with no mark IV are dropped — they are untraded strikes
        with no usable quote, and carrying them forward only pollutes the
        surface.
        """
        raw = self.get(
            "get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
            refresh,
        )

        quotes: list[OptionQuote] = []
        for row in raw:
            name = row.get("instrument_name", "")
            parsed = self._parse_instrument(name)
            if parsed is None:
                continue
            strike, kind = parsed
            if row.get("mark_iv") in (None, 0):
                continue
            underlying = row.get("underlying_price") or row.get("estimated_delivery_price")
            if not underlying:
                continue

            quotes.append(
                OptionQuote(
                    instrument=name,
                    underlying_price=float(underlying),
                    strike=strike,
                    expiry_timestamp_ms=self._expiry_ms(name),
                    kind=kind,
                    mark_price_coin=float(row.get("mark_price") or 0.0),
                    mark_iv=float(row["mark_iv"]),
                    bid_coin=_optional_float(row.get("bid_price")),
                    ask_coin=_optional_float(row.get("ask_price")),
                    open_interest=float(row.get("open_interest") or 0.0),
                    timestamp_ms=int(row.get("creation_timestamp") or time.time() * 1000),
                )
            )
        return quotes

    def ohlc(
        self,
        instrument: str = "BTC-PERPETUAL",
        resolution: str = "1D",
        days: int = 730,
        refresh: bool = False,
    ) -> dict[str, np.ndarray]:
        """Historical OHLC bars.

        Returns arrays keyed ``time`` (ms), ``open``, ``high``, ``low``,
        ``close``, ``volume`` — the inputs the range-based realised-variance
        estimators in :mod:`qar.data.realized` need.
        """
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 3600 * 1000
        # Round to the day so the cache key is stable across a session.
        day_ms = 24 * 3600 * 1000
        result = self.get(
            "get_tradingview_chart_data",
            {
                "instrument_name": instrument,
                "start_timestamp": (start_ms // day_ms) * day_ms,
                "end_timestamp": (end_ms // day_ms) * day_ms,
                "resolution": resolution,
            },
            refresh,
        )
        return {
            "time": np.asarray(result["ticks"], dtype=np.int64),
            "open": np.asarray(result["open"], dtype=float),
            "high": np.asarray(result["high"], dtype=float),
            "low": np.asarray(result["low"], dtype=float),
            "close": np.asarray(result["close"], dtype=float),
            "volume": np.asarray(result.get("volume", []), dtype=float),
        }

    # -- instrument-name parsing -----------------------------------------

    @staticmethod
    def _parse_instrument(name: str) -> tuple[float, str] | None:
        """``BTC-27JUN25-100000-C`` -> ``(100000.0, "call")``."""
        parts = name.split("-")
        if len(parts) != 4:
            return None
        try:
            strike = float(parts[2])
        except ValueError:
            return None
        kind = {"C": "call", "P": "put"}.get(parts[3].upper())
        if kind is None:
            return None
        return strike, kind

    @staticmethod
    def _expiry_ms(name: str) -> int:
        """Parse the ``27JUN25`` field into a settlement timestamp.

        Deribit options settle at 08:00 UTC on the expiry date.
        """
        from datetime import datetime, timezone

        parts = name.split("-")
        if len(parts) < 2:
            return 0
        try:
            date = datetime.strptime(parts[1], "%d%b%y").replace(
                hour=8, tzinfo=timezone.utc
            )
        except ValueError:
            return 0
        return int(date.timestamp() * 1000)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
