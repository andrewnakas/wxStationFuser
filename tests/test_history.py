"""History replay: out-of-sample forecasts, sampling, events and compact output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from wxfuser import history


def _oos(n_days=10):
    issues = pd.date_range("2025-01-01 03:00", periods=n_days * 2, freq="12h")
    rows = []
    for it in issues:
        for lead in range(1, 73):
            vt = it + pd.Timedelta(hours=lead)
            rows.append({"issue_time": it, "valid_time": vt, "lead_h": lead,
                         "obs": float(vt.day), "q05": 0.0, "q25": 1.0, "q50": float(vt.day) + 0.5,
                         "q75": 3.0, "q95": 4.0, "fc_hrrr": float(vt.day) + 3.0})
    return pd.DataFrame(rows)


def test_sampling_keeps_morning_snow_totals_and_six_hourly_temperature():
    f = _oos()
    snow = history.sample(f, "hn24_cm")
    assert set(pd.to_datetime(snow["valid_time"]).dt.hour) == {15}
    assert snow["lead_h"].max() <= 72
    temp = history.sample(f, "air_temp_c")
    assert set(pd.to_datetime(temp["valid_time"]).dt.hour) <= {0, 6, 12, 18}
    assert temp["lead_h"].max() <= 48


def test_events_pick_the_biggest_days_with_their_day_ahead_forecast():
    f = history.sample(_oos(), "hn24_cm")
    ev = history.events(f, ["hrrr"], n=3)
    assert [e["obs"] for e in ev] == [11.0, 10.0, 9.0]
    assert all(11 <= e["lead"] <= 30 for e in ev)
    assert ev[0]["raw"]["hrrr"] == 14.0


def test_summary_compares_on_identical_rows():
    f = _oos()
    s = history.summary(f, ["hrrr"])
    assert s["mae_calibrated"] == 0.5 and s["mae_hrrr"] == 3.0


def test_columnar_output_is_compact_and_null_safe():
    f = _oos(2).head(3)
    f.loc[1, "obs"] = np.nan
    c = history.columnar(f, ["hrrr"])
    assert c["obs"][1] is None
    assert len(c["issue"]) == 3 and isinstance(c["issue"][0], int)
