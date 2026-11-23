"""Logging utilities for the OSCAR / our_ddpg training pipeline.

Refactored to be wandb-first.  The Comet-ML path is preserved behind the
``comet_ml`` flag for backward compatibility but is no longer the default.

Key additions over the legacy version
--------------------------------------
* Jain's Fairness Index computed from per-station throughputs surfaced by the
  environment's ``info`` string.
* Rolling-window throughput tracker (window = 100 steps) with a one-shot
  ``convergence_epoch`` event logged to wandb when the rolling average
  exceeds 36.5 Mb/s.
* All ``wandb.log`` calls include a ``step`` key so they align on the same
  x-axis (requires the caller to have done ``wandb.define_metric("*",
  step_metric="step")``).
* ``DataFrame.append()`` (removed in pandas >= 2.0) replaced with
  ``pd.concat()``.
"""

import numpy as np
import pandas as pd
import wandb

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

try:
    from comet_ml import Experiment as CometExperiment  # optional dep
    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False

import glob
import json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONVERGENCE_THRESHOLD_MBPS: float = 36.5   # rolling-avg threshold
_CONVERGENCE_WINDOW:         int   = 100     # rolling window size (steps)


# ---------------------------------------------------------------------------
# Helper: Jain's Fairness Index
# ---------------------------------------------------------------------------

def jains_fairness_index(throughputs: np.ndarray) -> float:
    """Compute Jain's Fairness Index J = (sum x_i)^2 / (N * sum x_i^2).

    Args:
        throughputs: 1-D array of per-station throughput values (any unit).

    Returns:
        Scalar J in (0, 1].  Returns 1.0 for degenerate all-zero inputs.
    """
    x = np.asarray(throughputs, dtype=np.float64).flatten()
    sum_x  = float(np.sum(x))
    sum_x2 = float(np.sum(x ** 2))
    if sum_x2 < 1e-12:
        return 1.0  # all-zero -> perfectly equal by convention
    return (sum_x ** 2) / (len(x) * sum_x2)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger:
    """Step/episode telemetry logger for OSCAR-style agents.

    Supports wandb (default) and optionally Comet-ML.

    Parameters
    ----------
    comet_ml : bool
        When ``True`` log to Comet-ML (requires ``comet_ml`` package and a
        valid ``comet_token.json`` or hard-coded credentials).  When
        ``False`` (default) log exclusively to wandb.
    tags : list[str] | None
        Tags forwarded to the experiment backend.
    parameters : dict | None
        Hyper-parameters forwarded to the experiment backend.
    experiment : CometExperiment | None
        Pass an already-initialised Comet experiment to reuse it.
    """

    def __init__(
        self,
        comet_ml: bool = False,
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict] = None,
        experiment=None,
    ):
        self.stations: int = 5
        self.comet_ml = comet_ml and _COMET_AVAILABLE
        self.wandb_ml = not self.comet_ml  # wandb is default when comet off

        # Internal CSV log (always written regardless of backend)
        self.logs = pd.DataFrame(columns=["step", "name", "type", "value"])
        self.fname = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        # ------------------------------------------------------------------
        # Comet-ML initialisation (optional)
        # ------------------------------------------------------------------
        if self.comet_ml:
            if experiment is None:
                try:
                    json_loc = glob.glob("./**/comet_token.json")[0]
                    with open(json_loc, "r") as f:
                        kwargs = json.load(f)
                except IndexError:
                    kwargs = {
                        "api_key": "XXXXXXXXXXXX",
                        "project_name": "rl-in-wifi",
                        "workspace": "XYZ",
                    }
                self.experiment = CometExperiment(**kwargs)
            else:
                self.experiment = experiment

            if tags:
                self.experiment.add_tags(tags)
            if parameters:
                self.experiment.log_parameters(parameters)

        # ------------------------------------------------------------------
        # Throughput / fairness state
        # ------------------------------------------------------------------
        self.sent_mb: float      = 0.0
        self.step_time: float    = 0.01       # set by begin_logging()
        self.current_speed: float = 0.0

        # Rolling window for convergence tracking
        self._speed_window: deque = deque(maxlen=_CONVERGENCE_WINDOW)
        self._convergence_logged: bool = False

    # -----------------------------------------------------------------------
    # Low-level logging primitives
    # -----------------------------------------------------------------------

    def _append_csv(self, step, name, kind, value):
        """Append one record to the in-memory CSV log."""
        entry = pd.DataFrame(
            [{"step": step, "name": name, "type": kind, "value": value}]
        )
        self.logs = pd.concat([self.logs, entry], ignore_index=True)

    def log_parameter(self, param_name: str, param_value, commit: bool = True):
        """Log a single hyper-parameter."""
        if self.comet_ml:
            self.experiment.log_parameter(param_name, param_value)
        if self.wandb_ml:
            wandb.log({param_name: param_value}, commit=commit)
        self._append_csv(0, param_name, "parameter", param_value)

    def log_metric(
        self,
        metric_name: str,
        value,
        commit: bool = True,
        step: Optional[int] = None,
    ):
        """Log a scalar metric.

        Tensor values are automatically detached and converted to Python
        scalars before being dispatched to the backends.
        """
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = value.item()

        if self.comet_ml:
            self.experiment.log_metric(metric_name, value, step=step)
        if self.wandb_ml:
            payload: Dict = {metric_name: value}
            if step is not None:
                payload["step"] = step
            wandb.log(payload, commit=commit)

        self._append_csv(step, metric_name, "metric", value)

    def log_metrics(
        self, metrics: Dict, step: int, commit: bool = True
    ):
        """Log a dictionary of scalar metrics in one call."""
        for name, val in metrics.items():
            self.log_metric(name, val, step=step, commit=commit)

    # -----------------------------------------------------------------------
    # High-level helpers
    # -----------------------------------------------------------------------

    def begin_logging(
        self,
        episode_count: int,
        steps_per_ep: int,
        sigma: float,
        theta: float,
        step_time: float,
    ):
        """Log training hyper-parameters at run start."""
        self.step_time = step_time
        self.log_parameter("Episode count",     episode_count, commit=False)
        self.log_parameter("Steps per episode", steps_per_ep,  commit=False)
        self.log_parameter("theta",             theta,         commit=False)
        self.log_parameter("sigma",             sigma,         commit=False)
        self.log_parameter("Step time",         step_time,     commit=True)

    def log_round(
        self,
        states,
        reward,
        cumulative_reward,
        info,
        loss,
        observations,
        step: int,
    ):
        """Log per-step (round) metrics.

        The ``info`` field is expected to be a list/array of pipe-delimited
        strings  ``"mb_sent|CW|n_stations|fairness_index"``.  Columns
        beyond index 3 are ignored for future extensibility.

        Jain's Fairness Index is *also* recomputed here from per-station
        throughputs (derived from the state observations) whenever the
        ``states`` vector has length N > 1, providing a higher-fidelity
        signal than the scalar broadcasted by the environment.
        """
        # ------------------------------------------------------------------
        # Parse info string
        # ------------------------------------------------------------------
        if self.comet_ml:
            self.experiment.log_histogram_3d(states, name="Observations", step=step)

        info_rows = [[float(v) for v in row.split("|")] for row in info]
        info_arr = np.mean(np.array(info_rows, dtype=np.float32), axis=0)

        try:
            round_mb  = float(info_arr[0])
            CW        = float(info_arr[1])
            n_sta     = float(info_arr[2])
            fairness  = float(info_arr[3])  # environment-reported Jain's index
        except (IndexError, ValueError) as exc:
            print(f"[Logger] Unexpected info format: {info_arr}  ->  {exc}")
            raise

        self.stations = int(n_sta)

        # ------------------------------------------------------------------
        # Rolling throughput speed (Mb/s)
        # ------------------------------------------------------------------
        self._speed_window.append(round_mb)
        self.current_speed = float(
            np.mean(np.asarray(self._speed_window) / self.step_time)
        )
        self.sent_mb += round_mb

        # ------------------------------------------------------------------
        # Recompute Jain's FI from per-station obs when N > 1
        # ------------------------------------------------------------------
        states_arr = np.asarray(states, dtype=np.float32).flatten()
        if states_arr.size > 1:
            # obs are collision rates rho_i; per-station throughput proportional to (1 - rho_i)
            per_station_tput = np.maximum(1.0 - states_arr, 1e-6)
            jfi = jains_fairness_index(per_station_tput)
        else:
            jfi = fairness   # fall back to env-reported scalar

        # ------------------------------------------------------------------
        # CW distribution histogram (wandb only)
        # The station_cw_distribution histogram is built from the raw CW
        # exponent observations (values in [4, 10] as per the actor output).
        # ------------------------------------------------------------------
        if self.wandb_ml and states_arr.size > 0:
            wandb.log(
                {
                    "station_cw_distribution": wandb.Histogram(states_arr.tolist()),
                    "step": step,
                },
                commit=False,
            )

        # ------------------------------------------------------------------
        # Convergence tracker (rolling window = 100 steps)
        # Fires the convergence_epoch event exactly once.
        # ------------------------------------------------------------------
        _rolling_avg = (
            float(np.mean(self._speed_window) / self.step_time)
            if self._speed_window else 0.0
        )
        _convergence_payload: Dict = {}
        if not self._convergence_logged and _rolling_avg > _CONVERGENCE_THRESHOLD_MBPS:
            _convergence_payload["convergence_epoch"] = step
            self._convergence_logged = True

        # ------------------------------------------------------------------
        # Dispatch all scalar metrics in a single wandb.log call
        # (commit=False for all except the last to batch them together)
        # ------------------------------------------------------------------
        base_metrics: Dict = {
            "Current Collision Rate":    float(states_arr[0]) if states_arr.size else 0.0,
            "Round reward":              float(np.mean(reward)),
            "Per-ep reward":             float(np.mean(cumulative_reward)),
            "Megabytes sent":            self.sent_mb,
            "Round megabytes sent":      round_mb,
            "Chosen CW":                 CW,
            "Station count":             self.stations,
            "Current throughput":        self.current_speed,
            "throughput_rolling_avg_mbps": _rolling_avg,
            "Fairness index":            fairness,
            "jains_fairness_index":      jfi,
            **_convergence_payload,
        }
        self.log_metrics(base_metrics, step=step, commit=False)

        for i, obs in enumerate(observations):
            self.log_metric(f"Observation {i}", obs, step=step, commit=False)

        self.log_metrics(loss, step=step, commit=True)

    def log_episode(self, cumulative_reward: float, speed: float, step: int):
        """Log episode-boundary aggregate metrics."""
        self.log_metric("Cumulative reward", cumulative_reward, step=step, commit=False)
        self.log_metric("Speed",             speed,             step=step, commit=True)
        self.log_metric("MB sent",           self.sent_mb,      step=step, commit=True)

        # Reset per-episode accumulators
        self.sent_mb = 0.0
        self._speed_window = deque(maxlen=_CONVERGENCE_WINDOW)
        self.current_speed = 0.0

    def end(self):
        """Finalise experiment backends."""
        if self.comet_ml:
            self.experiment.end()
        # CSV persistence (uncomment if needed)
        # self.logs.to_csv(self.fname)
