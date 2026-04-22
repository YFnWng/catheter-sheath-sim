"""Real-time diagnostic plotter — subprocess-backed pyqtgraph window.

Runs in a fresh Python process (multiprocessing 'spawn') so it has its own
Qt 5.15 (from PyQt5) completely independent of SOFA's Qt 5.12.  No shared
memory is used; data arrives through a multiprocessing.Queue as small dicts
of numpy arrays (~5 KB per frame).

Two kinds of panels:

* **Spatial** — x-axis is arc length s (m).  Used for position, strain,
  internal force, contact force along the rod.
* **Time-series** — x-axis is simulation time (s) with a rolling window.
  Used for base translation, base rotation, tendon force, contact force
  totals.

Panels are selected via a ``panels`` tuple passed at construction.

The ``DiagnosticPlotter`` base class manages the subprocess and queue.
Subclasses override ``update()`` to pack domain-specific data into dicts
and call ``send(data)``.
"""
from __future__ import annotations

import multiprocessing
import sys
from collections import deque
from typing import Dict, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-panel builders (run in the subprocess)
# ---------------------------------------------------------------------------

_C = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
_CABLE_COLORS = [
    (50, 150, 50), (150, 50, 150), (50, 150, 150), (150, 150, 50),
]
_GT_SYM = "o"
_GT_SYM_SIZE = 6


def _make_panel(win, title: str, ylabel: str, xlabel: str = "s (m)"):
    import pyqtgraph as pg
    p = win.addPlot(title=title)
    p.setLabel("bottom", xlabel)
    p.setLabel("left", ylabel)
    p.addLegend(offset=(5, 5))
    p.showGrid(x=True, y=True, alpha=0.15)
    return p


# -- Spatial panels (x = arc length) ----------------------------------------

def _build_position(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Position", "m")
    ep = [p.plot(s_n, zeros_n, pen=pg.mkPen(_C[i], width=2), name=f"est {l}")
          for i, l in enumerate("XYZ")]
    gp = [p.plot(s_n, zeros_n, pen=None,
                 symbol=_GT_SYM, symbolSize=_GT_SYM_SIZE, symbolBrush=_C[i],
                 name=f"GT {l}")
          for i, l in enumerate("XYZ")]
    return dict(ep=ep, gp=gp)


def _build_strain(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Strain (κ₁,κ₂,τ)", "rad/m")
    es = [p.plot(s_n, zeros_n, pen=pg.mkPen(_C[i], width=2), name=f"est u{l}")
          for i, l in enumerate("xyz")]
    gs = [p.plot(s_s, zeros_s, pen=None,
                 symbol=_GT_SYM, symbolSize=_GT_SYM_SIZE, symbolBrush=_C[i],
                 name=f"GT u{l}")
          for i, l in enumerate("xyz")]
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    return dict(es=es, gs=gs)


def _build_internal_force(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Internal force", "N")
    en = [p.plot(s_n, zeros_n, pen=pg.mkPen(_C[i], width=2), name=f"N{l}")
          for i, l in enumerate("xyz")]
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    return dict(en=en)


def _build_contact_force(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "External force", "N")
    ef = [p.plot(s_n, zeros_n, pen=pg.mkPen(_C[i], width=2), name=f"est F{l}")
          for i, l in enumerate("xyz")]
    gf = [p.plot(s_n, zeros_n, pen=None,
                 symbol=_GT_SYM, symbolSize=_GT_SYM_SIZE, symbolBrush=_C[i],
                 name=f"GT F{l}")
          for i, l in enumerate("xyz")]
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    return dict(ef=ef, gf=gf)


# -- Time-series panels (x = time) ------------------------------------------

def _build_base_translation(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Base translation", "m", xlabel="t (s)")
    bt = [p.plot([], [], pen=pg.mkPen(_C[i], width=2), name=l)
          for i, l in enumerate("XYZ")]
    return dict(bt=bt)


def _build_base_rotation(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Base rotation", "deg", xlabel="t (s)")
    br = [p.plot([], [], pen=pg.mkPen(_C[i], width=2), name=l)
          for i, l in enumerate(["Roll", "Pitch", "Yaw"])]
    return dict(br=br)


def _build_tendon_force(win, s_n, s_s, zeros_n, zeros_s, **kw) -> Dict[str, list]:
    import pyqtgraph as pg
    n_cables = kw.get("n_cables", 1)
    p = _make_panel(win, "Tendon force", "N", xlabel="t (s)")
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    tf = [p.plot([], [], pen=pg.mkPen(_CABLE_COLORS[i % len(_CABLE_COLORS)], width=2),
                 name=f"Q{i}")
          for i in range(max(n_cables, 1))]
    return dict(tf=tf)


def _build_total_contact_force(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Total contact force", "N", xlabel="t (s)")
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    tcf = [p.plot([], [], pen=pg.mkPen(_C[i], width=2), name=l)
           for i, l in enumerate("XYZ")]
    tcf_norm = p.plot([], [], pen=pg.mkPen((0, 0, 0), width=2, style=2), name="|F|")
    return dict(tcf=tcf, tcf_norm=[tcf_norm])


def _build_tip_load(win, s_n, s_s, zeros_n, zeros_s, **_kw) -> Dict[str, list]:
    import pyqtgraph as pg
    p = _make_panel(win, "Tip load", "N", xlabel="t (s)")
    p.addLine(y=0, pen=pg.mkPen("k", width=0.5))
    tl = [p.plot([], [], pen=pg.mkPen(_C[i], width=2), name=l)
          for i, l in enumerate("XYZ")]
    tl_norm = p.plot([], [], pen=pg.mkPen((0, 0, 0), width=2, style=2), name="|F|")
    return dict(tl=tl, tl_norm=[tl_norm])


PANEL_BUILDERS = {
    # Spatial panels
    "position": _build_position,
    "strain": _build_strain,
    "internal_force": _build_internal_force,
    "contact_force": _build_contact_force,
    # Time-series panels
    "base_translation": _build_base_translation,
    "base_rotation": _build_base_rotation,
    "tendon_force": _build_tendon_force,
    "total_contact_force": _build_total_contact_force,
    "tip_load": _build_tip_load,
}

# Panels whose curves use rolling time buffers instead of fixed arc-length
_TIME_SERIES_PANELS = {"base_translation", "base_rotation",
                       "tendon_force", "total_contact_force",
                       "tip_load"}

# Data key -> curve key mapping for time-series panels
_TS_CURVE_KEYS = {
    "base_pos": "bt",
    "base_rot": "br",
    "cable_tensions": "tf",
    "total_contact": "tcf",
    "total_contact_norm": "tcf_norm",
    "tip_load": "tl",
}

_PANELS_PER_ROW = 2


# ---------------------------------------------------------------------------
# Window + apply (run in the subprocess)
# ---------------------------------------------------------------------------

def _build_window(
    n_nodes: int,
    n_sections: int,
    rod_length: float,
    panels: Sequence[str],
    title: str,
    size: Tuple[int, int],
    n_cables: int = 1,
):
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=True, useOpenGL=True, background="w", foreground="k")
    win = pg.GraphicsLayoutWidget(title=title)
    win.resize(*size)
    win.show()

    s_n = np.linspace(0.0, rod_length, n_nodes)
    s_s = s_n[:-1]
    zeros_n = np.zeros(n_nodes)
    zeros_s = np.zeros(n_sections)

    curves: Dict[str, list] = {}
    for idx, name in enumerate(panels):
        if idx > 0 and idx % _PANELS_PER_ROW == 0:
            win.nextRow()
        builder = PANEL_BUILDERS.get(name)
        if builder is not None:
            curves.update(builder(win, s_n, s_s, zeros_n, zeros_s,
                                  n_cables=n_cables))

    return win, curves, s_n, s_s


def _apply(curves: Dict[str, list], data: dict, s_n, s_s, ts_bufs):
    """Update curve data from a dict of numpy arrays.

    Spatial keys (x = arc-length, one frame per call):
        est_pos, gt_pos, est_str, gt_str, est_N, est_F, gt_F

    Time-series keys (x = rolling time buffer):
        t          : scalar — current simulation time
        base_pos   : (3,)   — base translation [x,y,z]
        base_rot   : (3,)   — base rotation [roll,pitch,yaw] in degrees
        cable_tensions : (n,) — tendon forces
        total_contact  : (3,) — total contact force [fx,fy,fz]
    """
    # -- Spatial panels --
    for i in range(3):
        if "ep" in curves and "est_pos" in data:
            curves["ep"][i].setData(s_n, data["est_pos"][:, i])
        if "gp" in curves and "gt_pos" in data:
            curves["gp"][i].setData(s_n, data["gt_pos"][:, i])
        if "es" in curves and "est_str" in data:
            curves["es"][i].setData(s_n, data["est_str"][:, i])
        if "gs" in curves and "gt_str" in data:
            curves["gs"][i].setData(s_s, data["gt_str"][:, i])
        if "en" in curves and "est_N" in data:
            curves["en"][i].setData(s_n, data["est_N"][:, i])
        if "ef" in curves and "est_F" in data:
            curves["ef"][i].setData(s_n, data["est_F"][:, i])
        if "gf" in curves and "gt_F" in data:
            curves["gf"][i].setData(s_n, data["gt_F"][:, i])

    # -- Time-series panels --
    if "t" not in data:
        return
    t = data["t"]
    ts_bufs["t"].append(t)
    t_arr = np.array(ts_bufs["t"])

    for data_key, curve_key in _TS_CURVE_KEYS.items():
        if curve_key not in curves or data_key not in data:
            continue
        val = np.asarray(data[data_key])
        if curve_key == "tcf_norm":
            ts_bufs[curve_key].append(float(val))
            curves[curve_key][0].setData(t_arr, np.array(ts_bufs[curve_key]))
        elif val.ndim == 0:
            ts_bufs.setdefault(curve_key + "_0", deque(maxlen=ts_bufs["t"].maxlen))
            ts_bufs[curve_key + "_0"].append(float(val))
            curves[curve_key][0].setData(t_arr, np.array(ts_bufs[curve_key + "_0"]))
        else:
            for i in range(min(len(val), len(curves[curve_key]))):
                buf_key = f"{curve_key}_{i}"
                ts_bufs.setdefault(buf_key, deque(maxlen=ts_bufs["t"].maxlen))
                ts_bufs[buf_key].append(float(val[i]))
                curves[curve_key][i].setData(t_arr, np.array(ts_bufs[buf_key]))

    # total_contact_norm: compute |F| from total_contact if present
    if "tcf_norm" in curves and "total_contact" in data:
        norm = float(np.linalg.norm(data["total_contact"]))
        ts_bufs.setdefault("tcf_norm", deque(maxlen=ts_bufs["t"].maxlen))
        ts_bufs["tcf_norm"].append(norm)
        curves["tcf_norm"][0].setData(t_arr, np.array(ts_bufs["tcf_norm"]))

    # tip_load_norm: compute |F| from tip_load if present
    if "tl_norm" in curves and "tip_load" in data:
        norm = float(np.linalg.norm(data["tip_load"]))
        ts_bufs.setdefault("tl_norm", deque(maxlen=ts_bufs["t"].maxlen))
        ts_bufs["tl_norm"].append(norm)
        curves["tl_norm"][0].setData(t_arr, np.array(ts_bufs["tl_norm"]))


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def run(
    queue,
    n_nodes: int,
    n_sections: int,
    rod_length: float,
    panels: Sequence[str] = ("position", "strain"),
    *,
    title: str = "Diagnostics",
    size: Tuple[int, int] = (1300, 450),
    window_seconds: float = 10.0,
    dt: float = 0.01,
    n_cables: int = 1,
) -> None:
    """Subprocess entry point.  Blocks until the window is closed."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import pyqtgraph as pg  # noqa: F401

    app = QApplication(sys.argv)
    win, curves, s_n, s_s = _build_window(n_nodes, n_sections, rod_length,
                                           panels, title, size,
                                           n_cables=n_cables)

    max_pts = int(window_seconds / dt) + 1
    ts_bufs: Dict[str, deque] = {"t": deque(maxlen=max_pts)}

    def _poll():
        latest = None
        try:
            while True:
                latest = queue.get_nowait()
        except Exception:
            pass
        if latest is not None:
            try:
                _apply(curves, latest, s_n, s_s, ts_bufs)
            except Exception:
                pass

    timer = QTimer()
    timer.timeout.connect(_poll)
    timer.start(33)
    sys.exit(app.exec_())


# ---------------------------------------------------------------------------
# DiagnosticPlotter — base class for scene-side plotting
# ---------------------------------------------------------------------------

class DiagnosticPlotter:
    """Subprocess-backed pyqtgraph diagnostic window.

    Spawns a ``run`` subprocess and provides ``send(data_dict)`` to push
    numpy arrays to the plotter.  Subclasses override ``update()`` to
    pack domain-specific data and call ``send()``.

    Parameters
    ----------
    n_nodes, n_sections, rod_length:
        Rod geometry for axis scaling.
    panels:
        Which panels to show (keys of ``PANEL_BUILDERS``).
    update_every:
        Only send data every N calls to ``send()`` (throttle).
    window_seconds:
        Rolling time window for time-series panels (seconds).
    dt:
        Simulation timestep for buffer sizing.
    n_cables:
        Number of tendon channels (for tendon_force panel).
    title, size:
        Window title and (width, height) in pixels.
    """

    def __init__(
        self,
        n_nodes: int,
        n_sections: int,
        rod_length: float,
        panels: Sequence[str] = ("position", "strain"),
        *,
        update_every: int = 5,
        window_seconds: float = 10.0,
        dt: float = 0.01,
        n_cables: int = 1,
        title: str = "Diagnostics",
        size: Tuple[int, int] = (1300, 450),
    ) -> None:
        self._n = n_nodes
        self._m = n_sections
        self._panels = tuple(panels)
        self._every = update_every
        self._counter = 0

        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue(maxsize=2)
        self._proc = ctx.Process(
            target=run,
            args=(self._queue, n_nodes, n_sections, rod_length, self._panels),
            kwargs=dict(title=title, size=size,
                        window_seconds=window_seconds, dt=dt,
                        n_cables=n_cables),
            daemon=True,
        )
        self._proc.start()

    @property
    def panels(self) -> Tuple[str, ...]:
        return self._panels

    def send(self, data: Dict[str, np.ndarray]) -> None:
        """Push a data dict to the plotter subprocess (throttled)."""
        self._counter += 1
        if self._counter % self._every != 0:
            return
        if not self._proc.is_alive():
            return
        try:
            self._queue.put_nowait(data)
        except Exception:
            pass
