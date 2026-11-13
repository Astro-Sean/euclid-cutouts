#!/usr/bin/python3
"""fits_viewer - Qt FITS image viewer with DS9-style contrast control.

Layout follows the RISOTTO scanner pattern: a control sidebar left of
a matplotlib canvas with a navigation toolbar, and a status bar along
the bottom. Contrast works the way DS9 and ginga do it: a histogram
of the data with draggable low/high cut markers, a stretch function
applied between the cuts and the colormap, and preset cut algorithms
(zscale, percentiles, min/max) for a sane starting window.

Usage: fits_viewer.py image1.fits [image2.fits ...]
"""

import sys
from pathlib import Path

# PySide6 first, PyQt6 fallback - same import order as RISOTTO, and
# Qt must be in before matplotlib so the canvas picks the same binding.
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        sys.exit(
            f"PySide6 or PyQt6 is required (interpreter: {sys.executable}): "
            "pip install pyside6"
        )

import math

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import (
    AsinhStretch,
    ImageNormalize,
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
    ZScaleInterval,
)
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from scipy import ndimage

# Stretch functions between the cut range and the colormap. "zscale"
# is DS9-style: linear display over zscale cuts, reapplied on select.
STRETCHES = {
    "zscale": LinearStretch(),
    "asinh": AsinhStretch(a=0.1),
    "log": LogStretch(a=1000),
    "sqrt": SqrtStretch(),
    "linear": LinearStretch(),
}

# Cut presets, matching the DS9 Scale menu entries.
PRESETS = ("zscale", "99%", "99.5%", "min/max")

# Colormaps kept to perceptually sane options (no jet/rainbow).
CMAPS = ("gray", "viridis", "plasma", "inferno", "magma", "bone")

SLIDER_STEPS = 1000


def preset_cuts(data, preset):
    """Low/high cut levels for a named algorithm."""
    if preset == "zscale":
        return tuple(ZScaleInterval().get_limits(data))
    if preset == "99%":
        return tuple(np.nanpercentile(data, (0.5, 99.5)))
    if preset == "99.5%":
        return tuple(np.nanpercentile(data, (0.25, 99.75)))
    return np.nanmin(data), np.nanmax(data)


class FitsViewer(QMainWindow):
    """Grid of FITS panels; contrast controls edit the active panel."""

    def __init__(self, files):
        super().__init__()
        self.setWindowTitle("euclid-cutouts viewer")
        self._files = [str(f) for f in files]
        self.active = 0
        self._drag = None  # 'lo'|'hi' while a histogram marker is held
        self._syncing = False  # guards widget setValue recursion
        self._mark = None  # SkyCoord of the crosshair, or None
        self._mark_scale = 1.0  # tick length factor from the slider
        self._mark_lines = []
        self._lo_line = self._hi_line = None  # histogram cut markers

        self.panels = [self._load_panel(f) for f in self._files]

        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_hist = None

        self._build_sidebar()
        right = QWidget()
        vbox = QVBoxLayout(right)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.toolbar)
        vbox.addWidget(self.canvas)

        central = QWidget()
        hbox = QHBoxLayout(central)
        hbox.addWidget(self._sidebar)
        hbox.addWidget(right, stretch=1)
        self.setCentralWidget(central)

        self.canvas.mpl_connect("button_press_event", self._canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._hist_drag)
        self.canvas.mpl_connect("button_release_event", self._hist_drop)
        self.canvas.mpl_connect("scroll_event", self._scroll_zoom)

        self._draw()
        self._set_active(0)

    @staticmethod
    def _load_panel(path):
        """Data + WCS + per-image contrast state for one FITS file."""
        with fits.open(path) as hdul:
            data, hdr = hdul[0].data, hdul[0].header
        wcs = WCS(hdr).celestial
        ps = wcs.pixel_scale_matrix
        if np.linalg.det(ps) > 0:
            data = data[:, ::-1]
            hdr = hdr.copy()
            hdr["CRPIX1"] = hdr["NAXIS1"] + 1 - hdr["CRPIX1"]
            for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2",
                      "CDELT1", "CDELT2"):
                hdr.pop(k, None)
            hdr["CD1_1"], hdr["CD1_2"] = -ps[0, 0], ps[0, 1]
            hdr["CD2_1"], hdr["CD2_2"] = -ps[1, 0], ps[1, 1]
            wcs = WCS(hdr).celestial
        finite = data[np.isfinite(data)]
        dmin, dmax = float(finite.min()), float(finite.max())
        # the histogram displays the data bulk; the full dmin-dmax
        # range stays reachable through the cut sliders
        hmin, hmax = np.nanpercentile(finite, (0.1, 99.9))
        counts, edges = np.histogram(
            finite, bins=200, range=(hmin, hmax)
        )
        lo, hi = preset_cuts(finite, "zscale")
        return {
            "path": str(path),
            "name": Path(path).name,
            "data": data,
            "wcs": wcs,
            "band": str(hdr.get("FILTER", "?")),
            "dmin": dmin,
            "dmax": dmax,
            "hmin": hmin,
            "hmax": hmax,
            "counts": counts,
            "edges": edges,
            "lo": lo,
            "hi": hi,
            "stretch": "zscale",
            "preset": "zscale",
            "cmap": "gray",
            "invert": False,
            "ax": None,
            "im": None,
        }

    def _build_sidebar(self):
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(240)
        box = QVBoxLayout(self._sidebar)

        title = QLabel("euclid-cutouts\nviewer")
        font = title.font()
        font.setBold(True)
        font.setPointSize(13)
        title.setFont(font)
        box.addWidget(title)

        self.image_list = QListWidget()
        for f in self._files:
            self.image_list.addItem(Path(f).name)
        self.image_list.setCurrentRow(0)
        self.image_list.currentRowChanged.connect(self._set_active)
        box.addWidget(self.image_list, stretch=1)

        contrast = QGroupBox("Contrast")
        cbox = QVBoxLayout(contrast)

        row = QHBoxLayout()
        row.addWidget(QLabel("Stretch"))
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(STRETCHES)
        self.stretch_combo.currentTextChanged.connect(
            self._stretch_changed
        )
        row.addWidget(self.stretch_combo)
        cbox.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cuts"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS)
        self.preset_combo.activated.connect(self._apply_preset)
        row.addWidget(self.preset_combo)
        zscale_btn = QPushButton("ZScale")
        zscale_btn.clicked.connect(self._zscale)
        row.addWidget(zscale_btn)
        cbox.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(CMAPS)
        self.cmap_combo.currentTextChanged.connect(self._refresh)
        row.addWidget(self.cmap_combo)
        cbox.addLayout(row)

        self.lo_label = QLabel("low")
        self.lo_slider = QSlider(Qt.Orientation.Horizontal)
        self.lo_slider.setRange(0, SLIDER_STEPS)
        self.hi_label = QLabel("high")
        self.hi_slider = QSlider(Qt.Orientation.Horizontal)
        self.hi_slider.setRange(0, SLIDER_STEPS)
        self.lo_slider.valueChanged.connect(self._slider_cuts)
        self.hi_slider.valueChanged.connect(self._slider_cuts)
        cbox.addWidget(self.lo_label)
        cbox.addWidget(self.lo_slider)
        cbox.addWidget(self.hi_label)
        cbox.addWidget(self.hi_slider)

        row = QHBoxLayout()
        self.invert_check = QCheckBox("Invert")
        self.invert_check.toggled.connect(self._refresh)
        self.keep_check = QCheckBox("Keep cuts")
        row.addWidget(self.invert_check)
        row.addWidget(self.keep_check)
        cbox.addLayout(row)
        box.addWidget(contrast)

        marker = QGroupBox("Crosshair")
        mbox = QVBoxLayout(marker)
        self.ra_edit = QLineEdit()
        self.ra_edit.setPlaceholderText("RA (deg or hh:mm:ss)")
        self.dec_edit = QLineEdit()
        self.dec_edit.setPlaceholderText("Dec (deg or dd:mm:ss)")
        mbox.addWidget(self.ra_edit)
        mbox.addWidget(self.dec_edit)
        row = QHBoxLayout()
        mark_btn = QPushButton("Mark")
        mark_btn.clicked.connect(self._mark_from_fields)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_mark)
        row.addWidget(mark_btn)
        row.addWidget(clear_btn)
        mbox.addLayout(row)
        self.click_check = QCheckBox("Click image to mark")
        mbox.addWidget(self.click_check)
        recentre_btn = QPushButton("Recentre")
        recentre_btn.setToolTip(
            "Centroid the nearest source to the crosshair"
        )
        recentre_btn.clicked.connect(self._recentre)
        mbox.addWidget(recentre_btn)
        self.mark_label = QLabel("crosshair size: 100%")
        self.mark_slider = QSlider(Qt.Orientation.Horizontal)
        self.mark_slider.setRange(10, 300)
        self.mark_slider.setValue(100)
        self.mark_slider.valueChanged.connect(self._mark_size)
        mbox.addWidget(self.mark_label)
        mbox.addWidget(self.mark_slider)
        box.addWidget(marker)
        box.addStretch(1)

    def _panel(self):
        return self.panels[self.active]

    def _set_active(self, idx):
        """Select the panel the contrast controls edit."""
        if not 0 <= idx < len(self.panels) or self.panels[idx]["ax"] is None:
            return
        if self.keep_check.isChecked() and idx != self.active:
            # carry the fractional cut window onto the new image
            old = self._panel()
            span = old["dmax"] - old["dmin"] or 1.0
            new = self.panels[idx]
            new["lo"] = new["dmin"] + (
                (old["lo"] - old["dmin"]) / span
            ) * (new["dmax"] - new["dmin"] or 1.0)
            new["hi"] = new["dmin"] + (
                (old["hi"] - old["dmin"]) / span
            ) * (new["dmax"] - new["dmin"] or 1.0)
            self._apply_norm(new)
        self.active = idx
        self._syncing = True
        if self.image_list.currentRow() != idx:
            self.image_list.setCurrentRow(idx)
        p = self._panel()
        self.stretch_combo.setCurrentText(p["stretch"])
        self.preset_combo.setCurrentText(p["preset"])
        self.cmap_combo.setCurrentText(p["cmap"])
        self.invert_check.setChecked(p["invert"])
        self._syncing = False
        self._highlight()
        self._draw_hist()
        self._sync_sliders()
        self.statusBar().showMessage(
            f"editing {idx + 1}/{len(self.panels)}: {p['name']}"
        )

    def _highlight(self):
        """Frame the active panel so edits have an obvious target."""
        for i, p in enumerate(self.panels):
            if p["ax"] is None:
                continue
            color = "tab:red" if i == self.active else "0.3"
            width = 1.6 if i == self.active else 0.8
            try:
                p["ax"].coords.frame.set_color(color)
                p["ax"].coords.frame.set_linewidth(width)
            except Exception:
                pass
        self.canvas.draw_idle()

    # --- per-panel contrast state ---

    def _norm_for(self, p):
        lo, hi = sorted((p["lo"], p["hi"]))
        return ImageNormalize(
            p["data"],
            interval=ManualInterval(vmin=lo, vmax=hi),
            stretch=STRETCHES[p["stretch"]],
        )

    def _cmap_for(self, p):
        return p["cmap"] + "_r" if p["invert"] else p["cmap"]

    def _apply_norm(self, p):
        if p["im"] is not None:
            p["im"].set_norm(self._norm_for(p))
            p["im"].set_cmap(self._cmap_for(p))

    def _stretch_changed(self, name):
        if self._syncing:
            return
        p = self._panel()
        p["stretch"] = name
        if name == "zscale":
            # zscale bundles its autocuts with the linear display
            finite = p["data"][np.isfinite(p["data"])]
            p["preset"] = "zscale"
            p["lo"], p["hi"] = preset_cuts(finite, "zscale")
        self._refresh()

    def _apply_preset(self, *_):
        p = self._panel()
        finite = p["data"][np.isfinite(p["data"])]
        p["preset"] = self.preset_combo.currentText()
        p["lo"], p["hi"] = preset_cuts(finite, p["preset"])
        self._refresh()

    def _zscale(self):
        """One-click reset to zscale cuts, like the DS9 scale menu."""
        self.preset_combo.setCurrentText("zscale")
        self._apply_preset()

    def _slider_cuts(self):
        if self._syncing:
            return
        p = self._panel()
        span = p["dmax"] - p["dmin"] or 1.0
        p["lo"] = p["dmin"] + (
            self.lo_slider.value() / SLIDER_STEPS
        ) * span
        p["hi"] = p["dmin"] + (
            self.hi_slider.value() / SLIDER_STEPS
        ) * span
        self._refresh()

    def _refresh(self, *_):
        if self._syncing:
            return
        p = self._panel()
        p["stretch"] = self.stretch_combo.currentText()
        p["cmap"] = self.cmap_combo.currentText()
        p["invert"] = self.invert_check.isChecked()
        self._apply_norm(p)
        lo, hi = sorted((p["lo"], p["hi"]))
        if self._lo_line is not None:
            self._lo_line.set_xdata([lo, lo])
            self._hi_line.set_xdata([hi, hi])
        self._sync_sliders()
        self.canvas.draw_idle()

    def _sync_sliders(self):
        p = self._panel()
        span = p["dmax"] - p["dmin"] or 1.0
        prev = self._syncing
        self._syncing = True
        self.lo_slider.setValue(
            int(round(SLIDER_STEPS * (p["lo"] - p["dmin"]) / span))
        )
        self.hi_slider.setValue(
            int(round(SLIDER_STEPS * (p["hi"] - p["dmin"]) / span))
        )
        self._syncing = prev
        self.lo_label.setText(f"low: {p['lo']:g}")
        self.hi_label.setText(f"high: {p['hi']:g}")

    # --- figure ---

    def _draw(self):
        self.fig.clf()
        n = len(self.panels)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        for i, p in enumerate(self.panels):
            ax = self.fig.add_subplot(
                rows, cols, i + 1, projection=p["wcs"], label=f"img{i}"
            )
            p["ax"] = ax
            p["im"] = ax.imshow(
                p["data"],
                origin="lower",
                cmap=self._cmap_for(p),
                norm=self._norm_for(p),
                interpolation="none",
            )
            ax.coords.grid(True, color="cyan", alpha=0.3, ls=":")
            for axis in ("ra", "dec"):
                ax.coords[axis].set_ticklabel(size=7)
            ax.set_title(p["band"], fontsize=9, loc="left")
            for axis, label in (("ra", "RA"), ("dec", "Dec")):
                ax.coords[axis].set_axislabel(label)

        self.ax_hist = self.fig.add_axes(
            [0.09, 0.03, 0.88, 0.09], label="histogram"
        )
        self.fig.subplots_adjust(
            left=0.07, right=0.98, top=0.94, bottom=0.16,
            hspace=0.35, wspace=0.3,
        )
        self._redraw_marks()

    def _draw_hist(self):
        p = self._panel()
        self.ax_hist.cla()
        self.ax_hist.fill_between(
            p["edges"][:-1],
            np.log10(np.clip(p["counts"], 1, None)),
            step="post",
            color="0.7",
            lw=0,
        )
        self.ax_hist.set_xlim(p["hmin"], p["hmax"])
        self.ax_hist.set_ylim(bottom=0)
        self.ax_hist.set_xticks([])
        self.ax_hist.set_yticks([])
        for spine in self.ax_hist.spines.values():
            spine.set_linewidth(0.5)
        self._lo_line = self.ax_hist.axvline(
            p["lo"], color="green", lw=1.2, picker=6
        )
        self._hi_line = self.ax_hist.axvline(
            p["hi"], color="red", lw=1.2, picker=6
        )
        self.ax_hist.set_title(
            p["name"], fontsize=7, loc="left", pad=1
        )
        self.canvas.draw_idle()

    # --- crosshair (shared sky position on every panel) ---

    def _mark_from_fields(self):
        ra_text = self.ra_edit.text().strip()
        dec_text = self.dec_edit.text().strip()
        if not dec_text:
            # a pasted "ra,dec" or "ra dec" pair in the RA field
            parts = ra_text.replace(",", " ").split()
            if len(parts) == 2:
                ra_text, dec_text = parts
        text = f"{ra_text} {dec_text}"
        coord = None
        for units in ((u.deg, u.deg), (u.hourangle, u.deg), None):
            try:
                coord = (
                    SkyCoord(text, unit=units)
                    if units
                    else SkyCoord(text)
                )
                break
            except Exception:
                continue
        if coord is None:
            self.statusBar().showMessage(
                f"cannot parse position: {text!r}"
            )
            return
        self._set_mark(coord)

    def _set_mark(self, coord):
        self._mark = coord
        self.statusBar().showMessage(
            f"crosshair at RA {coord.ra.deg:.7f}, Dec {coord.dec.deg:.7f}"
        )
        self._redraw_marks()

    def _clear_mark(self):
        self._mark = None
        self._redraw_marks()

    def _redraw_marks(self):
        for line in self._mark_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self._mark_lines = []
        if self._mark is not None:
            for p in self.panels:
                if p["ax"] is None:
                    continue
                x, y = np.asarray(
                    p["wcs"].world_to_pixel(self._mark)
                ).ravel()
                ny, nx = p["data"].shape
                if not (0 <= x < nx and 0 <= y < ny):
                    continue
                r1 = 0.02 * nx * self._mark_scale
                r2 = 0.06 * nx * self._mark_scale
                for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    (line,) = p["ax"].plot(
                        [x + dx * r1, x + dx * r2],
                        [y + dy * r1, y + dy * r2],
                        color="red",
                        lw=1.4,
                    )
                    self._mark_lines.append(line)
        self.canvas.draw_idle()

    def _mark_size(self, value):
        self._mark_scale = value / 100.0
        self.mark_label.setText(f"crosshair size: {value}%")
        self._redraw_marks()

    def _recentre(self):
        """Centroid the source nearest the crosshair on the active panel."""
        if self._mark is None:
            self.statusBar().showMessage("set a crosshair first")
            return
        p = self._panel()
        x, y = np.asarray(
            p["wcs"].world_to_pixel(self._mark)
        ).ravel()
        ny, nx = p["data"].shape
        scale = float(proj_plane_pixel_scales(p["wcs"])[0] * 3600.0)
        half = max(int(round(5.0 / scale)), 4)
        xc, yc = int(round(x)), int(round(y))
        x1, x2 = np.clip([xc - half, xc + half + 1], 0, nx)
        y1, y2 = np.clip([yc - half, yc + half + 1], 0, ny)
        sub = p["data"][y1:y2, x1:x2]
        if sub.size == 0:
            self.statusBar().showMessage("crosshair is off this image")
            return
        med, std = np.nanmedian(sub), np.nanstd(sub)
        mask = sub > med + 3 * std
        if mask.sum() < 4:
            self.statusBar().showMessage(
                "no source found near the crosshair"
            )
            return
        cy, cx = ndimage.center_of_mass(sub, mask, 1)
        coord = p["wcs"].pixel_to_world(x1 + cx, y1 + cy)
        self.statusBar().showMessage(
            f"re-centred: RA {coord.ra.deg:.7f}, Dec {coord.dec.deg:.7f}"
        )
        self._mark = coord
        self._redraw_marks()

    # --- histogram cut dragging (DS9 scale-window behaviour) ---
    # --- plus click-to-mark / panel selection on the images ---

    def _canvas_press(self, event):
        if event.button != 1:
            return
        if event.inaxes is self.ax_hist:
            if event.xdata is None:
                return
            p = self._panel()
            v = float(event.xdata)
            self._drag = (
                "lo" if abs(v - p["lo"]) <= abs(v - p["hi"]) else "hi"
            )
            if self._drag == "lo":
                p["lo"] = v
            else:
                p["hi"] = v
            self._refresh()
            return
        hit = next(
            (
                (i, p)
                for i, p in enumerate(self.panels)
                if event.inaxes is p["ax"]
            ),
            None,
        )
        if hit is None:
            return
        idx, p = hit
        if self.click_check.isChecked() and event.xdata is not None:
            coord = p["wcs"].pixel_to_world(event.xdata, event.ydata)
            self._set_mark(coord)
        elif idx != self.active:
            self._set_active(idx)

    def _scroll_zoom(self, event):
        """Wheel zoom on the panel under the cursor, centred there."""
        if event.xdata is None or event.inaxes in (None, self.ax_hist):
            return
        ax = event.inaxes
        if not any(ax is p["ax"] for p in self.panels):
            return
        factor = 0.8 if event.button == "up" else 1.25
        x, y = event.xdata, event.ydata
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        fx, fy = (x - xl) / (xr - xl), (y - yb) / (yt - yb)
        w, h = (xr - xl) * factor, (yt - yb) * factor
        ax.set_xlim(x - fx * w, x + (1 - fx) * w)
        ax.set_ylim(y - fy * h, y + (1 - fy) * h)
        self.canvas.draw_idle()

    def _hist_drag(self, event):
        if self._drag not in ("lo", "hi"):
            return
        if event.inaxes is not self.ax_hist or event.xdata is None:
            return
        p = self._panel()
        if self._drag == "lo":
            p["lo"] = float(event.xdata)
        else:
            p["hi"] = float(event.xdata)
        self._refresh()

    def _hist_drop(self, _event):
        self._drag = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self._set_active((self.active - 1) % len(self.panels))
        elif event.key() == Qt.Key.Key_Right:
            self._set_active((self.active + 1) % len(self.panels))
        elif event.key() == Qt.Key.Key_I:
            self.invert_check.toggle()
        elif event.key() == Qt.Key.Key_C:
            self._recentre()
        else:
            super().keyPressEvent(event)


def show(files):
    """Open the viewer window on `files`. Blocks until it closes."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = FitsViewer(files)
    win.resize(900, 800)
    win.show()
    app.exec()


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit("usage: fits_viewer.py image1.fits [image2.fits ...]")
    show(files)


if __name__ == "__main__":
    main()
