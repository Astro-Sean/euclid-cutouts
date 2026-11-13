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
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure

# Stretch functions between the cut range and the colormap. Asinh is
# the default: it keeps faint structure visible alongside saturated
# cores, which is the usual reason for opening the viewer.
STRETCHES = {
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
    def __init__(self, files):
        super().__init__()
        self.setWindowTitle("euclid-cutouts viewer")
        self._files = [str(f) for f in files]
        self._idx = 0
        self._drag = None  # 'lo'|'hi' while a histogram marker is held
        self._syncing = False  # guards slider setValue recursion
        self._mark = None  # SkyCoord of the crosshair, or None

        self.fig = Figure(figsize=(8, 7))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_img = None
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
        self.statusBar().showMessage(f"{len(self._files)} file(s)")

        self.canvas.mpl_connect("button_press_event", self._canvas_press)
        self.canvas.mpl_connect("motion_notify_event", self._hist_drag)
        self.canvas.mpl_connect("button_release_event", self._hist_drop)
        self._load(0)

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
        self.image_list.currentRowChanged.connect(self._load)
        box.addWidget(self.image_list, stretch=1)

        contrast = QGroupBox("Contrast")
        cbox = QVBoxLayout(contrast)

        row = QHBoxLayout()
        row.addWidget(QLabel("Stretch"))
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems(STRETCHES)
        self.stretch_combo.currentTextChanged.connect(self._refresh)
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
        box.addWidget(marker)
        box.addStretch(1)

    def _load(self, idx):
        if not 0 <= idx < len(self._files):
            return
        self._idx = idx
        with fits.open(self._files[idx]) as hdul:
            self.data = hdul[0].data
            hdr = hdul[0].header
        self.wcs = WCS(hdr).celestial
        self.band = str(hdr.get("FILTER", "?"))
        ps = self.wcs.pixel_scale_matrix
        if np.linalg.det(ps) > 0:
            self.data = self.data[:, ::-1]
            hdr = hdr.copy()
            hdr["CRPIX1"] = hdr["NAXIS1"] + 1 - hdr["CRPIX1"]
            for k in ("PC1_1", "PC1_2", "PC2_1", "PC2_2",
                      "CDELT1", "CDELT2"):
                hdr.pop(k, None)
            hdr["CD1_1"], hdr["CD1_2"] = -ps[0, 0], ps[0, 1]
            hdr["CD2_1"], hdr["CD2_2"] = -ps[1, 0], ps[1, 1]
            self.wcs = WCS(hdr).celestial

        finite = self.data[np.isfinite(self.data)]
        self.dmin, self.dmax = float(finite.min()), float(finite.max())
        # the histogram displays the data bulk; the full dmin-dmax
        # range stays reachable through the cut sliders
        self.hmin, self.hmax = np.nanpercentile(finite, (0.1, 99.9))
        self.counts, self.edges = np.histogram(
            finite, bins=200, range=(self.hmin, self.hmax)
        )
        if self.keep_check.isChecked() and hasattr(self, "_keep_frac"):
            lo_f, hi_f = self._keep_frac
            self._lo = self.dmin + lo_f * (self.dmax - self.dmin)
            self._hi = self.dmin + hi_f * (self.dmax - self.dmin)
        else:
            self._lo, self._hi = preset_cuts(
                finite, self.preset_combo.currentText()
            )
        self._draw()
        self.statusBar().showMessage(
            f"{idx + 1}/{len(self._files)}: {Path(self._files[idx]).name}"
        )

    def _apply_preset(self, *_):
        self._lo, self._hi = preset_cuts(
            self.data[np.isfinite(self.data)],
            self.preset_combo.currentText(),
        )
        self._draw()

    def _zscale(self):
        """One-click reset to zscale cuts, like the DS9 scale menu."""
        self.preset_combo.setCurrentText("zscale")
        self._apply_preset()

    def _cmap(self):
        name = self.cmap_combo.currentText()
        return name + "_r" if self.invert_check.isChecked() else name

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
        self._draw()

    def _clear_mark(self):
        self._mark = None
        self._draw()

    def _draw_mark(self):
        if self._mark is None:
            return
        x, y = self.wcs.world_to_pixel(self._mark)
        ny, nx = self.data.shape
        if not (0 <= x < nx and 0 <= y < ny):
            return  # target falls outside this image
        r1, r2 = 0.015 * nx, 0.05 * nx
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            self.ax_img.plot(
                [x + dx * r1, x + dx * r2],
                [y + dy * r1, y + dy * r2],
                color="red",
                lw=1.4,
            )

    def _slider_cuts(self):
        if self._syncing:
            return
        lo_f = self.lo_slider.value() / SLIDER_STEPS
        hi_f = self.hi_slider.value() / SLIDER_STEPS
        self._lo = self.dmin + lo_f * (self.dmax - self.dmin)
        self._hi = self.dmin + hi_f * (self.dmax - self.dmin)
        self._refresh()

    def _norm(self):
        lo, hi = sorted((self._lo, self._hi))
        return ImageNormalize(
            self.data,
            interval=ManualInterval(vmin=lo, vmax=hi),
            stretch=STRETCHES[self.stretch_combo.currentText()],
        )

    def _draw(self):
        self.fig.clf()
        self.ax_img = self.fig.add_subplot(
            1, 1, 1, projection=self.wcs, label="image"
        )
        self.ax_img.imshow(
            self.data,
            origin="lower",
            cmap=self._cmap(),
            norm=self._norm(),
            interpolation="none",
        )
        self._draw_mark()
        self.ax_img.coords.grid(True, color="cyan", alpha=0.3, ls=":")
        self.ax_img.coords["ra"].set_axislabel("RA", fontsize=10)
        self.ax_img.coords["dec"].set_axislabel("Dec", fontsize=10)
        for axis in ("ra", "dec"):
            self.ax_img.coords[axis].set_ticklabel(size=8)
        self.ax_img.set_title(
            f"{self.band}  -  {Path(self._files[self._idx]).name}",
            fontsize=9,
            loc="left",
        )

        self.ax_hist = self.fig.add_axes(
            [0.10, 0.04, 0.86, 0.10], label="histogram"
        )
        self.ax_hist.fill_between(
            self.edges[:-1],
            np.log10(np.clip(self.counts, 1, None)),
            step="post",
            color="0.7",
            lw=0,
        )
        self.ax_hist.set_xlim(self.hmin, self.hmax)
        self.ax_hist.set_ylim(bottom=0)
        self.ax_hist.set_xticks([])
        self.ax_hist.set_yticks([])
        for spine in self.ax_hist.spines.values():
            spine.set_linewidth(0.5)
        self._lo_line = self.ax_hist.axvline(
            self._lo, color="green", lw=1.2, picker=6
        )
        self._hi_line = self.ax_hist.axvline(
            self._hi, color="red", lw=1.2, picker=6
        )
        self.fig.subplots_adjust(
            left=0.10, right=0.97, top=0.93, bottom=0.18
        )
        self._sync_sliders()
        self.canvas.draw_idle()

    def _sync_sliders(self):
        span = self.dmax - self.dmin or 1.0
        self._syncing = True
        self.lo_slider.setValue(
            int(round(SLIDER_STEPS * (self._lo - self.dmin) / span))
        )
        self.hi_slider.setValue(
            int(round(SLIDER_STEPS * (self._hi - self.dmin) / span))
        )
        self._syncing = False
        self.lo_label.setText(f"low: {self._lo:g}")
        self.hi_label.setText(f"high: {self._hi:g}")

    def _refresh(self, *_):
        if self.ax_img is None:
            return
        lo, hi = sorted((self._lo, self._hi))
        norm = self._norm()
        norm.vmin, norm.vmax = lo, hi
        for artist in self.ax_img.images:
            artist.set_norm(norm)
            artist.set_cmap(self._cmap())
        if self._lo_line:
            self._lo_line.set_xdata([lo, lo])
        if self._hi_line:
            self._hi_line.set_xdata([hi, hi])
        self._keep_frac = (
            (self._lo - self.dmin) / (self.dmax - self.dmin or 1),
            (self._hi - self.dmin) / (self.dmax - self.dmin or 1),
        )
        self._sync_sliders()
        self.canvas.draw_idle()

    # --- histogram cut dragging (DS9 scale-window behaviour) ---
    # --- plus click-to-mark on the image when enabled ---

    def _hist_value(self, event):
        if event.inaxes is not self.ax_hist or event.xdata is None:
            return None
        return float(event.xdata)

    def _canvas_press(self, event):
        if event.button != 1:
            return
        if event.inaxes is self.ax_hist:
            v = self._hist_value(event)
            if v is None:
                return
            self._drag = (
                "lo" if abs(v - self._lo) <= abs(v - self._hi) else "hi"
            )
            if self._drag == "lo":
                self._lo = v
            else:
                self._hi = v
            self._refresh()
        elif (
            event.inaxes is self.ax_img
            and self.click_check.isChecked()
            and event.xdata is not None
        ):
            coord = self.wcs.pixel_to_world(event.xdata, event.ydata)
            self._set_mark(coord)

    def _hist_drag(self, event):
        if self._drag not in ("lo", "hi"):
            return
        v = self._hist_value(event)
        if v is None:
            return
        if self._drag == "lo":
            self._lo = v
        else:
            self._hi = v
        self._refresh()

    def _hist_drop(self, _event):
        self._drag = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.image_list.setCurrentRow(
                (self._idx - 1) % len(self._files)
            )
        elif event.key() == Qt.Key.Key_Right:
            self.image_list.setCurrentRow(
                (self._idx + 1) % len(self._files)
            )
        elif event.key() == Qt.Key.Key_I:
            self.invert_check.toggle()
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
