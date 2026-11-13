#!/usr/bin/env python
"""make_collage - multi-panel FITS collage with a target marker.

One WCS-projected panel per FITS file (2x2 when four are given, a single
panel for one). ZScale stretch, no interpolation, a small zoom box on the
target, a 5-arcsecond zoomed inset with notch markers, an N/E compass in
the bottom-right, a band label top-left, and the product id bottom-left.
Mirrored images are flipped
to north-up/east-left before display.

Usage:
    make_collage.py <ra> <dec> <out.png> <fits1> <fits2> ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import termprint as tp

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    LinearStretch,
    ZScaleInterval,
)
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


def add_compass(ax, wcs, shape, color="red"):
    """N and E arrows, bottom-right corner.

    Drawn in axes coordinates so WCS rotation only changes the arrow
    directions and can never push the arms or labels over the spines.
    The origin is shifted inward as needed for panels where E points
    right or N points down.
    """
    ny, nx = shape
    px0 = np.array([0.5 * nx, 0.5 * ny])
    c0 = wcs.pixel_to_world(*px0)
    vectors = {}
    for pa, label in ((0.0, "N"), (90.0, "E")):
        tip = wcs.world_to_pixel(
            c0.directional_offset_by(pa * u.deg, 30.0 * u.arcsec)
        )
        vec = (np.asarray(tip) - px0) / np.array([nx, ny])
        vectors[label] = vec / np.abs(vec).max()

    arm = 0.08
    label_arm = 1.35 * arm
    origin = np.array([0.90, 0.10])
    tips = np.array([origin + arm * v for v in vectors.values()])
    lbls = np.array([origin + label_arm * v for v in vectors.values()])
    # arrow tips get a small margin; labels a larger one since text
    # extends past its anchor point
    for pts, lo, hi in ((tips, 0.04, 0.96), (lbls, 0.075, 0.925)):
        origin += np.minimum(0.0, hi - pts[:, 0].max())
        origin += np.maximum(0.0, lo - pts[:, 0].min())
        origin += np.minimum(0.0, hi - pts[:, 1].max())
        origin += np.maximum(0.0, lo - pts[:, 1].min())

    ax.plot(
        origin[0],
        origin[1],
        marker="o",
        ms=5,
        mfc="none",
        mec=color,
        mew=1.2,
        transform=ax.transAxes,
    )
    for label, vec in vectors.items():
        ax.annotate(
            "",
            xy=tuple(origin + arm * vec),
            xycoords="axes fraction",
            xytext=tuple(origin),
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
        )
        ax.annotate(
            label,
            xy=tuple(origin + label_arm * vec),
            xycoords="axes fraction",
            color=color,
            fontsize=9,
            ha="center",
            va="center",
        )


def nice_scale_arcsec(field_as):
    """Largest round-number scale bar that stays well inside the field."""
    for s in (60, 30, 20, 10, 5, 2, 1):
        if s <= 0.35 * field_as:
            return s
    return 0.5


def add_scalebar(ax, arcsec, scale_as, shape, color="yellow", native_as=None):
    """Horizontal scale bar, bottom-left, above the product label."""
    ny, nx = shape
    sx, sy = scale_as
    x0, y0 = 0.05 * nx, 0.15 * ny
    bar = arcsec / sx
    ax.plot(
        [x0, x0 + bar], [y0, y0], color=color, lw=2.0, solid_capstyle="butt"
    )
    ax.text(
        x0 + bar / 2,
        y0 + 0.02 * ny,
        f'{arcsec:g}"',
        color=color,
        fontsize=8,
        ha="center",
        va="bottom",
    )
    px_label = (
        f'{sx:.2f}"/px' if np.isclose(sx, sy) else f'{sx:.2f}x{sy:.2f}"/px'
    )
    if native_as is not None and not np.isclose(sx, native_as, rtol=0.1):
        px_label += f" ({native_as:.2f} native)"
    ax.text(
        x0 + bar / 2,
        y0 - 0.02 * ny,
        px_label,
        color=color,
        fontsize=7,
        ha="center",
        va="top",
    )


def add_zoom_inset(ax, data, x0, y0, norm, scale_as, color="red"):
    """Top-right inset showing a 5-arcsec-wide zoom centred on the target."""
    ny, nx = data.shape
    sx, sy = scale_as
    half = max(int(round(2.5 / sx)), 3)
    x1, x2 = np.clip([x0 - half, x0 + half], 0, nx).astype(int)
    y1, y2 = np.clip(
        [y0 - int(round(2.5 / sy)), y0 + int(round(2.5 / sy))], 0, ny
    ).astype(int)
    crop = data[y1:y2, x1:x2]
    if crop.size == 0:
        return

    # Inset shares the parent's pixel coordinate space via extent, which lets
    # indicate_inset_zoom draw the connector box in the right place.
    axins = ax.inset_axes([0.70, 0.70, 0.28, 0.28], transform=ax.transAxes)
    axins.imshow(
        crop,
        origin="lower",
        cmap="gray",
        norm=norm,
        interpolation="none",
        extent=(x1 - 0.5, x2 - 0.5, y1 - 0.5, y2 - 0.5),
    )
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.0)

    r1, r2 = 0.25 * half, 0.7 * half
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        axins.plot(
            [x0 + dx * r1, x0 + dx * r2],
            [y0 + dy * r1, y0 + dy * r2],
            color=color,
            lw=1.2,
        )
    # 1-arcsec scale bar, bottom-left of the inset
    bar = 1.0 / sx
    bx, by = x1 + 0.08 * (x2 - x1), y1 + 0.09 * (y2 - y1)
    axins.plot(
        [bx, bx + bar], [by, by], color="yellow", lw=1.5,
        solid_capstyle="butt",
    )
    axins.text(
        bx + bar / 2,
        by + 0.06 * (y2 - y1),
        '1"',
        color="yellow",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    ax.indicate_inset_zoom(axins, edgecolor=color, alpha=0.6, lw=0.8)

BAND_ORDER = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]

# Throughput range shown under the band name: 50% cut-on/off edges of
# the total spectral response. NISP values from Euclid Collab.,
# Schirmer+2022 (A&A 662, A92, Table 1); VIS from the FWHM of
# Euclid_VIS.vis.dat. Escapes keep the source plain ASCII.
BAND_LABELS = {
    "VIS": "VIS (5400-9100 \u00c5)",
    "NIR_Y": "Y (0.950-1.212 \u00b5m)",
    "NIR_J": "J (1.168-1.567 \u00b5m)",
    "NIR_H": "H (1.522-2.021 \u00b5m)",
    "Y": "Y (0.950-1.212 \u00b5m)",
    "J": "J (1.168-1.567 \u00b5m)",
    "H": "H (1.522-2.021 \u00b5m)",
}

# Native detector pixel scale in arcsec. MER resamples NISP mosaics onto
# the VIS 0.1 arcsec grid, so a file can carry a finer scale than its
# detector; the label notes the native value when they differ.
NATIVE_SCALE = {
    "VIS": 0.1,
    "NIR_Y": 0.3,
    "NIR_J": 0.3,
    "NIR_H": 0.3,
    "Y": 0.3,
    "J": 0.3,
    "H": 0.3,
}


def band_sort_key(path):
    hdr = fits.getheader(path)
    band = str(hdr.get("FILTER", ""))
    return (BAND_ORDER.index(band) if band in BAND_ORDER else 99, path)


def render(files, ra, dec, out=None, show=True):
    """Draw the collage for `files` at position (ra, dec).

    `out` writes a PNG; `show` opens an interactive window. Both default
    off/None when called from euclid-cutouts, which picks one itself.
    """
    files = sorted((str(f) for f in files), key=band_sort_key)
    target = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")

    grid = 2 if len(files) > 1 else 1
    fig = plt.figure(figsize=(5 * grid, 5 * grid))
    for i, fname in enumerate(files[:4]):
        with fits.open(fname) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
        # normalize handedness: a positive-determinant pixel scale means
        # the image is mirrored relative to the N-up/E-left convention
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
        band = str(hdr.get("FILTER", "?"))
        inst = str(hdr.get("INSTRUME", "?"))

        ax = fig.add_subplot(grid, grid, i + 1, projection=wcs)
        norm = ImageNormalize(
            data, interval=ZScaleInterval(), stretch=LinearStretch()
        )
        ax.imshow(data, origin="lower", cmap="gray", norm=norm,
                  interpolation="none")
        scales = proj_plane_pixel_scales(wcs) * 3600.0
        add_compass(ax, wcs, data.shape)
        add_scalebar(
            ax, nice_scale_arcsec(data.shape[1] * scales[0]), scales,
            data.shape, native_as=NATIVE_SCALE.get(band),
        )
        x0, y0 = wcs.world_to_pixel(target)
        add_zoom_inset(ax, data, x0, y0, norm, scales)
        ax.coords.grid(True, color="cyan", alpha=0.3, ls=":")
        for axis in ("ra", "dec"):
            ax.coords[axis].set_axislabel(
                "RA" if axis == "ra" else "Dec", fontsize=10
            )
            ax.coords[axis].set_ticklabel(size=8)
        # EUC_<rel>_<band tokens>_<descriptor>_r<N>as.fits: the band may
        # contain '_' itself (DECAM_i), so the product id is everything
        # between the band tokens and the trailing size token, minus RAW
        # and timestamp tokens.
        stem = Path(fname).stem.split("_")
        descriptor = stem[2 + len(band.split("_")) : -1]
        product = "_".join(
            t
            for t in descriptor
            if t != "RAW" and not (t.startswith("T") and t[1:].isdigit())
        )
        ax.text(
            0.02,
            0.98,
            BAND_LABELS.get(band, band.removeprefix("NIR_")),
            transform=ax.transAxes,
            color="yellow",
            fontsize=9,
            ha="left",
            va="top",
        )
        ax.text(
            0.02,
            0.02,
            product,
            transform=ax.transAxes,
            color="yellow",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=200)
        return Path(out)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def main():
    ra, dec = float(sys.argv[1]), float(sys.argv[2])
    out = Path(sys.argv[3])
    render(sys.argv[4:], ra, dec, out=out)
    tp.ok(f"saved {out}")


if __name__ == "__main__":
    main()
