#!/usr/bin/python3
"""euclid-cutouts - fetch and display Euclid FITS images at a sky position.

Queries the ESA Euclid Science Archive (via astroquery.esa.euclid) for image
products whose footprint contains the position, fetches a cutout per band,
and shows them in a collage figure. Public DR1/Q1 data only, no login needed.
--download saves the FITS files (cutouts by default, or the full product with
--full); saved files get the standard keywords AutoPhOT looks up at ingest
(TELESCOP, INSTRUME, FILTER, DATE-OBS, MJD-OBS, EXPTIME, GAIN, RDNOISE,
SATURATE, AIRMASS) where the archive header lacks them.

Examples:
    euclid-cutouts 60.3372991,-49.9318588
    euclid-cutouts 60.3372991 -49.9318588 --download
    euclid-cutouts "06h04m00.32s -54d15m51.1s"
    euclid-cutouts 60.34 -49.93 -r 60 -b VIS,NIR_Y
    euclid-cutouts 60.34 -49.93 --list
    euclid-cutouts 60.34 -49.93 --include-raw   # raw detector cutouts too
    euclid-cutouts 2026abc                      # name resolved via Sesame
    euclid-cutouts 60.34 -49.93 --pub           # journal-quality PDF+PNG
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
import termprint as tp

try:
    import astropy.units as u
    import numpy as np
    from astropy import log as astropy_log
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS
    from astroquery.esa.euclid import Euclid
    from scipy import ndimage
    from scipy.signal import fftconvolve
    from scipy.spatial import cKDTree
except ModuleNotFoundError as err:
    tp.fail(
        f"{err}; wrong interpreter or missing deps - "
        f"run: pip install -r requirements.txt"
    )
    sys.exit(2)

astropy_log.setLevel("ERROR")

EUCLID_BANDS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]
# Band names are case-significant in the archive (DECAM_i, HSC_r2, ...), so
# aliases only cover the Euclid-native shortcuts.
BAND_ALIASES = {
    "vis": "VIS",
    "y": "NIR_Y",
    "j": "NIR_J",
    "h": "NIR_H",
}
# Server-side limit for the archive cutout service is 0.5 deg.
MAX_RADIUS_ARCSEC = 1800.0
# Products to search: mosaics are the deeper, uniform-processing option and
# are preferred over single-observation stacks when both cover a band.
PRODUCT_TABLES = (("mosaic_product", "MOSAIC"), ("observation_stack", "OBS"))

# Extra TAP columns fetched alongside the footprint query. Used to stamp the
# FITS keywords AutoPhOT expects (dates, exptime, zeropoint). Column names
# differ between releases, so they are keyed by (table, schema).
EXTRA_COLUMNS = {
    "mosaic_product": {
        "dr1": "version, reference_observation_date_time, "
        "zero_point, zero_point_error",
        "q1": "version, zero_point, zero_point_error",
    },
    "observation_stack": {
        "dr1": "version, start_time, duration, zero_point, zero_point_error",
        "q1": "version, start_time, duration, zero_point, zero_point_error",
    },
}

# Nominal detector values, used only when the product header lacks the key.
# Saturate is in raw DN; rate-unit images get it scaled by exp_nominal.
INSTRUMENT_DEFAULTS = {
    "VIS": {"gain": 3.2, "readnoise": 4.5, "saturate": 60000.0, "exp_nominal": 565.0},
    "NISP": {"gain": 2.1, "readnoise": 15.0, "saturate": 50000.0, "exp_nominal": 87.0},
    "DECAM": {"gain": 4.0, "readnoise": 5.0, "saturate": 60000.0, "exp_nominal": 90.0},
    "HSC": {"gain": 3.0, "readnoise": 4.5, "saturate": 65000.0, "exp_nominal": 120.0},
    "MEGACAM": {"gain": 1.8, "readnoise": 5.0, "saturate": 65000.0, "exp_nominal": 100.0},
    "GPC": {"gain": 1.3, "readnoise": 5.0, "saturate": 65000.0, "exp_nominal": 45.0},
}
DETECTOR_FALLBACK = {
    "gain": 1.0,
    "readnoise": 5.0,
    "saturate": 65535.0,
    "exp_nominal": 100.0,
}

# Band -> (instrument, raw_frame filter_name) for the raw-frame fallback.
# Raw NISP frames carry the bare letter; VIS raw uses its own name.
RAW_BAND_MAP = {
    "VIS": ("VIS", "VIS"),
    "NIR_Y": ("NISP", "Y"),
    "NIR_J": ("NISP", "J"),
    "NIR_H": ("NISP", "H"),
}


def parse_position(tokens):
    """Accept 'ra,dec', 'ra dec', or any SkyCoord-readable string."""
    text = " ".join(tokens).replace(",", " ").strip()
    parts = text.split()
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    for units in (None, (u.hourangle, u.deg), (u.deg, u.deg)):
        try:
            coord = SkyCoord(text, unit=units) if units else SkyCoord(text)
            return coord.ra.deg, coord.dec.deg
        except Exception:
            continue
    raise ValueError(f"cannot parse position: {' '.join(tokens)!r}")


def _name_variants(name):
    """SIMBAD alias guesses for a transient name like '2026abc'."""
    variants = [name]
    match = re.match(r"^(?:SN|AT)\s*(\d{4}[A-Za-z]+)$", name, re.IGNORECASE)
    if match:
        core = match.group(1)
    elif re.match(r"^\d{4}[A-Za-z]+$", name):
        core = name
    else:
        core = None
    if core:
        variants.append(core)
        for prefix in ("SN", "AT"):
            variants += [f"{prefix} {core}", f"{prefix}{core}"]
    seen, out = set(), []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def resolve_name(name):
    """Resolve an object name to ICRS degrees via Sesame.

    Sesame checks SIMBAD, NED and VizieR in one query. TNS names like
    2026abc are catalogued under SN/AT prefixes, so each spelling variant
    is tried in turn. Returns (ra, dec, variant) or None.
    """
    tp.info(f"resolving {name!r} via Sesame ...")
    for variant in _name_variants(name):
        try:
            coord = SkyCoord.from_name(variant, cache=False)
        except Exception:
            continue
        tp.ok(
            f"{name!r} -> {variant!r}: "
            f"RA {coord.ra.deg:.7f}, Dec {coord.dec.deg:.7f}"
        )
        return coord.ra.deg, coord.dec.deg, variant
    return None


def parse_bands(text):
    """None means no band filtering (external instruments included)."""
    if text is None or text.lower() == "euclid":
        return list(EUCLID_BANDS)
    if text.lower() == "all":
        return None
    return [
        BAND_ALIASES.get(tok.lower(), tok)
        for tok in text.replace(",", " ").split()
    ]


def coverage_query(schema, table, ra, dec, radius_deg):
    id_col = "tile_index" if table == "mosaic_product" else "observation_id"
    extras = EXTRA_COLUMNS.get(table, {}).get(schema, "version")
    return (
        f"SELECT {id_col} AS pid, instrument_name, filter_name, "
        f"file_name, file_path, {extras}, "
        f"DISTANCE(POINT('ICRS', {ra}, {dec}), POINT('ICRS', ra, dec)) AS dist "
        f"FROM {schema}.{table} WHERE ({schema}.{table}.fov IS NOT NULL AND "
        f"INTERSECTS(CIRCLE('ICRS', {ra}, {dec}, {radius_deg}), "
        f"{schema}.{table}.fov) = 1) ORDER BY dist ASC"
    )


def find_covering_products(schemas, ra, dec, radius_deg):
    products = []
    for schema in schemas:
        for table, kind in PRODUCT_TABLES:
            job = Euclid.launch_job(
                coverage_query(schema, table, ra, dec, radius_deg),
                output_format="votable_plain",
            )
            result = job.get_results() if job else None
            if result is None:
                continue
            for row in result:
                products.append(
                    {
                        "release": schema,
                        "kind": kind,
                        "id": str(row["pid"]),
                        "instrument": str(row["instrument_name"]),
                        "band": str(row["filter_name"]),
                        "file_name": str(row["file_name"]),
                        "file_path": str(row["file_path"]),
                        "dist": float(row["dist"]),
                        "version": _safe_int(row["version"]),
                        "timestamp": _file_timestamp(str(row["file_name"])),
                        "obs_iso": _first_text(
                            row,
                            ("reference_observation_date_time", "start_time"),
                        ),
                        "duration": _safe_float(_val(row, "duration")),
                        "zero_point": _val(row, "zero_point"),
                        "zero_point_err": _val(row, "zero_point_error"),
                    }
                )
    return products


def _val(row, key):
    """Return a table value, treating absent columns and masked cells as None."""
    try:
        v = row[key]
    except (KeyError, IndexError):
        return None
    if v is None or np.ma.is_masked(v):
        return None
    return v


def _first_text(row, keys):
    for key in keys:
        v = _val(row, key)
        if v:
            return str(v)
    return None


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _file_timestamp(file_name):
    match = re.search(r"(\d{8}T\d{6})", file_name)
    return match.group(1) if match else ""


def pick_products(products, wanted_bands, grab_all):
    """Keep one product per band unless grab_all; mosaics beat stacks."""
    products.sort(key=lambda p: p["timestamp"], reverse=True)
    ordered = sorted(
        products,
        key=lambda p: (
            p["kind"] != "MOSAIC",
            p["release"] != "dr1",
            p["dist"],
            -p["version"],
        ),
    )
    chosen, seen = [], set()
    wanted = None if wanted_bands is None else {b.lower() for b in wanted_bands}
    for p in ordered:
        if wanted is not None and p["band"].lower() not in wanted:
            continue
        key = p["file_name"] if grab_all else p["band"]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(p)
    return chosen


def cutout_name(p, radius_arcsec):
    return (
        f"EUC_{p['release']}_{p['band']}_{p['kind']}{p['id']}"
        f"_r{radius_arcsec:g}as.fits"
    )


def _iso_from_file_name(file_name):
    match = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", file_name)
    if not match:
        return None
    return "-".join(match.group(i) for i in (1, 2, 3)) + "T" + ":".join(
        match.group(i) for i in (4, 5, 6)
    )


def stamp_autophot_headers(fits_path, product):
    """Fill in the FITS keywords AutoPhOT looks up when ingesting images.

    check.py auto-detects GAIN, RDNOISE, SATURATE, AIRMASS, MJD-OBS, DATE-OBS
    and EXPTIME (plus TELESCOP/INSTRUME/FILTER for instrument matching).
    Existing archive values are kept; only missing or invalid keys are
    stamped, using archive metadata and nominal detector values.
    """
    det = INSTRUMENT_DEFAULTS.get(
        product["instrument"].upper(), DETECTOR_FALLBACK
    )
    with fits.open(fits_path, mode="update") as hdul:
        hdr = hdul[0].header
        # MER mosaic headers ship an invalid EXTNAME=None card that makes
        # astropy (and downstream tools) reject the file on write.
        if "EXTNAME" in hdr and hdr["EXTNAME"] is None:
            hdr["EXTNAME"] = "SCI"
        bunit = str(hdr.get("BUNIT", "")).lower()
        is_rate = "/s" in bunit
        in_electrons = "electron" in bunit

        hdr["TELESCOP"] = "EUCLID"
        hdr["INSTRUME"] = product["instrument"]
        if not str(hdr.get("FILTER", "")).strip():
            hdr["FILTER"] = product["band"]
        hdr["AIRMASS"] = 1.0
        if "RDNOISE" not in hdr:
            hdr["RDNOISE"] = det["readnoise"]
        if "GAIN" not in hdr:
            hdr["GAIN"] = 1.0 if in_electrons else det["gain"]
        if "SATURATE" not in hdr:
            hdr["SATURATE"] = (
                det["saturate"] / det["exp_nominal"]
                if is_rate
                else det["saturate"]
            )

        # Stacks put EXPTIME=0 on co-adds; the true integration is the
        # observation duration. Co-adds with no single exposure get 1.0 so
        # the data are self-consistent as per-second rates.
        if product["duration"] and product["duration"] > 0:
            hdr["EXPTIME"] = float(product["duration"])
        elif float(hdr.get("EXPTIME", 0.0) or 0.0) <= 0:
            hdr["EXPTIME"] = 1.0

        iso = product["obs_iso"] or _iso_from_file_name(product["file_name"])
        if iso:
            if "DATE-OBS" not in hdr:
                hdr["DATE-OBS"] = iso
            if "MJD-OBS" not in hdr:
                hdr["MJD-OBS"] = round(Time(iso).mjd, 6)
        elif "MJD-OBS" in hdr and "DATE-OBS" not in hdr:
            hdr["DATE-OBS"] = Time(hdr["MJD-OBS"], format="mjd").isot

        zp_keys = ("MAGZPT", "MAGZERO", "MAGZEROP", "PHOTZP", "ZP")
        if product["zero_point"] is not None and not any(
            k in hdr for k in zp_keys
        ):
            hdr["MAGZPT"] = float(product["zero_point"])
            if product["zero_point_err"] is not None:
                hdr["MAGZERR"] = float(product["zero_point_err"])

        hdr["COMMENT"] = "AutoPhOT keywords stamped by euclid-cutouts"


# --- Raw detector frames ------------------------------------------------
#
# Raw frames have no sky WCS: each detector extension stores its position
# in focal-plane millimetres (CTYPE Y_MOSA/Z_MOSA). The transform to the
# sky is recovered from the primary header's FOV_LON/LAT footprint polygon
# matched to the detector mosaic bounding box, then refined against an
# already-downloaded image through source matching.


def _tangent_offsets(ra_deg, dec_deg, ra0, dec0):
    """Gnomonic tangent offsets (E, N) in arcsec about the boresight."""
    a, d, a0, d0 = map(np.deg2rad, (ra_deg, dec_deg, ra0, dec0))
    den = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / den
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / den
    return np.degrees(xi) * 3600.0, np.degrees(eta) * 3600.0


def _corner_transform(hdul):
    """Similarity map from sky tangent arcsec to focal-plane mm.

    The four FOV_LON/LAT footprint corners are paired with the bounding
    box of the 16 detector WCS footprints; the pairing is chosen by least
    residual over cyclic orderings. Returns (scale, R, t, det_wcs, resid).
    """
    ph = hdul[0].header
    ra0, dec0 = ph["RA"], ph["DEC"]
    det_wcs = {}
    mm_min = np.array([np.inf, np.inf])
    mm_max = -mm_min
    for h in hdul:
        if h.name.endswith(".SCI") and h.data is not None:
            w = WCS(h.header)
            det_wcs[h.name] = w
            nx, ny = h.header["NAXIS1"], h.header["NAXIS2"]
            c = np.asarray(
                w.all_pix2world([[0, 0], [nx - 1, 0], [0, ny - 1],
                                 [nx - 1, ny - 1]], 0)
            )
            mm_min = np.minimum(mm_min, c.min(axis=0))
            mm_max = np.maximum(mm_max, c.max(axis=0))
    corners_mm = np.array(
        [[mm_min[0], mm_min[1]], [mm_max[0], mm_min[1]],
         [mm_max[0], mm_max[1]], [mm_min[0], mm_max[1]]]
    )
    lon = np.array([ph[f"FOV_LON{i}"] for i in range(1, 5)])
    lat = np.array([ph[f"FOV_LAT{i}"] for i in range(1, 5)])
    tan = np.vstack(_tangent_offsets(lon, lat, ra0, dec0)).T

    best = None
    for shift in range(4):
        for rev in (False, True):
            idx = np.roll(np.arange(4), shift)
            if rev:
                idx = idx[::-1]
            a, b = tan[idx], corners_mm
            za = a[:, 0] + 1j * a[:, 1]
            zb = b[:, 0] + 1j * b[:, 1]
            m = ((zb - zb.mean()) * np.conj(za - za.mean())).sum() / (
                np.abs(za - za.mean()) ** 2
            ).sum()
            t = zb.mean() - m * za.mean()
            resid = np.abs(m * za + t - zb).max()
            if best is None or resid < best[0]:
                best = resid, abs(m), np.angle(m), t
    resid, s, th, t = best
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return s, R, np.array([t.real, t.imag]), det_wcs, resid, (ra0, dec0)


def _sky_to_mm(tfm, ra_deg, dec_deg):
    s, R, t, _, _, (ra0, dec0) = tfm
    tan = np.array(_tangent_offsets(ra_deg, dec_deg, ra0, dec0))
    return s * (R @ tan) + t


def _mm_jacobian_deg_per_px(tfm, wdet):
    """Tangent-frame deg/pixel Jacobian of a detector, as a CD matrix."""
    s, R, _, _, _, _ = tfm
    if wdet.wcs.has_cd():
        cdmm = np.array(wdet.wcs.cd)
    else:
        cdmm = np.array(wdet.wcs.pc) @ np.diag(wdet.wcs.cdelt)
    return (R.T @ cdmm) / s / 3600.0


def _image_sources(data, pct=98.5, maxn=60):
    """Bright-source centroids, brightest first."""
    lab, n = ndimage.label(data > np.nanpercentile(data, pct))
    if n == 0:
        return np.empty((0, 2))
    cents = np.array(ndimage.center_of_mass(data, lab, range(1, n + 1)))
    peaks = np.array([data[int(y), int(x)] for y, x in cents])
    return cents[np.argsort(peaks)[::-1][:maxn]][:, ::-1]


def _refine_target(det_data, wdet, tfm, ra, dec, ref_path):
    """Pin the predicted target pixel to the reference image's field.

    The corner transform can be off by a few arcmin. Ref-source positions
    are predicted onto the detector, matched to real peaks for a rough
    affine, then a cross-correlation of the reprojected reference against
    the detector data fixes the last translation. Returns (detpix, n).
    """
    wdet_pred = lambda rr, dd: np.asarray(
        wdet.all_world2pix(*_sky_to_mm(tfm, rr, dd), 0)
    ).ravel()
    tpix0 = wdet_pred(ra, dec)
    with fits.open(ref_path) as hdul:
        rdata, rwcs = hdul[0].data, WCS(hdul[0].header)
    rsrc = _image_sources(rdata)
    if len(rsrc) < 4:
        return tpix0, 0
    world = rwcs.pixel_to_world(rsrc[:, 0], rsrc[:, 1])
    mm = np.array([_sky_to_mm(tfm, w.ra.deg, w.dec.deg) for w in world])
    pred_src = np.asarray(wdet.all_world2pix(mm[:, 0], mm[:, 1], 0)).T

    med, sd = np.median(det_data), det_data.std()
    lab, n = ndimage.label(det_data > med + 8 * sd)
    if n == 0:
        return tpix0, 0
    cents = np.array(
        ndimage.center_of_mass(det_data, lab, range(1, n + 1))
    )[:, ::-1]
    dists, idx = cKDTree(cents).query(pred_src)
    ok = dists < 15
    if ok.sum() < 4:
        return tpix0, int(ok.sum())
    pairs_p = cents[idx[ok]]
    pairs_s = np.vstack([world.ra.deg, world.dec.deg]).T[ok]
    A = np.vstack([pairs_p.T, np.ones(len(pairs_p))])
    sol, *_ = np.linalg.lstsq(A.T, pairs_s, rcond=None)
    M, b = sol[:2].T, sol[2]
    tpix = np.linalg.inv(M) @ (np.array([ra, dec]) - b)

    sz = 300
    cx, cy = int(round(tpix[0])), int(round(tpix[1]))
    ny, nx = det_data.shape
    if not (sz <= cx < nx - sz and sz <= cy < ny - sz):
        return tpix, int(ok.sum())
    xg, yg = np.meshgrid(
        np.arange(cx - sz, cx + sz), np.arange(cy - sz, cy + sz)
    )
    sky = M @ np.vstack([xg.ravel(), yg.ravel()]) + b[:, None]
    vx, vy = rwcs.all_world2pix(sky[0], sky[1], 0)
    rimg = ndimage.map_coordinates(
        rdata, [vy.reshape(xg.shape), vx.reshape(xg.shape)], order=1
    )
    jimg = det_data[cy - sz : cy + sz, cx - sz : cx + sz]
    corr = fftconvolve(
        jimg - np.median(jimg),
        (rimg - np.median(rimg))[::-1, ::-1],
        mode="same",
    )
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    return tpix + np.array([ix - sz, iy - sz]), int(ok.sum())


def extract_raw_cutout(raw_path, ra, dec, radius_as, ref_path, out_path):
    """Cut the target's detector image out of a raw frame as a sky FITS.

    Returns (detector_name, n_matched, primary_header) on success, or
    None when the position falls in a detector gap for this exposure.
    """
    with fits.open(raw_path) as hdul:
        tfm = _corner_transform(hdul)
        mm = _sky_to_mm(tfm, ra, dec)
        det_name, pred = None, None
        for name, w in tfm[3].items():
            pxy = np.asarray(w.all_world2pix(*mm, 0)).ravel()
            ny, nx = hdul[name].data.shape
            # wide margin: corner-transform error can be ~200 px, so a
            # slightly-off prediction must not hide real coverage
            if -300 < pxy[0] < nx + 300 and -300 < pxy[1] < ny + 300:
                det_name, pred = name, pxy
                break
        if det_name is None:
            return None
        det_data = hdul[det_name].data.astype(float)
        ph = dict(hdul[0].header)
        wdet = tfm[3][det_name]

    if ref_path:
        try:
            pred, n_match = _refine_target(
                det_data, wdet, tfm, ra, dec, ref_path
            )
        except Exception:
            n_match = 0
    else:
        n_match = 0

    # refined pixel must really be on the detector, not back in the gap
    ny, nx = det_data.shape
    if not (0 <= pred[0] < nx and 0 <= pred[1] < ny):
        return None

    cd_deg = _mm_jacobian_deg_per_px(tfm, wdet)
    scale_as = float(np.hypot(*cd_deg[:, 0]) * 3600.0)
    # Without a reference match the corner transform can miss the target
    # by a few arcmin, so fall back to the whole detector rather than a
    # cutout guaranteed not to contain it.
    if n_match >= 4:
        half = max(int(round(radius_as / scale_as)), 8)
        x0 = int(np.clip(int(round(pred[0])) - half, 0, nx - 1))
        y0 = int(np.clip(int(round(pred[1])) - half, 0, ny - 1))
    else:
        x0, y0 = 0, 0
        half = max(nx, ny)
    cut = det_data[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
    if cut.size == 0:
        return None

    hdr = fits.Header()
    hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
    hdr["CRVAL1"], hdr["CRVAL2"] = ra, dec
    hdr["CRPIX1"], hdr["CRPIX2"] = pred[0] - x0, pred[1] - y0
    hdr["CD1_1"], hdr["CD1_2"] = cd_deg[0, 0], cd_deg[0, 1]
    hdr["CD2_1"], hdr["CD2_2"] = cd_deg[1, 0], cd_deg[1, 1]
    hdr["BUNIT"] = "ADU"
    hdr["ORIGIN"] = f"raw {det_name} (uncalibrated)"
    fits.writeto(out_path, cut, hdr, overwrite=True)
    return det_name, n_match, ph


def raw_coverage_query(schema, ra, dec, radius_deg, instrument, filt):
    return (
        f"SELECT observation_id AS pid, instrument_name, filter_name, "
        f"file_name, file_path, "
        f"DISTANCE(POINT('ICRS', {ra}, {dec}), POINT('ICRS', ra, dec)) AS dist "
        f"FROM {schema}.raw_frame WHERE ({schema}.raw_frame.fov IS NOT NULL AND "
        f"INTERSECTS(CIRCLE('ICRS', {ra}, {dec}, {radius_deg}), "
        f"{schema}.raw_frame.fov) = 1) AND instrument_name='{instrument}' "
        f"AND filter_name='{filt}' ORDER BY file_name ASC"
    )


def fetch_raw_band(band, schemas, ra, dec, radius_as, outdir, ref_path,
                   grab_all, no_header_fix):
    """Download raw frames for a band and cut out the covering detector.

    Each covering exposure is fetched (~200 MB) until one has the target
    on a live detector; --all keeps going through every exposure.
    """
    instrument, filt = RAW_BAND_MAP[band]
    saved = []
    for schema in schemas:
        job = Euclid.launch_job(
            raw_coverage_query(
                schema, ra, dec, radius_as / 3600.0, instrument, filt
            ),
            output_format="votable_plain",
        )
        rows = job.get_results() if job else None
        if not rows or not len(rows):
            continue
        for row in rows:
            obs_id, fn = str(row["pid"]), str(row["file_name"])
            tp.info(f"fetching raw {band} exposure {fn[:50]} ...")
            tmp = tempfile.NamedTemporaryFile(
                suffix=".fits", delete=False
            ).name
            res = Euclid.get_product(
                file_name=fn, schema=schema, output_file=tmp
            )
            if not res or Path(tmp).stat().st_size == 0:
                Path(tmp).unlink(missing_ok=True)
                tp.warn("raw download failed or empty")
                continue
            tmp_out = Path(outdir) / f".raw_{band}_{obs_id}.fits"
            got = extract_raw_cutout(
                tmp, ra, dec, radius_as, ref_path, str(tmp_out)
            )
            Path(tmp).unlink(missing_ok=True)
            if got is None:
                tmp_out.unlink(missing_ok=True)
                tp.warn("position falls in a detector gap in this exposure")
                continue
            det, n_match, ph = got
            tstamp = _file_timestamp(fn)[-7:]
            out = outdir / (
                f"EUC_{schema}_{filt}_RAW_OBS{obs_id}_{det.split('.')[0]}"
                f"_{tstamp}_r{radius_as:g}as.fits"
            )
            tmp_out.rename(out)
            if not no_header_fix:
                try:
                    stamp_autophot_headers(
                        str(out),
                        {
                            "instrument": instrument,
                            "band": filt,
                            "duration": ph.get("EXPTIME"),
                            "obs_iso": ph.get("DATE-OBS"),
                            "zero_point": None,
                            "zero_point_err": None,
                            "file_name": fn,
                        },
                    )
                except Exception as err:
                    tp.warn(f"header stamping failed: {err}")
            note = "" if n_match else " (WCS approximate, no reference match)"
            tp.ok(f"saved {out.name}  [{det}]{note}")
            saved.append(out)
            if not grab_all:
                return saved
    return saved


def confirm_outdir(outdir):
    """Writing files straight into $HOME clutters it; make the user say so."""
    if outdir != Path.home():
        return
    msg = f"output directory is your home directory ({outdir})"
    if not sys.stdin.isatty():
        tp.warn(f"{msg}; continuing (non-interactive)")
        return
    if not tp.ask(f"{msg} - save files here?"):
        tp.info("aborted; pass -o DIR to pick another directory")
        sys.exit(1)


def print_product_table(products):
    tp.table(
        ["REL", "KIND", "BAND", "ID", "DIST(deg)", "FILE"],
        [
            (
                p["release"],
                p["kind"],
                p["band"],
                p["id"],
                f"{p['dist']:.3f}",
                p["file_name"],
            )
            for p in products
        ],
    )


def nearest_products(ra, dec):
    query = (
        f"SELECT tile_index AS pid, instrument_name, filter_name, "
        f"DISTANCE(POINT('ICRS', {ra}, {dec}), POINT('ICRS', ra, dec)) AS dist "
        f"FROM dr1.mosaic_product ORDER BY dist ASC"
    )
    job = Euclid.launch_job(query, output_format="votable_plain")
    return job.get_results() if job else None


def main():
    parser = argparse.ArgumentParser(
        prog="euclid-cutouts",
        description="Fetch and display Euclid archive FITS images at a "
        "sky position; --download saves them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "coords",
        nargs="+",
        help="RA Dec in degrees, 'ra,dec', a sexagesimal string, or an "
        "object name resolved via Sesame (e.g. 2026abc)",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=None,
        help="cutout radius in arcsec (default 30, max 1800)",
    )
    parser.add_argument(
        "-size",
        "--size",
        type=float,
        default=None,
        help="cutout width in arcsec; equals twice --radius",
    )
    parser.add_argument(
        "-b",
        "--bands",
        default="euclid",
        help="comma list like VIS,NIR_Y; 'euclid' = VIS+NIR_Y/J/H (default); "
        "'all' adds external bands (DECAM, HSC, MEGACAM, PANSTARRS)",
    )
    parser.add_argument(
        "--release",
        choices=["dr1", "q1", "both"],
        default="dr1",
        help="data release to search (default dr1; falls back to q1 if empty)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="fetch every covering product instead of one image per band",
    )
    parser.add_argument(
        "--include-raw",
        "--include_raw",
        action="store_true",
        help="for bands with no processed coverage, fetch raw detector "
        "frames and cut out the covering detector (uncalibrated; "
        "~200MB per exposure, slow)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="download full product files instead of cutouts (can be GBs)",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list covering products without downloading",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="output directory for saved files (default: cwd); "
        "implies --download",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="save the FITS files; without it the images are fetched "
        "and displayed but not kept",
    )
    parser.add_argument(
        "--no-plot",
        "--no_plot",
        action="store_true",
        help="do not display the figure; alone it only reports coverage",
    )
    parser.add_argument(
        "--pub",
        action="store_true",
        help="write a publication figure (journal column width, vector "
        "PDF with editable text + 300-dpi PNG) instead of displaying",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="open the Qt image viewer (histogram, stretch and cut "
        "controls, band stepping) instead of the static figure",
    )
    parser.add_argument(
        "--no-header-fix",
        action="store_true",
        help="keep archive headers untouched (skip AutoPhOT keyword stamping)",
    )
    # A sexagesimal Dec like '-54d15m51s' looks like a flag to argparse, so
    # option-shaped leftovers that start with a digit are re-admitted as
    # coordinate tokens. Anything else stays a usage error.
    args, extras = parser.parse_known_args()
    bad = [t for t in extras if not re.match(r"^-?\d", t)]
    if bad:
        parser.error(f"unrecognized arguments: {' '.join(bad)}")
    args.coords.extend(extras)

    try:
        ra, dec = parse_position(args.coords)
        target_name = None
    except ValueError:
        target_name = " ".join(args.coords)
        resolved = resolve_name(target_name)
        if resolved is None:
            tp.die(
                f"cannot parse position: {target_name!r} "
                f"(not a coordinate, and no catalog match)",
                2,
            )
        ra, dec, _ = resolved

    if args.radius is not None and args.size is not None:
        tp.die("use only one of --radius and --size", 2)
    radius_as = (
        args.radius
        if args.radius is not None
        else (args.size / 2.0 if args.size is not None else 30.0)
    )
    # an explicit -o means the user wants files, so it turns download on
    if args.outdir is not None:
        args.download = True
    else:
        args.outdir = "."
    if args.full and not args.download:
        tp.die("--full requires --download", 2)
    if args.pub and args.gui:
        tp.die("use only one of --pub and --gui", 2)
    if not args.full and not 0 < radius_as <= MAX_RADIUS_ARCSEC:
        tp.die(f"radius must be within (0, {MAX_RADIUS_ARCSEC:g}] arcsec", 2)

    bands = parse_bands(args.bands)
    schemas = ["dr1", "q1"] if args.release == "both" else [args.release]

    tp.header("Euclid archive cutouts")
    if target_name:
        tp.kv("Target", target_name)
    tp.kv("Position", f"RA {ra:.7f}, Dec {dec:.7f}")
    tp.kv(
        "Radius",
        "full product" if args.full else f"{radius_as:g} arcsec",
    )
    tp.kv("Release", " + ".join(schemas))

    tp.subheader("Coverage")
    tp.info(f"searching {', '.join(schemas)} footprints ...")
    products = find_covering_products(schemas, ra, dec, radius_as / 3600.0)

    # Q1 keeps older deep-field processing that DR1 superseded; check it only
    # when the default release has nothing at the position.
    if not products and args.release == "dr1":
        tp.info("no dr1 coverage; trying q1")
        products = find_covering_products(["q1"], ra, dec, radius_as / 3600.0)

    if not products:
        tp.fail(f"no Euclid image products cover ({ra:.7f}, {dec:.7f})")
        nearest = nearest_products(ra, dec)
        if nearest is not None and len(nearest):
            row = nearest[0]
            tp.info(
                f"nearest DR1 mosaic: TILE{row['pid']} {row['filter_name']}, "
                f"{float(row['dist']):.2f} deg away"
            )
        sys.exit(1)
    tp.ok(f"{len(products)} product(s) cover the position")

    if args.list:
        print_product_table(products)
        if args.include_raw:
            wanted = EUCLID_BANDS if bands is None else bands
            raw_rows = []
            for band in wanted:
                if band not in RAW_BAND_MAP:
                    continue
                inst, filt = RAW_BAND_MAP[band]
                for schema in schemas:
                    job = Euclid.launch_job(
                        raw_coverage_query(
                            schema, ra, dec, radius_as / 3600.0,
                            inst, filt,
                        ),
                        output_format="votable_plain",
                    )
                    rows = job.get_results() if job else None
                    for row in rows or []:
                        raw_rows.append(
                            (
                                schema,
                                "RAW",
                                filt,
                                str(row["pid"]),
                                f"{float(row['dist']):.3f}",
                                row["file_name"],
                            )
                        )
            if raw_rows:
                tp.subheader("Raw frames")
                tp.table(["REL", "KIND", "BAND", "ID", "DIST(deg)", "FILE"], raw_rows)
        return

    chosen = pick_products(products, bands, args.all)
    if not chosen:
        tp.fail("coverage exists but not for the requested bands")
        print_product_table(products)
        sys.exit(1)

    if args.no_plot and not args.download:
        covered = sorted({p["band"] for p in chosen})
        tp.ok(f"bands covered: {', '.join(covered)}")
        tp.info(
            "nothing fetched; drop --no-plot to display, "
            "or add --download to save"
        )
        return

    if args.download or args.pub:
        confirm_outdir(Path(args.outdir).expanduser().resolve())
    if args.download:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        # plot mode: files land in a temp dir and are deleted after the
        # window closes
        _tmpdir = tempfile.TemporaryDirectory(prefix="euclid_cutouts_")
        outdir = Path(_tmpdir.name)
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
    radius = radius_as * u.arcsec

    tp.subheader("Fetching")
    saved, failed = [], []
    for p in chosen:
        if args.full:
            out = outdir / p["file_name"]
            tp.info(
                f"downloading full {p['band']} {p['kind'].lower()} "
                f"{p['id']} (may be large)"
            )
            res = Euclid.get_product(
                file_name=p["file_name"],
                schema=p["release"],
                output_file=str(out),
            )
        else:
            out = outdir / cutout_name(p, radius_as)
            tp.info(
                f"downloading {p['band']} cutout from "
                f"{p['kind'].lower()} {p['id']}"
            )
            res = Euclid.get_cutout(
                file_path=f"{p['file_path']}/{p['file_name']}",
                instrument=p["instrument"],
                id=p["id"],
                coordinate=coord,
                radius=radius,
                output_file=str(out),
            )
        # the cutout service occasionally answers with an empty file
        if res and Path(res[0]).stat().st_size == 0:
            res = None
        if res and not args.no_header_fix:
            try:
                stamp_autophot_headers(res[0], p)
            except Exception as err:
                tp.warn(f"header stamping failed for {out.name}: {err}")
        if res:
            tp.ok(f"saved {out.name}")
            saved.append((p, res[0]))
        else:
            tp.warn(
                f"{p['band']} {p['kind'].lower()} {p['id']}: "
                f"download failed or empty"
            )
            failed.append((p, str(out)))

    raw_saved = []
    if args.include_raw:
        have = {p["band"].lower() for p, _ in saved}
        wanted = EUCLID_BANDS if bands is None else bands
        # any processed cutout from this run anchors the raw-frame WCS
        ref_path = saved[0][1] if saved else None
        for band in wanted:
            if band.lower() in have or band not in RAW_BAND_MAP:
                continue
            tp.info(f"no processed {band} coverage; trying raw frames")
            raw_saved += fetch_raw_band(
                band,
                schemas,
                ra,
                dec,
                radius_as,
                outdir,
                ref_path,
                args.all,
                args.no_header_fix,
            )

    if args.pub or (not args.download and not args.no_plot):
        paths = [str(p) for _, p in saved] + [str(p) for p in raw_saved]
        if not paths:
            tp.fail("no images to display")
            sys.exit(1)

        import make_collage

        if args.pub:
            fig_dir = Path(args.outdir).expanduser().resolve()
            fig_dir.mkdir(parents=True, exist_ok=True)
            base = fig_dir / f"euclid_collage_{ra:.4f}_{dec:.4f}.png"
            pdf = make_collage.render(paths, ra, dec, out=base, pub=True)
            tp.ok(f"wrote {pdf}")
            tp.ok(f"wrote {pdf.with_suffix('.png')}")
        elif args.gui:
            if not (
                os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY")
                or os.environ.get("QT_QPA_PLATFORM")
            ):
                tp.die("--gui needs a display", 2)
            try:
                import fits_viewer
            except SystemExit as err:
                tp.die(str(err), 2)
            tp.info("close the viewer window to exit")
            fits_viewer.show(paths)
            tp.info("Slán go fóill")
        else:
            import matplotlib

            if matplotlib.get_backend().lower() in (
                "agg",
                "cairo",
                "pdf",
                "pgf",
                "ps",
                "svg",
                "template",
            ):
                png = Path(
                    f"euclid_collage_{ra:.4f}_{dec:.4f}.png"
                ).resolve()
                make_collage.render(paths, ra, dec, out=png)
                tp.warn(f"no display available; wrote {png}")
            else:
                tp.info(
                    f"displaying {len(paths)} image(s); "
                    f"close the window to exit"
                )
                make_collage.render(paths, ra, dec, show=True)
                tp.info("Slán go fóill")
        if not args.download:
            return

    n_saved = len(saved) + len(raw_saved)
    if n_saved:
        tp.ok(f"{n_saved} file(s) in {outdir}")
    if failed and not n_saved:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        tp.info("Slán go fóill")
        sys.exit(130)
