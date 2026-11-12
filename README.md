# query_EUCLID

Terminal command that grabs Euclid archive images at a sky position.

It queries the ESA Euclid Science Archive (astroquery.esa.euclid) for image
products whose footprint covers the position, downloads a FITS cutout per
band, and stamps in the header keywords AutoPhOT needs at ingest
(TELESCOP, INSTRUME, FILTER, EXPTIME, GAIN, RDNOISE, SATURATE, DATE-OBS,
MJD-OBS). Public DR1/Q1 data only, no login required.

![VIS stack plus raw Y/J/H detector cutouts at RA 91.0014173, Dec -54.2642342](example_collage.png)

The example above comes from `query_EUCLID 91.0014173 -54.2642342
--include-raw` followed by `make_collage.py`. Only VIS has processed
coverage at this position, so Y/J/H are raw detector cutouts (DET44,
uncalibrated).

## Install

```bash
pip install -r requirements.txt
ln -s "$(pwd)/query_EUCLID.py" ~/.local/bin/query_EUCLID
```

## Usage

```bash
query_EUCLID 91.0014173 -54.2642342
```

Coordinates work as `ra dec`, `ra,dec`, or a sexagesimal string.

| flag | effect |
|---|---|
| `-r 60` | cutout radius in arcsec (default 30, max 1800) |
| `--size 90` | cutout width in arcsec; alternative to `-r` (size = 2 x radius) |
| `--download` | save into the current directory (same as `-o .`) |
| `-b VIS,NIR_Y` | band list; `euclid` (default) = VIS + NIR_Y/J/H, `all` adds DECAM/HSC/MEGACAM/PANSTARRS |
| `--release q1` / `both` | release to search; dr1 is the default, q1 is tried when dr1 has no coverage |
| `--all` | fetch every covering product, not one per band |
| `--include-raw` | for bands with no processed image, fetch raw detector frames and cut out the covering detector |
| `--full` | download the full product file instead of a cutout |
| `-l` | list covering products without downloading |
| `-o dir` | output directory |
| `--no-header-fix` | skip the keyword stamping |

## How it works

DR1 mosaics are checked first. Where no mosaic covers the position, the
tool falls back to single-observation stacks, then, with `--include-raw`,
to raw detector frames. Mosaic headers ship without observing metadata, so
the keywords above come from the TAP row and nominal detector values.

Raw frames need more work: each detector extension stores its footprint in
focal-plane millimetres rather than a sky WCS. The transform is recovered
by matching the exposure's FOV polygon to the detector mosaic, then refined
against a processed image downloaded in the same run (source matching plus
cross-correlation, good to a few arcsec). Without a reference image the
code writes out the whole covering detector instead of a cutout and marks
the WCS as approximate. Raw cutouts are uncalibrated single exposures -
useful for checking a source or a rough flux, not for photometry without
reduction.

## make_collage.py

```bash
python3 make_collage.py <ra> <dec> <out.png> <fits files...>
```

Builds a multi-panel figure with one WCS-projected panel per FITS file
(ZScale, no interpolation). Each panel shows a small zoom box on the
target, a 5-arcsecond zoomed inset with tight marker notches, an N/E
compass in the bottom-right, and a band label top-left. Mirrored images
are flipped to the usual north-up/east-left convention.
