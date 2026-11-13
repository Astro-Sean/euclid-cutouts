# euclid-cutouts

Terminal command that grabs Euclid archive images at a sky position.

It queries the ESA Euclid Science Archive (astroquery.esa.euclid) for image
products whose footprint covers the position, fetches a FITS cutout per
band, and displays them in a collage. `--download` saves the FITS files.

Public DR1/Q1 data only, no login required.



## Install

```bash
pip install -r requirements.txt
ln -s "$(pwd)/euclid-cutouts.py" ~/.local/bin/euclid-cutouts
```

## Usage

```bash
euclid-cutouts 60.3372991 -49.9318588
```

That fetches the covering cutouts and shows them in a figure. Nothing
is kept on disk unless `--download` is given. With no display
available, the figure is written to `euclid_collage_<ra>_<dec>.png`
instead.

Coordinates work as `ra dec`, `ra,dec`, or a sexagesimal string. A
non-coordinate argument is treated as an object name and resolved
through Sesame (SIMBAD, NED, VizieR); transient names like `2026abc`
are tried under the SN and AT prefixes automatically.

```bash
euclid-cutouts 2026abc
```

![DR1 mosaic cutouts in VIS and NIR Y/J/H at RA 60.3372991, Dec -49.9318588](example_collage.png)


| flag | effect |
|---|---|
| `-r 60` | cutout radius in arcsec (default 30, max 1800) |
| `--size 90` | cutout width in arcsec; alternative to `-r` (size = 2 x radius) |
| `--download` | save the FITS cutouts; without it they are displayed but not kept |
| `--no-plot` | do not display the figure; alone it only reports coverage |
| `-b VIS,NIR_Y` | band list; `euclid` (default) = VIS + NIR_Y/J/H, `all` adds DECAM/HSC/MEGACAM/PANSTARRS |
| `--release q1` / `both` | release to search; dr1 is the default, q1 is tried when dr1 has no coverage |
| `--all` | fetch every covering product, not one per band |
| `--include-raw` | for bands with no processed image, fetch raw detector frames and cut out the covering detector |
| `--full` | save the full product file instead of a cutout (needs `--download`) |
| `-l` | list covering products without fetching |
| `--pub` | write `euclid_collage_<ra>_<dec>.pdf` + `.png` at journal column size (vector text, 300 dpi) instead of displaying |
| `-o dir` | output directory for saved files (default: cwd); implies `--download` |
| `--no-header-fix` | skip the keyword stamping |



