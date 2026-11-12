# AGENTS.md

Working rules for the euclid-cutouts repository.

## Repository

https://github.com/Astro-Sean/euclid-cutouts

## What the project does

`euclid-cutouts` is a terminal command that fetches Euclid archive FITS
cutouts at a sky position. It queries the DR1/Q1 mosaic and
observation-stack footprint tables, fetches one cutout per band, and
displays them in a collage figure. `--download` saves the files and
stamps the header keywords AutoPhOT needs at ingest: TELESCOP,
INSTRUME, FILTER, EXPTIME, GAIN, RDNOISE, SATURATE, DATE-OBS, MJD-OBS.
Public archive data only, no login.

`--include-raw` falls back to raw NISP detector frames where no
processed product exists. Raw headers carry only focal-plane millimetre
coordinates, so the code rebuilds a sky WCS from the exposure FOV
polygon, refines it against a processed image by source matching and
cross-correlation, and cuts out the covering detector. Raw cutouts are
single uncalibrated exposures: fine for checking a source or a rough
flux, not for photometry without full reduction.

`make_collage.py` renders a figure with one WCS-projected panel per
FITS file: ZScale stretch, no interpolation, a 5-arcsecond zoomed inset
with notch markers on the target, an N/E compass bottom-right, a band
label top-left, and the product id bottom-left. Mirrored images are
flipped on load so every panel shows north up, east left.

## File map

- `euclid-cutouts.py` - the command: archive query, download, header
  stamping, raw extraction. Installed by symlinking to
  `~/.local/bin/euclid-cutouts`.
- `make_collage.py` - collage renderer. euclid-cutouts imports its
  `render()` for the default display mode; it also works standalone
  from the command line.
- `Euclid_VIS.vis.dat` - VIS bandpass curve (wavelength, transmission)
  for AutoPhOT filter calibration.
- `example_collage.png` - README figure, regenerated with
  `make_collage.py` when the layout changes.
- `test/` - working data, not committed.

## Commit rules

- One file per commit. Do not bundle changes across files into a
  single commit.
- Never acknowledge AI involvement anywhere in this repository: no AI
  tool names in file comments, docstrings, docs, commit messages, or
  the contributor list. No "Co-Authored-By" trailers, no "generated
  by" lines.
- Keep `README.md` in sync with the code. A new flag, option, or
  behavior change must come with a README update describing it.
- Commit messages: short imperative subject, body explains why.
- Plain ASCII only. No emojis, smart quotes, or em/en dashes.

## Writing rules

All prose in this repo - comments, docstrings, help text, README, docs,
commit messages - must follow the /humanizer skill checklist:

- Cut AI tells: not-X-but-Y contrasts, one-line closers, staged
  openers, forced triads, dashes as a universal connector, inflated
  significance, sales language, stock AI words, decorative bold, and
  filler.
- Every sentence must add something the reader did not already have.
- Keep every supported claim. Never add a fact, name, number, date,
  quote, or citation that is not already there.
- Write plain and direct, like the author did. Technical text stays
  neutral.

## Coding standards

- Python: black (line length 88), isort, ruff. 4-space indent,
  snake_case names.
- Comments explain why, not what.
- No silent fallbacks: failed downloads, empty files, and unmatched
  WCS refinement must be reported, never swallowed.
- No banned buzzwords in code or docs: delve, tapestry, testament,
  leverage, utilize, seamless, robust, crucial, paramount, landscape,
  navigate, intricate.

## Scientific invariants

- WCS consistency: any crop, flip, or cutout must update CRPIX/CD so
  the sky solution stays correct.
- Raw-frame outputs stay marked as uncalibrated. Do not present them
  as equivalent to processed stacks.
- Never modify files returned by the archive. Write cutouts to new
  files.
