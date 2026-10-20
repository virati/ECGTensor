# ECG Phase Atlas (web app)

Standalone Vite build of the ECG Phase Atlas artifact (`../atlas/ecg_phase_atlas_artifact.html`).
The rendering code in `src/main.js` is the artifact's code with two changes: three.js and d3 come
from npm (pinned to the same versions, three 0.160.0 and d3 7.9.0), and the data is fetched from
`public/ecg_data.json` instead of being inlined.

    npm install
    npm run dev        # local dev server
    npm run build      # static site in dist/ (relative paths, host anywhere)
    npm test           # pixel-parity tests vs the artifact (uses system Chrome)

Regenerate the data from the repo root with `python export_atlas.py` (writes `web/public/ecg_data.json`).

`tests/parity.spec.js` runs the same interaction script (method switching, mid-morph, face-on,
toggles, sliders, trace scrubbing, tooltips; desktop and phone widths) against the artifact and
the app under a frozen clock and compares full-page screenshots.
