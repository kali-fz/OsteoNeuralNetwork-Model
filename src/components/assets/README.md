# Globe map assets

These files are pinned local copies used by `src/components/globe.py`. Keeping
them in the repository means the homepage does not depend on a browser or
Streamlit-server network request to draw country shapes.

- `d3-array.min.js`: d3-array 3.2.4, ISC licence
- `d3-geo.min.js`: d3-geo 3.1.0, ISC licence
- `topojson-client.min.js`: topojson-client 3.1.0, ISC licence
- `countries-110m.json`: world-atlas 2.0.2, generated from Natural Earth data

The exact source URLs remain in `src/components/globe.py`.
