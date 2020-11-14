# Data
<p>The following artefacts were retained for the analysis: A, B, C, D, E, G, N, P, R and S. G was selected as inclusive of the text in K, and P was selected as representative of H-P-Q. The Santiago Staff (I) reflects a very particular genre and structure (also present in parts of G-K), and was left out of the analysis for now.</p>
<p>The corpus is provided as a dictionary with artefacts' names (letters) as keys. Values are themselves dictionaries with each line as key and a string of glyphs as value:</p>

```python
>>> from tablets import tablets, tablets_clean, tablets_simple
>>> tablets['B']['Br1']
'595-1-50.394s-4-2-595.1-50-301s-4-2-40-211-91-200-595.2-394-4t-2-595.2-50-394-4t-2-595.2-50-301s.4-2-211s:42-91-595s-600-50-381-4-2-306-325-430-53-430-17-430-4-2-208-200-2-22-305.74f-95-1-?-69*'
```

<p>In addition to Barthel's transliteration <i>ipsis litteris</i>, two other versions are provided. Glyphs in <code>tablets_clean</code> are stripped of the letters that identify variants, padded to three digits and reordered in cases of bottom-up ligatures. <code>tablets_simple</code> contain a more radical restructuring based on proposals such as those of Pozdniakov and Pozdniakov (<a href="http://pozdniakov.free.fr/publications/2007_Rapanui_Writing_and_the_Rapanui_Language.pdf">2007</a>) and Horley (<a href="https://kahualike.manoa.hawaii.edu/rnj/vol19/iss2/6/">2005</a>) (see <a href="./03_catalogue.html">Revising the Glyph Catalogue</a>):</p>

```python
>>> tablets_clean['B']['Br1']
'595-001-050-394-004-002-595-001-050-301-004-002-040-211-091-200-595-002-394-004-002-595-002-050-394-004-002-595-002-050-301-004-002-042-211-091-595-600-050-381-004-002-306-325-430-053-430-017-430-004-002-208-200-002-022-305-074-095-001-000-069'
>>> tablets_simple['B']['Br1']
'200-009-010-001-050-280-061-010-004-002-200-009-010-001-050-200-061-010-004-002-041-200-061-091-200-200-009-010-002-280-061-004-002-200-009-010-002-050-280-061-004-002-200-009-010-002-050-200-061-010-004-002-041-200-061-010-091-200-009-010-062-400-050-200-061-004-002-200-006-200-010-400-053-400-016-400-004-002-200-200-200-002-022-200-010-074-095-001-000-069'
```