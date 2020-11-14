# Data
<p>The following artefacts were retained for the analysis: A, B, C, D, E, G, N, P, R and S. G was selected as inclusive of the text in K, and P was selected as representative of H-P-Q. The Santiago Staff (I) reflects a very particular genre and structure (also present in parts of G-K), and was left out of the analysis for now.</p>
<p>The corpus is provided as a dictionary with artefacts' names (letters) as keys. Values are themselves dictionaries with each line as key and a string of glyphs separated by a dash as value:</p>
```python
from tablets import tablets, tablets_clean, tablets_simple

tablets['B']['Br1']
'595-1-50.394s-4-2-595.1-50-301s-4-2-40-211-91-200-595.2-394-4t-2-595.2-50-394-4t-2-595.2-50-301s.4-2-211s:42-91-595s-600-50-381-4-2-306-325-430-53-430-17-430-4-2-208-200-2-22-305.74f-95-1-?-69*'
```
