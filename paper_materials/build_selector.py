#!/usr/bin/env python3
"""Build the figure candidate selector HTML."""
import base64, os, json

subtypes = ['lpd', 'gpd', 'lrda', 'grda']
candidates_dir = 'paper_materials/candidates'

sections = {}
for sub in subtypes:
    data_file = f'paper_materials/figure_{sub}_examples_data.json'
    if not os.path.exists(data_file):
        continue
    meta_all = json.load(open(data_file))
    images = []
    for i, meta in enumerate(meta_all):
        path = os.path.join(candidates_dir, f'{sub}_{i:02d}.png')
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        images.append({
            'idx': i, 'b64': b64,
            'difficulty': meta.get('difficulty', '?'),
            'freq_bin': meta.get('freq_bin', '?'),
            'agreement_pct': meta.get('agreement_pct', 0),
            'jaccard': meta.get('jaccard', 0),
            'patient_id': meta.get('patient_id', '')[:40],
            'lat': meta.get('gt_lat') or meta.get('pred_lat', '?'),
            'freq': meta.get('gt_freq') or meta.get('pred_freq', 0),
        })
    sections[sub] = images

# Build HTML
parts = ["""<!DOCTYPE html>
<html><head><title>Figure Candidate Selector</title>
<style>
body { background:#f5f5f5; font-family:Consolas,monospace; padding:20px; max-width:1400px; margin:0 auto; }
h1 { color:#333; }
h2 { color:#555; margin-top:30px; border-bottom:2px solid #ccc; padding-bottom:5px; }
h3 { color:#777; margin-top:15px; }
.gallery { display:flex; flex-wrap:wrap; gap:12px; }
.card { background:white; border:3px solid #ddd; border-radius:8px; padding:6px; cursor:pointer; transition:all 0.2s; position:relative; }
.card:hover { border-color:#888; box-shadow:0 2px 8px rgba(0,0,0,0.15); }
.card.selected { border-color:#22aa44; background:#f0fff0; box-shadow:0 0 12px rgba(0,180,0,0.3); }
.card img { width:550px; height:auto; display:block; pointer-events:none; }
.card .badges { position:absolute; top:10px; right:10px; display:flex; gap:4px; pointer-events:none; }
.badge { padding:2px 6px; border-radius:3px; font-size:10px; font-weight:bold; color:white; }
.badge.Easy { background:#2a7d2a; }
.badge.Medium { background:#b87700; }
.badge.Hard { background:#c03030; }
.badge.freq { background:#336; }
.card .meta { font-size:10px; color:#666; padding:3px 0; pointer-events:none; }
.card .check { position:absolute; top:10px; left:10px; font-size:24px; color:#22aa44; display:none; pointer-events:none; }
.card.selected .check { display:block; }
.actions { position:sticky; top:0; z-index:100; margin:0 0 15px 0; padding:12px; background:#e8e8e8; border-radius:8px; border:1px solid #ccc; }
.actions button { padding:6px 14px; font-size:13px; cursor:pointer; border:none; border-radius:4px; margin-right:8px; }
.actions button.primary { background:#22aa44; color:white; }
.actions button.secondary { background:#666; color:white; }
#output { margin-top:10px; padding:8px; background:white; border:1px solid #ccc; border-radius:4px; font-size:11px; display:none; }
</style></head><body>
<h1>Paper Figure Candidate Selector</h1>
<p style="color:#666;font-size:12px;">Stratified by difficulty (IIIC expert agreement) and frequency. Click to select, then Generate.</p>

<div class="actions">
  <button class="primary" onclick="generateSelection()">Generate Selection</button>
  <button class="secondary" onclick="clearAll()">Clear All</button>
  <span id="selection-count" style="margin-left:15px;font-size:13px;">0 selected</span>
  <div id="output"></div>
</div>
"""]

for sub in subtypes:
    if sub not in sections:
        continue
    imgs = sections[sub]
    parts.append(f'<h2>{sub.upper()} ({len(imgs)} candidates)</h2>\n')

    by_diff = {}
    for img in imgs:
        by_diff.setdefault(img['difficulty'], []).append(img)

    for diff in ['Easy', 'Medium', 'Hard']:
        if diff not in by_diff:
            continue
        group = by_diff[diff]
        pct = group[0]['agreement_pct']
        parts.append(f'<h3>{diff} (~{pct:.0f}% agreement)</h3>\n<div class="gallery">\n')
        for img in group:
            parts.append(f'''<div class="card" data-sub="{sub}" data-idx="{img['idx']}" onclick="toggleCard(this)">
  <span class="check">&#10003;</span>
  <div class="badges">
    <span class="badge {img['difficulty']}">{img['agreement_pct']:.0f}%</span>
    <span class="badge freq">{img['freq_bin']} {img['freq']:.1f}Hz</span>
  </div>
  <img src="data:image/png;base64,{img['b64']}">
  <div class="meta">#{img['idx']} | {img['patient_id']} | lat={img['lat']}</div>
</div>\n''')
        parts.append('</div>\n')

parts.append("""
<script>
const selected = {};
function updateCount() {
  let n = 0;
  for (const k in selected) n += selected[k].size;
  document.getElementById('selection-count').textContent = n + ' selected';
}
function toggleCard(el) {
  el.classList.toggle('selected');
  const sub = el.dataset.sub, idx = parseInt(el.dataset.idx);
  if (!selected[sub]) selected[sub] = new Set();
  if (el.classList.contains('selected')) selected[sub].add(idx);
  else selected[sub].delete(idx);
  updateCount();
}
function clearAll() {
  document.querySelectorAll('.card.selected').forEach(c => c.classList.remove('selected'));
  for (const k in selected) selected[k] = new Set();
  updateCount();
  document.getElementById('output').style.display = 'none';
}
function generateSelection() {
  const result = {};
  for (const [sub, idxSet] of Object.entries(selected))
    if (idxSet.size > 0) result[sub] = [...idxSet].sort();
  const out = document.getElementById('output');
  out.style.display = 'block';
  out.innerHTML = '<strong>Selection:</strong><pre>' + JSON.stringify(result, null, 2) +
    '</pre><strong>Tell Claude:</strong><pre>Please regenerate figures with: ' +
    JSON.stringify(result) + '</pre>';
}
</script>
</body></html>""")

with open('paper_materials/figure_selector.html', 'w') as f:
    f.write(''.join(parts))
print('Selector ready')
