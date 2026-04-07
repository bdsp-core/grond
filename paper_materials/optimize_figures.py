#!/usr/bin/env python3
"""
Figure optimization loop using Gemini as visual critic.

Iteratively improves figure generation scripts by:
1. Rendering current figures
2. Sending to Gemini for visual critique
3. Rewriting the code based on feedback
4. Re-rendering and repeating

Supports multiple figure groups: characterization examples, PD pipeline,
RDA pipeline, and raw EEG examples.

Usage:
    conda run -n morgoth python paper_materials/optimize_figures.py [--group GROUP] [--rounds N]

    Groups: characterization, pd_pipeline, rda_pipeline, fig1_examples, all
    Default: all groups, 3 rounds each
"""

import os
import re
import sys
import json
import base64
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / 'figures'

GOOGLE_API_KEY = "REDACTED-GOOGLE-API-KEY"
MODEL_NAME = "gemini-2.5-flash"
LOG_DIR = SCRIPT_DIR / 'optimization_logs'
LOG_DIR.mkdir(exist_ok=True)

# ── Figure Groups ──

FIGURE_GROUPS = {
    'characterization': {
        'name': 'Characterization Examples (Fig 5-8)',
        'script': SCRIPT_DIR / 'render_figures.py',
        'render_cmd': [sys.executable, str(SCRIPT_DIR / 'render_figures.py'),
                       '--pick', '{"lpd":[0,1,2],"gpd":[0,1,2],"lrda":[0,1,2],"grda":[0,1,2]}'],
        'figure_files': [
            SCRIPT_DIR / 'figure_lpd_examples.png',
            SCRIPT_DIR / 'figure_gpd_examples.png',
            SCRIPT_DIR / 'figure_lrda_examples.png',
            SCRIPT_DIR / 'figure_grda_examples.png',
        ],
        'copy_targets': {
            'figure_lpd_examples.png': FIGURES_DIR / 'fig5_lpd_characterization.png',
            'figure_gpd_examples.png': FIGURES_DIR / 'fig6_gpd_characterization.png',
            'figure_lrda_examples.png': FIGURES_DIR / 'fig7_lrda_characterization.png',
            'figure_grda_examples.png': FIGURES_DIR / 'fig8_grda_characterization.png',
        },
        'caption': """Publication figures showing EEG characterization examples for
periodic and rhythmic patterns (LPD, GPD, LRDA, GRDA). Each figure contains 3 examples
at different difficulty levels (Easy/Medium/Hard based on inter-rater agreement).
Each row shows: (left) 10-second 19-channel average reference EEG with discharge timing
markers (PD) or narrowband overlay (RDA) and hemisphere shading, (right) Laplacian
topoplot (inferno colormap) with electrode labels and ACNS 2021 verbal description.
All 4 figures must have IDENTICAL styling — only the data differs.""",
    },
    'pd_pipeline': {
        'name': 'PD Pipeline (Fig 2)',
        'script': SCRIPT_DIR / 'generate_fig2_pd_pipeline.py',
        'render_cmd': [sys.executable, str(SCRIPT_DIR / 'generate_fig2_pd_pipeline.py')],
        'figure_files': [FIGURES_DIR / 'fig2_pd_pipeline.png'],
        'copy_targets': {},
        'caption': """Figure 2: PD Characterization Pipeline. Three-panel horizontal composite.
Panel A (left, 30%): Real 19-channel LPD EEG in average reference montage. Channels grouped
as L parasagittal, L temporal, midline, R parasagittal, R temporal with gaps between groups.
Panel B (center, 40%): Architecture flowchart — ChannelPD-Net (blue) at top feeds into 3
colored branches: Laterality Detection (green), HemiCET+DP Discharge Detection (red/salmon),
Discharge-Locked Topographic Localization (orange). Output boxes at bottom.
Panel C (right, 30%): Same EEG as Panel A but with red dashed discharge markers on involved
hemisphere, light blue hemisphere shading. Laplacian topoplot (inferno) inset in lower-right
corner with electrode labels. Verbal description text below topoplot.""",
    },
    'rda_pipeline': {
        'name': 'RDA Pipeline (Fig 3)',
        'script': SCRIPT_DIR / 'generate_fig3_rda_pipeline.py',
        'render_cmd': [sys.executable, str(SCRIPT_DIR / 'generate_fig3_rda_pipeline.py')],
        'figure_files': [FIGURES_DIR / 'fig3_rda_pipeline.png'],
        'copy_targets': {},
        'caption': """Figure 3: RDA Characterization Pipeline. Three-panel horizontal composite.
Panel A (left, 30%): Real 19-channel LRDA EEG in average reference montage.
Panel B (center, 40%): Architecture flowchart — W05 Iterative Narrowband Refinement (blue)
with Pass 1 (coarse) and Pass 2 (narrowband), feeding into 3 branches: Laterality Detection
(green), PLV×Amplitude Spatial Extent (purple), Narrowband Amplitude Topographic Localization (orange).
Panel C (right, 30%): Same EEG with green narrowband overlay at estimated frequency, light blue
hemisphere shading on involved side. Laplacian topoplot (inferno) inset in lower-right.
Verbal description text below topoplot.""",
    },
    'fig1_examples': {
        'name': 'Raw EEG Examples (Fig 1)',
        'script': SCRIPT_DIR / 'generate_fig0_examples.py',
        'render_cmd': [sys.executable, str(SCRIPT_DIR / 'generate_fig0_examples.py')],
        'figure_files': [FIGURES_DIR / 'fig1_eeg_examples.png'],
        'copy_targets': {},
        'caption': """Figure 1: Examples of periodic and rhythmic EEG patterns. 3×2 grid of
10-second EEG recordings in average reference montage (19 electrodes, standard 10-20).
A: Clear LPD (96% agreement). B: Clear GPD (90%). C: Clear LRDA (92%). D: Clear GRDA (86%).
E: Ambiguous LPD (58%). F: Ambiguous GRDA (50%). No algorithm annotations — raw EEG only.
Black traces on white background, scale bar in each panel. Clean, minimal, publication-ready.""",
    },
}

STYLE_GUIDE = """You are reviewing scientific EEG figures for publication in a top neurology journal.

Key standards:
- Clean, minimal design suitable for print at column width (~7 inches) or full page
- Fonts: 7-9pt for labels, 10-12pt for titles. All text must be readable when printed
- EEG traces: clean black lines on white background, consistent line width
- Channel labels: left-aligned, readable, properly spaced
- Topoplots: smooth interpolation, clear colorbar, head outline visible, electrode labels readable
- Hemisphere shading: subtle light blue, not distracting
- Discharge markers (PD): visible red dashed lines at correct locations
- Narrowband overlay (RDA): green traces overlaid on EEG channels
- Verbal descriptions: readable, properly positioned
- Flowchart boxes: clean rounded rectangles, distinct colors per branch, clear arrows
- White background throughout
- No unnecessary visual clutter
- Professional, publication-ready appearance
- Consistent styling across related figures"""


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_gemini(prompt: str, images_b64: list = None) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    contents = []

    if images_b64:
        for img_b64 in images_b64:
            contents.append(types.Part.from_bytes(
                mime_type="image/png",
                data=base64.b64decode(img_b64),
            ))
    contents.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=30000,
        ),
    )
    return response.text


def render_group(group: dict) -> bool:
    """Run the render command for a figure group. Return True if successful."""
    try:
        result = subprocess.run(
            group['render_cmd'],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  RENDER FAILED: {result.stderr[-500:]}")
            return False
        # Copy to final locations
        for src_name, dst_path in group.get('copy_targets', {}).items():
            src_path = SCRIPT_DIR / src_name
            if src_path.exists():
                shutil.copy(src_path, dst_path)
        print(f"  Rendered successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"  RENDER TIMED OUT")
        return False


def critique_group(group: dict, round_num: int) -> str:
    """Send figures to Gemini for critique."""
    images = []
    for fp in group['figure_files']:
        if fp.exists():
            images.append(image_to_base64(fp))

    prompt = f"""{STYLE_GUIDE}

Here are the current versions of the figure(s) for "{group['name']}" (iteration {round_num}).

Figure caption: {group['caption']}

Please critique these figures for publication readiness. Focus on:
1. Layout and spacing — are panels well-organized? Is there wasted space?
2. Typography — are fonts consistent, readable, properly sized?
3. EEG traces — are they clean, well-scaled, properly labeled?
4. Topoplots — smooth interpolation, clear color scale, head outline?
5. Any markers/overlays — visible, correctly placed?
6. Hemisphere shading — subtle but informative?
7. Verbal descriptions — readable, properly formatted?
8. Flowchart (if applicable) — clean boxes, clear arrows, readable text?
9. Overall professional appearance for a top journal

Be specific and actionable. List the top 5-8 most impactful improvements.
Do NOT suggest changes to the data — only the visual presentation and styling.
Focus on changes that can be made in matplotlib code."""

    print(f"  Sending {len(images)} figure(s) to Gemini for critique...")
    return call_gemini(prompt, images)


def rewrite_code(current_code: str, critique: str, group: dict, round_num: int) -> str:
    """Have Gemini rewrite the plotting code to address the critique."""
    prompt = f"""You are improving a matplotlib figure generation script for publication-quality EEG figures.

Here is the current critique (iteration {round_num}) for "{group['name']}":
{critique}

Here is the current Python code that generates the figure(s):
```python
{current_code}
```

Please rewrite the COMPLETE Python script, incorporating the improvements suggested
in the critique. Rules:
1. Keep ALL data loading, JSON parsing, computation, and algorithm logic exactly the same
2. Only modify plotting/styling/layout code (colors, fonts, spacing, sizes, positions, etc.)
3. The script must remain self-contained and runnable
4. Do NOT change file paths, data sources, or CLI argument handling
5. Do NOT add new library imports beyond what's already imported
6. Keep the same overall function structure
7. Keep MNE's plot_topomap for topoplots if present
8. Keep all EEG preprocessing (filtering, referencing) the same
9. Keep hemisphere shading and marker logic the same (only adjust styling)

Return ONLY the Python code inside a single ```python ... ``` block.
The code must be the COMPLETE script, not just changed parts."""

    print(f"  Asking Gemini to rewrite code...")
    response = call_gemini(prompt)

    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("  WARNING: Could not extract code from response")
    return ""


def optimize_group(group_name: str, group: dict, max_rounds: int):
    """Run the optimization loop for one figure group."""
    print(f"\n{'='*70}")
    print(f"  Optimizing: {group['name']}")
    print(f"{'='*70}")

    script_path = group['script']

    # Initial render
    print(f"\n[Initial] Rendering...")
    if not render_group(group):
        print("Initial render failed! Skipping this group.")
        return

    current_code = script_path.read_text()

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- Round {round_num}/{max_rounds} ---")

        # Backup
        backup_path = LOG_DIR / f'{group_name}_round{round_num-1}.py'
        backup_path.write_text(current_code)
        for fp in group['figure_files']:
            if fp.exists():
                shutil.copy(fp, LOG_DIR / f'{fp.stem}_{group_name}_round{round_num-1}.png')

        # Critique
        print(f"[Round {round_num}] Getting critique...")
        critique = critique_group(group, round_num)
        print(f"\n  Critique preview:\n{critique[:800]}...\n")
        (LOG_DIR / f'{group_name}_critique_round{round_num}.md').write_text(critique)

        # Rewrite
        print(f"[Round {round_num}] Rewriting code...")
        new_code = rewrite_code(current_code, critique, group, round_num)

        if not new_code:
            print("  Code rewrite failed. Stopping this group.")
            break

        (LOG_DIR / f'{group_name}_round{round_num}.py').write_text(new_code)
        script_path.write_text(new_code)

        # Re-render
        print(f"[Round {round_num}] Re-rendering...")
        if render_group(group):
            current_code = new_code
            print(f"  Round {round_num} complete!")
        else:
            print(f"  Render FAILED. Reverting to previous version.")
            script_path.write_text(current_code)
            render_group(group)
            break

    print(f"\n  {group['name']} optimization complete.")


def main():
    parser = argparse.ArgumentParser(description='Figure Optimization Loop (Gemini Critic)')
    parser.add_argument('--group', type=str, default='all',
                        choices=['all', 'characterization', 'pd_pipeline', 'rda_pipeline', 'fig1_examples'],
                        help='Which figure group to optimize')
    parser.add_argument('--rounds', type=int, default=3, help='Max optimization rounds per group')
    args = parser.parse_args()

    print("=" * 70)
    print("  Figure Optimization Loop (Gemini Critic)")
    print(f"  Groups: {args.group}, Rounds: {args.rounds}")
    print("=" * 70)

    if args.group == 'all':
        groups_to_run = list(FIGURE_GROUPS.keys())
    else:
        groups_to_run = [args.group]

    for group_name in groups_to_run:
        group = FIGURE_GROUPS[group_name]
        optimize_group(group_name, group, args.rounds)

    print(f"\n{'='*70}")
    print(f"  All optimizations complete. Logs in: {LOG_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
