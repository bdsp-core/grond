#!/usr/bin/env python3
"""
Figure optimization loop using Gemini as visual critic.

Iteratively improves render_figures.py by:
1. Rendering current figures
2. Sending to Gemini for visual critique
3. Rewriting the render code based on feedback
4. Re-rendering and repeating

Usage:
    conda run -n morgoth python paper_materials/optimize_figures.py

Requires: google-genai package (pip install google-genai)
"""

import os
import re
import sys
import json
import base64
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
RENDER_SCRIPT = SCRIPT_DIR / 'render_figures.py'
PICK = '{"lpd":[1,15,9],"gpd":[17,2,9],"lrda":[11,3,6],"grda":[0,4,9]}'
FIGURE_FILES = [
    SCRIPT_DIR / 'figure_lpd_examples.png',
    SCRIPT_DIR / 'figure_gpd_examples.png',
    SCRIPT_DIR / 'figure_lrda_examples.png',
    SCRIPT_DIR / 'figure_grda_examples.png',
]

GOOGLE_API_KEY = "REDACTED-GOOGLE-API-KEY"
MODEL_NAME = "gemini-2.5-flash"
MAX_ROUNDS = 3
LOG_DIR = SCRIPT_DIR / 'optimization_logs'
LOG_DIR.mkdir(exist_ok=True)

FIGURE_CAPTION = """Publication figures showing EEG characterization examples for
periodic and rhythmic patterns (LPD, GPD, LRDA, GRDA). Each figure contains 3 examples
at different difficulty levels (Easy/Medium/Hard based on inter-rater agreement).
Each row shows: (left) 10-second 18-channel bipolar EEG with discharge timing markers
and hemisphere shading, (center) MNE-interpolated topoplot showing spatial distribution
of channel involvement scores, (right) ACNS 2021 verbal description.
Target journal: high-impact neurology/neuroscience journal."""

STYLE_GUIDE = """You are reviewing scientific EEG figures for publication in a top neurology journal.

Key standards for these figures:
- Clean, minimal design suitable for print at column width (~7 inches)
- Fonts: 7-9pt for labels, 10-12pt for titles. All text must be readable when printed
- EEG traces: clean black lines on white background, consistent line width
- Channel labels: left-aligned, readable, properly spaced
- Topoplots: smooth interpolation, clear colorbar with scale, head outline visible
- Hemisphere shading: subtle light blue, not distracting
- Discharge markers: visible red dashed lines at correct locations
- Verbal descriptions: readable, properly positioned
- Consistent style across all 4 figures (LPD, GPD, LRDA, GRDA)
- Difficulty badges: clear but not dominant
- Time axis: clear 0-10s markings
- White background throughout
- No unnecessary visual clutter
- Professional, publication-ready appearance

IMPORTANT: All 4 figures must have IDENTICAL styling — only the data differs."""


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def call_gemini(prompt: str, images_b64: list[str] | None = None) -> str:
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


def render_figures() -> bool:
    """Run render_figures.py and return True if successful."""
    result = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), '--pick', PICK],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  RENDER FAILED: {result.stderr[-500:]}")
        return False
    print(f"  Rendered successfully")
    return True


def critique_figures(round_num: int) -> str:
    """Send all 4 figures to Gemini for critique."""
    images = []
    for fp in FIGURE_FILES:
        if fp.exists():
            images.append(image_to_base64(fp))

    prompt = f"""{STYLE_GUIDE}

Here are the current versions of 4 EEG characterization figures (iteration {round_num}).
The figures show LPD, GPD, LRDA, and GRDA examples respectively.

Figure caption: {FIGURE_CAPTION}

Please critique these figures for publication readiness. Focus on:
1. Layout and spacing — are panels well-organized? Is there wasted space?
2. Typography — are fonts consistent, readable, properly sized?
3. EEG traces — are they clean, well-scaled, properly labeled?
4. Topoplots — smooth interpolation, clear color scale, head outline?
5. Discharge markers — visible, correctly placed on involved hemisphere?
6. Hemisphere shading — subtle but informative?
7. Verbal descriptions — readable, properly formatted?
8. Cross-figure consistency — do all 4 figures have identical styling?
9. Overall professional appearance for a top journal

Be specific and actionable. List the top 5-8 most impactful improvements.
Do NOT suggest changes to the data — only the visual presentation and styling.
Focus on changes that can be made in matplotlib code."""

    print(f"  Sending {len(images)} figures to Gemini for critique...")
    return call_gemini(prompt, images)


def rewrite_code(current_code: str, critique: str, round_num: int) -> str:
    """Have Gemini rewrite the plotting code to address the critique."""
    prompt = f"""You are improving a matplotlib figure generation script for publication-quality EEG figures.

Here is the current critique (iteration {round_num}):
{critique}

Here is the current Python code that generates the figures:
```python
{current_code}
```

Please rewrite the COMPLETE Python script, incorporating the improvements suggested
in the critique. Rules:
1. Keep ALL data loading, JSON parsing, and computation logic exactly the same
2. Only modify plotting/styling/layout code (colors, fonts, spacing, sizes, etc.)
3. The script must remain self-contained and runnable
4. Do NOT change file paths, data sources, or the --pick argument handling
5. Do NOT add any new library imports that aren't already imported (matplotlib, numpy, json, mne, scipy are available)
6. Keep the same overall structure: render_subtype() calls draw_eeg_panel(), draw_topoplot(), draw_info_panel()
7. Keep MNE's plot_topomap for the topoplot (do not replace with manual interpolation)
8. All 4 figures must have IDENTICAL styling
9. Keep the notch filter + bandpass preprocessing in draw_eeg_panel
10. Keep the hemisphere shading logic (light blue, involved side only for LPD/LRDA, both for GPD/GRDA)
11. Keep discharge markers (red dashed lines on involved hemisphere)

Return ONLY the Python code inside a single ```python ... ``` block.
The code must be the COMPLETE script, not just changed parts."""

    print(f"  Asking Gemini to rewrite code...")
    response = call_gemini(prompt)

    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("  WARNING: Could not extract code from response")
    return ""


def main():
    print("=" * 70)
    print("  Figure Optimization Loop (Gemini Critic)")
    print("=" * 70)

    # Initial render
    print("\n[Initial] Rendering figures...")
    if not render_figures():
        print("Initial render failed!")
        return

    # Read current code
    current_code = RENDER_SCRIPT.read_text()

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"  Round {round_num}/{MAX_ROUNDS}")
        print(f"{'='*50}")

        # Backup current code
        backup_path = LOG_DIR / f'render_figures_round{round_num-1}.py'
        backup_path.write_text(current_code)

        # Backup current figures
        for fp in FIGURE_FILES:
            if fp.exists():
                shutil.copy(fp, LOG_DIR / f'{fp.stem}_round{round_num-1}.png')

        # Critique
        print(f"\n[Round {round_num}] Getting critique...")
        critique = critique_figures(round_num)
        print(f"\n  Critique:\n{critique[:1000]}...")

        # Save critique
        (LOG_DIR / f'critique_round{round_num}.md').write_text(critique)

        # Rewrite
        print(f"\n[Round {round_num}] Rewriting code...")
        new_code = rewrite_code(current_code, critique, round_num)

        if not new_code:
            print("  Code rewrite failed. Stopping.")
            break

        # Save new code
        (LOG_DIR / f'render_figures_round{round_num}.py').write_text(new_code)

        # Write to actual render script
        RENDER_SCRIPT.write_text(new_code)

        # Re-render
        print(f"\n[Round {round_num}] Re-rendering...")
        if render_figures():
            current_code = new_code
            print(f"  Round {round_num} complete!")
        else:
            print(f"  Render FAILED. Reverting to previous version.")
            RENDER_SCRIPT.write_text(current_code)
            render_figures()
            break

    # Final summary
    print(f"\n{'='*70}")
    print(f"  Optimization complete after {min(round_num, MAX_ROUNDS)} rounds")
    print(f"  Logs saved to: {LOG_DIR}")
    print(f"  Final figures: {[fp.name for fp in FIGURE_FILES]}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
