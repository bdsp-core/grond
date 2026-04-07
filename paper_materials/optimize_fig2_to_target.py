#!/usr/bin/env python3
"""
Iteratively revise generate_fig2_pd_pipeline.py to match PaperBanana target.

Shows Gemini both the target image (PaperBanana stylist version) and our
current matplotlib output, asks it to rewrite code to match the target's
layout/styling while keeping our real EEG data and annotations.

Usage:
    conda run -n morgoth python paper_materials/optimize_fig2_to_target.py [--rounds N]
"""

import os
import re
import sys
import base64
import shutil
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / 'figures'
LOG_DIR = SCRIPT_DIR / 'optimization_logs' / 'fig2_target'
LOG_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR_SCRIPT = SCRIPT_DIR / 'generate_fig2_pd_pipeline.py'
OUTPUT_FIGURE = FIGURES_DIR / 'fig2_pd_pipeline.png'
TARGET_FIGURE = SCRIPT_DIR / 'improved_figures' / 'paperbanana_fig2_pd_pipeline_target_diagram_stylist_desc0_base64_jpg.png'

GOOGLE_API_KEY = "REDACTED-GOOGLE-API-KEY"
MODEL_NAME = "gemini-2.5-flash"


def image_to_b64(path):
    return base64.b64encode(path.read_bytes()).decode('ascii')


def call_gemini(prompt, images_b64=None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_API_KEY)
    contents = []
    if images_b64:
        for img in images_b64:
            contents.append(types.Part.from_bytes(
                mime_type="image/png", data=base64.b64decode(img)))
    contents.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=MODEL_NAME, contents=contents,
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=60000))
    return response.text


def render():
    """Run the generator script. Returns True if successful."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR_SCRIPT)],
        capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  RENDER FAILED:\n{result.stderr[-500:]}")
        return False
    print("  Rendered OK")
    return True


def critique(round_num):
    """Show Gemini the target and current, get specific code suggestions."""
    target_b64 = image_to_b64(TARGET_FIGURE)
    current_b64 = image_to_b64(OUTPUT_FIGURE)

    prompt = f"""You are helping improve a scientific figure for a neurology journal.

IMAGE 1 (first image): TARGET — this is the design we want to match. It was generated
by PaperBanana (an AI figure design tool). Note its layout, fonts, colors, spacing,
and flowchart style. This is the GOAL.

IMAGE 2 (second image): CURRENT — this is our current matplotlib-generated figure.
It uses real EEG data and real algorithm output, which is correct and must be preserved.
But the visual design (layout, fonts, colors, spacing, flowchart boxes) needs to match
the target more closely.

IMPORTANT CONSTRAINTS:
- The real EEG data in Panels A and C must NOT change — same waveforms, same channels
- The discharge markers (red dashed lines) in Panel C must stay
- The hemisphere shading (light blue) in Panel C must stay
- The topoplot inset in Panel C must stay (inferno colormap)
- The verbal description must stay
- All data loading, filtering, and algorithm code must remain unchanged
- ONLY modify the visual styling: layout proportions, font sizes, colors, box styles,
  arrow styles, spacing, line widths, etc.

This is iteration {round_num}. Please list the top 5-8 most impactful changes to make
our matplotlib code look more like the target. Be very specific about:
- What to change (e.g., "the flowchart boxes", "the font size of channel labels")
- The current value (if you can tell)
- The target value
- Why it matters

Focus on the biggest visual differences between current and target."""

    print(f"  Sending target + current to Gemini for comparison...")
    return call_gemini(prompt, [target_b64, current_b64])


def rewrite(current_code, critique_text, round_num):
    """Have Gemini rewrite the code to match the target."""
    prompt = f"""You are improving a matplotlib script to match a target figure design.

Here is the critique (iteration {round_num}):
{critique_text}

Here is the current Python code:
```python
{current_code}
```

Please rewrite the COMPLETE Python script, making it look more like the target.
Rules:
1. Keep ALL data loading, EEG processing, topoplot generation, discharge detection unchanged
2. Keep the same import structure and file paths
3. Only modify plotting/styling code: fonts, colors, spacing, box styles, layout proportions, etc.
4. The script must remain runnable
5. Keep the three-panel A/B/C structure
6. Keep discharge markers, hemisphere shading, topoplot inset, verbal description
7. The flowchart in Panel B should be redesigned to match the target's style more closely
8. Focus on the specific changes from the critique

Return ONLY the complete Python code inside a single ```python ... ``` block."""

    print(f"  Asking Gemini to rewrite code...")
    response = call_gemini(prompt)
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("  WARNING: Could not extract code from response")
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("  Fig 2 Target-Matching Optimization")
    print(f"  Target: {TARGET_FIGURE.name}")
    print(f"  Rounds: {args.rounds}")
    print("=" * 70)

    if not TARGET_FIGURE.exists():
        print(f"ERROR: Target not found: {TARGET_FIGURE}")
        return

    # Initial render
    print("\n[Initial] Rendering...")
    if not render():
        print("Initial render failed!")
        return

    current_code = GENERATOR_SCRIPT.read_text()

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'='*50}")
        print(f"  Round {rnd}/{args.rounds}")
        print(f"{'='*50}")

        # Backup
        (LOG_DIR / f'code_round{rnd-1}.py').write_text(current_code)
        if OUTPUT_FIGURE.exists():
            shutil.copy(OUTPUT_FIGURE, LOG_DIR / f'fig2_round{rnd-1}.png')

        # Critique
        print(f"\n[Round {rnd}] Getting critique...")
        crit = critique(rnd)
        print(f"\n  Critique preview:\n{crit[:800]}...\n")
        (LOG_DIR / f'critique_round{rnd}.md').write_text(crit)

        # Rewrite
        print(f"[Round {rnd}] Rewriting code...")
        new_code = rewrite(current_code, crit, rnd)
        if not new_code:
            print("  Rewrite failed. Stopping.")
            break

        (LOG_DIR / f'code_round{rnd}.py').write_text(new_code)
        GENERATOR_SCRIPT.write_text(new_code)

        # Re-render
        print(f"[Round {rnd}] Re-rendering...")
        if render():
            current_code = new_code
            print(f"  Round {rnd} complete!")
        else:
            print(f"  Render FAILED. Reverting.")
            GENERATOR_SCRIPT.write_text(current_code)
            render()
            break

    print(f"\n{'='*70}")
    print(f"  Done. Logs in {LOG_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
