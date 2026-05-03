#!/usr/bin/env python3
"""
Composite Fig 2: PaperBanana Panel B + our real EEG Panels A and C.

1. Renders our matplotlib EEG panels (A and C) to a temp file
2. Crops Panel B from the PaperBanana stylist image
3. Composites them with drawn titles

Usage:
    conda run -n morgoth python paper_materials/composite_fig2.py
"""

import subprocess
import sys
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = SCRIPT_DIR / 'figures'

PB_IMAGE = SCRIPT_DIR / 'improved_figures' / 'paperbanana_fig2_pd_pipeline_target_diagram_stylist_desc0_base64_jpg.png'
GENERATOR = SCRIPT_DIR / 'generate_fig2_pd_pipeline.py'
OUTPUT = FIGURES_DIR / 'fig2_pd_pipeline.png'
RAW_OUTPUT = FIGURES_DIR / 'fig2_pd_pipeline_RAW.png'


def render_raw():
    """Render matplotlib EEG panels to a temp file."""
    # Temporarily redirect output
    code = GENERATOR.read_text()
    code_tmp = code.replace('fig2_pd_pipeline.png', 'fig2_pd_pipeline_RAW.png')
    tmp_script = Path('/tmp/gen_fig2_raw.py')
    tmp_script.write_text(code_tmp)

    # Insert project dir resolution at top of temp script
    preamble = f"import os; os.chdir('{PROJECT_DIR}')\n"
    tmp_script.write_text(preamble + code_tmp)

    result = subprocess.run(
        [sys.executable, str(tmp_script)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"Render failed: {result.stderr[-300:]}")
        return False
    if not RAW_OUTPUT.exists():
        print(f"ERROR: {RAW_OUTPUT} not created")
        return False
    print(f"Rendered raw: {RAW_OUTPUT}")
    return True


def composite():
    """Combine PaperBanana Panel B with our Panels A and C."""
    pb = Image.open(str(PB_IMAGE))
    ours = Image.open(str(RAW_OUTPUT))

    pb_w, pb_h = pb.size
    ours_w, ours_h = ours.size
    print(f"PaperBanana: {pb.size}, Ours: {ours.size}")

    # --- Crop regions ---
    # Our matplotlib: A is ~0-28%, gap, B is ~32-68%, gap, C is ~72-100%
    # We want just the EEG content from A and C (no titles, no Panel B)
    panel_a = ours.crop((0, 0, int(ours_w * 0.295), ours_h))

    # For Panel C, crop from where the EEG starts (skip the gap/B area)
    panel_c = ours.crop((int(ours_w * 0.715), 0, ours_w, ours_h))

    # PaperBanana Panel B: the middle ~52% of the image
    panel_b = pb.crop((int(pb_w * 0.22), 0, int(pb_w * 0.78), pb_h))

    # Scale Panel B to match our figure height
    scale = ours_h / panel_b.size[1]
    b_new_w = int(panel_b.size[0] * scale)
    panel_b_scaled = panel_b.resize((b_new_w, ours_h), Image.LANCZOS)

    print(f"Panel A: {panel_a.size}")
    print(f"Panel B (scaled): {panel_b_scaled.size}")
    print(f"Panel C: {panel_c.size}")

    # --- Build composite ---
    gap = 20  # small gap between panels
    total_w = panel_a.size[0] + gap + panel_b_scaled.size[0] + gap + panel_c.size[0]
    title_h = 90  # space for titles at top

    comp = Image.new('RGB', (total_w, ours_h + title_h), 'white')

    # Paste panels
    x = 0
    comp.paste(panel_a, (x, title_h))
    a_end = x + panel_a.size[0]

    x = a_end + gap
    comp.paste(panel_b_scaled, (x, title_h))
    b_end = x + panel_b_scaled.size[0]

    x = b_end + gap
    comp.paste(panel_c, (x, title_h))

    # --- Draw titles ---
    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 52)
    except:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 52)
        except:
            font = ImageFont.load_default()

    # A. Input
    draw.text((20, 20), 'A. Input', fill='black', font=font)

    # B. Pipeline Architecture — already in PaperBanana image, but add if cut off
    # (PaperBanana's title is at the top of panel_b, should be visible)

    # C. Output
    draw.text((b_end + gap + 20, 20), 'C. Output', fill='black', font=font)

    # --- Thin vertical separator lines ---
    draw.line([(a_end + gap//2, title_h), (a_end + gap//2, ours_h + title_h)],
              fill='#DDDDDD', width=2)
    draw.line([(b_end + gap//2, title_h), (b_end + gap//2, ours_h + title_h)],
              fill='#DDDDDD', width=2)

    # Save
    comp.save(str(OUTPUT), dpi=(300, 300))
    print(f"Saved composite: {comp.size} -> {OUTPUT}")

    # Cleanup
    RAW_OUTPUT.unlink()


def main():
    print("=" * 50)
    print("Compositing Fig 2")
    print("=" * 50)

    if not PB_IMAGE.exists():
        print(f"ERROR: PaperBanana image not found: {PB_IMAGE}")
        return

    print("\nStep 1: Rendering matplotlib EEG panels...")
    if not render_raw():
        return

    print("\nStep 2: Compositing...")
    composite()

    print("\nDone!")


if __name__ == '__main__':
    main()
