#!/usr/bin/env python3
"""
Improve a figure using Gemini's image generation, inspired by PaperBanana.

Sends the current figure + detailed description to Gemini and asks it to
generate an improved version as a new image.

Usage:
    python paper_materials/improve_figure_with_gemini.py --figure fig2
    python paper_materials/improve_figure_with_gemini.py --figure fig3
"""

import base64
import argparse
import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / 'figures'
OUTPUT_DIR = SCRIPT_DIR / 'improved_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

GOOGLE_API_KEY = "REDACTED-GOOGLE-API-KEY"

# ── Figure descriptions ──

FIGURE_DESCRIPTIONS = {
    'fig2': {
        'image_path': FIGURES_DIR / 'fig2_pd_pipeline.png',
        'description': """
FIGURE 2: PD (Periodic Discharge) Characterization Pipeline

This is a three-panel horizontal composite figure for a neurology journal paper.
It shows how our algorithm processes a 10-second EEG recording to characterize
lateralized periodic discharges (LPDs).

PANEL A (left, ~30% width): "Input: 19-Channel EEG"
- Shows a real 10-second, 19-channel EEG recording in common average reference montage
- Channels are grouped: left parasagittal (Fp1, F3, C3, P3), left temporal (F7, T3, T5, O1),
  midline (Fz, Cz, Pz), right parasagittal (Fp2, F4, C4, P4), right temporal (F8, T4, T6, O2)
- Small gaps between channel groups for visual separation
- Black traces on white background, showing clear periodic sharp discharges
- Channel labels on the left axis, time axis 0-10s at bottom
- This should look like a real clinical EEG display

PANEL B (center, ~40% width): "Pipeline Architecture"
- A clean flowchart showing the processing pipeline
- TOP: Blue box "ChannelPD-Net" — Per-channel CNN+Attention (×18), produces 18 PD Probabilities + 18 Frequency Estimates
- This feeds into THREE parallel branches via arrows:
  1. GREEN box (left): "Laterality Detection" — Compare L vs R hemisphere mean PD probability → Output: Left/Right, AUC = 0.963
  2. RED/SALMON box (center): "HemiCET+DP Discharge Detection" — 8-channel CET-UNet → Evidence trace, CNN+ACF Frequency Prior, Dynamic Programming with periodic prior, EM Refinement + Filtering → Output: discharge times t₁, t₂, ..., tₙ, Frequency = 1/median(IPI)
  3. ORANGE box (right): "Discharge-Locked Topographic Localization" — At each detected discharge time, extract 19-channel voltage. Laplacian-GFP Alignment, Two-Pass Template Refinement, GFP²-Weighted Averaging → Topoplot + Verbal Description
- BOTTOM: Three output boxes: "Laterality" | "Timing + Frequency" | "Spatial Localization"
- Use rounded rectangles with light colored fills, dark borders, clear arrows
- Professional, clean academic style

PANEL C (right, ~30% width): "Output: Characterized LPD"
- Shows the SAME EEG as Panel A, but now annotated with algorithm output:
  - Red dashed vertical lines at each detected discharge time
  - Light blue shading on the involved hemisphere's channels
- In the lower-right corner: a small circular topoplot (inferno colormap) showing the
  Laplacian topography of the mean discharge — this is a head-shaped circle with electrodes
  labeled (Fp1, F3, C3, etc.), hot colors showing the discharge maximum
- Below the topoplot: the verbal description in italic text, e.g.,
  "LPD, left sided (unilateral), at 1.1 Hz, left posterior temporal"

STYLE:
- Publication quality, suitable for Nature Neuroscience or similar
- Clean sans-serif fonts (Arial/Helvetica), 8-10pt
- Panel labels "A.", "B.", "C." in bold at top-left of each panel
- White background throughout
- The flowchart boxes should be clean, with distinct colors per branch
- Arrows should be clean and well-proportioned
- The topoplot should be clearly circular (not oval), with the inferno colormap
  showing a focal hot spot where the discharge is maximal
""",
    },
    'fig3': {
        'image_path': FIGURES_DIR / 'fig3_rda_pipeline.png',
        'description': """
FIGURE 3: RDA (Rhythmic Delta Activity) Characterization Pipeline

Same three-panel layout as Figure 2, but for rhythmic delta activity instead of periodic discharges.

PANEL A (left): "Input: 19-Channel EEG"
- Shows a real LRDA (lateralized rhythmic delta activity) recording
- Same channel layout as Figure 2 (19 channels, average reference)
- Should show clear rhythmic sinusoidal delta waves, larger on one side

PANEL B (center): "Pipeline Architecture"
- TOP: Blue box "W05: Iterative Narrowband Refinement"
- Pass 1 (light blue box): "Coarse Analysis" — Bandpass 0.5-3.5 Hz, Lateralization via mean variance per hemisphere, Frequency via Hilbert instantaneous frequency from top-3 dominant channels
- Pass 2 (medium blue box): "Narrowband Refinement" — Bandpass at est_freq ± 0.4 Hz, Refined Lateralization via envelope amplitude, Refined Frequency via Hilbert on dominant hemisphere
- Three branches:
  1. GREEN: "Laterality Detection" — L vs R narrowband amplitude, AUC = 0.837
  2. PURPLE: "Spatial Extent — PLV × Amplitude" — Per-channel phase coherence with dominant hemisphere × narrowband amplitude, Threshold → count/18
  3. ORANGE: "Topographic Localization" — Per-channel Hilbert amplitude envelope, Laplacian transform → Topoplot + Verbal Description
- Bottom outputs: "Laterality" | "Spatial Extent + Frequency" | "Spatial Localization"

PANEL C (right): "Output: Characterized LRDA"
- Same EEG with green narrowband overlay showing the rhythmic delta
- Light blue hemisphere shading on involved side
- Small circular topoplot (inferno) in lower-right showing narrowband amplitude distribution
- Verbal description below

STYLE: Same as Figure 2 — clean, professional, publication-ready.
""",
    },
}


def improve_figure(figure_key: str, n_rounds: int = 2):
    """Send figure to Gemini for improvement."""
    from google import genai
    from google.genai import types

    fig_info = FIGURE_DESCRIPTIONS[figure_key]
    image_path = fig_info['image_path']
    description = fig_info['description']

    if not image_path.exists():
        print(f"ERROR: {image_path} not found")
        return

    print(f"Improving {figure_key}: {image_path.name}")
    print(f"Description length: {len(description)} chars")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    image_bytes = image_path.read_bytes()

    for round_num in range(1, n_rounds + 1):
        print(f"\n--- Round {round_num}/{n_rounds} ---")

        prompt = f"""You are an expert scientific figure designer.

Here is a figure from an academic paper about automated EEG analysis.
Please generate an IMPROVED version of this figure based on the detailed description below.

The improved version should:
1. Be cleaner and more professional-looking
2. Have better typography (consistent font sizes, readable labels)
3. Have better spacing and layout (no wasted space, well-proportioned panels)
4. Have cleaner flowchart boxes with better visual hierarchy
5. Maintain the same three-panel A/B/C structure
6. Keep the same data content — just improve the visual presentation
7. Be suitable for a top-tier neurology journal

DETAILED FIGURE DESCRIPTION:
{description}

Please generate an improved version of this figure as an image.
"""

        contents = [
            types.Part.from_bytes(mime_type="image/png", data=image_bytes),
            types.Part.from_text(text=prompt),
        ]

        print("  Sending to Gemini...")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    response_modalities=["image", "text"],
                ),
            )

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    img_data = part.inline_data.data
                    if isinstance(img_data, str):
                        img_data = base64.b64decode(img_data)

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    out_path = OUTPUT_DIR / f'{figure_key}_improved_r{round_num}_{timestamp}.png'
                    out_path.write_bytes(img_data)
                    print(f"  Saved: {out_path}")

                    # Use this as input for next round
                    image_bytes = img_data
                    break
                elif hasattr(part, 'text') and part.text:
                    print(f"  Gemini text response: {part.text[:200]}...")
            else:
                print("  No image in response")

        except Exception as e:
            print(f"  Error: {e}")
            break

    print(f"\nDone! Check {OUTPUT_DIR} for improved versions.")


def main():
    parser = argparse.ArgumentParser(description='Improve figure with Gemini')
    parser.add_argument('--figure', type=str, required=True, choices=['fig2', 'fig3'],
                        help='Which figure to improve')
    parser.add_argument('--rounds', type=int, default=2, help='Number of improvement rounds')
    args = parser.parse_args()

    improve_figure(args.figure, args.rounds)


if __name__ == '__main__':
    main()
