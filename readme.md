# ComfyUI-Optical-Realism

**Trying to Kill the "Uncanny Valley" of uniform entropy?**

This node is an attempt at physics-based post-processing that uses a Depth Map to simulate **Atmospheric Perspective**, **Light Transport**, and **Physical Lens Imperfections**. It trys to turns a "generation" into a "photograph."

## The Problem
AI images often suffer from the **Frequency Distribution Problem**. High-frequency details (noise, texture) are distributed equally across the image. In real photography, physics dictates that:
1.  Distant objects have lower contrast and lifted blacks (Atmosphere).
2.  Bright background light bleeds over foreground edges (Light Wrap).
3.  Lenses are not mathematically perfect (Chromatic Aberration & Vignette).
4.  Film grain lives in the emulsion, not floating on top of the image (Depth-Aware Grain).

This node does some shit to address all that. Play with it or don't.

## Installation

1.  Clone this repo into your `custom_nodes` folder:
    ```bash
    git clone https://github.com/skatardude10/ComfyUI-Optical-Realism.git
    ```
2.  Restart ComfyUI.

## The Workflow

**Crucial:** This node requires a **Depth Map** to work its magic.
I highly recommend using **Depth Anything V2** (specifically the `vit_l` model for best edge detection).

**Wiring it up:**
1.  **Input Image** $\rightarrow$ `Remove Alpha (included utility)` $\rightarrow$ `Optical Realism` (Image Input)
2.  **Input Image** $\rightarrow$ `Depth Anything V2` $\rightarrow$ `Invert Image` $\rightarrow$ `Optical Realism` (Depth Input)
    *   *Note: My script assumes Black = Near, White = Far. If your depth model outputs the opposite, use an Invert Image node.*

## The Settings

Here is exactly what the sliders do.

### 💨 Atmospherics
*   **Atmosphere Enabled:** The master switch. Turns on the depth-based physics.
*   **Haze Strength:** How "thick" is the air? Low values = clear winter day. High values = humid rainforest or foggy street.
*   **Lift Blacks:** **The secret sauce.** Real shadows in the distance aren't pure black (`#000000`); they are dark atmospheric grey-blue. This lifts the background shadows to separate the subject from the environment.
*   **Depth Offset:** Pushes the "fog curtain" forward or backward. Negative values push it back (clearer foreground), positive values pull it close (macro feel).

### 📷 Optical Phenomena
*   **Light Wrap Strength:** Simulates "Bloom" or "Halation." It takes bright background light and bleeds it over the edges of the foreground subject. Kills the "cutout sticker" look.
*   **Chromatic Aberration:** Uses **Sub-Pixel Sampling** (not just resizing) to create a mathematically smooth, infinite-resolution lens fringing effect. Keep this low (`0.001` - `0.003`) for realism.
*   **Vignette Intensity:** Darkens the corners to mimic a physical lens barrel. Subtle framing that guides the eye to the center.

### 🎞️ Film Emulation
*   **Grain Power:** Adds texture. **Crucially, this is Depth-Aware.** The grain is sharp on the focused subject but gets softer/mushier in the blurred background, just like real film.
*   **Monochrome Grain:**
    *   `True` (Default): Simulates **Film Stock**. Noise affects luminance only.
    *   `False`: Simulates **Digital Sensors**. Independent RGB noise (Color noise).
*   **Highlight Roll-off:** AI likes to clip bright lights to pure white instantly. This adds a "shoulder" to the highlights, compressing them softly so they look creamy instead of harsh.

## Comparison



## Troubleshooting

**"RuntimeError: The size of tensor a (4) must match..."**
*   Your image has an Alpha channel (RGBA).
*   **Fix:** I included a helper node called **`Remove Alpha (RGBA to RGB)`**. Put this *before* the Optical Realism node and the Depth node.

## Credits
Obvious
