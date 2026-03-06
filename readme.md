# ComfyUI-Optical-Realism

**Subtle or Not attempt to bridge easily between AI generation and physical photography.**

AI images tend to have uniform frequency distribution—where noise, texture, and light are mathematically perfect across the entire frame. This node tries to capture more real photo feel, simulate physical lenses, atmospheric density, and film chemistry to introduce beautiful optical imperfections.

This custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) takes a standard RGB image and a **Depth Map**, acting as a virtual camera to simulate physical lens geometry, depth-of-field, light scattering, and film emulsion.

## The Problem vs. The Solution

| The "AI Look" | Real-World Photography |
| :--- | :--- |
| Perfectly sharp everywhere | **Depth of Field (Bokeh)** isolates the subject |
| Hard edges on backlit subjects | Bright light bleeds around edges (**Light Wrap** & **Halation**) |
| Uniform black shadows | Distant shadows are lifted and hazy (**Atmospherics**) |
| Mathematically straight lines | Glass lenses slightly curve light (**Lens Distortion**) |
| Perfect white light | Real film leans warm/cool (**Temp/Tint**) |
| Noise floating *over* the image | Film grain is sharp in-focus, but soft in the blur |

This node addresses all of these physics-based phenomena in a single pass. 

---

## 📷 Before & After
I like it subtle. You decide:

*(Left: Raw AI Generation | Right: Optical Realism Processing)*  S
| Before | After |
| :---: | :---: |
| <img src="examples/before.png" width="100%"> | <img src="examples/after.png" width="100%"> |
| <img src="examples/before2.png" width="100%"> | <img src="examples/after2.png" width="100%"> |

---

# Example Workflow in Image:
<img src="Example-Workflow.png" width="100%">

---

## ⚙️ Installation & Workflow

1. Clone this repository into your `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/skatardude10/ComfyUI-Optical-Realism.git
   ```
2. Restart ComfyUI.

### The Workflow 
**Crucial:** This node requires a **Depth Map** to calculate 3D space. We highly recommend using **Depth Anything V2** (specifically the `vit_l` model) for the most accurate edge detection. Example workflow includes other custom nodes you can choose to use or replace as you see fit.

**Basic Routing:**
1. **Input Image** → `Remove Alpha (included utility)` → `Optical Realism` (Image Input)
2. **Input Image** → `Depth Anything V2` → `Optical Realism` (Depth Input)
*(Note: The script assumes Black = Near, White = Far. If your depth model outputs the opposite, use an Invert Image node in between).*

<img src="examples/Core-Node.png" width="400">
<img src="examples/Core-Nodes.png" width="100%">

Optional upscalers like **SeedVR2** pair incredibly well with this pipeline to clean up details before optical processing, this is included in the Example Workflow image above, with a blank white mask to use here if you decide to use this.
<img src="whitemask.png" width="400">
<img src="examples/Upscaling-Nodes.png" width="100%">

---

## 🎛️ The Settings Explained

Here is a breakdown of what each parameter does under the hood.

### 📐 Camera & Lens Geometry
* **Lens Distortion:** Mimics physical lens curvature. Positive values (`0.05`) add barrel distortion (wide-angle bulge), while negative values add pincushioning. The script safely reflects the edges so you never get black borders.
  * <img src="examples/lens-distortion.png" width="100%">
* **Color Temperature & Tint:** Shifts the white balance while mathematically preserving overall luminance so highlights never blow out. 
  * *Tip: For a classic cinematic "Kodak" look that breathes life into skin tones, try `Temp: 0.10` and `Tint: -0.05`.*

### 💨 Atmospherics
* **Atmosphere Enabled:** The master switch for depth-based physics.
* **Haze Strength:** How "thick" is the air? High values push the background into a humid fog.
* **Lift Blacks:** **The secret sauce.** Real shadows in the distance aren't pure black (`#000000`); they are dark atmospheric grey-blue. This lifts background shadows to physically separate the subject from the environment.
* **Depth Offset:** Pushes the "fog curtain" forward or backward.

### 🔍 Depth of Field (Bokeh)
* **DoF Intensity:** Controls the size of the blur. *Note: This uses a custom Circular Disc Convolution, meaning out-of-focus background points will turn into beautiful, realistic circles (Bokeh) rather than muddy Gaussian smudges.*
* **DoF Auto Focus:** When enabled, the node analyzes the center 60% of the depth map and automatically locks focus onto your main subject based on depth percentiles.
* **DoF Sharpness Radius:** The "safe zone" around the focal point. Higher values (`0.35`) keep more of the subject crisp before the blur gently ramps up.

### 🌟 Optical Phenomena
* **Light Wrap Strength:** Takes bright background light and physically bleeds it over the edges of the foreground subject. Kills the "cutout sticker" look.
* **Pro-Mist Strength:** Simulates a physical glass diffusion filter (like a Tiffen Black Pro-Mist). It isolates high-luminance pixels and creates a gentle, warm bloom, taking the "digital edge" off sharp lines.
  * | Without Pro-Mist | With Pro-Mist |
    | :---: | :---: |
    | <img src="examples/pro-mist-1.png" width="100%"> | <img src="examples/pro-mist-2.png" width="100%"> |
* **Halation Strength:** Simulates the red/orange glow of bright light bouncing off the back of physical film strips. 
  * <img src="examples/Halation.png" width="100%">
* **Chromatic Aberration:** Uses sub-pixel sampling to create smooth, infinite-resolution color fringing on the edges of the frame. Keep low (`0.001 - 0.003`) for subtle realism.
  * <img src="examples/chromatic-circular.png" width="100%">
  * <img src="examples/chromatic-circular2.png" width="100%">
* **Vignette Intensity:** Darkens the corners to mimic a physical lens barrel, subtly guiding the eye to the center.

### 🎞️ Film Emulation
* **Grain Power:** Adds analog texture. **Crucially, this is Depth-Aware.** The grain is sharp on the focused subject but gets softer in the blurred background, perfectly matching real-world lens behavior.
  * <img src="examples/grain.png" width="100%">
* **Monochrome Grain:** `True` acts like classic Film Stock (luminance noise). `False` acts like a Digital Sensor (RGB color noise).
  * <img src="examples/mono-grain.png" width="100%">
* **Highlight Roll-off:** Prevents AI's tendency to instantly clip bright lights to harsh white. Adds a logarithmic "shoulder" to highlights, compressing them so they look creamy and smooth.

---

## 🛠️ Troubleshooting

**"RuntimeError: The size of tensor a (4) must match..."**
* Your image has an Alpha channel (RGBA) and PyTorch is expecting 3 RGB channels.
* **Fix:** I've included a helper node called **`Remove Alpha (RGBA to RGB)`**. Slot this right before the Optical Realism node.
