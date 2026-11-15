import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

class OpticalRealism:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                
                # --- CAMERA & LENS GEOMETRY ---
                "lens_distortion": ("FLOAT", {"default": -0.006, "min": -0.5, "max": 0.5, "step": 0.001}),
                
                # --- COLOR & TONE ---
                "color_temperature": ("FLOAT", {"default": 0.02, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": -0.01, "min": -1.0, "max": 1.0, "step": 0.01}),

                # --- ATMOSPHERICS ---
                "atmosphere_enabled": ("BOOLEAN", {"default": True}),
                "haze_strength": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lift_blacks": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_offset": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.01}),
                
                # --- DEPTH OF FIELD (BOKEH) ---
                "f_stop": (["Manual", "f/1.2", "f/1.4", "f/1.8", "f/2.0", "f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11", "f/16", "f/22"], {"default": "Manual"}),
                "dof_intensity": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_auto_focus": ("BOOLEAN", {"default": True}),
                "dof_sharpness_radius": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_focus_point": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),

                # --- OPTICAL PHENOMENA ---
                "light_wrap_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "promist_strength": ("FLOAT", {"default": 0.035, "min": 0.0, "max": 1.0, "step": 0.005}),
                "halation_strength": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chromatic_aberration": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 0.05, "step": 0.001}),
                "vignette_intensity": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # --- FILM EMULATION ---
                "grain_power": ("FLOAT", {"default": 0.009, "min": 0.0, "max": 0.5, "step": 0.001}),
                "monochrome_grain": ("BOOLEAN", {"default": True}), 
                "highlight_rolloff": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "PostProcessing/Physics"

    def process_image(self, image, depth_map, 
                      lens_distortion, color_temperature, tint,
                      atmosphere_enabled, haze_strength, lift_blacks, depth_offset, 
                      f_stop, dof_intensity, dof_auto_focus, dof_sharpness_radius, dof_focus_point,
                      light_wrap_strength, promist_strength, halation_strength, 
                      chromatic_aberration, vignette_intensity, 
                      grain_power, monochrome_grain, highlight_rolloff):
        
        # --- 1. SETUP & SAFETY ---
        device = image.device
        
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        b, h, w, c = image.shape
        
        # Depth Map Extraction & Resizing (Convention: 0.0 = Near/Foreground, 1.0 = Far/Background)
        if depth_map.shape[-1] == 3:
            depth = depth_map[..., 0] 
        else:
            depth = depth_map[..., 0]
        
        if depth.shape[1] != h or depth.shape[2] != w:
            depth = depth.unsqueeze(0).unsqueeze(0)
            depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=False)
            depth = depth.squeeze(0).squeeze(0)
            
        depth_mask = torch.clamp(depth + depth_offset, 0.0, 1.0).unsqueeze(-1)
        final_image = image.clone()

        # --- 2. LENS DISTORTION (Barrel/Pincushion - Bicubic Crispness) ---
        if lens_distortion != 0.0:
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            
            # Distance squared from center
            r2 = grid_x**2 + grid_y**2
            
            # Radial distortion factor
            f = 1.0 + lens_distortion * r2
            
            # Create sampling grid
            grid = torch.stack((grid_x * f, grid_y * f), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            
            # Distort Image using bicubic interpolation so fine detail is not softened
            final_image = F.grid_sample(
                final_image.permute(0, 3, 1, 2), grid,
                mode='bicubic', padding_mode='reflection', align_corners=False
            ).permute(0, 2, 3, 1)
            final_image = torch.clamp(final_image, 0.0, 1.0)
            
            # Distort Depth Map identically so 3D effects still align
            depth_mask = F.grid_sample(
                depth_mask.permute(0, 3, 1, 2), grid,
                mode='bilinear', padding_mode='reflection', align_corners=False
            ).permute(0, 2, 3, 1)

        # --- 3. WHITE BALANCE (Temperature & Tint) ---
        if color_temperature != 0.0 or tint != 0.0:
            r_gain = 1.0 + (color_temperature * 0.15) + (tint * 0.1)
            g_gain = 1.0 - (tint * 0.15)
            b_gain = 1.0 - (color_temperature * 0.15) + (tint * 0.1)
            
            luma_preservation = (0.299 * r_gain + 0.587 * g_gain + 0.114 * b_gain)
            r_gain /= luma_preservation
            g_gain /= luma_preservation
            b_gain /= luma_preservation
            
            gains = torch.tensor([r_gain, g_gain, b_gain], device=device).view(1, 1, 1, 3)
            final_image = torch.clamp(final_image * gains, 0.0, 1.0)

        # --- 4. DEPTH OF FIELD (Linear-Light Circular Bokeh Blur) ---
        if f_stop != "Manual":
            f_stop_map = {
                "f/1.2": {"intensity": 0.80, "radius": 0.05},
                "f/1.4": {"intensity": 0.60, "radius": 0.10},
                "f/1.8": {"intensity": 0.45, "radius": 0.15},
                "f/2.0": {"intensity": 0.40, "radius": 0.20},
                "f/2.8": {"intensity": 0.30, "radius": 0.30},
                "f/4.0": {"intensity": 0.20, "radius": 0.45},
                "f/5.6": {"intensity": 0.10, "radius": 0.60},
                "f/8.0": {"intensity": 0.05, "radius": 0.75},
                "f/11":  {"intensity": 0.02, "radius": 0.85},
                "f/16":  {"intensity": 0.00, "radius": 1.00},
                "f/22":  {"intensity": 0.00, "radius": 1.00},
            }
            dof_intensity = f_stop_map[f_stop]["intensity"]
            dof_sharpness_radius = f_stop_map[f_stop]["radius"]

        if dof_intensity > 0:
            if dof_auto_focus:
                # Focus on the foreground subject in the central 60% ROI
                h_start, h_end = int(h * 0.20), int(h * 0.80)
                w_start, w_end = int(w * 0.20), int(w * 0.80)
                roi = depth_mask[:, h_start:h_end, w_start:w_end, :] 
                roi_flat = roi.reshape(b, -1)
                target_focus = torch.quantile(roi_flat, 0.10, dim=1).view(b, 1, 1, 1)
                # In auto-focus mode, keep foreground up to the focal plane 100% sharp and only blur behind it
                dist_from_focus = torch.clamp(depth_mask - target_focus, min=0.0)
            else:
                target_focus = dof_focus_point
                dist_from_focus = torch.abs(depth_mask - target_focus)

            # Build Circular Disc Kernel (True Bokeh)
            radius = min(25, max(1, int(dof_intensity * 25.0)))
            kernel_size = radius * 2 + 1
            y_k, x_k = torch.meshgrid(
                torch.arange(-radius, radius + 1, device=device),
                torch.arange(-radius, radius + 1, device=device),
                indexing='ij'
            )
            
            mask_k = (x_k**2 + y_k**2 <= radius**2).float()
            kernel = mask_k / mask_k.sum()
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
            
            # Perform bokeh blur in linear light (gamma 2.2) so specular bokeh highlights bloom vividly
            # instead of averaging down into dull grey
            linear_img = torch.pow(torch.clamp(final_image, 1e-5, 1.0), 2.2).permute(0, 3, 1, 2)
            blurred_linear = F.conv2d(linear_img, kernel, padding=radius, groups=3)
            blurred_img = torch.pow(torch.clamp(blurred_linear.permute(0, 2, 3, 1), 1e-5, 1.0), 1.0 / 2.2)
            
            # Smooth Hermite (smoothstep) falloff outside sharpness radius so subject stays 100% razor sharp
            raw_blur = torch.clamp((dist_from_focus - dof_sharpness_radius) * 2.5, 0.0, 1.0)
            blur_mask = raw_blur * raw_blur * (3.0 - 2.0 * raw_blur)
            
            final_image = torch.lerp(final_image, blurred_img, blur_mask)

        # --- 5. ATMOSPHERIC PASS (Chroma-Preserving Aerial Perspective & 3D Depth Separation) ---
        if atmosphere_enabled and (haze_strength > 0 or lift_blacks > 0):
            # Keep foreground & subject planes (depth < 0.50) 100% untouched for maximum 3D pop & deep subject blacks
            far_depth = torch.clamp((depth_mask - 0.50) / 0.50, 0.0, 1.0)
            atmos_mask = torch.pow(far_depth, 1.6)
            
            luma = (0.299 * final_image[..., 0:1] + 0.587 * final_image[..., 1:2] + 0.114 * final_image[..., 2:3])
            # Strictly target dark shadows in the distance so vibrant midground/background colors
            # and highlights never get greyed out or darkened
            shadow_weight = torch.pow(torch.clamp(1.0 - luma * 1.15, 0.0, 1.0), 2.0)
            
            # Cool atmospheric Rayleigh scatter tone for distant shadow depth separation
            atmos_color = torch.tensor([0.58, 0.72, 0.90], device=device).view(1, 1, 1, 3)
            
            # Screen-blend atmospheric scatter into distant shadows only (lifts distant black floor cleanly)
            scatter_amount = atmos_mask * (haze_strength * 0.25 + lift_blacks * 0.85) * shadow_weight
            scatter_light = atmos_color * torch.clamp(scatter_amount, 0.0, 0.55)
            final_image = 1.0 - (1.0 - final_image) * (1.0 - scatter_light)
            final_image = torch.clamp(final_image, 0.0, 1.0)

        # --- 6. LIGHT WRAP (True Depth-Edge Background Bleed) ---
        if light_wrap_strength > 0:
            # Separate distant background emitter (depth > 0.58) from foreground subject receiver (depth < 0.58)
            bg_mask = torch.clamp((depth_mask - 0.58) / 0.20, 0.0, 1.0)
            fg_mask = 1.0 - bg_mask
            
            luma = 0.299 * final_image[..., 0:1] + 0.587 * final_image[..., 1:2] + 0.114 * final_image[..., 2:3]
            # Extract only bright background light to wrap onto foreground silhouette edges
            bright_bg_weight = bg_mask * torch.clamp((luma - 0.50) * 2.0, 0.0, 1.0)
            bright_bg = (final_image * bright_bg_weight).permute(0, 3, 1, 2)
            
            # Smooth Gaussian spill of background light across the occlusion boundary
            bg_bloom = TF.gaussian_blur(bright_bg, kernel_size=21, sigma=6.0).permute(0, 2, 3, 1)
            bg_reach = TF.gaussian_blur(bg_mask.permute(0, 3, 1, 2), kernel_size=21, sigma=6.0).permute(0, 2, 3, 1)
            
            # Mask strictly to the foreground silhouette edge (0 inside foreground interior, 0 in background)
            edge_wrap_zone = torch.clamp(fg_mask * bg_reach * 2.5, 0.0, 1.0)
            
            wrap_light = bg_bloom * light_wrap_strength * edge_wrap_zone
            final_image = 1.0 - (1.0 - final_image) * (1.0 - torch.clamp(wrap_light, 0.0, 1.0))
            final_image = torch.clamp(final_image, 0.0, 1.0)

        # --- 7. PRO-MIST / DIFFUSION FILTER (Specular Highlight Halo Only) ---
        if promist_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            luma = 0.299 * img_permuted[:, 0:1] + 0.587 * img_permuted[:, 1:2] + 0.114 * img_permuted[:, 2:3]
            
            # Isolate true highlights (> 0.65 luma) so midtones remain crisp and un-veiled
            high_mask = torch.pow(torch.clamp((luma - 0.65) / 0.35, 0.0, 1.0), 1.5)
            promist_source = img_permuted * high_mask
            
            pm_bloom = TF.gaussian_blur(promist_source, kernel_size=43, sigma=15.0).permute(0, 2, 3, 1)
            
            # Screen blend highlight bloom so highlights glow softly without clipping or washing out midtones
            final_image = 1.0 - (1.0 - final_image) * (1.0 - torch.clamp(pm_bloom * promist_strength, 0.0, 1.0))
            final_image = torch.clamp(final_image, 0.0, 1.0)

        # --- 8. HALATION (Warm Film Emulsion Edge Glow) ---
        if halation_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            luma = 0.299 * img_permuted[:, 0:1] + 0.587 * img_permuted[:, 1:2] + 0.114 * img_permuted[:, 2:3]
            
            # Target bright speculars (> 0.72 luma)
            hal_mask = torch.pow(torch.clamp((luma - 0.72) / 0.28, 0.0, 1.0), 1.5)
            hal_source = img_permuted * hal_mask
            
            hal_blur = TF.gaussian_blur(hal_source, kernel_size=31, sigma=7.0).permute(0, 2, 3, 1)
            
            # Receive halation primarily on darker/midtone edges adjacent to highlights,
            # preventing flat white surfaces from turning pink/peach
            luma_hwc = luma.permute(0, 2, 3, 1)
            edge_receive = torch.clamp(1.0 - torch.pow(luma_hwc, 1.5) * 0.85, 0.0, 1.0)
            
            red_halation = hal_blur[..., 0:1] * halation_strength * edge_receive
            green_halation = hal_blur[..., 1:2] * (halation_strength * 0.22) * edge_receive
            
            r_ch = final_image[..., 0:1] + red_halation
            g_ch = final_image[..., 1:2] + green_halation
            b_ch = final_image[..., 2:3]
            
            final_image = torch.clamp(torch.cat((r_ch, g_ch, b_ch), dim=-1), 0.0, 1.0)

        # --- 9. CHROMATIC ABERRATION (Bicubic Edge-Only Lateral CA) ---
        if chromatic_aberration > 0:
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            
            # Weight CA radially from center (r^2) so the center subject stays 100% sharp and aberration-free
            r2 = (grid_x**2 + grid_y**2).unsqueeze(0).unsqueeze(-1)
            ca_factor = chromatic_aberration * torch.clamp(r2, 0.0, 1.0)
            
            scale_r = 1.0 / (1.0 + ca_factor)
            scale_b = 1.0 / (1.0 - ca_factor)
            
            grid_r = base_grid * scale_r
            r_channel = final_image.permute(0, 3, 1, 2)[:, 0:1, :, :]
            r_sampled = F.grid_sample(r_channel, grid_r, mode='bicubic', padding_mode='border', align_corners=False)
            
            grid_b = base_grid * scale_b
            b_channel = final_image.permute(0, 3, 1, 2)[:, 2:3, :, :]
            b_sampled = F.grid_sample(b_channel, grid_b, mode='bicubic', padding_mode='border', align_corners=False)
            
            r_final = torch.clamp(r_sampled.permute(0, 2, 3, 1), 0.0, 1.0)
            b_final = torch.clamp(b_sampled.permute(0, 2, 3, 1), 0.0, 1.0)
            final_image = torch.cat((r_final, final_image[..., 1:2], b_final), dim=-1)

        # --- 10. LENS VIGNETTE (Natural Cosine-Fourth Optical Falloff) ---
        if vignette_intensity > 0:
            y_coords = torch.linspace(-1, 1, h, device=device).view(h, 1)
            x_coords = torch.linspace(-1, 1, w, device=device).view(1, w)
            radius = torch.sqrt(x_coords**2 + y_coords**2)
            # Smooth corner falloff starting further out (0.55) so the central subject stays bright and vivid
            corner_ramp = torch.clamp((radius - 0.55) / 0.85, 0.0, 1.0)
            smooth_corner = corner_ramp * corner_ramp * (3.0 - 2.0 * corner_ramp)
            vignette_mask = (1.0 - smooth_corner * vignette_intensity * 0.75).unsqueeze(0).unsqueeze(-1)
            final_image = final_image * vignette_mask

        # --- 11. PHYSICALLY ACCURATE FILM GRAIN (Emulsion Curve & Clump Based) ---
        if grain_power > 0:
            raw_noise = torch.randn_like(final_image)
            raw_noise_permuted = raw_noise.permute(0, 3, 1, 2)
            clumped_noise = TF.gaussian_blur(raw_noise_permuted, kernel_size=3, sigma=0.8)
            clumped_noise = clumped_noise.permute(0, 2, 3, 1) * 1.5 
            
            if monochrome_grain:
                mono = clumped_noise.mean(dim=-1, keepdim=True)
                clumped_noise = mono.repeat(1, 1, 1, 3)
            else:
                mono = clumped_noise.mean(dim=-1, keepdim=True)
                clumped_noise = torch.lerp(mono, clumped_noise, 0.4)
            
            emulsion_curve = final_image * (1.0 - final_image) * 4.0
            emulsion_curve = torch.clamp(emulsion_curve + 0.01, 0.0, 1.0)
            
            final_image = final_image + (clumped_noise * grain_power * emulsion_curve)

        # --- 12. HIGHLIGHT ROLL-OFF (True Highlight-Only Knee Shoulder) ---
        if highlight_rolloff > 0:
            # Only compress above knee = 0.70 so shadows, midtones, and color saturation remain 100% punchy
            knee = 0.70
            headroom = 1.0 - knee
            excess = torch.clamp(final_image - knee, min=0.0)
            compressed_excess = excess / (1.0 + (excess / headroom) * highlight_rolloff * 0.75)
            final_image = torch.where(final_image > knee, knee + compressed_excess, final_image)

        final_image = torch.clamp(final_image, 0.0, 1.0)
        return (final_image,)

class RemoveAlphaChannel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_alpha"
    CATEGORY = "Image/Utils"

    def remove_alpha(self, image):
        if image.shape[-1] == 4:
            return (image[..., :3],)
        return (image,)

NODE_CLASS_MAPPINGS = {
    "OpticalRealism": OpticalRealism,
    "RemoveAlphaChannel": RemoveAlphaChannel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpticalRealism": "Optical Realism & Physics",
    "RemoveAlphaChannel": "Remove Alpha (RGBA to RGB)"
}
