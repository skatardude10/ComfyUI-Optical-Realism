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
                "lens_distortion": ("FLOAT", {"default": -0.01, "min": -0.5, "max": 0.5, "step": 0.001}),
                
                # --- COLOR & TONE ---
                "color_temperature": ("FLOAT", {"default": 0.05, "min": -1.0, "max": 1.0, "step": 0.05}),
                "tint": ("FLOAT", {"default": -0.05, "min": -1.0, "max": 1.0, "step": 0.05}),

                # --- ATMOSPHERICS ---
                "atmosphere_enabled": ("BOOLEAN", {"default": True}),
                "haze_strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lift_blacks": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_offset": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.05}),
                
                # --- DEPTH OF FIELD (BOKEH) ---
                "dof_intensity": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_auto_focus": ("BOOLEAN", {"default": True}),
                "dof_sharpness_radius": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dof_focus_point": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),

                # --- OPTICAL PHENOMENA ---
                "light_wrap_strength": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05}),
                "promist_strength": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "halation_strength": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "chromatic_aberration": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 0.05, "step": 0.001}),
                "vignette_intensity": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05}),
                
                # --- FILM EMULATION ---
                "grain_power": ("FLOAT", {"default": 0.012, "min": 0.0, "max": 0.5, "step": 0.001}),
                "monochrome_grain": ("BOOLEAN", {"default": True}), 
                "highlight_rolloff": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "PostProcessing/Physics"

    def process_image(self, image, depth_map, 
                      lens_distortion, color_temperature, tint,
                      atmosphere_enabled, haze_strength, lift_blacks, depth_offset, 
                      dof_intensity, dof_auto_focus, dof_sharpness_radius, dof_focus_point,
                      light_wrap_strength, promist_strength, halation_strength, 
                      chromatic_aberration, vignette_intensity, 
                      grain_power, monochrome_grain, highlight_rolloff):
        
        # --- 1. SETUP & SAFETY ---
        device = image.device
        
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        b, h, w, c = image.shape
        
        # Depth Map Extraction & Resizing
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

        # --- 2. LENS DISTORTION (Barrel/Pincushion) ---
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
            
            # Distort Image (using reflection to fill pulled edges seamlessly)
            final_image = F.grid_sample(final_image.permute(0, 3, 1, 2), grid, mode='bilinear', padding_mode='reflection', align_corners=False).permute(0, 2, 3, 1)
            
            # Distort Depth Map identically so 3D effects still align!
            depth_mask = F.grid_sample(depth_mask.permute(0, 3, 1, 2), grid, mode='bilinear', padding_mode='reflection', align_corners=False).permute(0, 2, 3, 1)

        # --- 3. WHITE BALANCE (Temperature & Tint) ---
        if color_temperature != 0.0 or tint != 0.0:
            # Shift RGB channels based on standard color science weights
            r_gain = 1.0 + (color_temperature * 0.15) + (tint * 0.1)
            g_gain = 1.0 - (tint * 0.15)
            b_gain = 1.0 - (color_temperature * 0.15) + (tint * 0.1)
            
            # Normalize to preserve overall luminance (prevent blowouts)
            luma_preservation = (0.299 * r_gain + 0.587 * g_gain + 0.114 * b_gain)
            r_gain /= luma_preservation
            g_gain /= luma_preservation
            b_gain /= luma_preservation
            
            gains = torch.tensor([r_gain, g_gain, b_gain], device=device).view(1, 1, 1, 3)
            final_image = torch.clamp(final_image * gains, 0.0, 1.0)

        # --- 4. DEPTH OF FIELD (Circular Bokeh Blur) ---
        if dof_intensity > 0:
            if dof_auto_focus:
                # ROI: Center 60% of the frame
                h_start, h_end = int(h * 0.2), int(h * 0.8)
                w_start, w_end = int(w * 0.2), int(w * 0.8)
                roi = depth_mask[:, h_start:h_end, w_start:w_end, :] 
                roi_flat = roi.reshape(b, -1)
                
                # Auto-Focus Target (10th percentile for body/subject priority)
                target_focus = torch.quantile(roi_flat, 0.10, dim=1).view(b, 1, 1, 1)
            else:
                target_focus = dof_focus_point

            # Build Circular Disc Kernel (True Bokeh)
            radius = min(25, max(1, int(dof_intensity * 25.0))) # Cap radius for VRAM safety
            kernel_size = radius * 2 + 1
            y_k, x_k = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing='ij')
            
            # 1 inside circle, 0 outside
            mask_k = (x_k**2 + y_k**2 <= radius**2).float().to(device)
            kernel = mask_k / mask_k.sum() # Normalize 
            
            # Reshape for grouped Convolution: (out_channels, in_channels/groups, kH, kW)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
            
            img_permuted = final_image.permute(0, 3, 1, 2)
            # Apply blur per-channel (groups=3)
            blurred_img = F.conv2d(img_permuted, kernel, padding=radius, groups=3)
            blurred_img = blurred_img.permute(0, 2, 3, 1)
            
            # Distance and Deadzone Blending
            dist_from_focus = torch.abs(depth_mask - target_focus)
            blur_mask = torch.clamp(dist_from_focus - dof_sharpness_radius, 0.0, 1.0)
            blur_mask = torch.clamp(blur_mask * 4.0, 0.0, 1.0) # Sharp falloff ramp
            
            final_image = torch.lerp(final_image, blurred_img, blur_mask)

        # --- 5. ATMOSPHERIC PASS (Haze) ---
        if atmosphere_enabled:
            atmos_color = torch.tensor([0.16, 0.19, 0.25], device=device).view(1, 1, 1, 3)
            atmos_mask = torch.pow(depth_mask, 1.5) * haze_strength
            final_image = torch.lerp(final_image, atmos_color, atmos_mask * lift_blacks)
            grayscale = 0.299 * final_image[..., 0] + 0.587 * final_image[..., 1] + 0.114 * final_image[..., 2]
            grayscale = grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
            final_image = torch.lerp(final_image, grayscale, atmos_mask * 0.5)

        # --- 6. LIGHT WRAP (Depth Bloom) ---
        if light_wrap_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            bloom = F.avg_pool2d(img_permuted, kernel_size=21, stride=1, padding=10)
            bloom = bloom.permute(0, 2, 3, 1)
            near_mask = 1.0 - depth_mask
            screen_blend = 1 - (1 - final_image) * (1 - (bloom * light_wrap_strength))
            final_image = torch.lerp(final_image, screen_blend, near_mask * 0.5)

        # --- 7. PRO-MIST / DIFFUSION FILTER ---
        if promist_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            luma = 0.299 * img_permuted[:, 0:1] + 0.587 * img_permuted[:, 1:2] + 0.114 * img_permuted[:, 2:3]
            
            # Isolate mid-high luminance
            high_mask = torch.clamp((luma - 0.4) * 2.0, 0.0, 1.0)
            promist_source = img_permuted * high_mask
            
            # Massive, soft global blur
            pm_bloom = TF.gaussian_blur(promist_source, kernel_size=43, sigma=15.0)
            pm_bloom = pm_bloom.permute(0, 2, 3, 1)
            
            # Additive mix to lower micro-contrast
            final_image = final_image + (pm_bloom * promist_strength)
            final_image = torch.clamp(final_image, 0.0, 1.0)

        # --- 8. HALATION (Film Glow) ---
        if halation_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            luma = 0.299 * img_permuted[:, 0:1] + 0.587 * img_permuted[:, 1:2] + 0.114 * img_permuted[:, 2:3]
            
            # Isolate extreme highlights (strict threshold)
            hal_mask = torch.clamp((luma - 0.6) * 2.5, 0.0, 1.0)
            hal_source = img_permuted * hal_mask
            
            # Spread highlight scatter
            hal_blur = TF.gaussian_blur(hal_source, kernel_size=31, sigma=8.0)
            hal_blur = hal_blur.permute(0, 2, 3, 1)
            
            # Extract Red Channel scatter and add back safely
            red_halation = hal_blur[..., 0:1] * halation_strength
            r = final_image[..., 0:1] + red_halation
            gb = final_image[..., 1:3]
            
            final_image = torch.cat((r, gb), dim=-1)
            final_image = torch.clamp(final_image, 0.0, 1.0)

        # --- 9. CHROMATIC ABERRATION ---
        if chromatic_aberration > 0:
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            scale_r = 1.0 / (1.0 + chromatic_aberration)
            scale_b = 1.0 / (1.0 - chromatic_aberration)
            
            grid_r = base_grid * scale_r
            r_channel = final_image.permute(0, 3, 1, 2)[:, 0:1, :, :]
            r_sampled = F.grid_sample(r_channel, grid_r, mode='bilinear', padding_mode='border', align_corners=False)
            
            grid_b = base_grid * scale_b
            b_channel = final_image.permute(0, 3, 1, 2)[:, 2:3, :, :]
            b_sampled = F.grid_sample(b_channel, grid_b, mode='bilinear', padding_mode='border', align_corners=False)
            
            r_final = r_sampled.permute(0, 2, 3, 1)
            b_final = b_sampled.permute(0, 2, 3, 1)
            final_image = torch.cat((r_final, final_image[..., 1:2], b_final), dim=-1)

        # --- 10. LENS VIGNETTE ---
        if vignette_intensity > 0:
            y_coords = torch.linspace(-1, 1, h).view(h, 1).to(device)
            x_coords = torch.linspace(-1, 1, w).view(1, w).to(device)
            radius = torch.sqrt(x_coords**2 + y_coords**2)
            vignette_mask = 1.0 - (torch.clamp(radius - 0.4, 0, 1) * vignette_intensity)
            vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(-1)
            final_image = final_image * vignette_mask

        # --- 11. ADAPTIVE GRAIN ---
        if grain_power > 0:
            if monochrome_grain:
                noise_gray = torch.randn((b, h, w, 1), device=device) * grain_power
                noise = noise_gray.repeat(1, 1, 1, 3) * 1.2 
            else:
                noise = torch.randn_like(final_image) * grain_power
            
            luminance = 0.299 * final_image[..., 0] + 0.587 * final_image[..., 1] + 0.114 * final_image[..., 2]
            luminance = luminance.unsqueeze(-1)
            luma_mask = 1.0 - torch.abs(luminance * 2.0 - 1.0)
            
            # Aggressive Depth Falloff
            depth_grain_mask = 1.0 - (depth_mask * 0.85)
            depth_grain_mask = torch.clamp(depth_grain_mask, 0.0, 1.0)
            
            final_image = final_image + (noise * luma_mask * depth_grain_mask)

        # --- 12. HIGHLIGHT ROLL-OFF ---
        if highlight_rolloff > 0:
            final_image = final_image / (1.0 + final_image * highlight_rolloff * 0.5)

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