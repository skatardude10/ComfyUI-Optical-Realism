import torch
import torch.nn.functional as F
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
                
                # --- ATMOSPHERICS ---
                "atmosphere_enabled": ("BOOLEAN", {"default": True}),
                "haze_strength": ("FLOAT", {"default": 0.80, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lift_blacks": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_offset": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.05}),
                
                # --- OPTICAL PHENOMENA ---
                "light_wrap_strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
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

    def process_image(self, image, depth_map, atmosphere_enabled, haze_strength, lift_blacks, depth_offset, 
                      light_wrap_strength, chromatic_aberration, vignette_intensity, 
                      grain_power, monochrome_grain, highlight_rolloff):
        
        # --- 1. SETUP & SAFETY ---
        device = image.device
        
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        b, h, w, c = image.shape
        
        # Depth Map Alignment
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

        # --- 2. ATMOSPHERIC PASS ---
        if atmosphere_enabled:
            atmos_color = torch.tensor([0.16, 0.19, 0.25], device=device).view(1, 1, 1, 3)
            atmos_mask = torch.pow(depth_mask, 1.5) * haze_strength
            final_image = torch.lerp(final_image, atmos_color, atmos_mask * lift_blacks)
            grayscale = 0.299 * final_image[..., 0] + 0.587 * final_image[..., 1] + 0.114 * final_image[..., 2]
            grayscale = grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
            final_image = torch.lerp(final_image, grayscale, atmos_mask * 0.5)

        # --- 3. LIGHT WRAP ---
        if light_wrap_strength > 0:
            img_permuted = final_image.permute(0, 3, 1, 2)
            bloom = F.avg_pool2d(img_permuted, kernel_size=21, stride=1, padding=10)
            bloom = bloom.permute(0, 2, 3, 1)
            near_mask = 1.0 - depth_mask
            screen_blend = 1 - (1 - final_image) * (1 - (bloom * light_wrap_strength))
            final_image = torch.lerp(final_image, screen_blend, near_mask * 0.5)

        # --- 4. CHROMATIC ABERRATION (SUB-PIXEL SAMPLING) ---
        if chromatic_aberration > 0:
            # We use grid_sample for infinite precision (no pixel snapping)
            
            # Create a normalized grid [-1, 1]
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
            
            # Stack to create (H, W, 2)
            base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
            
            # Calculate Scale Factors
            # Red channel zooms IN slightly (scale grid DOWN)
            # Blue channel zooms OUT slightly (scale grid UP)
            scale_r = 1.0 / (1.0 + chromatic_aberration)
            scale_b = 1.0 / (1.0 - chromatic_aberration)
            
            # Sample Red
            grid_r = base_grid * scale_r
            r_channel = final_image.permute(0, 3, 1, 2)[:, 0:1, :, :] # B, 1, H, W
            r_sampled = F.grid_sample(r_channel, grid_r, mode='bilinear', padding_mode='border', align_corners=False)
            
            # Sample Blue
            grid_b = base_grid * scale_b
            b_channel = final_image.permute(0, 3, 1, 2)[:, 2:3, :, :]
            b_sampled = F.grid_sample(b_channel, grid_b, mode='bilinear', padding_mode='border', align_corners=False)
            
            # Permute back to B, H, W, C
            r_final = r_sampled.permute(0, 2, 3, 1)
            b_final = b_sampled.permute(0, 2, 3, 1)
            
            # Combine (R_new, G_orig, B_new)
            final_image = torch.cat((r_final, final_image[..., 1:2], b_final), dim=-1)

        # --- 5. LENS VIGNETTE ---
        if vignette_intensity > 0:
            y_coords = torch.linspace(-1, 1, h).view(h, 1).to(device)
            x_coords = torch.linspace(-1, 1, w).view(1, w).to(device)
            radius = torch.sqrt(x_coords**2 + y_coords**2)
            vignette_mask = 1.0 - (torch.clamp(radius - 0.4, 0, 1) * vignette_intensity)
            vignette_mask = vignette_mask.unsqueeze(0).unsqueeze(-1)
            final_image = final_image * vignette_mask

        # --- 6. ADAPTIVE GRAIN ---
        if grain_power > 0:
            if monochrome_grain:
                noise_gray = torch.randn((b, h, w, 1), device=device) * grain_power
                noise = noise_gray.repeat(1, 1, 1, 3)
            else:
                noise = torch.randn_like(final_image) * grain_power
            
            luminance = 0.299 * final_image[..., 0] + 0.587 * final_image[..., 1] + 0.114 * final_image[..., 2]
            luminance = luminance.unsqueeze(-1)
            luma_mask = 1.0 - torch.abs(luminance * 2.0 - 1.0)
            depth_grain_mask = 1.0 - (depth_mask * 0.5)
            final_image = final_image + (noise * luma_mask * depth_grain_mask)

        # --- 7. HIGHLIGHT ROLL-OFF ---
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