from .optical_realism import OpticalRealism, RemoveAlphaChannel

NODE_CLASS_MAPPINGS = {
    "OpticalRealism": OpticalRealism,
    "RemoveAlphaChannel": RemoveAlphaChannel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpticalRealism": "Optical Realism & Physics",
    "RemoveAlphaChannel": "Remove Alpha (RGBA to RGB)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("LOADED: Optical Realism & Alpha Fixer mappings registered.")