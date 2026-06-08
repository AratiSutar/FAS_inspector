import json
from config.settings import settings

try:
    with open(settings.variant_map_path) as f:
        VARIANT_MAP = json.load(f)
except FileNotFoundError:
    raise RuntimeError(
        f"variant_map.json not found at {settings.variant_map_path}. "
        f"Make sure config/variant_map.json exists."
    )