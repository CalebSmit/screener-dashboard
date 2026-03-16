"""
Configuration presets for different investment styles and factor tilts.

Presets override the factor_weights section of config.yaml while preserving
all other settings. Each preset represents a different investment philosophy.
"""

# Balanced: Current default weights (equal representation)
PRESET_BALANCED = {
    "name": "Balanced",
    "description": "Balanced exposure across all factors (default)",
    "factor_weights": {
        "valuation": 22,
        "quality": 22,
        "growth": 13,
        "momentum": 13,
        "risk": 10,
        "revisions": 10,
        "size": 5,
        "investment": 5,
    }
}

# Value: Heavy tilt toward valuation and quality metrics
# Rationale: Value investing seeks cheap, high-quality stocks
PRESET_VALUE = {
    "name": "Value",
    "description": "Heavy tilt toward valuation + quality (cheap, durable businesses)",
    "factor_weights": {
        "valuation": 30,
        "quality": 25,
        "growth": 8,
        "momentum": 10,
        "risk": 10,
        "revisions": 8,
        "size": 5,
        "investment": 4,
    }
}

# Growth: Heavy tilt toward growth, momentum, and revisions
# Rationale: Growth investing seeks rapid earnings expansion
PRESET_GROWTH = {
    "name": "Growth",
    "description": "Heavy tilt toward growth + momentum + revisions (expanding earnings)",
    "factor_weights": {
        "valuation": 12,
        "quality": 15,
        "growth": 25,
        "momentum": 20,
        "risk": 10,
        "revisions": 10,
        "size": 4,
        "investment": 4,
    }
}

# Momentum: Heavy tilt toward recent price momentum and revisions
# Rationale: Momentum strategies exploit trending price action and sentiment
PRESET_MOMENTUM = {
    "name": "Momentum",
    "description": "Heavy tilt toward momentum + revisions (trending price & sentiment)",
    "factor_weights": {
        "valuation": 10,
        "quality": 12,
        "growth": 10,
        "momentum": 30,
        "risk": 15,
        "revisions": 13,
        "size": 5,
        "investment": 5,
    }
}

# Registry: maps preset names to preset dicts
PRESETS = {
    "balanced": PRESET_BALANCED,
    "value": PRESET_VALUE,
    "growth": PRESET_GROWTH,
    "momentum": PRESET_MOMENTUM,
}


def get_preset(preset_name: str) -> dict | None:
    """Retrieve a preset by name (case-insensitive).
    
    Args:
        preset_name: Name of the preset (e.g., 'value', 'growth')
    
    Returns:
        Preset dict or None if not found
    """
    return PRESETS.get(preset_name.lower())


def list_presets() -> list[str]:
    """Return list of available preset names."""
    return list(PRESETS.keys())


def apply_preset(cfg: dict, preset_name: str) -> dict:
    """Apply a preset's factor_weights to a config dict.
    
    Args:
        cfg: Configuration dict loaded from config.yaml
        preset_name: Name of the preset to apply
    
    Returns:
        Modified config dict with preset factor_weights applied
    
    Raises:
        ValueError: If preset_name is not found
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {', '.join(list_presets())}")
    
    # Make a copy to avoid modifying original
    cfg_copy = cfg.copy()
    
    # Apply the preset's factor_weights
    cfg_copy["factor_weights"] = preset["factor_weights"].copy()
    
    return cfg_copy


def print_preset_info():
    """Print a formatted table of all available presets."""
    print("\n" + "="*80)
    print("  CONFIGURATION PRESETS")
    print("="*80)
    
    for preset_name in list_presets():
        preset = PRESETS[preset_name]
        print(f"\n  {preset_name.upper()}: {preset['description']}")
        print(f"  {'-'*70}")
        weights = preset["factor_weights"]
        print(f"    {'Factor':<18} {'Weight':>8}")
        for factor, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"      {factor:<16} {weight:>6}%")
    
    print("\n" + "="*80 + "\n")
