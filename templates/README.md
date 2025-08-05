# Valorant Template Library

This directory contains template images for detecting UI elements, abilities, weapons, and game states in Valorant gameplay videos.

## Directory Structure

```
templates/
├── agents/          # Agent portraits and indicators
├── weapons/         # Weapon icons and indicators  
├── abilities/       # Ability icons and cooldowns
├── ui_elements/     # HUD elements, buttons, indicators
```

## Adding Templates

1. **Take Screenshots**: Capture high-quality screenshots during gameplay
2. **Crop Precisely**: Crop to exact UI element (no extra background)
3. **Save as PNG**: Use PNG format for best quality
4. **Organize by Category**: Place in the appropriate subdirectory
5. **Use Descriptive Names**: e.g., `phantom_icon.png`, `jett_ability_q.png`

## Template Guidelines

- **Resolution**: Native game resolution preferred (1920x1080)
- **Quality**: Sharp, clear images without compression artifacts
- **Consistency**: Similar lighting and UI state across templates
- **Size**: Crop tightly to the specific element

## Usage

Templates are automatically loaded by the hybrid detection system. No manual configuration required.

Run `python hybrid_main.py --setup_templates` to ensure directory structure is created.
