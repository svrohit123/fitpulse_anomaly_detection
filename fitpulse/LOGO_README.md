# FitPulse Logo Assets

This directory contains logo files and generation scripts for the FitPulse Anomaly Detection project.

## Files

### Static Logos
- `logo.svg` - Main SVG logo (200x60)
- `favicon.svg` - Square favicon version (32x32)

### Generated Logos
Run `python logo_generator.py` to generate:
- `logos/logo_small.svg` - Small version (120x36)
- `logos/logo_medium.svg` - Medium version (200x60) 
- `logos/logo_large.svg` - Large version (300x90)
- `logos/logo_small_light.png` - Small light theme PNG
- `logos/logo_small_dark.png` - Small dark theme PNG
- `logos/logo_medium_light.png` - Medium light theme PNG
- `logos/logo_medium_dark.png` - Medium dark theme PNG
- `logos/logo_large_light.png` - Large light theme PNG
- `logos/logo_large_dark.png` - Large dark theme PNG

## Logo Design

The FitPulse logo features:
- **Heart icon** (red) - Represents health and fitness monitoring
- **Pulse line** (green) - Represents data analysis and anomaly detection
- **Modern typography** - Clean, professional font
- **Color scheme**:
  - Primary: Blue (#2563eb)
  - Heart: Red (#ef4444) 
  - Pulse: Green (#16a34a)
  - Text: Dark gray (#1f2937)

## Usage

### In Streamlit App
The logo is automatically included in the sidebar with theme-aware styling.

### In Documentation
Use the SVG versions for web/print documentation as they scale perfectly.

### In Presentations
Use the PNG versions with appropriate size for your slide dimensions.

## Customization

To modify colors or styling, edit the SVG files directly or update the `logo_generator.py` script.

## Requirements

For generating PNG logos:
```bash
pip install Pillow
```

## Brand Guidelines

- Use the heart + pulse combination for brand recognition
- Maintain consistent color palette
- Ensure adequate spacing around the logo
- Use appropriate size for context (minimum 120px width for readability)
