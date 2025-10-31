"""
Logo Generator for FitPulse
Creates various logo formats and sizes for the project.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import io

def create_logo_svg(size="medium"):
    """Create SVG logo with different sizes"""
    sizes = {
        "small": (120, 36),
        "medium": (200, 60), 
        "large": (300, 90)
    }
    
    width, height = sizes.get(size, sizes["medium"])
    scale = width / 200
    
    svg_content = f'''<svg width="{width}" height="{height}" viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
  <!-- Background circle -->
  <circle cx="30" cy="30" r="25" fill="#2563eb" opacity="0.1"/>
  
  <!-- Heart icon -->
  <path d="M30 45 C30 45, 15 35, 15 25 C15 20, 18 17, 22 17 C25 17, 28 20, 30 25 C32 20, 35 17, 38 17 C42 17, 45 20, 45 25 C45 35, 30 45, 30 45 Z" 
        fill="#ef4444" stroke="#dc2626" stroke-width="1"/>
  
  <!-- Pulse line -->
  <path d="M50 30 L55 20 L60 40 L65 10 L70 50 L75 30 L80 35 L85 25 L90 45 L95 15 L100 40 L105 20 L110 35 L115 30" 
        stroke="#16a34a" stroke-width="3" fill="none" stroke-linecap="round"/>
  
  <!-- Pulse dots -->
  <circle cx="55" cy="20" r="2" fill="#16a34a"/>
  <circle cx="65" cy="10" r="2" fill="#16a34a"/>
  <circle cx="70" cy="50" r="2" fill="#16a34a"/>
  <circle cx="75" cy="30" r="2" fill="#16a34a"/>
  <circle cx="85" cy="45" r="2" fill="#16a34a"/>
  <circle cx="95" cy="15" r="2" fill="#16a34a"/>
  <circle cx="100" cy="40" r="2" fill="#16a34a"/>
  <circle cx="105" cy="20" r="2" fill="#16a34a"/>
  <circle cx="110" cy="35" r="2" fill="#16a34a"/>
  
  <!-- Text -->
  <text x="125" y="25" font-family="Arial, sans-serif" font-size="{int(24 * scale)}" font-weight="bold" fill="#1f2937">FitPulse</text>
  <text x="125" y="42" font-family="Arial, sans-serif" font-size="{int(12 * scale)}" fill="#6b7280">Anomaly Detection</text>
</svg>'''
    
    return svg_content

def create_logo_png(size="medium", theme="light"):
    """Create PNG logo using PIL"""
    sizes = {
        "small": (120, 36),
        "medium": (200, 60),
        "large": (300, 90)
    }
    
    width, height = sizes.get(size, sizes["medium"])
    
    # Create image
    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Colors
    colors = {
        "light": {
            "bg": "#ffffff",
            "text": "#1f2937", 
            "subtext": "#6b7280",
            "heart": "#ef4444",
            "pulse": "#16a34a"
        },
        "dark": {
            "bg": "#0f1116",
            "text": "#e8eaed",
            "subtext": "#9aa0a6", 
            "heart": "#ef4444",
            "pulse": "#16a34a"
        }
    }
    
    color = colors.get(theme, colors["light"])
    
    # Draw background circle
    draw.ellipse([5, 5, 55, 55], fill=color["bg"], outline="#2563eb", width=2)
    
    # Draw heart (simplified)
    heart_points = [
        (30, 45), (15, 35), (15, 25), (18, 17), (22, 17), (25, 20), (30, 25),
        (35, 20), (38, 17), (42, 17), (45, 25), (45, 35), (30, 45)
    ]
    draw.polygon(heart_points, fill=color["heart"])
    
    # Draw pulse line
    pulse_points = [(50, 30), (55, 20), (60, 40), (65, 10), (70, 50), (75, 30), 
                   (80, 35), (85, 25), (90, 45), (95, 15), (100, 40), (105, 20), 
                   (110, 35), (115, 30)]
    
    for i in range(len(pulse_points) - 1):
        draw.line([pulse_points[i], pulse_points[i+1]], fill=color["pulse"], width=3)
    
    # Draw pulse dots
    for x, y in pulse_points[1:-1]:
        draw.ellipse([x-2, y-2, x+2, y+2], fill=color["pulse"])
    
    # Add text (simplified - would need proper font for production)
    try:
        # Try to use a system font
        font_large = ImageFont.truetype("arial.ttf", int(24 * width / 200))
        font_small = ImageFont.truetype("arial.ttf", int(12 * width / 200))
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw text
    draw.text((125, 15), "FitPulse", fill=color["text"], font=font_large)
    draw.text((125, 40), "Anomaly Detection", fill=color["subtext"], font=font_small)
    
    return img

def generate_all_logos():
    """Generate all logo variations"""
    os.makedirs("logos", exist_ok=True)
    
    # Generate SVG logos
    for size in ["small", "medium", "large"]:
        svg_content = create_logo_svg(size)
        with open(f"logos/logo_{size}.svg", "w") as f:
            f.write(svg_content)
    
    # Generate PNG logos
    for size in ["small", "medium", "large"]:
        for theme in ["light", "dark"]:
            img = create_logo_png(size, theme)
            img.save(f"logos/logo_{size}_{theme}.png")
    
    print("Generated logos:")
    print("- SVG: small, medium, large")
    print("- PNG: small/medium/large × light/dark themes")
    print("Check the 'logos' folder for all files.")

if __name__ == "__main__":
    generate_all_logos()
