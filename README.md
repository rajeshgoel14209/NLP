import webcolors

def rgb_to_color_name(rgb):
    try:
        # Try to get the exact color name
        color_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        # If exact match not found, get the closest color name
        closest_name = None
        min_distance = float("inf")
        for hex_value, name in webcolors.CSS3_NAMES_TO_HEX.items():
            r, g, b = webcolors.hex_to_rgb(hex_value)
            distance = (rgb[0] - r) ** 2 + (rgb[1] - g) ** 2 + (rgb[2] - b) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        color_name = f"Closest match: {closest_name}"
    return color_name

# Example Usage
rgb_color = (123, 200, 150)
print(f"RGB {rgb_color} is closest to color: {rgb_to_color_name(rgb_color)}")
