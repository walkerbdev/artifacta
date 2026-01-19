"""
Image generation functions for e2e tests
Functions for generating synthetic test images
"""

import os
import tempfile

from PIL import Image


def generate_synthetic_images(count=5, size=(28, 28)):
    """Generate small synthetic images programmatically

    Returns list of (filename, filepath) tuples
    """
    images = []
    for i in range(count):
        # Create gradient pattern image
        img = Image.new("L", size)  # Grayscale
        pixels = []
        for y in range(size[1]):
            for x in range(size[0]):
                # Create different patterns for each image
                if i == 0:  # Horizontal gradient
                    val = int(255 * x / size[0])
                elif i == 1:  # Vertical gradient
                    val = int(255 * y / size[1])
                elif i == 2:  # Diagonal gradient
                    val = int(255 * (x + y) / (size[0] + size[1]))
                elif i == 3:  # Circular pattern
                    cx, cy = size[0] // 2, size[1] // 2
                    dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                    val = int(255 * (1 - min(dist / (size[0] / 2), 1)))
                else:  # Checkerboard
                    val = 255 if (x // 4 + y // 4) % 2 == 0 else 0
                pixels.append(val)
        img.putdata(pixels)

        # Save to temp file
        temp_path = os.path.join(tempfile.gettempdir(), f"sample_{i}.png")
        img.save(temp_path, "PNG")
        images.append((f"sample_{i}.png", temp_path))

    return images
