import numpy as np
from PIL import Image

# Load image
img_path = "/Users/khizirsiddiqui/.gemini/antigravity/brain/631ba749-e98e-4b1f-abe2-632d217673e6/uploaded_image_1768382842466.png"
img = Image.open(img_path)
img_arr = np.array(img)

print(f"Image shape: {img_arr.shape}")

# Define some sample regions manually based on the image visual
# Road seems to be the gray main area.
# Grass is green.
# Kerbs are red/white.

# Let's take a central patch for road (assuming car is somewhat central or road is there)
# Looking at the image, road is roughly in the middle-right curving up.
# Let's sample a few points.

# Sample 1: Gray Road
# Approx coordinates (based on 400x600 image visual guess)
# The image uploaded might be the full render or the crop.
# Let's check unique colors to see the palette.
unique_colors, counts = np.unique(img_arr.reshape(-1, 3), axis=0, return_counts=True)
sorted_indices = np.argsort(counts)[::-1]

print("Top 10 most frequent colors (RGB):")
for i in range(min(10, len(unique_colors))):
    idx = sorted_indices[i]
    color = unique_colors[idx]
    count = counts[idx]
    print(f"Color: {color}, Count: {count}")

