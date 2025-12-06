#!/usr/bin/env python3
"""Quick map viewer"""
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Load map - prefer cleaned version
if os.path.exists('track_map_cleaned.npy'):
    print("📂 Loading CLEANED map")
    grid = np.load('track_map_cleaned.npy')
    with open('track_map_cleaned_meta.json', 'r') as f:
        meta = json.load(f)
    map_type = "CLEANED"
else:
    print("📂 Loading ORIGINAL map")
    grid = np.load('track_map.npy')
    with open('track_map_meta.json', 'r') as f:
        meta = json.load(f)
    map_type = "ORIGINAL"

# Stats
free = np.sum(grid == 0)
walls = np.sum(grid == 100)
unknown = np.sum(grid == -1)
total = grid.size

print("=" * 50)
print("MAP STATS")
print("=" * 50)
print(f"Size: {grid.shape[0]}x{grid.shape[1]} @ {meta['resolution']}m/cell")
print(f"Free (white):    {free:6d} ({100*free/total:5.1f}%)")
print(f"Walls (black):   {walls:6d} ({100*walls/total:5.1f}%)")
print(f"Unknown (gray):  {unknown:6d} ({100*unknown/total:5.1f}%)")
print("=" * 50)

# Display
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Map colors: -1=gray (unknown), 0=white (free), 100=black (walls)
display = np.zeros_like(grid, dtype=float)
display[grid == -1] = 0.5  # Gray
display[grid == 0] = 1.0   # White
display[grid == 100] = 0.0 # Black

ax.imshow(display, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.set_title(f"{map_type} Track Map\nFree: {100*free/total:.1f}%  |  Walls: {100*walls/total:.1f}%", 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")

plt.tight_layout()
plt.savefig('map_view.png', dpi=150, bbox_inches='tight')
print("💾 Saved: map_view.png")
plt.show()

