#!/usr/bin/env python3
"""
DEBUG: Check what the map actually looks like
"""
import numpy as np
import matplotlib.pyplot as plt
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

print(f"\nMap info:")
print(f"  Shape: {grid.shape}")
print(f"  Resolution: {meta['resolution']}m")
print(f"  Origin: ({meta['origin_x']}, {meta['origin_y']})")

# Count values
unique, counts = np.unique(grid, return_counts=True)
print(f"\nGrid values:")
for val, count in zip(unique, counts):
    pct = 100 * count / grid.size
    if val == -1:
        print(f"  {val} (unknown): {count} cells ({pct:.1f}%)")
    elif val == 0:
        print(f"  {val} (free):    {count} cells ({pct:.1f}%)")
    elif val == 100:
        print(f"  {val} (occupied): {count} cells ({pct:.1f}%)")

# Show the map with proper interpretation
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Raw map
ax = axes[0]
ax.imshow(grid, cmap='gray', origin='lower', vmin=-1, vmax=100)
ax.set_title('Raw Map\n(dark=unknown, gray=free, white=walls)')
ax.set_xlabel('X (grid)')
ax.set_ylabel('Y (grid)')

# Free space only
ax = axes[1]
free_map = (grid == 0).astype(float)
ax.imshow(free_map, cmap='gray', origin='lower')
ax.set_title('Free Space Only\n(white=driveable, black=not)')
ax.set_xlabel('X (grid)')

# Obstacles only
ax = axes[2]
obstacle_map = (grid == 100).astype(float)
ax.imshow(obstacle_map, cmap='gray', origin='lower')
ax.set_title('Obstacles Only\n(white=walls, black=not)')
ax.set_xlabel('X (grid)')

plt.tight_layout()
plt.savefig('map_debug.png', dpi=150)
print("\n✅ Saved map_debug.png")
plt.show()

print("\n🔍 DIAGNOSIS:")
print("Look at the plots:")
print("1. Raw map should show the track")
print("2. Free space (white) should be the driveable area")
print("3. Obstacles (white) should be the walls")
print("\nIf free space is mostly EMPTY, that's the problem!")
print("The centerline needs LOTS of free space to work.")

