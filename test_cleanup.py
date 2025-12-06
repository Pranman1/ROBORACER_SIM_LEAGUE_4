#!/usr/bin/env python3
"""
Test the map cleanup on your existing map
Shows before/after
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
import json

print("Loading map...")
grid_original = np.load('track_map.npy')
grid = grid_original.copy()

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

print(f"Original map:")
print(f"  Free: {np.sum(grid == 0)}")
print(f"  Walls: {np.sum(grid == 100)}")
print(f"  Unknown: {np.sum(grid == -1)}")

# Apply cleanup
print("\nApplying cleanup...")

# 1. Fill gaps in walls
obstacle_map = (grid == 100)
kernel = np.ones((5, 5), dtype=bool)
obstacles_filled = binary_closing(obstacle_map, kernel)

# Make walls slightly thicker
obstacles_thick = binary_dilation(obstacles_filled, np.ones((2, 2)))

print("✅ Filled wall gaps")

# 2. Find convex hull
oy, ox = np.where(obstacles_thick)
if len(ox) >= 10:
    points = np.column_stack([ox, oy])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # 3. Fill interior
    grid_y, grid_x = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    hull_path = MplPath(hull_points)
    inside_hull = hull_path.contains_points(grid_points).reshape(grid.shape)
    
    # Fill interior with free space
    grid[inside_hull & ~obstacles_thick] = 0
    
    # Update with thick obstacles
    grid[obstacles_thick] = 100
    
    print("✅ Filled interior")
else:
    print("❌ Not enough obstacles for hull")

print(f"\nCleaned map:")
print(f"  Free: {np.sum(grid == 0)}")
print(f"  Walls: {np.sum(grid == 100)}")
print(f"  Unknown: {np.sum(grid == -1)}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.imshow(grid_original, cmap='gray', origin='lower', vmin=-1, vmax=100)
ax.set_title('Original Map\n(dark=unknown, gray=free, white=walls)')

ax = axes[1]
ax.imshow(grid, cmap='gray', origin='lower', vmin=-1, vmax=100)
ax.set_title('Cleaned Map\n(walls filled, interior marked free)')

plt.tight_layout()
plt.savefig('map_cleanup_comparison.png', dpi=150)
print("\n💾 Saved map_cleanup_comparison.png")
plt.show()

# Save cleaned map
np.save('track_map_cleaned.npy', grid)
print("💾 Saved track_map_cleaned.npy")

print("\n✅ Cleanup complete! Check the comparison image.")
print("If it looks good, the track_mapper.py code will do this automatically!")

