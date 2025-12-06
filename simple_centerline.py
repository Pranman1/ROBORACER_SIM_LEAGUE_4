#!/usr/bin/env python3
"""
SUPER SIMPLE: Just find the middle of the driveable area
Treats both FREE (0) and UNKNOWN (-1) as driveable
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_opening
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

print(f"Map: {grid.shape} @ {RESOLUTION}m")

# Count what we have
n_free = np.sum(grid == 0)
n_obstacle = np.sum(grid == 100)
n_unknown = np.sum(grid == -1)
print(f"Free: {n_free}, Obstacles: {n_obstacle}, Unknown: {n_unknown}")

# KEY FIX: Treat ONLY obstacles as obstacles, everything else is driveable!
obstacle_map = (grid == 100).astype(bool)

# Clean obstacles - remove noise
print("Cleaning obstacles...")
cleaned_obstacles = binary_opening(obstacle_map, np.ones((10, 10)))

# Driveable area = NOT obstacles
driveable = ~cleaned_obstacles

print(f"Driveable cells: {np.sum(driveable)}")

# Distance transform - find centerline
print("Computing distance transform...")
dist_map = distance_transform_edt(driveable)

# Find pixels that are FAR from walls (centerline)
# Use 70th percentile - not too aggressive
threshold = np.percentile(dist_map[dist_map > 0], 70)
print(f"Distance threshold: {threshold * RESOLUTION:.3f}m")

centerline_mask = dist_map > threshold

cy, cx = np.where(centerline_mask)
print(f"Found {len(cx)} centerline points")

if len(cx) < 10:
    print("❌ TOO FEW POINTS! Map might be bad.")
    exit(1)

# Convert to world coordinates
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle to create loop
center_x = np.mean(wx)
center_y = np.mean(wy)
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)
wx = wx[sorted_idx]
wy = wy[sorted_idx]

# Smooth
window = 5
if len(wx) >= window:
    wx = np.convolve(wx, np.ones(window)/window, mode='valid')
    wy = np.convolve(wy, np.ones(window)/window, mode='valid')

# Subsample to 80 points
if len(wx) > 80:
    step = len(wx) // 80
    wx = wx[::step]
    wy = wy[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.imshow(driveable, cmap='gray', origin='lower')
ax.set_title(f'Driveable Area ({np.sum(driveable)} cells)')

ax = axes[1]
im = ax.imshow(dist_map, cmap='hot', origin='lower')
ax.set_title('Distance from Walls')
plt.colorbar(im, ax=ax)

ax = axes[2]
gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)
ax.imshow(grid, cmap='gray', origin='lower', alpha=0.5, vmin=-1, vmax=100)
ax.plot(gx, gy, 'g-', linewidth=3, label=f'Racing Line ({len(waypoints)} pts)')
ax.plot(gx[0], gy[0], 'ro', markersize=10, label='Start')
ax.legend()
ax.set_title('Final Racing Line')

plt.tight_layout()
plt.savefig('simple_centerline.png', dpi=150)
print("\n💾 Saved simple_centerline.png")

with open('centerline_simple.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline_simple.json")

plt.show()

print("\n✅ If this looks good, I'll add it to track_mapper.py!")

