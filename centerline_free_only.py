#!/usr/bin/env python3
"""
Use ONLY the free space (grid == 0) - the actual driven track
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_closing, binary_opening
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

print(f"Map: {grid.shape} @ {RESOLUTION}m")

# ONLY use free space (0) - the white area in debug map
free_space = (grid == 0)
n_free = np.sum(free_space)
print(f"Free space cells: {n_free}")

if n_free < 100:
    print("❌ Almost no free space! Map might be bad.")
    exit(1)

# Clean up - fill small holes
print("Cleaning free space...")
free_cleaned = binary_closing(free_space, np.ones((5, 5)))

# Distance transform - find centerline of the free space
print("Computing centerline...")
dist_map = distance_transform_edt(free_cleaned)

# Take points far from edges of free space
# Use 60th percentile - conservative
threshold = np.percentile(dist_map[dist_map > 0], 60)
print(f"Distance threshold: {threshold * RESOLUTION:.3f}m")

centerline_mask = dist_map > threshold
cy, cx = np.where(centerline_mask)
print(f"Found {len(cx)} centerline points")

if len(cx) < 10:
    print("❌ Too few points!")
    exit(1)

# Convert to world coordinates
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle to create a loop
center_x = np.mean(wx)
center_y = np.mean(wy)
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)
wx = wx[sorted_idx]
wy = wy[sorted_idx]

# Smooth the path
window = 5
if len(wx) >= window:
    wx = np.convolve(wx, np.ones(window)/window, mode='valid')
    wy = np.convolve(wy, np.ones(window)/window, mode='valid')

# Subsample to 80 points
if len(wx) > 80:
    step = len(wx) // 80
    wx = wx[::step]
    wy = wy[::step]

# Close the loop
waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]
if len(waypoints) > 0:
    waypoints.append(waypoints[0])

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Free space
ax = axes[0]
ax.imshow(free_space, cmap='gray', origin='lower')
ax.set_title(f'Free Space Only ({n_free} cells)')

# Distance map
ax = axes[1]
im = ax.imshow(dist_map, cmap='hot', origin='lower')
ax.set_title('Distance from Edges')
plt.colorbar(im, ax=ax)

# Final line
ax = axes[2]
gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)
ax.imshow(free_space, cmap='gray', origin='lower', alpha=0.7)
ax.plot(gx, gy, 'g-', linewidth=3, label=f'Racing Line ({len(waypoints)} pts)')
ax.plot(gx[0], gy[0], 'ro', markersize=10, label='Start')
ax.legend()
ax.set_title('Racing Line on Track')

plt.tight_layout()
plt.savefig('centerline_free.png', dpi=150)
print("\n💾 Saved centerline_free.png")

# Save waypoints
with open('centerline_final.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline_final.json")

plt.show()

print("\n✅ This should follow the actual track!")
print("If it looks good, I'll add this to track_mapper.py")

