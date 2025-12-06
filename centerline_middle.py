#!/usr/bin/env python3
"""
SUPER SIMPLE: Find waypoints through the MIDDLE of the white areas
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_closing
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

# Get free space only (the white area you drove)
free_space = (grid == 0)
print(f"Free space: {np.sum(free_space)} cells")

# Clean it up a tiny bit
free_cleaned = binary_closing(free_space, np.ones((3, 3)))

# Distance transform - find middle of the track
dist_map = distance_transform_edt(free_cleaned)

# Use 40th percentile - gets a nice STRIP through the middle
threshold = np.percentile(dist_map[dist_map > 0], 40)
print(f"Distance threshold: {threshold * RESOLUTION:.3f}m")

centerline_mask = dist_map > threshold
cy, cx = np.where(centerline_mask)
print(f"Found {len(cx)} middle points")

# Convert to world
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Find center of track
center_x = np.mean(wx)
center_y = np.mean(wy)

# Sort by angle to make loop
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)
wx = wx[sorted_idx]
wy = wy[sorted_idx]

# Light smooth (NO convolution - keep it simple)
# Just take rolling average manually
window = 3
wx_smooth = []
wy_smooth = []
for i in range(len(wx)):
    indices = [i, (i+1)%len(wx), (i+2)%len(wx)]
    wx_smooth.append(np.mean([wx[j] for j in indices]))
    wy_smooth.append(np.mean([wy[j] for j in indices]))

wx = np.array(wx_smooth)
wy = np.array(wy_smooth)

# Subsample to 80-100 points
target = 80
if len(wx) > target:
    step = len(wx) // target
    wx = wx[::step]
    wy = wy[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]

# Close loop
if len(waypoints) > 0:
    waypoints.append(waypoints[0])

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Free space
ax = axes[0]
ax.imshow(free_space, cmap='gray', origin='lower')
ax.set_title('Your Driven Track (Free Space)')

# Distance map
ax = axes[1]
im = ax.imshow(dist_map, cmap='hot', origin='lower')
plt.colorbar(im, ax=ax)
ax.contour(centerline_mask, colors='cyan', linewidths=2)
ax.set_title(f'Middle Strip (40th percentile)')

# Final line
ax = axes[2]
gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)

ax.imshow(free_space, cmap='gray', origin='lower', alpha=0.7)
ax.plot(gx, gy, 'g-', linewidth=3, label=f'Racing Line ({len(waypoints)} pts)')
ax.scatter(gx, gy, c='lime', s=30, alpha=0.8, zorder=5)
ax.plot(gx[0], gy[0], 'ro', markersize=12, label='Start', zorder=10)
ax.legend()
ax.set_title('Waypoints Through Middle')

plt.tight_layout()
plt.savefig('centerline_middle.png', dpi=150)
print("\n💾 Saved centerline_middle.png")

with open('centerline_middle.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline_middle.json")

plt.show()

print("\n✅ This takes points from the MIDDLE of your driven path!")
print("40th percentile = nice continuous strip through center")

