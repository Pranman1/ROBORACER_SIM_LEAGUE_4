#!/usr/bin/env python3
"""
Extract track boundaries (edges) and find centerline between them
Like the cone-based approach but using map boundaries
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

# Get free space
free_space = (grid == 0)
print(f"Free space: {np.sum(free_space)} cells")

# Find the BOUNDARIES of the free space (track edges)
# Erode free space to find inner boundary
inner = binary_erosion(free_space, np.ones((3, 3)))

# The boundary is: free space but NOT inner
boundary = free_space & ~inner

# Get boundary points
by, bx = np.where(boundary)

print(f"Boundary points: {len(bx)}")

# Convert to world coordinates
wx_boundary = grid_origin_x + bx * RESOLUTION
wy_boundary = grid_origin_y + by * RESOLUTION

# Find center of free space
cy, cx = np.where(free_space)
center_x = np.mean(grid_origin_x + cx * RESOLUTION)
center_y = np.mean(grid_origin_y + cy * RESOLUTION)

print(f"Track center: ({center_x:.2f}, {center_y:.2f})")

# For each boundary point, calculate angle from center
angles = np.arctan2(wy_boundary - center_y, wx_boundary - center_x)

# Sort boundary by angle
sorted_idx = np.argsort(angles)
wx_boundary = wx_boundary[sorted_idx]
wy_boundary = wy_boundary[sorted_idx]
angles = angles[sorted_idx]

# Find centerline: average position at each angle
# Group boundary points by angle (left and right walls)
angle_bins = np.linspace(-np.pi, np.pi, 200)
wx_center = []
wy_center = []

for i in range(len(angle_bins) - 1):
    # Find points in this angle range
    mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
    if np.sum(mask) > 0:
        # Average all boundary points at this angle (finds middle)
        wx_center.append(np.mean(wx_boundary[mask]))
        wy_center.append(np.mean(wy_boundary[mask]))

wx_center = np.array(wx_center)
wy_center = np.array(wy_center)

print(f"Centerline points: {len(wx_center)}")

# Smooth slightly
window = 3
if len(wx_center) >= window:
    wx_smooth = np.convolve(wx_center, np.ones(window)/window, mode='valid')
    wy_smooth = np.convolve(wy_center, np.ones(window)/window, mode='valid')
else:
    wx_smooth = wx_center
    wy_smooth = wy_center

# Subsample to 80 waypoints
target = 80
if len(wx_smooth) > target:
    step = len(wx_smooth) // target
    wx_smooth = wx_smooth[::step]
    wy_smooth = wy_smooth[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx_smooth, wy_smooth)]

# Close loop
if len(waypoints) > 0:
    waypoints.append(waypoints[0])

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Show boundary detection
ax = axes[0]
ax.imshow(free_space, cmap='gray', origin='lower', alpha=0.5)
ax.scatter(bx, by, c='red', s=1, alpha=0.5, label='Boundaries')
ax.set_title(f'Track Boundaries ({len(bx)} points)')
ax.legend()

# Show centerline
ax = axes[1]
ax.imshow(free_space, cmap='gray', origin='lower', alpha=0.5)

# Convert back to grid for plotting
gx_center = ((wx_smooth - grid_origin_x) / RESOLUTION).astype(int)
gy_center = ((wy_smooth - grid_origin_y) / RESOLUTION).astype(int)

ax.plot(gx_center, gy_center, 'g-', linewidth=3, label=f'Centerline ({len(waypoints)} pts)')
ax.plot(gx_center[0], gy_center[0], 'ro', markersize=10, label='Start')
ax.legend()
ax.set_title('Centerline Between Boundaries')

plt.tight_layout()
plt.savefig('centerline_boundaries.png', dpi=150)
print("\n💾 Saved centerline_boundaries.png")

# Save
with open('centerline.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline.json")

plt.show()

print("\n✅ Centerline computed from track boundaries!")
print("This finds the middle between the track edges - like cone-based approaches!")

