#!/usr/bin/env python3
"""
Test centerline computation from saved map
Run this to experiment with different algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing, gaussian_filter
import json

# Load the saved map
print("Loading map...")
grid = np.load('track_map.npy')

# Load metadata
with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']
grid_size = meta['width']

print(f"Map size: {grid_size}x{grid_size}")
print(f"Resolution: {RESOLUTION}m")

# Show the original map
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(grid, cmap='gray', origin='lower')
plt.title('Original Map')
plt.colorbar()

# Algorithm 1: Heavy cleaning + distance transform
print("\n=== Algorithm 1: Heavy Cleaning ===")
obstacle_map = (grid == 100).astype(bool)

# Opening: removes small objects
kernel_large = np.ones((15, 15), dtype=bool)
cleaned = binary_opening(obstacle_map, kernel_large)
cleaned = binary_closing(cleaned, kernel_large)

plt.subplot(1, 3, 2)
plt.imshow(cleaned, cmap='gray', origin='lower')
plt.title('Cleaned Map')

# Distance transform
free_space = ~cleaned
dist_map = distance_transform_edt(free_space)

# Smooth distance map
smoothed_dist = gaussian_filter(dist_map.astype(float), sigma=2)

# Take top 80th percentile (points far from walls)
threshold = np.percentile(smoothed_dist[smoothed_dist > 0], 80)
centerline_mask = smoothed_dist > threshold

cy, cx = np.where(centerline_mask)
print(f"Found {len(cx)} centerline points")

# Convert to world coords
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle to create loop
center_x, center_y = np.mean(wx), np.mean(wy)
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)
wx_sorted = wx[sorted_idx]
wy_sorted = wy[sorted_idx]

# Smooth with moving average
window = 5
wx_smooth = np.convolve(wx_sorted, np.ones(window)/window, mode='valid')
wy_smooth = np.convolve(wy_sorted, np.ones(window)/window, mode='valid')

# Subsample
step = max(1, len(wx_smooth) // 80)
wx_final = wx_smooth[::step]
wy_final = wy_smooth[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx_final, wy_final)]

# Convert back to grid coords for plotting
gx_plot = ((wx_final - grid_origin_x) / RESOLUTION).astype(int)
gy_plot = ((wy_final - grid_origin_y) / RESOLUTION).astype(int)

plt.subplot(1, 3, 3)
plt.imshow(grid, cmap='gray', origin='lower', alpha=0.5)
plt.plot(gx_plot, gy_plot, 'g-', linewidth=2, label='Racing Line')
plt.plot(gx_plot[0], gy_plot[0], 'ro', markersize=10, label='Start')
plt.title(f'Racing Line ({len(waypoints)} waypoints)')
plt.legend()

plt.tight_layout()
plt.show()

# Save waypoints
print(f"\nSaving {len(waypoints)} waypoints to centerline_test.json")
with open('centerline_test.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)

print("\n✅ Done! Check the plot and centerline_test.json")
print("If this looks good, we'll add it back to track_mapper.py")

