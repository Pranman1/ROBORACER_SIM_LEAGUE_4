#!/usr/bin/env python3
"""
Test MULTIPLE centerline algorithms
Try different approaches and compare!
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_opening, binary_closing
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
import json

# Load the saved map
print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']
grid_size = meta['width']

print(f"Map: {grid_size}x{grid_size} @ {RESOLUTION}m")

obstacle_map = (grid == 100).astype(bool)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 15))

# ========== Algorithm 1: Distance Ridge ==========
print("\n=== Algorithm 1: Distance Ridge ===")
ax = axes[0, 0]

# Clean
cleaned = binary_opening(obstacle_map, np.ones((15, 15), dtype=bool))
cleaned = binary_closing(cleaned, np.ones((15, 15), dtype=bool))

free_space = ~cleaned
dist_map = distance_transform_edt(free_space)
smoothed_dist = gaussian_filter(dist_map, sigma=3)

# Take ridge (top 85%)
threshold = np.percentile(smoothed_dist[smoothed_dist > 0], 85)
mask = smoothed_dist > threshold

cy, cx = np.where(mask)
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle
angles = np.arctan2(wy - np.mean(wy), wx - np.mean(wx))
sorted_idx = np.argsort(angles)
wx, wy = wx[sorted_idx], wy[sorted_idx]

# Smooth
window = 7
wx = np.convolve(wx, np.ones(window)/window, mode='valid')
wy = np.convolve(wy, np.ones(window)/window, mode='valid')

# Subsample
step = max(1, len(wx) // 80)
wx, wy = wx[::step], wy[::step]

gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)

ax.imshow(grid, cmap='gray', origin='lower', alpha=0.5)
ax.plot(gx, gy, 'g-', linewidth=2)
ax.plot(gx[0], gy[0], 'ro', markersize=10)
ax.set_title(f'Algo 1: Distance Ridge ({len(wx)} pts)')

# ========== Algorithm 2: Thinned Distance ==========
print("\n=== Algorithm 2: Thinned Distance ===")
ax = axes[0, 1]

# More aggressive cleaning
cleaned2 = binary_erosion(obstacle_map, np.ones((20, 20)))
cleaned2 = binary_dilation(cleaned2, np.ones((20, 20)))

free_space2 = ~cleaned2
dist_map2 = distance_transform_edt(free_space2)

# Take only very high distance points
threshold2 = np.percentile(dist_map2[dist_map2 > 0], 90)
mask2 = dist_map2 > threshold2

cy2, cx2 = np.where(mask2)
wx2 = grid_origin_x + cx2 * RESOLUTION
wy2 = grid_origin_y + cy2 * RESOLUTION

angles2 = np.arctan2(wy2 - np.mean(wy2), wx2 - np.mean(wx2))
sorted_idx2 = np.argsort(angles2)
wx2, wy2 = wx2[sorted_idx2], wy2[sorted_idx2]

# Heavy smoothing
window2 = 10
wx2 = np.convolve(wx2, np.ones(window2)/window2, mode='valid')
wy2 = np.convolve(wy2, np.ones(window2)/window2, mode='valid')

step2 = max(1, len(wx2) // 80)
wx2, wy2 = wx2[::step2], wy2[::step2]

gx2 = ((wx2 - grid_origin_x) / RESOLUTION).astype(int)
gy2 = ((wy2 - grid_origin_y) / RESOLUTION).astype(int)

ax.imshow(grid, cmap='gray', origin='lower', alpha=0.5)
ax.plot(gx2, gy2, 'b-', linewidth=2)
ax.plot(gx2[0], gy2[0], 'ro', markersize=10)
ax.set_title(f'Algo 2: Thinned ({len(wx2)} pts)')

# ========== Algorithm 3: Moderate Threshold ==========
print("\n=== Algorithm 3: Moderate Threshold ===")
ax = axes[1, 0]

cleaned3 = binary_opening(obstacle_map, np.ones((10, 10)))
free_space3 = ~cleaned3
dist_map3 = distance_transform_edt(free_space3)

# 75th percentile
threshold3 = np.percentile(dist_map3[dist_map3 > 0], 75)
mask3 = dist_map3 > threshold3

cy3, cx3 = np.where(mask3)
wx3 = grid_origin_x + cx3 * RESOLUTION
wy3 = grid_origin_y + cy3 * RESOLUTION

angles3 = np.arctan2(wy3 - np.mean(wy3), wx3 - np.mean(wx3))
sorted_idx3 = np.argsort(angles3)
wx3, wy3 = wx3[sorted_idx3], wy3[sorted_idx3]

window3 = 5
wx3 = np.convolve(wx3, np.ones(window3)/window3, mode='valid')
wy3 = np.convolve(wy3, np.ones(window3)/window3, mode='valid')

step3 = max(1, len(wx3) // 80)
wx3, wy3 = wx3[::step3], wy3[::step3]

gx3 = ((wx3 - grid_origin_x) / RESOLUTION).astype(int)
gy3 = ((wy3 - grid_origin_y) / RESOLUTION).astype(int)

ax.imshow(grid, cmap='gray', origin='lower', alpha=0.5)
ax.plot(gx3, gy3, 'r-', linewidth=2)
ax.plot(gx3[0], gy3[0], 'ro', markersize=10)
ax.set_title(f'Algo 3: Moderate ({len(wx3)} pts)')

# ========== Algorithm 4: Conservative ==========
print("\n=== Algorithm 4: Conservative ===")
ax = axes[1, 1]

# Light cleaning
cleaned4 = binary_opening(obstacle_map, np.ones((8, 8)))
free_space4 = ~cleaned4
dist_map4 = distance_transform_edt(free_space4)

# 70th percentile
threshold4 = np.percentile(dist_map4[dist_map4 > 0], 70)
mask4 = dist_map4 > threshold4

cy4, cx4 = np.where(mask4)
wx4 = grid_origin_x + cx4 * RESOLUTION
wy4 = grid_origin_y + cy4 * RESOLUTION

angles4 = np.arctan2(wy4 - np.mean(wy4), wx4 - np.mean(wx4))
sorted_idx4 = np.argsort(angles4)
wx4, wy4 = wx4[sorted_idx4], wy4[sorted_idx4]

window4 = 8
wx4 = np.convolve(wx4, np.ones(window4)/window4, mode='valid')
wy4 = np.convolve(wy4, np.ones(window4)/window4, mode='valid')

step4 = max(1, len(wx4) // 80)
wx4, wy4 = wx4[::step4], wy4[::step4]

gx4 = ((wx4 - grid_origin_x) / RESOLUTION).astype(int)
gy4 = ((wy4 - grid_origin_y) / RESOLUTION).astype(int)

ax.imshow(grid, cmap='gray', origin='lower', alpha=0.5)
ax.plot(gx4, gy4, 'c-', linewidth=2)
ax.plot(gx4[0], gy4[0], 'ro', markersize=10)
ax.set_title(f'Algo 4: Conservative ({len(wx4)} pts)')

plt.tight_layout()
plt.savefig('centerline_comparison.png', dpi=150)
print("\n✅ Saved comparison to centerline_comparison.png")
plt.show()

# Save best one (you can change which one)
print("\nWhich looks best?")
print("1 = Green (Algo 1)")
print("2 = Blue (Algo 2)")  
print("3 = Red (Algo 3)")
print("4 = Cyan (Algo 4)")
choice = input("Enter 1-4: ")

if choice == '1':
    waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]
elif choice == '2':
    waypoints = [[float(x), float(y)] for x, y in zip(wx2, wy2)]
elif choice == '3':
    waypoints = [[float(x), float(y)] for x, y in zip(wx3, wy3)]
elif choice == '4':
    waypoints = [[float(x), float(y)] for x, y in zip(wx4, wy4)]
else:
    print("Invalid choice, using Algo 3")
    waypoints = [[float(x), float(y)] for x, y in zip(wx3, wy3)]

with open('centerline_best.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)

print(f"\n💾 Saved {len(waypoints)} waypoints to centerline_best.json")
print("Once you're happy, we'll add this algorithm back to track_mapper.py!")

