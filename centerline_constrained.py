#!/usr/bin/env python3
"""
Centerline that STAYS on the track - no going outside!
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

print(f"Map: {grid.shape} @ {RESOLUTION}m")

# ONLY free space
free_space = (grid == 0)
n_free = np.sum(free_space)
print(f"Free space: {n_free} cells")

# Clean up
free_cleaned = binary_closing(free_space, np.ones((5, 5)))

# Distance transform
dist_map = distance_transform_edt(free_cleaned)

# Take centerline - 50th percentile (middle of the track)
threshold = np.percentile(dist_map[dist_map > 0], 50)
print(f"Distance threshold: {threshold * RESOLUTION:.3f}m")

centerline_mask = dist_map > threshold
cy, cx = np.where(centerline_mask)
print(f"Found {len(cx)} centerline points")

# Convert to world coords
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle
center_x, center_y = np.mean(wx), np.mean(wy)
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)
wx = wx[sorted_idx]
wy = wy[sorted_idx]

# LIGHT smoothing - don't leave the track!
window = 3  # Smaller window
if len(wx) >= window:
    wx_smooth = []
    wy_smooth = []
    for i in range(len(wx)):
        # Get neighbors (wrap around)
        indices = [(i + j - window//2) % len(wx) for j in range(window)]
        wx_avg = np.mean([wx[idx] for idx in indices])
        wy_avg = np.mean([wy[idx] for idx in indices])
        
        # Check if smoothed point is still in free space
        gx_check = int((wx_avg - grid_origin_x) / RESOLUTION)
        gy_check = int((wy_avg - grid_origin_y) / RESOLUTION)
        
        if 0 <= gx_check < grid.shape[1] and 0 <= gy_check < grid.shape[0]:
            if grid[gy_check, gx_check] == 0:  # Still in free space
                wx_smooth.append(wx_avg)
                wy_smooth.append(wy_avg)
            else:  # Smoothing would go outside - keep original
                wx_smooth.append(wx[i])
                wy_smooth.append(wy[i])
        else:
            wx_smooth.append(wx[i])
            wy_smooth.append(wy[i])
    
    wx = np.array(wx_smooth)
    wy = np.array(wy_smooth)

# Subsample to 80-100 points
target_points = 80
if len(wx) > target_points:
    step = len(wx) // target_points
    wx = wx[::step]
    wy = wy[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]

# Close the loop
if len(waypoints) > 0:
    waypoints.append(waypoints[0])

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.imshow(free_space, cmap='gray', origin='lower')
ax.set_title('Free Space (Track)')

ax = axes[1]
im = ax.imshow(dist_map, cmap='hot', origin='lower')
ax.set_title('Distance from Edges')
plt.colorbar(im, ax=ax)

ax = axes[2]
gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)

# Show free space
ax.imshow(free_space, cmap='gray', origin='lower', alpha=0.7)

# Plot line
ax.plot(gx, gy, 'g-', linewidth=3, label=f'Racing Line ({len(waypoints)} pts)')
ax.plot(gx[0], gy[0], 'ro', markersize=10, label='Start')

# Verify all points are on track
off_track = 0
for i, (x, y) in enumerate(zip(gx, gy)):
    if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        if grid[y, x] != 0:
            off_track += 1
            ax.plot(x, y, 'rx', markersize=8)  # Mark off-track points in red

if off_track > 0:
    print(f"⚠️  WARNING: {off_track} waypoints are off-track!")
else:
    print("✅ All waypoints are ON track!")

ax.legend()
ax.set_title('Racing Line (Constrained to Track)')

plt.tight_layout()
plt.savefig('centerline_constrained.png', dpi=150)
print("\n💾 Saved centerline_constrained.png")

with open('centerline_constrained.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline_constrained.json")

plt.show()

print("\n✅ Line should stay ON the track now!")

