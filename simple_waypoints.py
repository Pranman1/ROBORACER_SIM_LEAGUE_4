#!/usr/bin/env python3
"""
DEAD SIMPLE: Just extract waypoints from the recorded free space
"""
import numpy as np
import matplotlib.pyplot as plt
import json

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

# Get ALL free space points (what we actually drove)
free_space = (grid == 0)
cy, cx = np.where(free_space)

print(f"Found {len(cx)} free space points")

# Convert to world coordinates
wx = grid_origin_x + cx * RESOLUTION
wy = grid_origin_y + cy * RESOLUTION

# Sort by angle from center to create a loop
center_x = np.mean(wx)
center_y = np.mean(wy)
angles = np.arctan2(wy - center_y, wx - center_x)
sorted_idx = np.argsort(angles)

wx = wx[sorted_idx]
wy = wy[sorted_idx]

# Subsample to ~100 waypoints
target = 100
step = max(1, len(wx) // target)
wx = wx[::step]
wy = wy[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]

# Close the loop
if len(waypoints) > 0:
    waypoints.append(waypoints[0])

print(f"✅ Created {len(waypoints)} waypoints")

# Plot
plt.figure(figsize=(12, 10))

# Show the track
plt.imshow(grid, cmap='gray', origin='lower', alpha=0.5, vmin=-1, vmax=100)

# Plot waypoints
gx = ((wx - grid_origin_x) / RESOLUTION).astype(int)
gy = ((wy - grid_origin_y) / RESOLUTION).astype(int)

plt.plot(gx, gy, 'g-', linewidth=3, label=f'Waypoints ({len(waypoints)} pts)')
plt.scatter(gx, gy, c='lime', s=20, alpha=0.6, zorder=5)
plt.plot(gx[0], gy[0], 'ro', markersize=15, label='Start', zorder=10)

plt.title('Simple Waypoints from Recorded Track')
plt.legend()
plt.tight_layout()
plt.savefig('simple_waypoints.png', dpi=150)
print("💾 Saved simple_waypoints.png")

# Save
with open('centerline.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline.json")

plt.show()

print("\n✅ DONE! These are literally the points from your map.")
print("No algorithms, no smoothing, just YOUR recorded path!")

