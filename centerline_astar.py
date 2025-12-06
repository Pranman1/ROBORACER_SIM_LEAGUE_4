#!/usr/bin/env python3
"""
Use A* pathfinding to find optimal path around the track!
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing
import json
from queue import PriorityQueue

print("Loading map...")
grid = np.load('track_map.npy')

with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

RESOLUTION = meta['resolution']
grid_origin_x = meta['origin_x']
grid_origin_y = meta['origin_y']

# Free space only
free_space = (grid == 0)
free_cleaned = binary_closing(free_space, np.ones((5, 5)))

print(f"Free space: {np.sum(free_cleaned)} cells")

# Find the track loop - get points along the outer edge
cy, cx = np.where(free_cleaned)

# Find a start point (leftmost point)
start_idx = np.argmin(cx)
start = (cy[start_idx], cx[start_idx])

# Find a goal point (rightmost point - opposite side of track)
goal_idx = np.argmax(cx)
goal = (cy[goal_idx], cx[goal_idx])

print(f"Start: {start}, Goal: {goal}")

# Simple A* to find path through free space
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, free_map):
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while not frontier.empty():
        _, current = frontier.get()
        
        if current == goal:
            break
        
        # 8-connected neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                neighbor = (current[0] + dy, current[1] + dx)
                
                # Check bounds and if free
                if (0 <= neighbor[0] < free_map.shape[0] and 
                    0 <= neighbor[1] < free_map.shape[1] and
                    free_map[neighbor]):
                    
                    new_cost = cost_so_far[current] + 1
                    
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(goal, neighbor)
                        frontier.put((priority, neighbor))
                        came_from[neighbor] = current
    
    # Reconstruct path
    if goal not in came_from:
        return None
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    
    return path

print("Running A* pathfinding...")
path1 = astar(start, goal, free_cleaned)

if path1 is None:
    print("❌ No path found!")
    exit(1)

print(f"Found path with {len(path1)} points")

# Now find path back from goal to start (complete the loop)
print("Completing the loop...")
path2 = astar(goal, start, free_cleaned)

if path2:
    # Combine paths (remove duplicate at connection point)
    full_path = path1 + path2[1:]
else:
    full_path = path1

print(f"Full loop: {len(full_path)} points")

# Convert to world coordinates
gy, gx = zip(*full_path)
gy, gx = np.array(gy), np.array(gx)

wx = grid_origin_x + gx * RESOLUTION
wy = grid_origin_y + gy * RESOLUTION

# Subsample to 80-100 waypoints
target = 80
if len(wx) > target:
    step = len(wx) // target
    wx = wx[::step]
    wy = wy[::step]
    gx = gx[::step]
    gy = gy[::step]

waypoints = [[float(x), float(y)] for x, y in zip(wx, wy)]

print(f"✅ Final: {len(waypoints)} waypoints")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

ax = axes[0]
ax.imshow(free_cleaned, cmap='gray', origin='lower')
ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
ax.plot(goal[1], goal[0], 'bo', markersize=15, label='Goal')
ax.set_title('Track with Start/Goal')
ax.legend()

ax = axes[1]
ax.imshow(free_cleaned, cmap='gray', origin='lower', alpha=0.7)
ax.plot(gx, gy, 'g-', linewidth=3, label=f'A* Path ({len(waypoints)} pts)')
ax.plot(gx[0], gy[0], 'ro', markersize=10, label='Start')
ax.legend()
ax.set_title('A* Racing Line')

plt.tight_layout()
plt.savefig('centerline_astar.png', dpi=150)
print("\n💾 Saved centerline_astar.png")

with open('centerline_astar.json', 'w') as f:
    json.dump({'waypoints': waypoints, 'num_waypoints': len(waypoints)}, f, indent=2)
print("💾 Saved centerline_astar.json")

plt.show()

print("\n✅ A* guarantees path stays on track!")
print("If this looks good, I'll add it to track_mapper.py")

