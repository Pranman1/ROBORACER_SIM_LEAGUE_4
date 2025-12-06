#!/usr/bin/env python3
"""
Centerline with Pathfinding - NO WALL CROSSING!
Uses distance transform + Dijkstra to find valid racing line
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import ConvexHull
import json
from heapq import heappush, heappop

def load_map():
    """Load the cleaned map"""
    import os
    if os.path.exists('track_map_cleaned.npy'):
        print("📂 Loading CLEANED map")
        map_data = np.load('track_map_cleaned.npy')
        with open('track_map_cleaned_meta.json', 'r') as f:
            meta = json.load(f)
    else:
        print("📂 Loading original map")
        map_data = np.load('track_map.npy')
        with open('track_map_meta.json', 'r') as f:
            meta = json.load(f)
    return map_data, meta

def find_midline_waypoints(grid, num_waypoints=80):
    """Find waypoints along the midline (farthest from walls)"""
    
    # Distance transform - distance to nearest wall
    free = (grid == 0)
    dist_map = distance_transform_edt(free)
    
    print(f"📍 Distance map computed, max distance: {dist_map.max():.1f} cells")
    
    # Find centerline points (high distance values)
    threshold = np.percentile(dist_map[dist_map > 0], 60)  # Top 40%
    centerline_mask = dist_map > threshold
    
    cy, cx = np.where(centerline_mask)
    print(f"   Found {len(cx)} centerline candidates")
    
    if len(cx) < 10:
        print("❌ Too few centerline points!")
        return None
    
    # Sort by angle to create ordered waypoints
    center_x, center_y = np.mean(cx), np.mean(cy)
    angles = np.arctan2(cy - center_y, cx - center_x)
    sorted_idx = np.argsort(angles)
    
    cx = cx[sorted_idx]
    cy = cy[sorted_idx]
    
    # Subsample to target number
    if len(cx) > num_waypoints:
        step = len(cx) // num_waypoints
        cx = cx[::step]
        cy = cy[::step]
    
    waypoints = list(zip(cx, cy))
    print(f"✅ Selected {len(waypoints)} midline waypoints")
    
    return waypoints, dist_map

def dijkstra_path(grid, start, goal):
    """Find shortest path on grid that STAYS in free space"""
    
    # Check if start/goal are valid
    if grid[start[1], start[0]] != 0 or grid[goal[1], goal[0]] != 0:
        return None
    
    rows, cols = grid.shape
    visited = set()
    heap = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    
    # 8-connected neighbors
    neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    
    while heap:
        current_cost, current = heappop(heap)
        
        if current == goal:
            # Reconstruct path
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]
        
        if current in visited:
            continue
        
        visited.add(current)
        
        cx, cy = current
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            
            # Check bounds
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            
            # Must be free space
            if grid[ny, nx] != 0:
                continue
            
            neighbor = (nx, ny)
            
            # Cost = distance (diagonal = sqrt(2))
            move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
            new_cost = cost_so_far[current] + move_cost
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + abs(nx - goal[0]) + abs(ny - goal[1])  # A* heuristic
                heappush(heap, (priority, neighbor))
                came_from[neighbor] = current
    
    return None  # No path found

def connect_waypoints_with_pathfinding(grid, waypoints):
    """Connect waypoints using Dijkstra - NO WALL CROSSING!"""
    
    print(f"\n🔗 Connecting {len(waypoints)} waypoints with pathfinding...")
    
    full_path = []
    failed = 0
    
    for i in range(len(waypoints)):
        start = waypoints[i]
        goal = waypoints[(i + 1) % len(waypoints)]  # Wrap around
        
        path_segment = dijkstra_path(grid, start, goal)
        
        if path_segment is None:
            print(f"   ⚠️  Failed to connect waypoint {i} -> {i+1}")
            failed += 1
            # Fallback: just add the start point
            full_path.append(start)
        else:
            # Add all points except the last (to avoid duplicates)
            full_path.extend(path_segment[:-1])
    
    print(f"✅ Path complete! {len(full_path)} points total")
    if failed > 0:
        print(f"⚠️  {failed} segments couldn't be connected")
    
    return full_path

def grid_to_world(path, meta):
    """Convert grid coordinates to world coordinates"""
    res = meta['resolution']
    origin_x = meta['origin_x']
    origin_y = meta['origin_y']
    
    world_path = []
    for (gx, gy) in path:
        wx = origin_x + gx * res
        wy = origin_y + gy * res
        world_path.append([float(wx), float(wy)])
    
    return world_path

def smooth_path(path, window=5):
    """Smooth path with moving average"""
    if len(path) < window:
        return path
    
    path_array = np.array(path)
    smoothed = np.copy(path_array)
    
    for i in range(len(path)):
        start = max(0, i - window // 2)
        end = min(len(path), i + window // 2 + 1)
        smoothed[i] = np.mean(path_array[start:end], axis=0)
    
    return smoothed.tolist()

def visualize(grid, waypoints, full_path, meta):
    """Visualize the results"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Left: Grid with waypoints and path
    ax = axes[0]
    
    # Show free space
    free = (grid == 0)
    ax.imshow(free, cmap='Greys', origin='lower', alpha=0.3)
    
    # Plot waypoints
    if waypoints:
        wx, wy = zip(*waypoints)
        ax.scatter(wx, wy, c='cyan', s=100, zorder=5, label='Midline Waypoints', 
                  edgecolors='blue', linewidths=2, marker='o')
    
    # Plot full path
    if full_path:
        px, py = zip(*full_path)
        ax.plot(px, py, 'g-', linewidth=2, label='Connected Path', alpha=0.8)
        
        # Highlight start
        ax.scatter(px[0], py[0], c='red', s=300, marker='*', 
                  zorder=10, label='Start', edgecolors='darkred', linewidths=2)
    
    ax.set_title("Grid: Waypoints + Dijkstra Path", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    
    # Right: World coordinates (what the car will follow)
    ax = axes[1]
    ax.imshow(free, cmap='Greys', origin='lower', alpha=0.3)
    
    if full_path:
        world_path = grid_to_world(full_path, meta)
        world_array = np.array(world_path)
        
        # Convert back to grid for plotting
        res = meta['resolution']
        origin_x = meta['origin_x']
        origin_y = meta['origin_y']
        
        pgx = (world_array[:, 0] - origin_x) / res
        pgy = (world_array[:, 1] - origin_y) / res
        
        ax.plot(pgx, pgy, 'lime', linewidth=3, label='Final Racing Line')
        ax.scatter(pgx[::10], pgy[::10], c='yellow', s=50, zorder=5, 
                  edgecolors='orange', linewidths=1)
        ax.scatter(pgx[0], pgy[0], c='red', s=300, marker='*', zorder=10, 
                  label='Start', edgecolors='darkred', linewidths=2)
    
    ax.set_title("Final Racing Line (No Wall Crossing!)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('centerline_pathfinding.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: centerline_pathfinding.png")
    plt.show()

def save_waypoints(path, meta):
    """Save waypoints to JSON"""
    if path is None or len(path) == 0:
        return
    
    # Convert to world coordinates
    world_path = grid_to_world(path, meta)
    
    # Smooth
    world_path = smooth_path(world_path, window=5)
    
    # Subsample if too many points
    if len(world_path) > 100:
        step = len(world_path) // 100
        world_path = world_path[::step]
    
    data = {
        'waypoints': world_path,
        'num_waypoints': len(world_path),
        'method': 'distance_transform_dijkstra'
    }
    
    with open('centerline_pathfinding.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved: centerline_pathfinding.json ({len(world_path)} waypoints)")

def main():
    print("=" * 60)
    print("🗺️  CENTERLINE WITH PATHFINDING (NO WALL CROSSING!)")
    print("=" * 60)
    
    # Load map
    grid, meta = load_map()
    print(f"📊 Map: {grid.shape[0]}x{grid.shape[1]} @ {meta['resolution']}m/cell")
    
    free = np.sum(grid == 0)
    walls = np.sum(grid == 100)
    print(f"   Free: {100*free/grid.size:.1f}%  |  Walls: {100*walls/grid.size:.1f}%")
    
    # Find midline waypoints
    print("\n1️⃣ Finding midline waypoints...")
    result = find_midline_waypoints(grid, num_waypoints=50)
    
    if result is None:
        print("❌ Failed to find waypoints")
        return
    
    waypoints, dist_map = result
    
    # Connect with pathfinding
    print("\n2️⃣ Connecting waypoints with Dijkstra...")
    full_path = connect_waypoints_with_pathfinding(grid, waypoints)
    
    if full_path is None or len(full_path) == 0:
        print("❌ Failed to create path")
        return
    
    print(f"\n✅ SUCCESS! Generated {len(full_path)} path points")
    
    # Save
    save_waypoints(full_path, meta)
    
    # Visualize
    visualize(grid, waypoints, full_path, meta)

if __name__ == '__main__':
    main()

