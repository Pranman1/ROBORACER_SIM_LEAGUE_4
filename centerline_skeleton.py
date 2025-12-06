#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, uniform_filter1d
from skimage import measure
from skimage.graph import route_through_array
import json
import os
import pandas as pd
import sys

# Increase recursion limit
sys.setrecursionlimit(50000)

def load_map_data():
    if os.path.exists('track_map_clea2ed.npy'):
        print("📂 Loading cleaned map...")
        grid = np.load('track_map_clean2d.npy')
        with open('track_map_cleaned_me2a.json', 'r') as f:
            meta = json.load(f)
    else:
        print("📂 Loading original map...")
        grid = np.load('track_map.npy')
        with open('track_map_meta.json', 'r') as f:
            meta = json.load(f)
    return grid, meta

def clean_map(grid):
    binary_track = (grid == 0).astype(int)
    labeled_array, num_features = measure.label(binary_track, return_num=True, connectivity=2)
    if num_features < 2: return grid
    max_label = 0
    max_size = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled_array == i)
        if size > max_size:
            max_size = size
            max_label = i
    new_grid = np.ones_like(grid) * 100 
    new_grid[labeled_array == max_label] = 0
    return new_grid

def get_costmap(grid):
    binary_track = (grid == 0)
    dist_map = distance_transform_edt(binary_track)
    max_dist = np.max(dist_map)
    cost_map = max_dist - dist_map
    cost_map[grid != 0] = np.inf 
    return cost_map, dist_map

def get_quadrant_checkpoints(dist_map):
    valid_y, valid_x = np.where(dist_map > 0)
    center_y = np.mean(valid_y)
    center_x = np.mean(valid_x)
    h, w = dist_map.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    angles = np.arctan2(y_grid - center_y, x_grid - center_x)
    
    # Order: BottomLeft, BottomRight, TopRight, TopLeft
    sectors = [(-np.pi, -np.pi/2), (-np.pi/2, 0), (0, np.pi/2), (np.pi/2, np.pi)]
    checkpoints = []
    
    for i, (start_ang, end_ang) in enumerate(sectors):
        angle_mask = (angles >= start_ang) & (angles < end_ang)
        valid_mask = angle_mask & (dist_map > 0)
        if np.sum(valid_mask) == 0: continue
        sector_dist = dist_map.copy()
        sector_dist[~valid_mask] = -1
        best_idx = np.argmax(sector_dist)
        checkpoints.append(np.unravel_index(best_idx, dist_map.shape))
        
    return checkpoints

def route_segments(cost_map, checkpoints):
    full_path = []
    num_points = len(checkpoints)
    for i in range(num_points):
        start = checkpoints[i]
        end = checkpoints[(i + 1) % num_points]
        try:
            indices, weight = route_through_array(cost_map, start, end, fully_connected=True, geometric=True)
            segment = np.array(indices)
            if i > 0: full_path.append(segment[1:])
            else: full_path.append(segment)
        except: return None
    return np.vstack(full_path)

def resample_path(path, num_points=200):
    """
    Interpolates the path to have exactly `num_points` evenly spaced.
    """
    # Calculate cumulative distance
    dists = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cumulative_dist = np.r_[0, np.cumsum(dists)]
    total_dist = cumulative_dist[-1]
    
    # Create new uniform distances
    new_dists = np.linspace(0, total_dist, num_points)
    
    # Interpolate x and y
    x_new = np.interp(new_dists, cumulative_dist, path[:,0])
    y_new = np.interp(new_dists, cumulative_dist, path[:,1])
    
    return np.column_stack((x_new, y_new))

def main():
    # 1. Generate Raw Loop
    grid, meta = load_map_data()
    clean_grid = clean_map(grid)
    cost_map, dist_map = get_costmap(clean_grid)
    checkpoints = get_quadrant_checkpoints(dist_map) # Green, Cyan, Yellow, Orange
    path_pixels = route_segments(cost_map, checkpoints)

    # 2. Convert to World Coords
    res = meta['resolution']
    ox = meta['origin_x']
    oy = meta['origin_y']
    
    py = path_pixels[:, 0]
    px = path_pixels[:, 1]
    wx = (px * res) + ox
    wy = (py * res) + oy
    
    # 3. Smooth
    window = 20
    wx = uniform_filter1d(wx, size=window, mode='wrap')
    wy = uniform_filter1d(wy, size=window, mode='wrap')
    path_world = np.column_stack((wx, wy))

    # --- CUSTOM ORDERING LOGIC ---
    print("\n🔄 Reordering Path: Yellow -> Blue...")
    
    # Anchor 3 is Yellow (Index 2 in checkpoints)
    # Anchor 2 is Blue/Cyan (Index 1 in checkpoints)
    # Convert Anchor 3 to world coords to find it on the line
    y_anch, x_anch = checkpoints[2] 
    yellow_world = np.array([(x_anch * res) + ox, (y_anch * res) + oy])
    
    # Find index closest to Yellow
    dists = np.linalg.norm(path_world - yellow_world, axis=1)
    start_idx = np.argmin(dists)
    
    # Roll array so Yellow is at 0
    path_rolled = np.roll(path_world, -start_idx, axis=0)
    
    # Check Direction: We want to go to Blue (Checkpoints[1])
    # Currently loop is 1->2->3->4. 
    # Rolled is 3->4->1->2. (Yellow -> Orange -> Green -> Blue)
    # User wants Yellow -> Blue. This is REVERSE of generated order.
    
    print("   Detected Direction: Needs Reversal")
    # Reverse then Roll again to be safe
    path_reversed = path_world[::-1]
    dists_rev = np.linalg.norm(path_reversed - yellow_world, axis=1)
    start_idx_rev = np.argmin(dists_rev)
    final_path_raw = np.roll(path_reversed, -start_idx_rev, axis=0)
    
    # 4. Resample to 200 points
    print(f"📏 Resampling from {len(final_path_raw)} to 200 points...")
    final_path = resample_path(final_path_raw, num_points=200)

    # 5. Output Array
    print("\n" + "="*40)
    print("📋 PYTHON ARRAY (Copy this into your bot)")
    print("="*40)
    print("WAYPOINTS = np.array([")
    for x, y in final_path:
        print(f"    [{x:.3f}, {y:.3f}],")
    print("])")
    print("="*40 + "\n")

    # 6. Save & Plot
    df = pd.DataFrame({'x': final_path[:,0], 'y': final_path[:,1], 'v': 2.0})
    df.to_csv('raceline.csv', index=False)
    
    plt.figure(figsize=(10,10))
    plt.imshow(clean_grid, cmap='gray', origin='lower')
    # Plot converting world back to pixels
    px_plot = (final_path[:,0] - ox) / res
    py_plot = (final_path[:,1] - oy) / res
    
    plt.plot(px_plot, py_plot, 'r-', linewidth=2)
    # Plot first few points to show direction
    plt.scatter(px_plot[0], py_plot[0], c='yellow', s=150, label='Start (Yellow)', zorder=10)
    plt.scatter(px_plot[10], py_plot[10], c='orange', s=50, label='Direction', zorder=10)
    plt.scatter(px_plot[50], py_plot[50], c='cyan', s=50, label='Towards Blue', zorder=10)
    
    plt.legend()
    plt.title("Final Exported Raceline (200 pts)")
    plt.show()

if __name__ == "__main__":
    main()