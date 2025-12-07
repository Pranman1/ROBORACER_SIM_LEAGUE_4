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
    if os.path.exists('track_map_cle22ed.npy'):
        print("📂 Loading cleaned map...")
        grid = np.load('track_map_cleaned.npy')
        with open('track_map_cleaned_meta.json', 'r') as f:
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
    """Find 4 checkpoints: TOP, BOTTOM, MIDDLE-LEFT, MIDDLE-RIGHT"""
    print("📍 Finding TOP/BOTTOM/LEFT/RIGHT checkpoints...")
    from scipy.ndimage import maximum_filter, gaussian_filter
    
    # Find centerline/ridge points (high distance values)
    threshold = np.percentile(dist_map[dist_map > 0], 50)
    
    # Smooth distance map
    dist_smooth = gaussian_filter(dist_map, sigma=2)
    
    # Find local maxima (ridge points)
    local_max = maximum_filter(dist_smooth, size=7)
    ridge = (dist_smooth == local_max) & (dist_map > threshold)
    
    ry, rx = np.where(ridge)
    
    if len(rx) < 10:
        print("   ❌ Too few ridge points!")
        return []
    
    print(f"   Found {len(rx)} ridge points")
    
    # Get distances for all ridge points
    dists = np.array([dist_map[ry[i], rx[i]] for i in range(len(rx))])
    
    # Find bounds
    min_y, max_y = np.min(ry), np.max(ry)
    min_x, max_x = np.min(rx), np.max(rx)
    mid_y = (min_y + max_y) / 2
    mid_x = (min_x + max_x) / 2
    
    checkpoints = []
    
    # 1. TOP-MOST (highest Y, widest)
    top_mask = ry > (max_y - (max_y - min_y) * 0.3)  # Top 30%
    if np.any(top_mask):
        top_idx = np.where(top_mask)[0]
        best = top_idx[np.argmax(dists[top_idx])]
        checkpoints.append((ry[best], rx[best]))
        print(f"   ✓ TOP: ({ry[best]},{rx[best]}) width={dists[best]:.1f}")
    
    # 2. BOTTOM-MOST (lowest Y, widest)
    bot_mask = ry < (min_y + (max_y - min_y) * 0.3)  # Bottom 30%
    if np.any(bot_mask):
        bot_idx = np.where(bot_mask)[0]
        best = bot_idx[np.argmax(dists[bot_idx])]
        checkpoints.append((ry[best], rx[best]))
        print(f"   ✓ BOTTOM: ({ry[best]},{rx[best]}) width={dists[best]:.1f}")
    
    # 3. MIDDLE-LEFT (middle Y, leftmost X, widest)
    mid_mask = (ry > min_y + (max_y - min_y) * 0.3) & (ry < max_y - (max_y - min_y) * 0.3)
    left_mask = mid_mask & (rx < mid_x)
    if np.any(left_mask):
        left_idx = np.where(left_mask)[0]
        best = left_idx[np.argmax(dists[left_idx])]
        checkpoints.append((ry[best], rx[best]))
        print(f"   ✓ MID-LEFT: ({ry[best]},{rx[best]}) width={dists[best]:.1f}")
    
    # 4. MIDDLE-RIGHT (middle Y, rightmost X, widest)
    right_mask = mid_mask & (rx > mid_x)
    if np.any(right_mask):
        right_idx = np.where(right_mask)[0]
        best = right_idx[np.argmax(dists[right_idx])]
        checkpoints.append((ry[best], rx[best]))
        print(f"   ✓ MID-RIGHT: ({ry[best]},{rx[best]}) width={dists[best]:.1f}")
    
    print(f"\n   Total checkpoints: {len(checkpoints)}")
    return checkpoints

def route_segments(cost_map, checkpoints):
    print(f"\n🔗 Routing between {len(checkpoints)} checkpoints...")
    
    # Reorder: Go 1 -> 4 -> 3 -> 2 -> 1 (indices: 0 -> 3 -> 2 -> 1 -> 0)
    ordered_checkpoints = [checkpoints[0], checkpoints[3], checkpoints[2], checkpoints[1]]
    print(f"   Route order: CP1 -> CP4 -> CP3 -> CP2 -> CP1")
    
    full_path = []
    num_points = len(ordered_checkpoints)
    
    for i in range(num_points):
        start = ordered_checkpoints[i]
        end = ordered_checkpoints[(i + 1) % num_points]
        
        # Labels for display: [1, 4, 3, 2]
        labels = [1, 4, 3, 2]
        print(f"   Segment CP{labels[i]} -> CP{labels[(i+1)%num_points]}: ({start[0]},{start[1]}) -> ({end[0]},{end[1]})")
        
        try:
            indices, weight = route_through_array(cost_map, start, end, fully_connected=True, geometric=True)
            segment = np.array(indices)
            print(f"      ✅ SUCCESS: {len(segment)} points, cost: {weight:.1f}")
            
            if i > 0: 
                full_path.append(segment[1:])
            else: 
                full_path.append(segment)
                
        except Exception as e:
            print(f"      ❌ FAILED: {e}")
            print(f"      Start cost: {cost_map[start[0], start[1]]:.3f}")
            print(f"      End cost: {cost_map[end[0], end[1]]:.3f}")
            
            # Check if there's ANY valid path
            valid_between = np.sum(cost_map[min(start[0],end[0]):max(start[0],end[0])+1, 
                                             min(start[1],end[1]):max(start[1],end[1])+1] != np.inf)
            print(f"      Valid cells in rectangle: {valid_between}")
            return None
    
    if len(full_path) == 0:
        print("   ❌ No segments succeeded!")
        return None
        
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
    
    plt.figure(figsize=(12,12))
    plt.imshow(clean_grid, cmap='gray', origin='lower')
    
    # Plot the 4 checkpoints FIRST (so we can see them)
    checkpoint_colors = ['red', 'cyan', 'yellow', 'orange']
    checkpoint_names = ['Q1', 'Q2', 'Q3', 'Q4']
    for i, cp in enumerate(checkpoints):
        plt.scatter(cp[1], cp[0], c=checkpoint_colors[i], s=300, marker='*', 
                   edgecolors='black', linewidths=2, zorder=15, 
                   label=f'{checkpoint_names[i]} Checkpoint')
    
    # Plot converting world back to pixels
    px_plot = (final_path[:,0] - ox) / res
    py_plot = (final_path[:,1] - oy) / res
    
    plt.plot(px_plot, py_plot, 'lime', linewidth=3, label='Racing Line', alpha=0.8)
    
    # Plot first few points to show direction
    plt.scatter(px_plot[0], py_plot[0], c='lime', s=200, label='Start', zorder=12, 
               edgecolors='green', linewidths=3)
    plt.scatter(px_plot[50], py_plot[50], c='blue', s=100, label='Midpoint', zorder=11)
    
    plt.legend()
    plt.title("Final Exported Raceline (200 pts)")
    plt.show()

if __name__ == "__main__":
    main()