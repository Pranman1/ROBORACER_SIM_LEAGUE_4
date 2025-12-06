#!/usr/bin/env python3
"""
Delaunay Triangulation Racing Line Generator
Uses wall boundaries to create optimal path through track
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
import json

def load_map():
    """Load the saved map"""
    # Try cleaned version first, fall back to original
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

def delaunay_centerline(grid, meta):
    """Generate centerline using DISTANCE TRANSFORM for midline points"""
    
    from scipy.ndimage import distance_transform_edt
    
    # Get free space
    free = (grid == 0)
    
    # Distance transform - distance to nearest wall
    dist_map = distance_transform_edt(free)
    
    print(f"📍 Distance map computed, max distance: {dist_map.max():.1f} cells")
    
    # Find midline points (far from walls)
    threshold = np.percentile(dist_map[dist_map > 0], 60)  # Top 40%
    midline_mask = dist_map > threshold
    
    my, mx = np.where(midline_mask)
    
    if len(mx) < 10:
        print("❌ Not enough midline points!")
        return None, None, None
    
    print(f"📍 Found {len(mx)} midline points")
    
    # Convert to world coordinates
    res = meta['resolution']
    origin_x = meta['origin_x']
    origin_y = meta['origin_y']
    
    midline_pts = np.column_stack([
        origin_x + mx * res,
        origin_y + my * res
    ])
    
    # Sort by angle to create ordered path
    cx, cy = np.mean(midline_pts[:, 0]), np.mean(midline_pts[:, 1])
    angles = np.arctan2(midline_pts[:, 1] - cy, midline_pts[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    
    path = midline_pts[sorted_idx]
    
    print(f"✅ Sorted {len(path)} midline points by angle")
    
    # Smooth path
    from scipy.ndimage import uniform_filter1d
    path[:, 0] = uniform_filter1d(path[:, 0], size=7, mode='wrap')
    path[:, 1] = uniform_filter1d(path[:, 1], size=7, mode='wrap')
    
    # Subsample to ~80 waypoints
    if len(path) > 80:
        step = len(path) // 80
        path = path[::step]
    
    # Close loop
    path = np.vstack([path, path[0]])
    
    print(f"🎯 Final path: {len(path)} waypoints")
    
    return path, midline_pts, dist_map

def visualize(grid, meta, path, midline_pts, dist_map):
    """Visualize the results"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Distance map with midline points
    ax = axes[0]
    
    # Show distance map
    if dist_map is not None:
        im = ax.imshow(dist_map, cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='Distance to wall (cells)')
    else:
        ax.imshow(grid, cmap='gray', origin='lower')
    
    ax.set_title("Distance Transform + Midline Points", fontsize=14, fontweight='bold')
    ax.axis('equal')
    
    # Plot midline points in grid coords
    if midline_pts is not None:
        res = meta['resolution']
        origin_x = meta['origin_x']
        origin_y = meta['origin_y']
        
        mgx = (midline_pts[:, 0] - origin_x) / res
        mgy = (midline_pts[:, 1] - origin_y) / res
        ax.scatter(mgx, mgy, c='red', s=2, alpha=0.6, label='Midline Points')
    
    # Right: Clean visualization with racing line
    ax = axes[1]
    
    # Show free space
    free = (grid == 0)
    ax.imshow(free, cmap='Greys', origin='lower', alpha=0.3)
    
    # Plot path in grid coords
    if path is not None:
        px_grid = (path[:, 0] - origin_x) / res
        py_grid = (path[:, 1] - origin_y) / res
        ax.plot(px_grid, py_grid, 'g-', linewidth=3, label='Racing Line')
        ax.scatter(px_grid, py_grid, c='lime', s=50, zorder=5, edgecolors='green', linewidths=2)
        
        # Highlight start
        ax.scatter(px_grid[0], py_grid[0], c='red', s=200, marker='*', 
                  zorder=10, label='Start', edgecolors='darkred', linewidths=2)
    
    ax.set_title("Midline Racing Line", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('delaunay_result.png', dpi=150, bbox_inches='tight')
    print("💾 Saved: delaunay_result.png")
    plt.show()

def save_waypoints(path):
    """Save waypoints to JSON"""
    if path is None:
        return
    
    data = {
        'waypoints': path.tolist(),
        'num_waypoints': len(path),
        'method': 'distance_transform_midline'
    }
    
    with open('centerline_delaunay.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved: centerline_delaunay.json ({len(path)} waypoints)")

def main():
    print("=" * 60)
    print("📍 DISTANCE TRANSFORM MIDLINE RACING LINE")
    print("=" * 60)
    
    # Load map
    grid, meta = load_map()
    print(f"📊 Map: {grid.shape[0]}x{grid.shape[1]} @ {meta['resolution']}m/cell")
    
    free = np.sum(grid == 0)
    walls = np.sum(grid == 100)
    print(f"   Free: {100*free/grid.size:.1f}%  |  Walls: {100*walls/grid.size:.1f}%")
    
    # Generate centerline
    path, midline_pts, dist_map = delaunay_centerline(grid, meta)
    
    if path is not None:
        print(f"\n✅ SUCCESS! Generated {len(path)} waypoints")
        save_waypoints(path)
        visualize(grid, meta, path, midline_pts, dist_map)
    else:
        print("\n❌ FAILED to generate path")
        visualize(grid, meta, None, None, None)

if __name__ == '__main__':
    main()
