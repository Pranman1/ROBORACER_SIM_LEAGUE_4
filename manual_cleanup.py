#!/usr/bin/env python3
"""
Manual Map Cleanup - Load, Clean, Save
"""
import numpy as np
import json
from scipy.ndimage import binary_closing, binary_dilation
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt

print("=" * 60)
print("MANUAL MAP CLEANUP")
print("=" * 60)

# Load
print("\n1️⃣ Loading map...")
grid = np.load('track_map.npy')
with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

grid_orig = grid.copy()

print(f"   Loaded: {grid.shape[0]}x{grid.shape[1]}")
free_before = np.sum(grid == 0)
wall_before = np.sum(grid == 100)
unknown_before = np.sum(grid == -1)
print(f"   Before: Free={free_before} ({100*free_before/grid.size:.1f}%) Walls={wall_before} ({100*wall_before/grid.size:.1f}%)")

# Clean
print("\n2️⃣ Cleaning...")

from scipy.ndimage import binary_closing, binary_dilation, binary_opening, label, binary_erosion, binary_fill_holes

# Step 0: Remove tiny black spots (noise smaller than real walls)
print("   🧹 Removing tiny black noise...")
walls = (grid == 100)
labeled_walls, num_wall_features = label(walls)

# Remove wall blobs smaller than 50 pixels
min_wall_size = 50
for i in range(1, num_wall_features + 1):
    wall_blob_size = np.sum(labeled_walls == i)
    if wall_blob_size < min_wall_size:
        # Convert small black spots to white (free space)
        grid[labeled_walls == i] = 0

removed_count = num_wall_features - np.sum([np.sum(labeled_walls == i) >= min_wall_size for i in range(1, num_wall_features + 1)])
print(f"   ✓ Removed {removed_count} tiny black spots")

# Step 1: Fill gaps in walls AGGRESSIVELY (make them solid and thick)
free_space = (grid == 0)
walls = (grid == 100)

# Close LARGE gaps in walls
walls_closed = binary_closing(walls, np.ones((9, 9)))
# Make walls thicker
walls_thick = binary_dilation(walls_closed, np.ones((4, 4)))
print("   ✓ Filled wall gaps and made thick")

# Step 2: Expand free space into nearby unknown (balanced expansion)
free_expanded = binary_dilation(free_space, np.ones((13, 13)))
free_expanded = free_expanded & ~walls_thick
print("   ✓ Expanded free space")

# Step 3: Keep only LARGEST connected component
labeled, num_features = label(free_expanded)
if num_features > 0:
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_component = sizes.argmax()
    free_expanded = (labeled == largest_component)
    print(f"   ✓ Kept largest track region (removed {num_features-1} isolated areas)")

# Step 4: Fill ALL holes inside free space (unknown areas surrounded by white)
free_filled = binary_fill_holes(free_expanded)
print("   ✓ Filled ALL holes surrounded by white")

# Step 5: Remove overlap with walls
free_final = free_filled & ~walls_thick

# Step 6: Add THIN perimeter around all white areas
free_inner = binary_erosion(free_final, np.ones((2, 2)))
perimeter = free_final & ~free_inner

# Combine walls + perimeter
walls_final = walls_thick | perimeter
print("   ✓ Added thin perimeter walls")

# Final free space (away from walls)
free_final = free_final & ~walls_final

# Apply to grid
grid[:] = -1  # Reset to unknown
grid[free_final] = 0  # Free space (cleaned track)
grid[walls_final] = 100  # Walls (super thick + perimeter)
print("   ✓ Applied cleaned map")

free_after = np.sum(grid == 0)
wall_after = np.sum(grid == 100)
unknown_after = np.sum(grid == -1)

print(f"\n   After: Free={free_after} ({100*free_after/grid.size:.1f}%) Walls={wall_after} ({100*wall_after/grid.size:.1f}%)")
print(f"   Change: Free +{free_after-free_before} Walls +{wall_after-wall_before}")

# Save
print("\n3️⃣ Saving cleaned map...")
np.save('track_map_cleaned.npy', grid)
print("   ✓ Saved to: track_map_cleaned.npy")

# Also save metadata
with open('track_map_cleaned_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("   ✓ Saved metadata: track_map_cleaned_meta.json")

# Visualize
print("\n5️⃣ Visualizing...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Before
ax = axes[0]
display = np.zeros_like(grid_orig, dtype=float)
display[grid_orig == -1] = 0.5
display[grid_orig == 0] = 1.0
display[grid_orig == 100] = 0.0
ax.imshow(display, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.set_title(f"BEFORE Cleanup\nFree: {100*free_before/grid.size:.1f}%", fontsize=14, fontweight='bold')

# After
ax = axes[1]
display = np.zeros_like(grid, dtype=float)
display[grid == -1] = 0.5
display[grid == 0] = 1.0
display[grid == 100] = 0.0
ax.imshow(display, cmap='gray', origin='lower', vmin=0, vmax=1)
ax.set_title(f"AFTER Cleanup\nFree: {100*free_after/grid.size:.1f}%", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('cleanup_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: cleanup_comparison.png")
plt.show()

print("\n✅ DONE!")
print("   - Cleaned map saved as: track_map_cleaned.npy")
print("   - Original map unchanged: track_map.npy")
print("   - Now update Delaunay script to use track_map_cleaned.npy")

