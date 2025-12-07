#!/usr/bin/env python3
"""
SIMPLE 3-STEP CLEANUP: Melt → Fill → Re-Wall
"""
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, label

print("=" * 60)
print("SIMPLE 3-STEP CLEANUP")
print("=" * 60)

# Load
print("\n1️⃣ Loading map...")
grid = np.load('track_map.npy')
with open('track_map_meta.json', 'r') as f:
    meta = json.load(f)

grid_orig = grid.copy()
print(f"   Loaded: {grid.shape[0]}x{grid.shape[1]}")

# STEP 1: MELT small black noise (keep only BIG walls)
print("\n2️⃣ STEP 1: Melting small black noise...")
walls = (grid == 100)
labeled_walls, num_walls = label(walls)

min_wall_size = 100  # Keep walls bigger than 100 pixels
for i in range(1, num_walls + 1):
    if np.sum(labeled_walls == i) < min_wall_size:
        # Remove small black spots → make white
        grid[labeled_walls == i] = 0

removed = num_walls - np.sum([np.sum(labeled_walls == i) >= min_wall_size for i in range(1, num_walls + 1)])
print(f"   ✓ Melted {removed} small black spots")

# STEP 2: FILL white areas (expand + fill holes)
print("\n3️⃣ STEP 2: Filling white track...")
free = (grid == 0)

# Expand white to fill gaps
free_expanded = binary_dilation(free, structure=np.ones((15, 15)))
print("   ✓ Expanded white")

# Keep only largest white blob (main track)
labeled_free, num_free = label(free_expanded)
if num_free > 0:
    sizes = np.bincount(labeled_free.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    free_main = (labeled_free == largest)
    print(f"   ✓ Kept largest track (removed {num_free-1} blobs)")
else:
    free_main = free_expanded

# Fill all holes inside track
free_filled = binary_fill_holes(free_main)
print("   ✓ Filled holes")

# STEP 3: RE-ADD big black walls
print("\n4️⃣ STEP 3: Re-adding big walls...")
# Find original BIG walls (not small noise)
big_walls = np.zeros_like(grid, dtype=bool)
for i in range(1, num_walls + 1):
    if np.sum(labeled_walls == i) >= min_wall_size:
        big_walls |= (labeled_walls == i)

# Make walls thick and solid
walls_thick = binary_closing(big_walls, structure=np.ones((7, 7)))
walls_thick = binary_dilation(walls_thick, structure=np.ones((3, 3)))
print("   ✓ Thickened walls")

# Add perimeter around white
free_inner = binary_erosion(free_filled, structure=np.ones((2, 2)))
perimeter = free_filled & ~free_inner
walls_final = walls_thick | perimeter
print("   ✓ Added perimeter")

# Final track = white minus walls
track_final = free_filled & ~walls_final

# Build final grid
new_grid = np.ones_like(grid) * -1  # Unknown
new_grid[track_final] = 0  # Free (white)
new_grid[walls_final] = 100  # Walls (black)

# Save
print("\n💾 Saving...")
np.save('track_map_cleaned.npy', new_grid)
with open('track_map_cleaned_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("   ✓ Saved track_map_cleaned.npy")

# Visualize
print("\n📊 Visualizing...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax = axes[0]
vis_before = np.zeros_like(grid_orig, dtype=float)
vis_before[grid_orig == -1] = 0.5
vis_before[grid_orig == 0] = 1.0
vis_before[grid_orig == 100] = 0.0
ax.imshow(vis_before, cmap='gray', origin='lower')
ax.set_title("BEFORE: Noisy")

ax = axes[1]
vis_after = np.zeros_like(new_grid, dtype=float)
vis_after[new_grid == -1] = 0.5
vis_after[new_grid == 0] = 1.0
vis_after[new_grid == 100] = 0.0
ax.imshow(vis_after, cmap='gray', origin='lower')
ax.set_title("AFTER: Melt→Fill→Wall")

plt.tight_layout()
plt.savefig('cleanup_comparison.png', dpi=150)
print("   ✓ Saved: cleanup_comparison.png")
plt.show()

print("\n✅ DONE! Clean map saved.")
