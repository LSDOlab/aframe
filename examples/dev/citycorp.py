"""
Citicorp Center (601 Lexington Ave, New York) – 3D Structural Truss Node Plot
==============================================================================
The Citicorp Center (1977, SOM / William LeMessurier) is renowned for its
unique structural system:
  • Four 9-storey mega-column stilts positioned at the midpoint of each
    facade (not at the corners) rising to a transfer level (~35 m).
  • Above the transfer level, four corner columns carry the main frame.
  • Signature large-panel chevron (V-shaped) bracing on every facade,
    spanning multiple floor bands, ties the frame together.
  • A tuned mass damper was added later to control wind sway.

Approximate geometry used (SI, metres):
  Plan:   50 m × 50 m  (≈ 164 ft)
  Height: 279 m        (≈ 915 ft / 59 floors)
  Stilt height: 35 m   (≈ 9 floors × ~3.9 m/floor)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa – registers 3-D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BUILDING GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
W  = 50.0    # plan width / depth  [m]
H  = 279.0   # total height        [m]
hw = W / 2   # half-width = 25 m

Z_BASE     = 0.0
Z_TRANSFER = 35.0   # top of stilt region (≈ floor 9)

# ── Plan coordinates ──────────────────────────────────────────────────────────
#   Stilt mega-columns are at the mid-point of each of the four faces
stilt_plan  = [( hw,   0),   # East
               (-hw,   0),   # West
               (  0,  hw),   # North
               (  0, -hw)]   # South

#   Corner columns form the main structural frame above the transfer level
corner_plan = [( hw,  hw),   # NE
               (-hw,  hw),   # NW
               (-hw, -hw),   # SW
               ( hw, -hw)]   # SE

# ── Height schedules ──────────────────────────────────────────────────────────
z_stilt   = np.linspace(Z_BASE, Z_TRANSFER, 6)    # 5 stilt bands
z_main    = np.linspace(Z_TRANSFER, H, 13)         # 12 main-frame floor bands
z_chevron = np.linspace(Z_TRANSFER, H, 7)          # 6 chevron tier boundaries

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD NODE DICTIONARY  (key = rounded (x,y,z), value = index)
# ─────────────────────────────────────────────────────────────────────────────
node_dict: dict[tuple, int] = {}
nodes: list[tuple] = []

def add_node(x: float, y: float, z: float) -> int:
    key = (round(x, 2), round(y, 2), round(z, 2))
    if key not in node_dict:
        node_dict[key] = len(nodes)
        nodes.append(key)
    return node_dict[key]

# --- Group A: stilt-column nodes ---
stilt_node_ids: list[int] = []
for z in z_stilt:
    for (x, y) in stilt_plan:
        stilt_node_ids.append(add_node(x, y, z))

# --- Group B: corner-column nodes (transfer → roof) ---
corner_node_ids: list[int] = []
for z in z_main:
    for (x, y) in corner_plan:
        corner_node_ids.append(add_node(x, y, z))

# --- Group C: face-midpoint nodes above transfer (mega-column continued) ---
mid_node_ids: list[int] = []
for z in z_main:
    for (x, y) in stilt_plan:
        mid_node_ids.append(add_node(x, y, z))

# --- Group D: corner nodes at chevron tier boundaries ---
for z in z_chevron:
    for (x, y) in corner_plan:
        add_node(x, y, z)

# --- Group E: face-midpoint nodes at chevron tier boundaries ---
for z in z_chevron:
    for (x, y) in stilt_plan:
        add_node(x, y, z)

nodes_arr = np.array(nodes)   # shape (N, 3)
N = len(nodes_arr)
print(f"Total nodes: {N}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUILD MEMBER (EDGE) LIST
# ─────────────────────────────────────────────────────────────────────────────
members: list[tuple[int, int]] = []

def connect(x1, y1, z1, x2, y2, z2):
    k1 = (round(x1, 2), round(y1, 2), round(z1, 2))
    k2 = (round(x2, 2), round(y2, 2), round(z2, 2))
    if k1 in node_dict and k2 in node_dict:
        members.append((node_dict[k1], node_dict[k2]))

# ── A. Stilt verticals ────────────────────────────────────────────────────────
for (x, y) in stilt_plan:
    for i in range(len(z_stilt) - 1):
        connect(x, y, z_stilt[i], x, y, z_stilt[i + 1])

# ── B. Corner-column verticals (transfer → roof) ──────────────────────────────
for (x, y) in corner_plan:
    for i in range(len(z_main) - 1):
        connect(x, y, z_main[i], x, y, z_main[i + 1])

# ── C. Face-midpoint column verticals (transfer → roof) ───────────────────────
for (x, y) in stilt_plan:
    for i in range(len(z_main) - 1):
        connect(x, y, z_main[i], x, y, z_main[i + 1])

# ── D. Perimeter floor beams at each main-frame level ─────────────────────────
for z in z_main:
    for i in range(4):
        x1, y1 = corner_plan[i]
        x2, y2 = corner_plan[(i + 1) % 4]
        connect(x1, y1, z, x2, y2, z)

# ── E. Face beams: corners ↔ face-midpoint at chevron tier levels ──────────────
#    Maps each face-midpoint to the two corners on that face
face_corners = {
    ( 0,  hw): [(-hw,  hw), ( hw,  hw)],  # N-mid → NW, NE
    ( 0, -hw): [(-hw, -hw), ( hw, -hw)],  # S-mid → SW, SE
    ( hw,  0): [( hw,  hw), ( hw, -hw)],  # E-mid → NE, SE
    (-hw,  0): [(-hw,  hw), (-hw, -hw)],  # W-mid → NW, SW
}
for z in z_chevron:
    for (mx, my), cnrs in face_corners.items():
        for (cx, cy) in cnrs:
            connect(mx, my, z, cx, cy, z)

# ── F. Chevron bracing (signature V-braces on each facade) ────────────────────
#    Pattern: from each face-midpoint node at the TOP of a tier, two diagonal
#    members descend to the corresponding face-corner nodes at the BOTTOM.
#    This produces the iconic inverted-V (chevron) silhouette.
n_tiers = len(z_chevron) - 1
for ti in range(n_tiers):
    z_bot = z_chevron[ti]
    z_top = z_chevron[ti + 1]
    for (mx, my), cnrs in face_corners.items():
        for (cx, cy) in cnrs:
            connect(cx, cy, z_bot, mx, my, z_top)

# ── G. Transfer-level diagonals: stilt tops → nearest corners ─────────────────
zt = Z_TRANSFER
for (mx, my), cnrs in face_corners.items():
    for (cx, cy) in cnrs:
        connect(mx, my, zt, cx, cy, zt)

# ── H. Stilt-level cross-bracing (diagonal X-braces in each stilt bay) ────────
#   Pairs of stilts that share a common face get cross-braced
stilt_pairs = [
    ((  0,  hw), (  0, -hw)),   # N–S pair (share East/West elevation)
    (( hw,   0), (-hw,   0)),   # E–W pair (share North/South elevation)
]
for i in range(len(z_stilt) - 1):
    z_lo, z_hi = z_stilt[i], z_stilt[i + 1]
    for (x1, y1), (x2, y2) in stilt_pairs:
        connect(x1, y1, z_lo, x2, y2, z_hi)
        connect(x1, y1, z_hi, x2, y2, z_lo)

print(f"Total members: {len(members)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CATEGORISE NODES FOR COLOUR-CODING
# ─────────────────────────────────────────────────────────────────────────────
stilt_set    = set(stilt_node_ids)
corner_set   = set(corner_node_ids)
mid_set      = set(mid_node_ids)
transfer_ids = {node_dict[(round(x, 2), round(y, 2), round(Z_TRANSFER, 2))]
                for (x, y) in stilt_plan + corner_plan
                if (round(x, 2), round(y, 2), round(Z_TRANSFER, 2)) in node_dict}

node_colors = []
for idx in range(N):
    if idx in stilt_set and nodes_arr[idx, 2] < Z_TRANSFER - 0.01:
        node_colors.append('#E74C3C')   # red   – stilt column
    elif idx in transfer_ids:
        node_colors.append('#F39C12')   # amber – transfer level
    elif idx in corner_set:
        node_colors.append('#2ECC71')   # green – corner column
    else:
        node_colors.append('#3498DB')   # blue  – face-midpoint / chevron apex

node_sizes = [50 if idx in transfer_ids else 20 for idx in range(N)]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FIGURE  –  3D scatter + wireframe
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 20))

# Helper to set up a sub-axis with common aesthetics
def setup_ax(ax, elev, azim, title):
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, color='white', pad=6)
    ax.set_xlabel('X  [m]', fontsize=8, labelpad=4)
    ax.set_ylabel('Y  [m]', fontsize=8, labelpad=4)
    ax.set_zlabel('Z  [m]', fontsize=8, labelpad=4)
    ax.tick_params(labelsize=7)
    ax.set_facecolor('#0d1117')
    # Subtle grid
    ax.xaxis.pane.fill = False;  ax.yaxis.pane.fill = False;  ax.zaxis.pane.fill = False
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

# ── Main large view (isometric) ───────────────────────────────────────────────
ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')

# Draw members with colour-coded line types
chevron_start = len(list(connect.__code__.co_varnames))   # rough separator (unused)
# Separate chevron members vs other members for styling
chevron_indices = set()
mem_arr = np.array(members)

# Mark chevron members (F: those added in section F above)
# We re-identify them: both endpoints have same x,y as a face-midpoint or corner
def is_chevron(i, j):
    xi, yi, zi = nodes_arr[i]
    xj, yj, zj = nodes_arr[j]
    # A chevron member spans two different z-levels and connects a face corner to a face midpoint
    if abs(zi - zj) < 0.1:
        return False
    mid_pts  = {(round(x,2), round(y,2)) for (x, y) in stilt_plan}
    corn_pts = {(round(x,2), round(y,2)) for (x, y) in corner_plan}
    pi = (round(xi, 2), round(yi, 2))
    pj = (round(xj, 2), round(yj, 2))
    return (pi in mid_pts and pj in corn_pts) or (pi in corn_pts and pj in mid_pts)

for idx, (i, j) in enumerate(members):
    xi, yi, zi = nodes_arr[i]
    xj, yj, zj = nodes_arr[j]
    if is_chevron(i, j):
        ax1.plot([xi, xj], [yi, yj], [zi, zj],
                 color='#F39C12', linewidth=1.4, alpha=0.85, zorder=3)
    else:
        ax1.plot([xi, xj], [yi, yj], [zi, zj],
                 color='#5DADE2', linewidth=0.7, alpha=0.55, zorder=2)

# Draw nodes
ax1.scatter(nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2],
            c=node_colors, s=node_sizes, zorder=5, edgecolors='none')

setup_ax(ax1, elev=18, azim=-55, title='Isometric View')

# Annotation leader lines
ax1.text( hw + 3,  0, Z_TRANSFER / 2, 'Stilt\ncolumns', fontsize=7,
          color='#E74C3C', ha='left')
ax1.text(-hw - 3, -hw, Z_TRANSFER + 20, 'Corner\ncolumns', fontsize=7,
          color='#2ECC71', ha='right')
ax1.text( hw + 3,  0, (Z_TRANSFER + H) / 2, 'Chevron\nbrace', fontsize=7,
          color='#F39C12', ha='left')
ax1.text(0, 0, Z_TRANSFER - 3, f'Transfer\nlevel ~{Z_TRANSFER:.0f} m', fontsize=7,
          color='#F39C12', ha='center')

# Title block
fig.text(0.5, 0.97,
         'Citicorp Center  —  601 Lexington Ave, New York\n'
         'Structural Truss: Nodal Positions (3-D)',
         ha='center', va='top', fontsize=14, fontweight='bold',
         color='white', linespacing=1.6)

# ── Small elevation views ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax3 = fig.add_subplot(2, 2, 4, projection='3d')

for ax, elev, azim, title in [
    (ax2,  5,   0, 'North Elevation (Y-Z plane)'),
    (ax3, 85, -90, 'Plan View (X-Y plane)'),
]:
    for i, j in members:
        xi, yi, zi = nodes_arr[i]
        xj, yj, zj = nodes_arr[j]
        color = '#F39C12' if is_chevron(i, j) else '#5DADE2'
        lw    = 1.2       if is_chevron(i, j) else 0.6
        ax.plot([xi, xj], [yi, yj], [zi, zj],
                color=color, linewidth=lw, alpha=0.7)
    ax.scatter(nodes_arr[:, 0], nodes_arr[:, 1], nodes_arr[:, 2],
               c=node_colors, s=12, zorder=5, edgecolors='none')
    setup_ax(ax, elev, azim, title)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color='#E74C3C', label='Stilt mega-column node'),
    mpatches.Patch(color='#F39C12', label='Transfer-level / chevron-apex node'),
    mpatches.Patch(color='#2ECC71', label='Corner-column node'),
    mpatches.Patch(color='#3498DB', label='Frame / floor-beam node'),
    plt.Line2D([0], [0], color='#F39C12', lw=2,  label='Chevron brace member'),
    plt.Line2D([0], [0], color='#5DADE2', lw=0.8, label='Column / floor-beam member'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=8, framealpha=0.3, labelcolor='white',
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
# plt.savefig('/mnt/user-data/outputs/citicorp_truss.png',
#             dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
# print("Plot saved → citicorp_truss.png")
plt.show()