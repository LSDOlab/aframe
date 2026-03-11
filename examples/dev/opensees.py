import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# MODEL PARAMETERS
# -----------------------------

NStory = 5      # number of stories
NBayX = 4       # number of bays in X
NBayZ = 3       # number of bays in Z

StoryH = 3.5    # story height (m)
BayX = 6.0      # bay width in X (m)
BayZ = 6.0      # bay width in Z (m)

# -----------------------------
# GENERATE NODES
# -----------------------------

nodes = {}
node_id = 1

for k in range(NStory + 1):
    z = k * StoryH
    for j in range(NBayZ + 1):
        y = j * BayZ
        for i in range(NBayX + 1):
            x = i * BayX
            nodes[node_id] = (x, y, z)
            node_id += 1

print("Total nodes:", len(nodes))

# -----------------------------
# HELPER FUNCTION
# -----------------------------

def node_index(i, j, k):
    """
    Convert grid index to node number
    """
    return k*(NBayX+1)*(NBayZ+1) + j*(NBayX+1) + i + 1


# -----------------------------
# GENERATE ELEMENTS
# -----------------------------

columns = []
beams_x = []
beams_z = []

# Columns
for k in range(NStory):
    for j in range(NBayZ + 1):
        for i in range(NBayX + 1):

            n1 = node_index(i, j, k)
            n2 = node_index(i, j, k+1)

            columns.append((n1, n2))

# Beams in X direction
for k in range(1, NStory+1):
    for j in range(NBayZ + 1):
        for i in range(NBayX):

            n1 = node_index(i, j, k)
            n2 = node_index(i+1, j, k)

            beams_x.append((n1, n2))

# Beams in Z direction
for k in range(1, NStory+1):
    for j in range(NBayZ):
        for i in range(NBayX + 1):

            n1 = node_index(i, j, k)
            n2 = node_index(i, j+1, k)

            beams_z.append((n1, n2))

print("Columns:", len(columns))
print("Beams X:", len(beams_x))
print("Beams Z:", len(beams_z))

# -----------------------------
# PLOT STRUCTURE
# -----------------------------

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
for nid, coord in nodes.items():
    ax.scatter(*coord)

# Plot elements
def draw_elements(elements):
    for n1, n2 in elements:
        x = [nodes[n1][0], nodes[n2][0]]
        y = [nodes[n1][1], nodes[n2][1]]
        z = [nodes[n1][2], nodes[n2][2]]
        ax.plot(x, y, z)

draw_elements(columns)
draw_elements(beams_x)
draw_elements(beams_z)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("OpenSees Example 8 Generic 3D Frame")

ax.set_box_aspect([1,1,1])

plt.show()