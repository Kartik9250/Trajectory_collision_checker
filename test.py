import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# PARAMETERS
# -----------------------------
GRID_SIZE = 50

# USER-DEFINED WAYPOINTS (ANY NUMBER)
waypoints = [
    (5, 5),
    (18, 1),
    (10, 30),
    (45, 40)
]

# -----------------------------
# GRID & OBSTACLES
# -----------------------------
grid = np.zeros((GRID_SIZE, GRID_SIZE))

def mark_line(grid, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    n = max(abs(x2 - x1), abs(y2 - y1))
    for i in range(n + 1):
        x = int(round(x1 + (x2 - x1) * i / n))
        y = int(round(y1 + (y2 - y1) * i / n))
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid[y, x] = 1

# Predefined obstacle lines
mark_line(grid, (10, 10), (40, 10))
mark_line(grid, (25, 15), (25, 40))

# -----------------------------
# GENERATE STRAIGHT PATH THROUGH WAYPOINTS
# -----------------------------
def straight_segment(p1, p2, n=100):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return list(zip(xs, ys))

full_path = []
for i in range(len(waypoints) - 1):
    seg = straight_segment(waypoints[i], waypoints[i + 1])
    if i > 0:
        seg = seg[1:]
    full_path.extend(seg)

# Nested list output (as requested)
path_points = [(round(x, 2), round(y, 2)) for x, y in full_path]
print("Generated path:")
print(path_points)

# -----------------------------
# COLLISION DETECTION
# -----------------------------
collision_flags = []
collision_points = []

for x, y in full_path:
    gx, gy = int(round(x)), int(round(y))
    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and grid[gy, gx] == 1:
        collision_flags.append(True)
        collision_points.append((gx, gy))
    else:
        collision_flags.append(False)

# Remove duplicate collision coordinates
collision_points = list(dict.fromkeys(collision_points))

# -----------------------------
# ANIMATION: COLLISION VISUALIZATION
# -----------------------------
fig, ax = plt.subplots()
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect('equal')
ax.grid(True)

# Obstacles (BLACK)
ys, xs = np.where(grid == 1)
ax.plot(xs, ys, 'ks', markersize=4)

# Waypoints
wx, wy = zip(*waypoints)
ax.plot(wx, wy, 'bo', markersize=6)

line, = ax.plot([], [], 'b-', lw=2)
coll_scatter = ax.scatter([], [], c='red', s=50)
texts = []

def update(i):
    # Draw path progressively
    xs = [p[0] for p in full_path[:i+1]]
    ys = [p[1] for p in full_path[:i+1]]
    line.set_data(xs, ys)

    # Clear old annotations
    for t in texts:
        t.remove()
    texts.clear()

    cx, cy = [], []
    for k in range(i + 1):
        if collision_flags[k]:
            gx = int(round(full_path[k][0]))
            gy = int(round(full_path[k][1]))
            cx.append(gx)
            cy.append(gy)
            texts.append(
                ax.text(gx + 0.3, gy + 0.3,
                        f"({gx},{gy})",
                        fontsize=8,
                        color='red')
            )

    if cx:
        coll_scatter.set_offsets(np.c_[cx, cy])
    else:
        coll_scatter.set_offsets(np.empty((0, 2)))

    return line, coll_scatter

ani = FuncAnimation(fig,
                    update,
                    frames=len(full_path),
                    interval=30,
                    blit=False)

plt.show()
