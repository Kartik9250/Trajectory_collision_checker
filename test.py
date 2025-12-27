import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 50

waypoints = [
    (5, 5),
    (18, 1),
    (10, 30),
    (45, 40),
    (40, 30),
    (22, 25),
    (35, 5)
]

TEXT_OFFSET = 1.0
OFFSET_MODE = "right"  # "right" or "down"

obstacle_lines = [
    ((10, 10), (40, 10)),
    ((25, 15), (25, 40))
]

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def straight_segment(p1, p2, n=120):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return list(zip(xs, ys))

full_path = []
for i in range(len(waypoints) - 1):
    seg = straight_segment(waypoints[i], waypoints[i + 1])
    if i > 0:
        seg = seg[1:]
    full_path.extend(seg)

collision_indices = []
collision_points = []

for i in range(len(full_path) - 1):
    p1 = full_path[i]
    p2 = full_path[i + 1]

    for o1, o2 in obstacle_lines:
        if segments_intersect(p1, p2, o1, o2):
            collision_indices.append(i)
            collision_points.append(((p1[0] + p2[0]) / 2,
                                     (p1[1] + p2[1]) / 2))

fig, ax = plt.subplots()
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect("equal")
ax.grid(True)

for (x1, y1), (x2, y2) in obstacle_lines:
    ax.plot([x1, x2], [y1, y2], "k-", lw=0.8)

wx, wy = zip(*waypoints)
ax.plot(wx, wy, "bo", markersize=6)

path_line, = ax.plot([], [], "b-", lw=2)
collision_scatter = ax.scatter([], [], c="red", s=60)
texts = []

def update(frame):
    xs = [p[0] for p in full_path[:frame + 1]]
    ys = [p[1] for p in full_path[:frame + 1]]
    path_line.set_data(xs, ys)

    for t in texts:
        t.remove()
    texts.clear()

    cx, cy = [], []

    for idx, (ix, iy) in zip(collision_indices, collision_points):
        if idx <= frame:
            
            cx.append(ix)
            cy.append(iy)

            if OFFSET_MODE == "right":
                tx, ty = ix + TEXT_OFFSET, iy
            else:
                tx, ty = ix, iy - TEXT_OFFSET

            texts.append(
                ax.text(
                    tx, ty,
                    f"({round(ix,1)}, {round(iy,1)})",
                    fontsize=8,
                    color="red"
                )
            )

    if cx:
        collision_scatter.set_offsets(np.c_[cx, cy])
    else:
        collision_scatter.set_offsets(np.empty((0, 2)))

    return path_line, collision_scatter

ani = FuncAnimation(
    fig,
    update,
    frames=len(full_path),
    interval=30,
    blit=False
)

plt.show()
