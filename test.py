import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

GRID_SIZE = 50

# ================= USER INPUT =================

waypoints = [
    (5, 5),
    (20, 5),
    (20, 30),
    (48, 43)
]

obstacle_paths = [
    [(10, 10), (40, 10), (41, 15), (30, 18)],
    [(5, 16), (5, 10), (10, 30), (20, 42), (25, 40), (29, 25)],
    [(29, 48), (40, 46), (45, 30)]
]

SAFETY_BUFFER = 2.5   # <<< change buffer distance here

TEXT_OFFSET = 2.5
OFFSET_MODE = "right"   # "right" or "down"

# =================================================


def straight_segment(p1, p2, n=120):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return list(zip(xs, ys))


def point_to_segment_distance(P, A, B):
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    AB = B - A
    AP = P - A

    AB_len_sq = np.dot(AB, AB)
    
    if AB_len_sq == 0:  # A and B are the same point
        return np.linalg.norm(AP), A
    
    t = np.dot(AP, AB) / AB_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest = A + t * AB
    return np.linalg.norm(P - closest), closest


# ---- Convert obstacle paths → segments ----
obstacle_segments = []
for path in obstacle_paths:
    for i in range(len(path) - 1):
        obstacle_segments.append((path[i], path[i + 1]))


# ---- Build full waypoint path ----
full_path = []
for i in range(len(waypoints) - 1):
    seg = straight_segment(waypoints[i], waypoints[i + 1])
    if i > 0:
        seg = seg[1:]
    full_path.extend(seg)


# ---- Detect buffer collisions and find entry/exit points ----
collision_indices = []
collision_points = []
collision_closest = []

in_collision = False
collision_zones = []  # List of (entry_idx, exit_idx, entry_pos, exit_pos)

for i, p in enumerate(full_path):
    has_collision = False
    closest_point = None
    
    for A, B in obstacle_segments:
        dist, closest = point_to_segment_distance(p, A, B)
        if dist <= SAFETY_BUFFER:
            has_collision = True
            closest_point = tuple(closest)
            collision_indices.append(i)
            collision_points.append(p)
            collision_closest.append(closest_point)
            break
    
    # Track entry and exit points
    if has_collision and not in_collision:
        # Entry point
        entry_idx = i
        entry_pos = p
        entry_closest = closest_point
        in_collision = True
    elif not has_collision and in_collision:
        # Exit point (previous point was last collision)
        exit_idx = i - 1
        exit_pos = full_path[exit_idx]
        collision_zones.append((entry_idx, exit_idx, entry_pos, exit_pos))
        in_collision = False

# Handle case where collision continues to the end
if in_collision:
    exit_idx = len(full_path) - 1
    exit_pos = full_path[exit_idx]
    collision_zones.append((entry_idx, exit_idx, entry_pos, exit_pos))


# ---- Plot setup ----
fig, ax = plt.subplots()
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect("equal")
ax.grid(True)

# Draw obstacle paths with proper buffer visualization
for path in obstacle_paths:
    xs, ys = zip(*path)
    # Draw the actual obstacle path
    ax.plot(xs, ys, "k-", lw=2)
    
    # Draw buffer zone more accurately
    for i in range(len(path) - 1):
        # Create a polygon around each segment
        A = np.array(path[i])
        B = np.array(path[i + 1])
        
        # Vector along the segment
        AB = B - A
        AB_len = np.linalg.norm(AB)
        
        if AB_len > 0:
            # Perpendicular vector
            perp = np.array([-AB[1], AB[0]]) / AB_len * SAFETY_BUFFER
            
            # Create rectangle around segment
            corners = [
                A + perp, B + perp, B - perp, A - perp
            ]
            
            poly_xs = [c[0] for c in corners] + [corners[0][0]]
            poly_ys = [c[1] for c in corners] + [corners[0][1]]
            ax.fill(poly_xs, poly_ys, color="gray", alpha=0.2)
        
        # Add circles at endpoints for rounded buffer
        circle1 = plt.Circle(path[i], SAFETY_BUFFER, color='gray', alpha=0.2)
        ax.add_patch(circle1)
    
    # Add circle at last point
    circle_last = plt.Circle(path[-1], SAFETY_BUFFER, color='gray', alpha=0.2)
    ax.add_patch(circle_last)

# Draw waypoints
wx, wy = zip(*waypoints)
ax.plot(wx, wy, "bo", markersize=8, label="Waypoints", zorder=5)

# Add legend items manually to avoid duplicates
ax.plot([], [], "k-", lw=2, label="Other drones")
path_line, = ax.plot([], [], "b-", lw=2, label="Drone path")
collision_scatter = ax.scatter([], [], c="red", s=80, zorder=10, label="Collisions")

texts = []
violation_lines = []


def update(frame):
    xs = [p[0] for p in full_path[:frame + 1]]
    ys = [p[1] for p in full_path[:frame + 1]]
    path_line.set_data(xs, ys)

    for t in texts:
        t.remove()
    texts.clear()
    
    for line in violation_lines:
        line.remove()
    violation_lines.clear()

    cx, cy = [], []

    # Mark all collision points up to current frame
    for idx, drone_pos, obs_pos in zip(collision_indices, collision_points, collision_closest):
        if idx <= frame:
            cx.append(drone_pos[0])
            cy.append(drone_pos[1])

    # Mark entry and exit points with labels
    for entry_idx, exit_idx, entry_pos, exit_pos in collision_zones:
        if entry_idx <= frame:
            # Mark entry point
            if OFFSET_MODE == "right":
                tx, ty = entry_pos[0] + TEXT_OFFSET, entry_pos[1]
            else:
                tx, ty = entry_pos[0], entry_pos[1] - TEXT_OFFSET

            texts.append(
                ax.text(
                    tx, ty,
                    f"({round(entry_pos[0],1)}, {round(entry_pos[1],1)})",
                    fontsize=8,
                    color="red",
                    zorder=11,
                    ha='left' if OFFSET_MODE == "right" else 'center',
                    va='center' if OFFSET_MODE == "right" else 'top'
                )
            )
        
        if exit_idx <= frame:
            # Mark exit point
            if OFFSET_MODE == "right":
                tx, ty = exit_pos[0] + TEXT_OFFSET, exit_pos[1]
            else:
                tx, ty = exit_pos[0], exit_pos[1] - TEXT_OFFSET

            texts.append(
                ax.text(
                    tx, ty,
                    f"({round(exit_pos[0],1)}, {round(exit_pos[1],1)})",
                    fontsize=8,
                    color="red",
                    zorder=11,
                    ha='left' if OFFSET_MODE == "right" else 'center',
                    va='center' if OFFSET_MODE == "right" else 'top'
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

plt.legend(loc='upper left')
plt.title('Drone Path Collision Detection')
plt.show()