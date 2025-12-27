import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

# -----------------------------
# PARAMETERS
# -----------------------------
GRID_SIZE = 50
SAFETY_BUFFER = 3

start = (5, 5)
end   = (45, 45)

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

# obstacle lines (T shape)
mark_line(grid, (10, 10), (40, 10))
mark_line(grid, (25, 15), (25, 40))

# -----------------------------
# INFLATE OBSTACLES (SAFETY BUFFER)
# -----------------------------
def inflate_obstacles(grid, buffer_cells):
    inflated = grid.copy()
    obs = np.argwhere(grid == 1)

    for y, x in obs:
        for dy in range(-buffer_cells, buffer_cells + 1):
            for dx in range(-buffer_cells, buffer_cells + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                    if np.hypot(dx, dy) <= buffer_cells:
                        inflated[ny, nx] = 1
    return inflated

inflated_grid = inflate_obstacles(grid, SAFETY_BUFFER)

# -----------------------------
# A* PATH PLANNER (8-connected)
# -----------------------------
def astar(grid, start, goal):
    def h(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}

    moves = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                if grid[ny, nx] == 1:
                    continue
                neighbor = (nx, ny)
                ng = g[current] + np.hypot(dx, dy)
                if neighbor not in g or ng < g[neighbor]:
                    g[neighbor] = ng
                    f = ng + h(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
    return []

safe_path = astar(inflated_grid, start, end)
#print("Safe path length:", len(safe_path))

# -----------------------------
# ANIMATION 1: SAFE PATH
# -----------------------------
fig1, ax1 = plt.subplots()
ax1.set_xlim(0, GRID_SIZE)
ax1.set_ylim(0, GRID_SIZE)
ax1.set_aspect('equal')
ax1.grid(True)

# buffer (gray)
ys, xs = np.where(inflated_grid == 1)
ax1.plot(xs, ys, 's', color='lightgray', markersize=4)

# original obstacles (black)
ys, xs = np.where(grid == 1)
ax1.plot(xs, ys, 'ks', markersize=4)

ax1.plot(start[0], start[1], 'go', markersize=8)
ax1.plot(end[0], end[1], 'ro', markersize=8)

line1, = ax1.plot([], [], 'r-', lw=2)

def update1(i):
    xs = [p[0] for p in safe_path[:i+1]]
    ys = [p[1] for p in safe_path[:i+1]]
    line1.set_data(xs, ys)
    return line1,

ani1 = FuncAnimation(fig1, update1,
                     frames=len(safe_path),
                     interval=50,
                     blit=False)

# -----------------------------
# NAIVE STRAIGHT PATH
# -----------------------------
def straight_line_path(p1, p2, n=200):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    return list(zip(xs, ys))

naive_path = straight_line_path(start, end)

collision_flags = []
for x, y in naive_path:
    gx, gy = int(round(x)), int(round(y))
    if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
        collision_flags.append(inflated_grid[gy, gx] == 1)
    else:
        collision_flags.append(False)

# -----------------------------
# ANIMATION 2: CONFLICT VISUALIZATION
# -----------------------------
fig2, ax2 = plt.subplots()
ax2.set_xlim(0, GRID_SIZE)
ax2.set_ylim(0, GRID_SIZE)
ax2.set_aspect('equal')
ax2.grid(True)

# buffer first
ys, xs = np.where(inflated_grid == 1)
ax2.plot(xs, ys, 's', color='lightgray', markersize=4)

# original obstacles
ys, xs = np.where(grid == 1)
ax2.plot(xs, ys, 'ks', markersize=4)

ax2.plot(start[0], start[1], 'go', markersize=8)
ax2.plot(end[0], end[1], 'ro', markersize=8)

line2, = ax2.plot([], [], 'b--', lw=2)
conf_scatter = ax2.scatter([], [], c='red', s=50)
texts = []

def update2(i):
    xs = [p[0] for p in naive_path[:i+1]]
    ys = [p[1] for p in naive_path[:i+1]]
    line2.set_data(xs, ys)

    for t in texts:
        t.remove()
    texts.clear()

    cx, cy = [], []
    for k in range(i + 1):
        if collision_flags[k]:
            gx = int(round(naive_path[k][0]))
            gy = int(round(naive_path[k][1]))
            cx.append(gx)
            cy.append(gy)
            texts.append(
                ax2.text(gx + 0.3, gy + 0.3,
                         f"({gx},{gy})",
                         fontsize=8,
                         color='red')
            )

    if cx:
        conf_scatter.set_offsets(np.c_[cx, cy])
    else:
        conf_scatter.set_offsets(np.empty((0, 2)))

    return line2, conf_scatter

ani2 = FuncAnimation(fig2, update2,
                     frames=len(naive_path),
                     interval=30,
                     blit=False)

plt.show()
