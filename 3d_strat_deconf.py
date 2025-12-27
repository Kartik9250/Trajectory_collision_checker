import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import proj3d

GRID_SIZE = 50

# ================= USER INPUT =================

waypoints = [
    (5, 5, 10),
    (20, 5, 25),
    (20, 30, 20),
    (45, 40, 35)
]

obstacle_paths = [
    [(10, 10, 15), (40, 10, 15), (41, 15, 20), (30, 18, 22)],
    [(5, 16, 8), (5, 10, 12), (10, 30, 18), (20, 42, 25), (25, 40, 28), (29, 25, 20)],
    [(29, 48, 30), (40, 46, 35), (45, 30, 32)]
]

SAFETY_BUFFER = 3   # <<< change buffer distance here

TEXT_OFFSET = 1.5
OFFSET_MODE = "right"   # "right" or "down"

# =================================================


def straight_segment(p1, p2, n=120):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    zs = np.linspace(p1[2], p2[2], n)
    return list(zip(xs, ys, zs))


def point_to_segment_distance_3d(P, A, B):
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
        dist, closest = point_to_segment_distance_3d(p, A, B)
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


# ---- Create cylindrical buffer zones for visualization ----
def create_cylinder_around_segment(A, B, radius, n_points=20):
    """Create vertices for a cylinder around a line segment"""
    A = np.array(A)
    B = np.array(B)
    
    # Direction vector
    v = B - A
    length = np.linalg.norm(v)
    
    if length == 0:
        return None
    
    v = v / length
    
    # Find perpendicular vectors
    if abs(v[2]) < 0.9:
        perp1 = np.cross(v, [0, 0, 1])
    else:
        perp1 = np.cross(v, [1, 0, 0])
    
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(v, perp1)
    
    # Create circle points
    theta = np.linspace(0, 2*np.pi, n_points)
    
    # Circles at both ends
    circles = []
    for point in [A, B]:
        circle = []
        for t in theta:
            p = point + radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
            circle.append(p)
        circles.append(np.array(circle))
    
    return circles[0], circles[1]


# ---- Plot setup ----
fig = plt.figure(figsize=(14, 10))

# Make room for controls at the bottom
ax = fig.add_subplot(111, projection='3d', position=[0.05, 0.15, 0.9, 0.8])

# Store initial view limits for home button
initial_xlim = (0, GRID_SIZE)
initial_ylim = (0, GRID_SIZE)
initial_zlim = (0, GRID_SIZE)
initial_elev = 25
initial_azim = 45

ax.set_xlim(initial_xlim)
ax.set_ylim(initial_ylim)
ax.set_zlim(initial_zlim)

# Set colored axes with larger font
ax.set_xlabel('X', color='red', fontsize=14, fontweight='bold')
ax.set_ylabel('Y', color='green', fontsize=14, fontweight='bold')
ax.set_zlabel('Z', color='blue', fontsize=14, fontweight='bold')

# Color the axis lines
ax.xaxis.line.set_color('red')
ax.yaxis.line.set_color('green')
ax.zaxis.line.set_color('blue')

# Color the tick labels
ax.tick_params(axis='x', colors='red', labelsize=11)
ax.tick_params(axis='y', colors='green', labelsize=11)
ax.tick_params(axis='z', colors='blue', labelsize=11)

# Color the axis panes
ax.xaxis.pane.set_edgecolor('red')
ax.yaxis.pane.set_edgecolor('green')
ax.zaxis.pane.set_edgecolor('blue')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Enable proper mouse interactions including zoom
from matplotlib.backend_bases import MouseButton
ax.mouse_init(rotate_btn=MouseButton.LEFT, zoom_btn=MouseButton.RIGHT)

# Store the current zoom level
current_zoom = [1.0]

# Add Home button
ax_home = plt.axes([0.02, 0.05, 0.08, 0.04])
btn_home = Button(ax_home, 'Home', color='lightblue', hovercolor='skyblue')

def home_view(event):
    ax.set_xlim(initial_xlim)
    ax.set_ylim(initial_ylim)
    ax.set_zlim(initial_zlim)
    ax.view_init(elev=initial_elev, azim=initial_azim)
    current_zoom[0] = 1.0
    zoom_slider.set_val(1.0)
    fig.canvas.draw_idle()

btn_home.on_clicked(home_view)

# Add Zoom slider
ax_zoom = plt.axes([0.15, 0.05, 0.7, 0.03])
zoom_slider = Slider(ax_zoom, 'Zoom', 0.1, 3.0, valinit=1.0, valstep=0.1)

def update_zoom(val):
    zoom = zoom_slider.val
    current_zoom[0] = zoom
    
    # Calculate center point
    cx = (initial_xlim[0] + initial_xlim[1]) / 2
    cy = (initial_ylim[0] + initial_ylim[1]) / 2
    cz = (initial_zlim[0] + initial_zlim[1]) / 2
    
    # Calculate range based on zoom
    x_range = (initial_xlim[1] - initial_xlim[0]) / zoom
    y_range = (initial_ylim[1] - initial_ylim[0]) / zoom
    z_range = (initial_zlim[1] - initial_zlim[0]) / zoom
    
    # Set new limits centered on the center point
    ax.set_xlim(cx - x_range/2, cx + x_range/2)
    ax.set_ylim(cy - y_range/2, cy + y_range/2)
    ax.set_zlim(cz - z_range/2, cz + z_range/2)
    
    fig.canvas.draw_idle()

zoom_slider.on_changed(update_zoom)

# Store all important points for hover detection
hover_points = []
hover_labels = []

# Add waypoints
for i, wp in enumerate(waypoints):
    hover_points.append(wp)
    hover_labels.append(f"Waypoint {i+1}\n({wp[0]}, {wp[1]}, {wp[2]})")

# Add collision entry/exit points
for entry_idx, exit_idx, entry_pos, exit_pos in collision_zones:
    hover_points.append(entry_pos)
    hover_labels.append(f"Entry\n({round(entry_pos[0],1)}, {round(entry_pos[1],1)}, {round(entry_pos[2],1)})")
    hover_points.append(exit_pos)
    hover_labels.append(f"Exit\n({round(exit_pos[0],1)}, {round(exit_pos[1],1)}, {round(exit_pos[2],1)})")

# Store helper lines
helper_lines = []
hover_text = None

def on_mouse_move(event):
    global helper_lines, hover_text
    
    if event.inaxes != ax:
        # Remove helper lines if mouse is outside the plot
        for line in helper_lines:
            line.remove()
        helper_lines = []
        if hover_text is not None:
            hover_text.remove()
            hover_text = None
        fig.canvas.draw_idle()
        return
    
    # Find the closest point to mouse
    if event.xdata is None or event.ydata is None:
        return
    
    min_dist = float('inf')
    closest_point = None
    closest_label = None
    
    for point, label in zip(hover_points, hover_labels):
        # Project 3D point to 2D screen coordinates
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
        
        # Calculate distance in screen space
        dist = np.sqrt((x2 - event.xdata)**2 + (y2 - event.ydata)**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_point = point
            closest_label = label
    
    # Only show helper lines if mouse is close enough (threshold in data coordinates)
    threshold = 5.0  # Adjust this value to change sensitivity
    
    if min_dist < threshold and closest_point is not None:
        # Remove old helper lines
        for line in helper_lines:
            line.remove()
        helper_lines = []
        if hover_text is not None:
            hover_text.remove()
            hover_text = None
        
        px, py, pz = closest_point
        
        # Draw helper lines from point to axes with proper colors
        # Line from point down to XY plane (Z direction - blue)
        line, = ax.plot([px, px], [py, py], [0, pz], 
                       color='blue', linestyle='--', alpha=0.5, linewidth=2)
        helper_lines.append(line)
        
        # Line from point to YZ plane (X direction - red)
        line, = ax.plot([0, px], [py, py], [pz, pz], 
                       color='red', linestyle='--', alpha=0.5, linewidth=2)
        helper_lines.append(line)
        
        # Line from point to XZ plane (Y direction - green)
        line, = ax.plot([px, px], [0, py], [pz, pz], 
                       color='green', linestyle='--', alpha=0.5, linewidth=2)
        helper_lines.append(line)
        
        # Show label
        hover_text = ax.text(px, py, pz + 2, closest_label, 
                           fontsize=9, color='black', 
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           zorder=100)
        
        fig.canvas.draw_idle()
    else:
        # Remove helper lines if mouse moved away
        for line in helper_lines:
            line.remove()
        helper_lines = []
        if hover_text is not None:
            hover_text.remove()
            hover_text = None
        fig.canvas.draw_idle()

# Connect the mouse move event
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Draw obstacle paths
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

for path in obstacle_paths:
    xs, ys, zs = zip(*path)
    ax.plot(xs, ys, zs, "k-", lw=2)
    
    # Draw buffer zones as connected cylinders with spheres at joints
    for i in range(len(path) - 1):
        A = np.array(path[i])
        B = np.array(path[i + 1])
        
        result = create_cylinder_around_segment(A, B, SAFETY_BUFFER, n_points=20)
        if result:
            circle1, circle2 = result
            
            # Draw cylinder surface with fill (no edges)
            for j in range(len(circle1)):
                next_j = (j + 1) % len(circle1)
                xs = [circle1[j][0], circle2[j][0], circle2[next_j][0], circle1[next_j][0]]
                ys = [circle1[j][1], circle2[j][1], circle2[next_j][1], circle1[next_j][1]]
                zs = [circle1[j][2], circle2[j][2], circle2[next_j][2], circle1[next_j][2]]
                
                verts = [list(zip(xs, ys, zs))]
                poly = Poly3DCollection(verts, alpha=0.3, facecolor='yellow', edgecolor='none', linewidth=0)
                ax.add_collection3d(poly)
    
    # Draw smooth spheres at ALL waypoints (including joints) with no grid lines
    for point in path:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = SAFETY_BUFFER * np.outer(np.cos(u), np.sin(v)) + point[0]
        y = SAFETY_BUFFER * np.outer(np.sin(u), np.sin(v)) + point[1]
        z = SAFETY_BUFFER * np.outer(np.ones(np.size(u)), np.cos(v)) + point[2]
        ax.plot_surface(x, y, z, color='yellow', alpha=0.3, linewidth=0, edgecolor='none', shade=True)

# Draw waypoints
wx, wy, wz = zip(*waypoints)
ax.scatter(wx, wy, wz, c='blue', s=100, label="Waypoints", zorder=5)

# Initialize plot elements
path_line, = ax.plot([], [], [], "b-", lw=2, label="Drone path")
collision_scatter = ax.scatter([], [], [], c="red", s=80, zorder=10, label="Collisions")


# Add legend
ax.plot([], [], [], "k-", lw=2, label="Other drones")
ax.legend(loc='upper left')
ax.set_title('3D Drone Path Collision Detection')

# Draw the complete path and all collisions at once (no animation)
xs = [p[0] for p in full_path]
ys = [p[1] for p in full_path]
zs = [p[2] for p in full_path]
path_line.set_data(xs, ys)
path_line.set_3d_properties(zs)

# Mark all collision points
cx, cy, cz = [], [], []
for idx, drone_pos, obs_pos in zip(collision_indices, collision_points, collision_closest):
    cx.append(drone_pos[0])
    cy.append(drone_pos[1])
    cz.append(drone_pos[2])

# Mark all entry and exit points with labels
for entry_idx, exit_idx, entry_pos, exit_pos in collision_zones:
    # Mark entry point
    if OFFSET_MODE == "right":
        tx, ty, tz = entry_pos[0] + TEXT_OFFSET, entry_pos[1], entry_pos[2]
    else:
        tx, ty, tz = entry_pos[0], entry_pos[1] - TEXT_OFFSET, entry_pos[2]

    ax.text(
        tx, ty, tz,
        f"({round(entry_pos[0],1)}, {round(entry_pos[1],1)}, {round(entry_pos[2],1)})",
        fontsize=8,
        color="red",
        zorder=11
    )
    
    # Mark exit point
    if OFFSET_MODE == "right":
        tx, ty, tz = exit_pos[0] + TEXT_OFFSET, exit_pos[1], exit_pos[2]
    else:
        tx, ty, tz = exit_pos[0], exit_pos[1] - TEXT_OFFSET, exit_pos[2]

    ax.text(
        tx, ty, tz,
        f"({round(exit_pos[0],1)}, {round(exit_pos[1],1)}, {round(exit_pos[2],1)})",
        fontsize=8,
        color="red",
        zorder=11
    )

if cx:
    collision_scatter._offsets3d = (cx, cy, cz)

# Set initial view angle
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()