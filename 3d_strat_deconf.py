import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import proj3d

GRID_SIZE = 50

# ================= USER INPUT =================

# User drone path
waypoints = [
    (20, 0, 20),
    (20, 40, 20),
]
user_drone_speed = 2  # units per second
user_start_time = 0.0   # seconds

# Predefined drone paths
obstacle_paths = [
    [(0, 10, 20), (40, 10, 20)],
    [(0, 20, 20), (40, 20, 20)],
    [(0, 30, 20), (40, 30, 20)]
]
obstacle_speeds = [2.0, 2.0, 2.0]  # units per second for each path
obstacle_start_times = [0.0, 0.0, 0.0]  # start time in seconds for each path

SAFETY_BUFFER = 3   # <<< change buffer distance here

TEXT_OFFSET = 1.5
OFFSET_MODE = "right"   # "right" or "down"

# =================================================


def straight_segment(p1, p2, n=120):
    # This n is now overridden by the dynamic calculation in the main loop
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    zs = np.linspace(p1[2], p2[2], n)
    return list(zip(xs, ys, zs))


def calculate_path_length(path):
    """Calculate total length of a path"""
    total_length = 0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        total_length += np.linalg.norm(p2 - p1)
    return total_length


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


# Calculate times for obstacle paths
obstacle_path_times = []
for path, speed, start_time in zip(obstacle_paths, obstacle_speeds, obstacle_start_times):
    times = []
    cumulative_distance = 0
    
    for i, point in enumerate(path):
        if i == 0:
            times.append(start_time)
        else:
            distance = np.linalg.norm(np.array(point) - np.array(path[i - 1]))
            cumulative_distance += distance
            times.append(start_time + (cumulative_distance / speed))
    
    obstacle_path_times.append(times)


# ---- Convert obstacle paths → segments with time info ----
obstacle_segments = []
obstacle_segment_times = []

for path, times in zip(obstacle_paths, obstacle_path_times):
    for i in range(len(path) - 1):
        obstacle_segments.append((path[i], path[i + 1]))
        obstacle_segment_times.append((times[i], times[i + 1]))


def get_obstacle_position_at_time(segment_start, segment_end, time_start, time_end, query_time):
    """Get the position of an obstacle drone at a specific time along a segment"""
    if query_time < time_start or query_time > time_end:
        return None
    
    # Linear interpolation of position based on time
    t = (query_time - time_start) / (time_end - time_start)
    pos = np.array(segment_start) + t * (np.array(segment_end) - np.array(segment_start))
    return pos


# ---- Build full waypoint path with time tracking ----
full_path = []
path_times = []  # Time at each point along the path
cumulative_distance = 0

# --- CORRECTION: High-resolution time-stepping ---
TIME_RESOLUTION = 0.01 # Check every 0.01 seconds

for i in range(len(waypoints) - 1):
    segment_length = np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))
    segment_duration = segment_length / user_drone_speed
    
    # Calculate required points for the desired resolution
    n_points = max(2, int(segment_duration / TIME_RESOLUTION))
    
    seg = straight_segment(waypoints[i], waypoints[i + 1], n=n_points)
    
    for j, point in enumerate(seg):
        if i > 0 and j == 0:
            continue  # Skip first point of subsequent segments to avoid duplicates
        
        # Calculate time based on distance traveled
        if i == 0 and j == 0:
            point_time = user_start_time
        else:
            if j > 0:
                prev_point = seg[j - 1]
            else:
                prev_point = full_path[-1]
            
            distance = np.linalg.norm(np.array(point) - np.array(prev_point))
            cumulative_distance += distance
            point_time = user_start_time + (cumulative_distance / user_drone_speed)
        
        full_path.append(point)
        path_times.append(point_time)


# ---- Detect actual time-based collisions with safety buffer ----
collision_indices = []
collision_points = []
collision_closest = []
collision_times = []
collision_obstacle_positions = []  # Store actual obstacle position at collision time

in_collision = False
collision_zones = []

for i, (user_pos, user_time) in enumerate(zip(full_path, path_times)):
    has_collision = False
    closest_point = None
    closest_obstacle_pos = None
    
    # Check against each obstacle segment
    for (seg_start, seg_end), (time_start, time_end) in zip(obstacle_segments, obstacle_segment_times):
        # Get obstacle position at this time
        obstacle_pos = get_obstacle_position_at_time(seg_start, seg_end, time_start, time_end, user_time)
        
        if obstacle_pos is not None:
            # Calculate distance between user drone and obstacle drone at this time
            dist = np.linalg.norm(np.array(user_pos) - obstacle_pos)
            
            # Check if within safety buffer
            if dist <= SAFETY_BUFFER:
                has_collision = True
                closest_point = tuple(obstacle_pos)
                closest_obstacle_pos = tuple(obstacle_pos)
                collision_indices.append(i)
                collision_points.append(user_pos)
                collision_closest.append(closest_point)
                collision_times.append(user_time)
                collision_obstacle_positions.append(closest_obstacle_pos)
                break
    
    # Track entry and exit points
    if has_collision and not in_collision:
        # Entry point
        entry_idx = i
        entry_pos = user_pos
        entry_time = user_time
        entry_obstacle_pos = closest_obstacle_pos
        in_collision = True
    elif not has_collision and in_collision:
        # Exit point (previous point was last collision)
        exit_idx = i - 1
        exit_pos = full_path[exit_idx]
        exit_time = path_times[exit_idx]
        collision_zones.append((entry_idx, exit_idx, entry_pos, exit_pos, entry_time, exit_time))
        in_collision = False

# Handle case where collision continues to the end
if in_collision:
    exit_idx = len(full_path) - 1
    exit_pos = full_path[exit_idx]
    exit_time = path_times[exit_idx]
    collision_zones.append((entry_idx, exit_idx, entry_pos, exit_pos, entry_time, exit_time))

# Calculate total times and print summary
user_total_time = path_times[-1] - user_start_time
obstacle_total_times = [(times[-1] - times[0]) for times in obstacle_path_times]

print(f"User drone path: Start={user_start_time:.2f}s, End={path_times[-1]:.2f}s, Duration={user_total_time:.2f}s")
for i, (start, times) in enumerate(zip(obstacle_start_times, obstacle_path_times)):
    print(f"Obstacle drone {i+1}: Start={start:.2f}s, End={times[-1]:.2f}s, Duration={obstacle_total_times[i]:.2f}s")

print(f"\nTime-based collisions detected: {len(collision_zones)} zone(s)")
if len(collision_zones) == 0:
    print("  No actual collisions - drones pass through areas at different times!")
else:
    for i, (entry_idx, exit_idx, entry_pos, exit_pos, entry_time, exit_time) in enumerate(collision_zones):
        print(f"  Zone {i+1}: Entry at t={entry_time:.2f}s, Exit at t={exit_time:.2f}s, Duration={(exit_time-entry_time):.2f}s")


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

# Draw obstacle paths
for path_idx, (path, times) in enumerate(zip(obstacle_paths, obstacle_path_times)):
    xs, ys, zs = zip(*path)
    ax.plot(xs, ys, zs, "k-", lw=2)
    
    # Add time annotations for obstacle path waypoints
    for i, (point, time) in enumerate(zip(path, times)):
        ax.text(point[0], point[1], point[2] + 1.5, f"t={time:.1f}s", 
                fontsize=7, color='black', ha='center', alpha=0.7)
    
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
ax.scatter(wx, wy, wz, c='blue', s=100, label="User Waypoints", zorder=5)

# Add time annotations for user waypoints
for i, (wp, idx) in enumerate(zip(waypoints, [0] + [len(straight_segment(waypoints[j], waypoints[j+1])) - 1 + sum(len(straight_segment(waypoints[k], waypoints[k+1])) - 1 for k in range(j)) for j in range(len(waypoints)-1)])):
    if i < len(path_times):
        # Find the closest time in path_times for this waypoint
        wp_array = np.array(wp)
        min_dist = float('inf')
        wp_time = user_start_time
        for pt, t in zip(full_path, path_times):
            dist = np.linalg.norm(np.array(pt) - wp_array)
            if dist < min_dist:
                min_dist = dist
                wp_time = t
        
        ax.text(wp[0], wp[1], wp[2] + 1.5, f"t={wp_time:.1f}s", 
                fontsize=8, color='blue', ha='center')

# Initialize plot elements
path_line, = ax.plot([], [], [], "b-", lw=2, label="Drone path")
collision_scatter = ax.scatter([], [], [], c="red", s=80, zorder=10, label="Collisions")

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

if cx:
    collision_scatter._offsets3d = (cx, cy, cz)

# Mark entry and exit points with labels including time
for entry_idx, exit_idx, entry_pos, exit_pos, entry_time, exit_time in collision_zones:
    # Mark entry point
    if OFFSET_MODE == "right":
        tx, ty, tz = entry_pos[0] + TEXT_OFFSET, entry_pos[1], entry_pos[2]
    else:
        tx, ty, tz = entry_pos[0], entry_pos[1] - TEXT_OFFSET, entry_pos[2]

    ax.text(
        tx, ty, tz,
        f"Entry @ t={entry_time:.2f}s\n({round(entry_pos[0],1)}, {round(entry_pos[1],1)}, {round(entry_pos[2],1)})",
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
        f"Exit @ t={exit_time:.2f}s\n({round(exit_pos[0],1)}, {round(exit_pos[1],1)}, {round(exit_pos[2],1)})",
        fontsize=8,
        color="red",
        zorder=11
    )

# Set initial view angle
ax.view_init(elev=25, azim=45)

# Add legend
ax.plot([], [], [], "k-", lw=2, label="Other drones")
ax.legend(loc='upper left')
ax.set_title('3D Drone Path Collision Detection')

# Store all important points for hover detection
hover_points = []
hover_labels = []

# Add waypoints
for i, wp in enumerate(waypoints):
    # Find the time for this waypoint
    wp_array = np.array(wp)
    min_dist = float('inf')
    wp_time = user_start_time
    for pt, t in zip(full_path, path_times):
        dist = np.linalg.norm(np.array(pt) - wp_array)
        if dist < min_dist:
            min_dist = dist
            wp_time = t
    
    hover_points.append(wp)
    hover_labels.append(f"Waypoint {i+1}\nt={wp_time:.2f}s\n({wp[0]}, {wp[1]}, {wp[2]})")

# Add collision entry/exit points
for entry_idx, exit_idx, entry_pos, exit_pos, entry_time, exit_time in collision_zones:
    hover_points.append(entry_pos)
    hover_labels.append(f"Entry @ t={entry_time:.2f}s\n({round(entry_pos[0],1)}, {round(entry_pos[1],1)}, {round(entry_pos[2],1)})")
    hover_points.append(exit_pos)
    hover_labels.append(f"Exit @ t={exit_time:.2f}s\n({round(exit_pos[0],1)}, {round(exit_pos[1],1)}, {round(exit_pos[2],1)})")

# Store helper lines
helper_lines = []
hover_text = None
last_closest_point = [None]  # Track last hovered point to avoid unnecessary redraws
is_dragging = [False]  # Track if user is dragging
hover_enabled = [True]  # Global hover enable/disable

def on_button_press(event):
    is_dragging[0] = True
    # Clear any existing hover effects immediately when starting to drag
    global helper_lines, hover_text
    if helper_lines or hover_text is not None:
        for line in helper_lines:
            line.remove()
        helper_lines = []
        if hover_text is not None:
            hover_text.remove()
            hover_text = None
        last_closest_point[0] = None

def on_button_release(event):
    is_dragging[0] = False

def on_mouse_move(event):
    global helper_lines, hover_text
    
    # Skip all hover processing while dragging
    if is_dragging[0]:
        return
    
    # Skip if hover is disabled
    if not hover_enabled[0]:
        return
    
    if event.inaxes != ax:
        # Only clean up if there's something to clean
        if helper_lines or hover_text is not None:
            for line in helper_lines:
                try:
                    line.remove()
                except:
                    pass
            helper_lines = []
            if hover_text is not None:
                try:
                    hover_text.remove()
                except:
                    pass
                hover_text = None
            last_closest_point[0] = None
        return
    
    # Find the closest point to mouse (only if not dragging)
    if event.xdata is None or event.ydata is None:
        return
    
    min_dist = float('inf')
    closest_point = None
    closest_label = None
    
    try:
        for point, label in zip(hover_points, hover_labels):
            # Project 3D point to 2D screen coordinates
            x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
            
            # Calculate distance in screen space
            dist = np.sqrt((x2 - event.xdata)**2 + (y2 - event.ydata)**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_point = point
                closest_label = label
    except:
        return
    
    # Only show helper lines if mouse is close enough
    threshold = 5.0
    
    if min_dist < threshold and closest_point is not None:
        # Only redraw if hovering over a different point
        if last_closest_point[0] == closest_point:
            return
        
        last_closest_point[0] = closest_point
        
        # Remove old helper lines
        for line in helper_lines:
            try:
                line.remove()
            except:
                pass
        helper_lines = []
        if hover_text is not None:
            try:
                hover_text.remove()
            except:
                pass
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
        if last_closest_point[0] is not None:
            for line in helper_lines:
                try:
                    line.remove()
                except:
                    pass
            helper_lines = []
            if hover_text is not None:
                try:
                    hover_text.remove()
                except:
                    pass
                hover_text = None
            last_closest_point[0] = None

# Connect the mouse events with lower priority to not interfere with built-in rotation
fig.canvas.mpl_connect('button_press_event', on_button_press)
fig.canvas.mpl_connect('button_release_event', on_button_release)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

plt.show()