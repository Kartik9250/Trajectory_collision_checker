import numpy as np
from typing import List, Tuple, Dict, Any, Optional
# =====================================================
# ================= GLOBAL CONFIG =====================
# =====================================================

TIME_RESOLUTION = 0.01

DEFAULT_USER_START_TIME = 0.0
DEFAULT_SAFETY_BUFFER = 3.0

DEFAULT_OBSTACLE_PATHS = [
    [(0, 10, 20), (40, 10, 20)],
    [(0, 20, 20), (40, 20, 20)],
    [(0, 30, 20), (40, 30, 20)]
]

DEFAULT_OBSTACLE_SPEEDS = [2.0, 2.0, 2.0]
DEFAULT_OBSTACLE_START_TIMES = [0.0, 0.0, 0.0]

# =====================================================
# ================= HELPER FUNCTIONS ==================
# =====================================================

def straight_segment(p1, p2, n):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    zs = np.linspace(p1[2], p2[2], n)
    return list(zip(xs, ys, zs))


def calculate_path_length(path):
    return sum(
        np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
        for i in range(len(path) - 1)
    )


def get_obstacle_position_at_time(A, B, t0, t1, tq):
    if tq < t0 or tq > t1:
        return None
    u = (tq - t0) / (t1 - t0)
    return np.array(A) + u * (np.array(B) - np.array(A))


# =====================================================
# ================= CORE API FUNCTION =================
# =====================================================

def run_collision_check(
    waypoints: List[Tuple[float, float, float]],
    user_end_time: float,
    obstacle_paths: List[List[Tuple[float, float, float]]],
    safety_buffer: float,
    user_start_time: Optional[float] = None,
    obstacle_speeds: Optional[List[float]] = None,
    obstacle_start_times: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Time-based 3D collision detection.

    Args:
        waypoints (list[tuple[float, float, float]]):
            User trajectory waypoints in 3D.
        user_end_time (float):
            Time at which the user trajectory ends.
        obstacle_paths (list[list[tuple[float, float, float]]]):
            List of obstacle trajectories, each defined by waypoints.
        safety_buffer (float):
            Minimum allowed distance between user and obstacle.

        user_start_time (float, optional):
            Start time of the user trajectory. Defaults to 0.0.
        obstacle_speeds (list[float], optional):
            Speeds for each obstacle trajectory.
        obstacle_start_times (list[float], optional):
            Start times for each obstacle trajectory.

    Returns:
        dict:
            Collision information containing:
            - collision_count (int)
            - collision_zones (list[dict])
            - user_path (list[tuple])
            - user_times (list[float])
    """

    # ---------- Defaults ----------
    if user_start_time is None:
        user_start_time = DEFAULT_USER_START_TIME

    if obstacle_speeds is None:
        obstacle_speeds = DEFAULT_OBSTACLE_SPEEDS[:len(obstacle_paths)]

    if obstacle_start_times is None:
        obstacle_start_times = DEFAULT_OBSTACLE_START_TIMES[:len(obstacle_paths)]

    # ---------- Build user path with time ----------
    full_path = []
    path_times = []

    total_dist = calculate_path_length(waypoints)
    total_time = user_end_time - user_start_time
    time_cursor = user_start_time

    for i in range(len(waypoints) - 1):
        p1, p2 = np.array(waypoints[i]), np.array(waypoints[i + 1])
        seg_dist = np.linalg.norm(p2 - p1)

        seg_time = (seg_dist / total_dist) * total_time if total_dist else total_time
        n_pts = max(2, int(seg_time / TIME_RESOLUTION))

        segment = straight_segment(p1, p2, n_pts)

        for j, pt in enumerate(segment):
            if i > 0 and j == 0:
                continue
            alpha = j / (n_pts - 1)
            full_path.append(pt)
            path_times.append(time_cursor + alpha * seg_time)

        time_cursor += seg_time

    # ---------- Build obstacle segments with time ----------
    obstacle_segments = []
    obstacle_times = []

    for path, speed, start in zip(obstacle_paths, obstacle_speeds, obstacle_start_times):
        times = [start]
        for i in range(1, len(path)):
            d = np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
            times.append(times[-1] + d / speed)

        for i in range(len(path) - 1):
            obstacle_segments.append((path[i], path[i + 1]))
            obstacle_times.append((times[i], times[i + 1]))

    # ---------- Collision Detection ----------
    collision_zones = []
    in_collision = False

    for i, (u_pos, u_time) in enumerate(zip(full_path, path_times)):
        hit = False

        for (A, B), (t0, t1) in zip(obstacle_segments, obstacle_times):
            obs_pos = get_obstacle_position_at_time(A, B, t0, t1, u_time)
            if obs_pos is None:
                continue

            if np.linalg.norm(np.array(u_pos) - obs_pos) <= safety_buffer:
                hit = True
                break

        if hit and not in_collision:
            entry_idx = i
            entry_time = u_time
            entry_pos = u_pos
            in_collision = True

        elif not hit and in_collision:
            exit_idx = i - 1
            collision_zones.append({
                "entry_time": entry_time,
                "exit_time": path_times[exit_idx],
                "entry_pos": entry_pos,
                "exit_pos": full_path[exit_idx],
                "duration": path_times[exit_idx] - entry_time
            })
            in_collision = False

    # ---------- Final result ----------
    return {
        "collision_count": len(collision_zones),
        "collision_zones": collision_zones,
        "user_path": full_path,
        "user_times": path_times,
        "obstacle_paths": obstacle_paths,
        "safety_buffer": safety_buffer
    }


# =====================================================
# ================= DIRECT EXECUTION ==================
# =====================================================

if __name__ == "__main__":

    waypoints = [
        (20, 0, 20),
        (20, 40, 20)
    ]

    result = run_collision_check(
        waypoints=waypoints,
        user_end_time=20.0,
        obstacle_paths=DEFAULT_OBSTACLE_PATHS,
        safety_buffer=DEFAULT_SAFETY_BUFFER
    )

    print(f"\nCollisions detected: {result['collision_count']}")
    for i, z in enumerate(result["collision_zones"], 1):
        print(
            f"Zone {i}: "
            f"Entry @ t={z['entry_time']:.2f}s → "
            f"Exit @ t={z['exit_time']:.2f}s "
            f"(Duration {z['duration']:.2f}s)"
        )
