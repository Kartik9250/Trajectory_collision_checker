from strat_deconf import run_collision_check

# User path
waypoints = [
    (10, 0, 20),
    (20, 40, 20)
]

# Obstacle paths
obstacle_paths = [
    [(0, 10, 20), (40, 10, 20)],
    [(0, 20, 20), (40, 20, 20)]
]

# Run collision check
result = run_collision_check(
    waypoints=waypoints,
    user_end_time=20.0,
    obstacle_paths=obstacle_paths,
    safety_buffer=0.0
)

# Check result
if result["collision_count"] == 0:
    print("✅ No collisions detected")
else:
    print(f"❌ {result['collision_count']} collision zone(s) detected")
    for i, z in enumerate(result["collision_zones"], 1):
        print(
            f"  Zone {i}: "
            f"{z['entry_time']:.2f}s → {z['exit_time']:.2f}s "
            f"(Δt={z['duration']:.2f}s)"
        )
