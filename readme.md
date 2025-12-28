# Trajectory Deconfliction & Collision Checker

## Overview

A lightweight, time-parameterized 3D trajectory safety checker for robotics and motion planning.

The tool evaluates a waypoint-defined user trajectory against one or more moving obstacle trajectories, accounting for speed, timing, and a configurable safety buffer. If a conflict occurs, it reports precise collision time windows and 3D locations.

---

## Features

* Multiple waypoints per trajectory
* Time-based (spatiotemporal) collision checking
* Configurable safety buffer
* Multiple moving obstacles
* Structured collision output (time & position)

---

## Files

### `strat_deconf.py`

Core library module exposing `run_collision_check()`.
Use this file for integration into planners, simulations, or ROS pipelines.

### `sample_callback.py`

Minimal example showing how to call the collision checker and interpret results.

### `3d_strat_deconf.py`

Standalone, self-contained variant with defaults and a direct execution block.
Useful for quick testing.

### `test.py`

Legacy visualization-based prototype.
Not required for core functionality.

---

## Prerequisites

* Python3 verion:3.13.9(tested), for installation visit [Python's website](https://www.python.org/downloads/) 
* Matplotlib and numpy, to install run `pip install -r requirements.txt`

---

## Typical Use

1. Define user waypoints, end_time, safety_buffer distance, and obstacle waypoints
2. Call `run_collision_check()`
3. Use collision results for replanning or safety handling

---

## License

Apache-2.0 license 
