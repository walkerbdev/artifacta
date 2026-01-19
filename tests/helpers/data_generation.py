"""
Data generation functions for e2e tests
Functions for generating domain-specific simulation data and mock system metrics
"""

import csv
import os
import tempfile

import numpy as np


def generate_csv_data(filename="training_data.csv", rows=20):
    """Generate small CSV file with sample tabular data

    Returns (filename, filepath) tuple
    """
    temp_path = os.path.join(tempfile.gettempdir(), filename)

    with open(temp_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["feature_1", "feature_2", "feature_3", "target"])
        # Data rows
        for _i in range(rows):
            f1 = np.random.randn()
            f2 = np.random.randn()
            f3 = np.random.randn()
            target = 1 if (f1 + f2 + f3) > 0 else 0
            writer.writerow([f"{f1:.4f}", f"{f2:.4f}", f"{f3:.4f}", target])

    return filename, temp_path


def generate_simulation_data(config, seed=42):
    """
    Generate physics simulation data that varies based on config parameters.

    Args:
        config: Config dict with timestep, integration_method, etc.
        seed: Random seed for reproducibility

    Returns:
        dict with energy and particle state data
    """
    np.random.seed(seed)

    timestep = config.get("timestep", 0.01)
    # Smaller timestep = more stable energy conservation
    stability_factor = 0.01 / timestep  # smaller timestep = higher stability
    noise_level = min(2.0, 1.0 / stability_factor)

    time_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    total_energy = 150

    # Generate fluctuating energies that sum to constant (with more noise for larger timesteps)
    kinetic = []
    potential = []
    for _ in time_points:
        ke = 100 + np.random.uniform(-noise_level, noise_level)
        kinetic.append(ke)
        potential.append(total_energy - ke)

    return {
        "kinetic_energy": kinetic,
        "potential_energy": potential,
        "total_energy": [total_energy] * len(time_points),
        "time_values": time_points,
    }


def generate_ab_test_data(config, seed=42):
    """
    Generate A/B test data that varies based on config parameters.

    Args:
        config: Config dict with traffic_split, min_sample_size, etc.
        seed: Random seed

    Returns:
        dict with conversion data
    """
    np.random.seed(seed)

    traffic_split = config.get("traffic_split", 0.5)
    min_sample = config.get("min_sample_size", 1000)

    # More samples = tighter distribution, more confident results
    n_control = int(min_sample * traffic_split)
    n_variant = int(min_sample * (1 - traffic_split))

    # Variant performs better (realistic for A/B test)
    base_control_rate = 0.124
    base_variant_rate = 0.156

    # Add some variation based on sample size (larger sample = closer to true rate)
    sample_noise = 1000 / min_sample  # smaller samples = more noise

    control_vals = [
        base_control_rate + np.random.normal(0, 0.01 * sample_noise) for _ in range(n_control)
    ]
    variant_vals = [
        base_variant_rate + np.random.normal(0, 0.01 * sample_noise) for _ in range(n_variant)
    ]

    return {
        "values": control_vals + variant_vals,
        "groups": ["Control"] * n_control + ["Variant A"] * n_variant,
        "n_control": n_control,
        "n_variant": n_variant,
    }


def generate_climate_data(config, seed=42):
    """
    Generate climate model data that varies based on config parameters.

    Args:
        config: Config dict with model_resolution, ensemble_size, etc.
        seed: Random seed

    Returns:
        dict with temperature anomaly data
    """
    np.random.seed(seed)

    resolution = config.get("model_resolution", "1deg")
    ensemble_size = config.get("ensemble_size", 10)

    # Higher resolution and larger ensemble = smoother, more accurate trends
    resolution_factor = 0.5 if "0.5" in resolution else 1.0
    ensemble_factor = ensemble_size / 10

    noise_level = 0.05 / (resolution_factor * ensemble_factor)

    years = list(range(2010, 2020))
    base_trend = np.linspace(-0.2, 0.7, len(years))

    global_anomaly = (base_trend + np.random.normal(0, noise_level, len(years))).tolist()
    northern = (base_trend + 0.05 + np.random.normal(0, noise_level * 1.2, len(years))).tolist()
    southern = (base_trend - 0.05 + np.random.normal(0, noise_level * 0.8, len(years))).tolist()

    return {
        "global_anomaly": global_anomaly,
        "northern_hemisphere": northern,
        "southern_hemisphere": southern,
        "years": years,
    }


def generate_path_planning_data(config, seed=42):
    """
    Generate robotics path planning data based on config.

    Args:
        config: Config dict with max_iterations, step_size, etc.
        seed: Random seed

    Returns:
        dict with trajectory data
    """
    np.random.seed(seed)

    max_iter = config.get("max_iterations", 1000)
    step_size = config.get("step_size", 0.5)

    # More iterations and smaller steps = smoother, more optimal path
    smoothness = (max_iter / 1000) * (0.5 / step_size)
    noise = 0.3 / smoothness

    # Path from (0,0) to (10,10)
    base_x = [0, 2, 5, 7, 10]
    base_y = [0, 3, 2, 6, 10]

    x = [bx + np.random.uniform(-noise, noise) for bx in base_x]
    y = [by + np.random.uniform(-noise, noise) for by in base_y]

    # Velocity varies with step size
    base_vel = 1.0 / step_size
    velocity = [max(0.3, base_vel + np.random.uniform(-0.2, 0.2)) for _ in range(5)]

    return {
        "x": x,
        "y": y,
        "velocity": velocity,
        "time": [0, 2, 4, 6, 8],
    }


def generate_finance_backtest_data(config, seed=42):
    """
    Generate finance backtest data based on config parameters.

    Args:
        config: Config dict with lookback_period, stop_loss, etc.
        seed: Random seed

    Returns:
        dict with portfolio and benchmark data
    """
    np.random.seed(seed)

    lookback = config.get("lookback_period", 20)
    stop_loss = config.get("stop_loss", 0.02)

    # Longer lookback period → smoother, potentially better performance
    # Tighter stop loss → less drawdown but potentially lower returns
    smoothness = lookback / 20
    risk_factor = stop_loss / 0.02

    # Generate portfolio returns
    days = 10
    initial_value = 100000

    # Better parameters = better risk-adjusted returns
    daily_return_mean = 0.008 * smoothness * (1 + (1 - risk_factor) * 0.3)
    daily_return_std = 0.02 * risk_factor

    portfolio_values = [initial_value]
    for _ in range(days - 1):
        ret = np.random.normal(daily_return_mean, daily_return_std)
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    # Benchmark grows steadily
    benchmark_values = [initial_value * (1 + 0.005 * i) for i in range(days)]

    return {
        "portfolio_value": portfolio_values,
        "benchmark": benchmark_values,
        "days": list(range(days)),
    }


def generate_mock_system_metrics(profile, num_samples=50):
    """
    Generate realistic mock system metrics based on workload profile.

    Mimics what SystemMonitor would capture:
    - CPU: System & process usage
    - Memory: System & process RAM usage
    - Disk: I/O throughput
    - Network: Sent/received bytes
    - GPU: Utilization, memory, temperature, power (single GPU)

    Args:
        profile: "cpu-intensive", "memory-intensive", or "balanced"
        num_samples: Number of time points to generate

    Returns:
        dict of metric_name -> list of values
    """
    timestamps = list(range(num_samples))

    # Base patterns with realistic noise
    if profile == "cpu-intensive":
        # High CPU, moderate memory
        cpu_base = np.linspace(30, 95, num_samples)
        cpu_noise = np.random.normal(0, 5, num_samples)
        cpu_percent = np.clip(cpu_base + cpu_noise, 20, 100).tolist()

        proc_cpu_base = np.linspace(25, 85, num_samples)
        proc_cpu_percent = np.clip(
            proc_cpu_base + np.random.normal(0, 4, num_samples), 15, 95
        ).tolist()

        memory_base = np.linspace(2.5, 4.2, num_samples)
        memory_used_gb = np.clip(memory_base + np.random.normal(0, 0.2, num_samples), 2, 8).tolist()

    elif profile == "memory-intensive":
        # Moderate CPU, high memory
        cpu_base = np.linspace(40, 65, num_samples)
        cpu_percent = np.clip(cpu_base + np.random.normal(0, 3, num_samples), 30, 80).tolist()

        proc_cpu_base = np.linspace(30, 55, num_samples)
        proc_cpu_percent = np.clip(
            proc_cpu_base + np.random.normal(0, 3, num_samples), 25, 70
        ).tolist()

        memory_base = np.linspace(3.0, 12.5, num_samples)
        memory_used_gb = np.clip(
            memory_base + np.random.normal(0, 0.5, num_samples), 2.5, 16
        ).tolist()

    else:  # balanced
        # Low CPU, low memory
        cpu_base = np.linspace(15, 35, num_samples)
        cpu_percent = np.clip(cpu_base + np.random.normal(0, 2, num_samples), 10, 50).tolist()

        proc_cpu_base = np.linspace(10, 25, num_samples)
        proc_cpu_percent = np.clip(
            proc_cpu_base + np.random.normal(0, 2, num_samples), 5, 40
        ).tolist()

        memory_base = np.linspace(1.8, 2.8, num_samples)
        memory_used_gb = np.clip(
            memory_base + np.random.normal(0, 0.1, num_samples), 1.5, 4
        ).tolist()

    # Process memory (RSS) in MB - correlates with memory_used_gb
    proc_memory_rss = (
        np.array(memory_used_gb) * 300 + np.random.normal(0, 50, num_samples)
    ).tolist()

    # Memory percent
    memory_percent = (
        np.array(memory_used_gb) * 6.25 + np.random.normal(0, 1, num_samples)
    ).tolist()

    # Disk I/O - slight variations
    disk_in = np.cumsum(np.random.uniform(0.5, 2.0, num_samples)).tolist()
    disk_out = np.cumsum(np.random.uniform(0.2, 1.0, num_samples)).tolist()

    # Network - minimal for local training
    network_sent = np.cumsum(np.random.uniform(0.01, 0.1, num_samples)).tolist()
    network_recv = np.cumsum(np.random.uniform(0.01, 0.1, num_samples)).tolist()

    # Thread count - varies slightly
    thread_count = (np.random.randint(8, 16, num_samples)).tolist()

    # GPU metrics (simulating single GPU at index 0)
    if profile == "cpu-intensive":
        gpu_util_base = np.linspace(10, 25, num_samples)
        gpu_util = np.clip(gpu_util_base + np.random.normal(0, 3, num_samples), 5, 40).tolist()

        gpu_mem_base = np.linspace(15, 30, num_samples)
        gpu_mem = np.clip(gpu_mem_base + np.random.normal(0, 2, num_samples), 10, 40).tolist()

        gpu_temp_base = np.linspace(45, 55, num_samples)
        gpu_temp = np.clip(gpu_temp_base + np.random.normal(0, 2, num_samples), 40, 65).tolist()

        gpu_power_base = np.linspace(50, 80, num_samples)
        gpu_power = np.clip(gpu_power_base + np.random.normal(0, 5, num_samples), 40, 100).tolist()

    elif profile == "memory-intensive":
        gpu_util_base = np.linspace(60, 85, num_samples)
        gpu_util = np.clip(gpu_util_base + np.random.normal(0, 5, num_samples), 50, 95).tolist()

        gpu_mem_base = np.linspace(70, 95, num_samples)
        gpu_mem = np.clip(gpu_mem_base + np.random.normal(0, 3, num_samples), 65, 98).tolist()

        gpu_temp_base = np.linspace(65, 78, num_samples)
        gpu_temp = np.clip(gpu_temp_base + np.random.normal(0, 2, num_samples), 60, 83).tolist()

        gpu_power_base = np.linspace(180, 250, num_samples)
        gpu_power = np.clip(
            gpu_power_base + np.random.normal(0, 10, num_samples), 160, 280
        ).tolist()

    else:  # balanced
        gpu_util_base = np.linspace(40, 65, num_samples)
        gpu_util = np.clip(gpu_util_base + np.random.normal(0, 4, num_samples), 30, 75).tolist()

        gpu_mem_base = np.linspace(35, 55, num_samples)
        gpu_mem = np.clip(gpu_mem_base + np.random.normal(0, 3, num_samples), 25, 65).tolist()

        gpu_temp_base = np.linspace(50, 65, num_samples)
        gpu_temp = np.clip(gpu_temp_base + np.random.normal(0, 2, num_samples), 45, 72).tolist()

        gpu_power_base = np.linspace(120, 160, num_samples)
        gpu_power = np.clip(gpu_power_base + np.random.normal(0, 8, num_samples), 100, 180).tolist()

    # GPU memory allocated bytes (correlates with gpu_mem %)
    gpu_mem_total_gb = 24
    gpu_mem_allocated_bytes = (np.array(gpu_mem) / 100 * gpu_mem_total_gb * 1024**3).tolist()

    # GPU power percent (assume 350W TDP)
    gpu_power_limit = 350
    gpu_power_percent = (np.array(gpu_power) / gpu_power_limit * 100).tolist()

    return {
        "timestamp": timestamps,
        "cpu_percent": [round(v, 2) for v in cpu_percent],
        "proc.cpu.percent": [round(v, 2) for v in proc_cpu_percent],
        "proc.cpu.threads": thread_count,
        "memory_percent": [round(v, 2) for v in memory_percent],
        "memory_used_gb": [round(v, 2) for v in memory_used_gb],
        "proc.memory.rssMB": [round(v, 2) for v in proc_memory_rss],
        "disk.in": [round(v, 2) for v in disk_in],
        "disk.out": [round(v, 2) for v in disk_out],
        "network.sent": [round(v, 2) for v in network_sent],
        "network.recv": [round(v, 2) for v in network_recv],
        "gpu.0.gpu": [round(v, 2) for v in gpu_util],
        "gpu.0.memory": [round(v, 2) for v in gpu_mem],
        "gpu.0.memoryAllocated": [round(v, 2) for v in gpu_mem],
        "gpu.0.memoryAllocatedBytes": [int(v) for v in gpu_mem_allocated_bytes],
        "gpu.0.temp": [round(v, 1) for v in gpu_temp],
        "gpu.0.powerWatts": [round(v, 2) for v in gpu_power],
        "gpu.0.powerPercent": [round(v, 2) for v in gpu_power_percent],
    }
