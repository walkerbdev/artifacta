"""System monitoring via background thread for automatic metrics capture.

This module implements a background monitoring daemon that periodically captures
system-level metrics (CPU, memory, disk, network, GPU) and emits them to the
tracking server. The design follows a non-invasive pattern: monitoring runs in
a daemon thread and fails silently to avoid impacting the main training loop.

Architecture:
    - Background Thread: Runs in daemon mode (won't block Python exit)
    - Periodic Sampling: Configurable interval (default 30 seconds)
    - Graceful Degradation: All metric capture wrapped in try/except
    - GPU Support: Optional via pynvml (NVIDIA GPUs only)
    - Process-Level: Tracks both system-wide and current process metrics

Monitoring Strategy:
    The monitor captures two categories of metrics:

    1. System-wide metrics (all processes):
       - CPU percent, memory usage, disk I/O, network I/O
       - Provides context about overall system load

    2. Process-specific metrics (current Python process):
       - CPU percent, thread count, RSS memory, memory percent
       - Helps identify if training is bottlenecked by system resources

GPU Monitoring Algorithm:
    1. Try to import pynvml and initialize NVML at __init__
    2. If successful, set has_gpu=True and store pynvml reference
    3. In _capture_metrics(), iterate over all GPU devices
    4. For each GPU, capture:
       - Utilization (GPU compute %, memory %)
       - Memory (used bytes, allocated %)
       - Temperature (Celsius)
       - Power (watts, percent of limit)
       - Clock speeds (SM, memory, graphics in MHz)
       - Memory errors (corrected/uncorrected ECC errors)
       - Encoder utilization (for video encoding workloads)
    5. Wrap each metric in try/except (some GPUs don't support all metrics)

Background Thread Lifecycle:
    1. start() -> Create daemon thread, set running=True, start loop
    2. Loop: capture metrics, emit via HTTP, sleep for interval
    3. stop() -> Set running=False, join thread with 5s timeout
    4. Cleanup: Call nvmlShutdown() if GPU monitoring was enabled

Performance Considerations:
    - psutil.oneshot() context manager batches system calls for efficiency
    - CPU percent uses interval=1 for accuracy (blocks for 1 second)
    - Process CPU uses interval=None (non-blocking, uses cached value)
    - All exceptions suppressed to avoid crashing the monitoring thread
    - Daemon thread ensures no zombie threads after program exit

Why daemon thread:
    Daemon threads don't prevent the Python interpreter from exiting.
    If training finishes, the main thread exits, and the monitor thread
    is automatically terminated. This avoids requiring users to explicitly
    call stop() in all cases.
"""

import contextlib
import threading
import time

import psutil


class SystemMonitor:
    """Background thread for logging system metrics."""

    def __init__(self, interval=30, http_emitter=None):
        """Initialize SystemMonitor.

        Args:
            interval: Seconds between metric captures (default: 30).
            http_emitter: HTTPEmitter to emit metrics to API.
        """
        self.http_emitter = http_emitter
        self.interval = interval
        self.running = False
        self.thread = None
        self.process = psutil.Process()  # Current process for process-level metrics

        # Try to import GPU monitoring
        self.has_gpu = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.has_gpu = True
            self.pynvml = pynvml
        except Exception:
            pass

    def start(self):
        """Start monitoring in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        # Cleanup GPU if initialized
        if self.has_gpu:
            with contextlib.suppress(Exception):
                self.pynvml.nvmlShutdown()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metrics = self._capture_metrics()
                # Emit to API Gateway (single source of truth)
                if self.http_emitter:
                    self.http_emitter.emit_metrics(metrics)
            except Exception:
                # Silently continue on errors
                pass

            time.sleep(self.interval)

    def _capture_metrics(self):
        """Capture current system metrics."""
        metrics = {
            "_source": "system",
            "timestamp": int(time.time() * 1000),  # Unix timestamp in milliseconds
        }

        # System-wide CPU
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)

        # Process-level CPU metrics
        try:
            with self.process.oneshot():
                metrics["proc.cpu.percent"] = round(self.process.cpu_percent(interval=None), 2)
                metrics["proc.cpu.threads"] = self.process.num_threads()
        except Exception:
            pass

        # System-wide memory
        vm = psutil.virtual_memory()
        metrics["memory_percent"] = round(vm.percent, 2)
        metrics["memory_used_gb"] = round(vm.used / 1024**3, 2)

        # Process-level memory metrics
        try:
            with self.process.oneshot():
                mem_info = self.process.memory_info()
                metrics["proc.memory.rssMB"] = round(mem_info.rss / 1024**2, 2)
                metrics["proc.memory.percent"] = round(self.process.memory_percent(), 2)
                metrics["proc.memory.availableMB"] = round(vm.available / 1024**2, 2)
        except Exception:
            pass

        # Disk I/O
        try:
            disk = psutil.disk_io_counters()
            if disk:
                metrics["disk.in"] = round(disk.read_bytes / 1024**2, 2)  # MB read
                metrics["disk.out"] = round(disk.write_bytes / 1024**2, 2)  # MB written
        except Exception:
            pass

        # Disk usage (check common mount points)
        disk_paths = ["/"]  # Default to root, can be configured
        for path in disk_paths:
            try:
                usage = psutil.disk_usage(path)
                # Sanitize path for metric name (/ becomes root)
                path_key = "root" if path == "/" else path.replace("/", "_").strip("_")
                metrics[f"disk.{path_key}.usagePercent"] = round(usage.percent, 2)
                metrics[f"disk.{path_key}.usageGB"] = round(usage.used / 1024**3, 2)
            except Exception:
                pass

        # Network I/O
        try:
            net = psutil.net_io_counters()
            if net:
                metrics["network.sent"] = round(net.bytes_sent / 1024**2, 2)  # MB sent
                metrics["network.recv"] = round(net.bytes_recv / 1024**2, 2)  # MB received
        except Exception:
            pass

        # GPU metrics
        if self.has_gpu:
            try:
                device_count = self.pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)

                    # Basic utilization and memory
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = self.pynvml.nvmlDeviceGetTemperature(handle, 0)

                    metrics[f"gpu.{i}.gpu"] = util.gpu
                    metrics[f"gpu.{i}.memory"] = round((mem.used / mem.total) * 100, 2)
                    metrics[f"gpu.{i}.memoryAllocated"] = round((mem.used / mem.total) * 100, 2)
                    metrics[f"gpu.{i}.memoryAllocatedBytes"] = mem.used
                    metrics[f"gpu.{i}.temp"] = temp

                    # Power usage
                    try:
                        power_mw = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                        power_watts = power_mw / 1000.0
                        metrics[f"gpu.{i}.powerWatts"] = round(power_watts, 2)

                        # Get power limit for percentage calculation
                        try:
                            power_limit_mw = self.pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                            power_limit_watts = power_limit_mw / 1000.0
                            if power_limit_watts > 0:
                                metrics[f"gpu.{i}.powerPercent"] = round(
                                    (power_watts / power_limit_watts) * 100, 2
                                )
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # Clock speeds
                    try:
                        sm_clock = self.pynvml.nvmlDeviceGetClockInfo(
                            handle, self.pynvml.NVML_CLOCK_SM
                        )
                        metrics[f"gpu.{i}.smClock"] = sm_clock
                    except Exception:
                        pass

                    try:
                        mem_clock = self.pynvml.nvmlDeviceGetClockInfo(
                            handle, self.pynvml.NVML_CLOCK_MEM
                        )
                        metrics[f"gpu.{i}.memoryClock"] = mem_clock
                    except Exception:
                        pass

                    try:
                        graphics_clock = self.pynvml.nvmlDeviceGetClockInfo(
                            handle, self.pynvml.NVML_CLOCK_GRAPHICS
                        )
                        metrics[f"gpu.{i}.graphicsClock"] = graphics_clock
                    except Exception:
                        pass

                    # Memory errors
                    try:
                        corrected_errors = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                            handle,
                            self.pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED,
                            self.pynvml.NVML_VOLATILE_ECC,
                            self.pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                        )
                        metrics[f"gpu.{i}.correctedMemoryErrors"] = corrected_errors
                    except Exception:
                        pass

                    try:
                        uncorrected_errors = self.pynvml.nvmlDeviceGetMemoryErrorCounter(
                            handle,
                            self.pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                            self.pynvml.NVML_VOLATILE_ECC,
                            self.pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                        )
                        metrics[f"gpu.{i}.unCorrectedMemoryErrors"] = uncorrected_errors
                    except Exception:
                        pass

                    # Encoder utilization
                    try:
                        encoder_util, _ = self.pynvml.nvmlDeviceGetEncoderUtilization(handle)
                        metrics[f"gpu.{i}.encoderUtilization"] = encoder_util
                    except Exception:
                        pass

            except Exception:
                pass

        return metrics
