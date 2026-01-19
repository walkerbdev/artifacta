import time

import GPUtil
import psutil


class SystemMonitor:
    def __init__(self, profile="balanced"):
        self.profile = profile
        self.metrics_history = []

    def collect_metrics(self):
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        proc = psutil.Process()
        proc_cpu = proc.cpu_percent()
        proc_threads = proc.num_threads()

        # Memory metrics
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_used_gb = mem.used / (1024**3)
        proc_mem = proc.memory_info().rss / (1024**2)  # MB

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024**2)
        disk_write_mb = disk_io.write_bytes / (1024**2)

        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / (1024**2)
        net_recv_mb = net_io.bytes_recv / (1024**2)

        # GPU metrics (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                gpu_mem = gpu.memoryUtil * 100
                gpu_temp = gpu.temperature
                gpu_power = gpu.powerDraw
            else:
                gpu_util = gpu_mem = gpu_temp = gpu_power = 0
        except Exception:
            gpu_util = gpu_mem = gpu_temp = gpu_power = 0

        return {
            "cpu_percent": cpu_percent,
            "proc_cpu_percent": proc_cpu,
            "proc_threads": proc_threads,
            "memory_percent": memory_percent,
            "memory_used_gb": memory_used_gb,
            "proc_memory_mb": proc_mem,
            "disk_read_mb": disk_read_mb,
            "disk_write_mb": disk_write_mb,
            "net_sent_mb": net_sent_mb,
            "net_recv_mb": net_recv_mb,
            "gpu_util": gpu_util,
            "gpu_mem": gpu_mem,
            "gpu_temp": gpu_temp,
            "gpu_power": gpu_power,
        }

    def monitor(self, duration=60, interval=1):
        start_time = time.time()
        while time.time() - start_time < duration:
            metrics = self.collect_metrics()
            metrics["timestamp"] = time.time() - start_time
            self.metrics_history.append(metrics)
            time.sleep(interval)

        return self.metrics_history
