"""
OMNI-SYSTEM ULTIMATE - Monitoring Dashboard
Real-time system monitoring and performance analytics.
"""

import asyncio
import json
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class MonitoringDashboard:
    """
    Ultimate Monitoring Dashboard with real-time analytics.
    Comprehensive system monitoring and alerting.
    """

    def __init__(self, base_path: str = "/Users/thealchemist/OMNI-SYSTEM-ULTIMATE"):
        self.base_path = Path(base_path)
        self.metrics = {}
        self.alerts = []
        self.thresholds = {}
        self.monitoring_active = False
        self.logger = logging.getLogger("Monitoring-Dashboard")

    async def initialize(self) -> bool:
        """Initialize monitoring dashboard."""
        try:
            # Setup metrics collection
            await self._setup_metrics_collection()

            # Initialize alerting system
            await self._init_alerting_system()

            # Start monitoring thread
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            self.logger.info("Monitoring Dashboard initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Monitoring Dashboard initialization failed: {e}")
            return False

    async def _setup_metrics_collection(self):
        """Setup metrics collection."""
        self.metrics = {
            "cpu": {"usage": 0, "cores": psutil.cpu_count()},
            "memory": {"usage": 0, "total": psutil.virtual_memory().total},
            "disk": {"usage": 0, "total": psutil.disk_usage('/').total},
            "network": {"connections": 0, "bytes_sent": 0, "bytes_recv": 0},
            "processes": {"count": 0}
        }

    async def _init_alerting_system(self):
        """Initialize alerting system."""
        self.thresholds = {
            "cpu_usage": 80,
            "memory_usage": 85,
            "disk_usage": 90
        }

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                self._collect_metrics()

                # Check thresholds
                self._check_thresholds()

                # Sleep for 5 seconds
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")

    def _collect_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            self.metrics["cpu"]["usage"] = psutil.cpu_percent(interval=1)

            # Memory metrics
            mem = psutil.virtual_memory()
            self.metrics["memory"]["usage"] = mem.percent

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics["disk"]["usage"] = disk.percent

            # Network metrics
            net = psutil.net_io_counters()
            self.metrics["network"]["bytes_sent"] = net.bytes_sent
            self.metrics["network"]["bytes_recv"] = net.bytes_recv
            self.metrics["network"]["connections"] = len(psutil.net_connections())

            # Process metrics
            self.metrics["processes"]["count"] = len(psutil.pids())

        except Exception as e:
            self.logger.error(f"Metrics collection error: {e}")

    def _check_thresholds(self):
        """Check if metrics exceed thresholds."""
        alerts = []

        if self.metrics["cpu"]["usage"] > self.thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {self.metrics['cpu']['usage']}%")

        if self.metrics["memory"]["usage"] > self.thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {self.metrics['memory']['usage']}%")

        if self.metrics["disk"]["usage"] > self.thresholds["disk_usage"]:
            alerts.append(f"High disk usage: {self.metrics['disk']['usage']}%")

        if alerts:
            self.alerts.extend(alerts)
            self.logger.warning("Alerts triggered: " + ", ".join(alerts))

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        return {
            "metrics": self.metrics,
            "alerts": self.alerts[-10:],  # Last 10 alerts
            "thresholds": self.thresholds,
            "monitoring_active": self.monitoring_active
        }

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()

    async def health_check(self) -> bool:
        """Health check for monitoring dashboard."""
        try:
            return self.monitoring_active and len(self.metrics) > 0
        except:
            return False

# Global monitoring dashboard instance
monitoring_dashboard = None

async def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get or create monitoring dashboard."""
    global monitoring_dashboard
    if not monitoring_dashboard:
        monitoring_dashboard = MonitoringDashboard()
        await monitoring_dashboard.initialize()
    return monitoring_dashboard
