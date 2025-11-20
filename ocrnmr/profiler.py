"""Performance profiler for OCR NMR."""

import time
import json
import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List
import threading

logger = logging.getLogger(__name__)


class Profiler:
    """Singleton profiler to track performance metrics."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Profiler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.enabled = False
        self.output_file = None
        self.metrics = defaultdict(list)
        self.timers = {}
        self.start_time = None
        self._initialized = True
        self._thread_local = threading.local()

    def enable(self, output_file: str):
        """Enable profiling and set output file."""
        self.enabled = True
        self.output_file = output_file
        self.start_time = time.time()
        logger.info(f"Profiling enabled. Output will be written to {output_file}")

    def start_timer(self, key: str) -> None:
        """Start a timer for a given key."""
        if not self.enabled:
            return
        
        # Use thread-local storage to handle potential threading
        if not hasattr(self._thread_local, 'timers'):
            self._thread_local.timers = {}
            
        self._thread_local.timers[key] = time.time()

    def stop_timer(self, key: str) -> None:
        """Stop a timer and record duration."""
        if not self.enabled:
            return
            
        if not hasattr(self._thread_local, 'timers') or key not in self._thread_local.timers:
            return
            
        start_t = self._thread_local.timers.pop(key)
        duration = time.time() - start_t
        self.add_metric(key, duration)

    def add_metric(self, key: str, value: Any) -> None:
        """Add a metric value."""
        if not self.enabled:
            return
            
        # We need to be thread-safe when writing to the shared metrics dict
        with self._lock:
            self.metrics[key].append(value)

    def save_results(self) -> None:
        """Save profiling results to JSON file."""
        if not self.enabled or not self.output_file:
            return

        duration = time.time() - self.start_time
        
        summary = {
            "total_duration": duration,
            "metrics": {}
        }

        with self._lock:
            for key, values in self.metrics.items():
                if not values:
                    continue
                
                # Assuming numeric values for now
                try:
                    avg_val = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    total_val = sum(values)
                    
                    summary["metrics"][key] = {
                        "count": len(values),
                        "total": total_val,
                        "avg": avg_val,
                        "min": min_val,
                        "max": max_val
                    }
                except TypeError:
                    # Handle non-numeric metrics if any
                    summary["metrics"][key] = {
                        "count": len(values),
                        "values": values
                    }

        try:
            import json
            with open(self.output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Profiling results saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling results: {e}")

# Global instance
profiler = Profiler()

