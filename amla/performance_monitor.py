"""
Performance Monitoring and Metrics Collection
Tracks response times, cache hit rates, and system health
"""

import time
from collections import defaultdict, deque
from typing import Dict, List
import json
from datetime import datetime
import statistics


class PerformanceMonitor:
    """Track system performance metrics"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        # Response time tracking
        self.response_times = defaultdict(lambda: deque(maxlen=window_size))
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Feature extraction breakdown
        self.layer_timings = defaultdict(lambda: deque(maxlen=window_size))
        
        # API endpoint metrics
        self.endpoint_calls = defaultdict(int)
        self.endpoint_errors = defaultdict(int)
        
        # Historical logs
        self.logs = deque(maxlen=1000)
    
    def start_timer(self, operation_name: str) -> float:
        """Start timing an operation"""
        return time.time()
    
    def end_timer(self, operation_name: str, start_time: float, metadata: Dict = None):
        """End timing and record metric"""
        elapsed = time.time() - start_time
        self.response_times[operation_name].append(elapsed)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "elapsed_ms": round(elapsed * 1000, 2),
            "metadata": metadata or {}
        }
        self.logs.append(log_entry)
    
    def record_layer_timing(self, layer_name: str, elapsed: float):
        """Record feature extraction layer timing"""
        self.layer_timings[layer_name].append(elapsed)
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def record_api_call(self, endpoint: str, success: bool = True):
        """Record API call"""
        self.endpoint_calls[endpoint] += 1
        if not success:
            self.endpoint_errors[endpoint] += 1
    
    def get_avg_response_time(self, operation: str) -> float:
        """Get average response time for operation"""
        times = self.response_times.get(operation, [])
        return statistics.mean(times) if times else 0.0
    
    def get_p95_response_time(self, operation: str) -> float:
        """Get 95th percentile response time"""
        times = self.response_times.get(operation, [])
        if len(times) < 2:
            return max(times) if times else 0.0
        return sorted(times)[int(len(times) * 0.95)]
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(hit_rate, 2),
            "total_requests": total
        }
    
    def get_layer_stats(self) -> Dict:
        """Get feature extraction layer performance"""
        stats = {}
        for layer_name, timings in self.layer_timings.items():
            if timings:
                stats[layer_name] = {
                    "avg_time_ms": round(statistics.mean(timings) * 1000, 2),
                    "min_time_ms": round(min(timings) * 1000, 2),
                    "max_time_ms": round(max(timings) * 1000, 2),
                    "samples": len(timings)
                }
        return stats
    
    def get_api_stats(self) -> Dict:
        """Get API endpoint statistics"""
        stats = {}
        for endpoint, calls in self.endpoint_calls.items():
            errors = self.endpoint_errors.get(endpoint, 0)
            success_rate = ((calls - errors) / calls * 100) if calls > 0 else 100.0
            
            avg_time = self.get_avg_response_time(f"api_{endpoint}")
            p95_time = self.get_p95_response_time(f"api_{endpoint}")
            
            stats[endpoint] = {
                "calls": calls,
                "errors": errors,
                "success_rate": round(success_rate, 2),
                "avg_response_time_ms": round(avg_time * 1000, 2),
                "p95_response_time_ms": round(p95_time * 1000, 2)
            }
        return stats
    
    def get_full_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cache_statistics": self.get_cache_stats(),
            "layer_performance": self.get_layer_stats(),
            "api_statistics": self.get_api_stats(),
            "system_operations": {
                op: {
                    "avg_ms": round(self.get_avg_response_time(op) * 1000, 2),
                    "p95_ms": round(self.get_p95_response_time(op) * 1000, 2),
                    "samples": len(self.response_times[op])
                }
                for op in self.response_times.keys()
            }
        }
    
    def get_recent_logs(self, n: int = 50) -> List[Dict]:
        """Get recent log entries"""
        return list(reversed(list(self.logs)[-n:]))
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.response_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.layer_timings.clear()
        self.endpoint_calls.clear()
        self.endpoint_errors.clear()
        self.logs.clear()


# Global performance monitor instance
_monitor = None


def get_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


def reset_monitor():
    """Reset global monitor"""
    global _monitor
    _monitor = PerformanceMonitor()
