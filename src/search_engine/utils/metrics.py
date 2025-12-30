"""
Metrics and monitoring for the search engine.
Tracks pipeline health, query performance, and data quality.
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    timestamp: str
    result_count: int
    latency_ms: float
    cache_hit: bool


@dataclass
class SystemMetrics:
    """Overall system metrics."""
    total_queries: int
    total_documents: int
    cache_hit_rate: float
    avg_query_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    index_size_bytes: int
    vocabulary_size: int


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring.
    """
    
    def __init__(self):
        self.query_history: List[QueryMetrics] = []
        self.max_history = 1000  # Keep last 1000 queries
    
    def record_query(
        self,
        query: str,
        result_count: int,
        latency_ms: float,
        cache_hit: bool = False
    ):
        """Record a query execution."""
        metrics = QueryMetrics(
            query=query,
            timestamp=datetime.now().isoformat(),
            result_count=result_count,
            latency_ms=latency_ms,
            cache_hit=cache_hit
        )
        self.query_history.append(metrics)
        
        # Keep only recent history
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]
    
    def get_system_metrics(
        self,
        total_documents: int,
        index_size_bytes: int,
        vocabulary_size: int,
        cache_hit_rate: float
    ) -> SystemMetrics:
        """Calculate aggregate system metrics."""
        if not self.query_history:
            return SystemMetrics(
                total_queries=0,
                total_documents=total_documents,
                cache_hit_rate=cache_hit_rate,
                avg_query_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                index_size_bytes=index_size_bytes,
                vocabulary_size=vocabulary_size
            )
        
        latencies = [q.latency_ms for q in self.query_history]
        latencies_sorted = sorted(latencies)
        
        n = len(latencies_sorted)
        p50 = latencies_sorted[n // 2] if n > 0 else 0.0
        p95 = latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1] if n > 0 else 0.0
        
        return SystemMetrics(
            total_queries=len(self.query_history),
            total_documents=total_documents,
            cache_hit_rate=cache_hit_rate,
            avg_query_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            index_size_bytes=index_size_bytes,
            vocabulary_size=vocabulary_size
        )
    
    def get_recent_queries(self, limit: int = 20) -> List[QueryMetrics]:
        """Get most recent queries."""
        return self.query_history[-limit:]
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query statistics."""
        if not self.query_history:
            return {
                'total': 0,
                'avg_latency_ms': 0.0,
                'cache_hits': 0,
                'cache_misses': 0
            }
        
        cache_hits = sum(1 for q in self.query_history if q.cache_hit)
        cache_misses = len(self.query_history) - cache_hits
        
        return {
            'total': len(self.query_history),
            'avg_latency_ms': sum(q.latency_ms for q in self.query_history) / len(self.query_history),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / len(self.query_history) if self.query_history else 0.0
        }
    
    def export_metrics(self, filepath: str = 'metrics.json'):
        """Export metrics to JSON file."""
        data = {
            'system_metrics': asdict(self.get_system_metrics(0, 0, 0, 0.0)),
            'query_stats': self.get_query_stats(),
            'recent_queries': [asdict(q) for q in self.get_recent_queries(50)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

