//! SSE Monitoring and Diagnostics System
//!
//! This module provides comprehensive monitoring and diagnostics functionality
//! for SSE endpoints used in agent communication. It includes real-time
//! monitoring, health checks, performance metrics, and diagnostic tools.
//!
//! ## Features
//!
//! - Real-time SSE endpoint monitoring
//! - Health status tracking and alerts
//! - Performance metrics collection
//! - Connection diagnostics and troubleshooting
//! - SSE event logging and analysis
//! - Automated recovery mechanisms
//! - Integration with port management system

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, Instant};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::time::{interval, sleep};
use tracing::{info, warn, error, debug};
use crate::port_allocator::PortAllocator;
use crate::sse_validator::SSEValidator;
use crate::sse_manager::SSEManager;

/// SSE monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEMonitoringConfig {
    /// Monitoring interval in seconds
    pub monitor_interval_secs: u64,
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    /// Maximum number of failed attempts before alert
    pub max_failed_attempts: u32,
    /// Timeout for health checks in seconds
    pub health_check_timeout_secs: u64,
    /// Whether to enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Whether to enable detailed logging
    pub enable_detailed_logging: bool,
    /// Performance metrics collection
    pub enable_performance_metrics: bool,
}

impl Default for SSEMonitoringConfig {
    fn default() -> Self {
        Self {
            monitor_interval_secs: 5,
            health_check_interval_secs: 30,
            max_failed_attempts: 3,
            health_check_timeout_secs: 10,
            enable_auto_recovery: true,
            enable_detailed_logging: true,
            enable_performance_metrics: true,
        }
    }
}

/// SSE endpoint health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SSEEndpointHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl SSEEndpointHealth {
    /// Check if the endpoint is in a healthy state
    pub fn is_healthy(&self) -> bool {
        matches!(self, SSEEndpointHealth::Healthy)
    }

    /// Get the health status as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            SSEEndpointHealth::Healthy => "healthy",
            SSEEndpointHealth::Degraded => "degraded",
            SSEEndpointHealth::Unhealthy => "unhealthy",
            SSEEndpointHealth::Unknown => "unknown",
        }
    }
}

/// SSE endpoint metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEndpointMetrics {
    /// Endpoint URL
    pub endpoint_url: String,
    /// Current health status
    pub health_status: SSEEndpointHealth,
    /// Last successful connection time
    pub last_success_time: Option<SystemTime>,
    /// Number of successful connections
    pub successful_connections: u64,
    /// Number of failed connections
    pub failed_connections: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Total bytes sent
    pub total_bytes_sent: u64,
    /// Total bytes received
    pub total_bytes_received: u64,
    /// Number of events processed
    pub events_processed: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Error rate (errors per minute)
    pub error_rate_per_minute: f64,
    /// Last error message
    pub last_error: Option<String>,
}

impl Default for SSEEndpointMetrics {
    fn default() -> Self {
        Self {
            endpoint_url: String::new(),
            health_status: SSEEndpointHealth::Unknown,
            last_success_time: None,
            successful_connections: 0,
            failed_connections: 0,
            avg_response_time_ms: 0.0,
            total_bytes_sent: 0,
            total_bytes_received: 0,
            events_processed: 0,
            uptime_seconds: 0,
            error_rate_per_minute: 0.0,
            last_error: None,
        }
    }
}

/// SSE event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type (e.g., "message", "error", "connect", "disconnect")
    pub event_type: String,
    /// Event data
    pub data: String,
    /// Source endpoint
    pub source: String,
    /// Event severity
    pub severity: SSEEventSeverity,
}

/// SSE event severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SSEEventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl SSEEventSeverity {
    /// Get the severity as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            SSEEventSeverity::Info => "info",
            SSEEventSeverity::Warning => "warning",
            SSEEventSeverity::Error => "error",
            SSEEventSeverity::Critical => "critical",
        }
    }
}

/// SSE monitoring system
pub struct SSEMonitor {
    config: SSEMonitoringConfig,
    port_allocator: Arc<PortAllocator>,
    sse_validator: Arc<SSEValidator>,
    sse_manager: Arc<SSEManager>,
    
    /// Monitored endpoints
    monitored_endpoints: Arc<Mutex<HashMap<String, SSEEndpointMetrics>>>,
    
    /// Event log
    event_log: Arc<Mutex<Vec<SSEEvent>>>,
    
    /// Health check handles
    health_check_handles: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
    
    /// Performance metrics
    performance_metrics: Arc<Mutex<HashMap<String, SSEPerformanceMetrics>>>,
}

/// SSE performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEPerformanceMetrics {
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Connection success rate
    pub connection_success_rate: f64,
    /// Data throughput (bytes per second)
    pub data_throughput_bps: f64,
    /// Error rate
    pub error_rate: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

impl Default for SSEPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_response_time_ms: 0.0,
            connection_success_rate: 0.0,
            data_throughput_bps: 0.0,
            error_rate: 0.0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
        }
    }
}

impl SSEMonitor {
    /// Create a new SSE monitoring system
    pub fn new(
        config: SSEMonitoringConfig,
        port_allocator: Arc<PortAllocator>,
        sse_validator: Arc<SSEValidator>,
        sse_manager: Arc<SSEManager>,
    ) -> Self {
        Self {
            config,
            port_allocator,
            sse_validator,
            sse_manager,
            monitored_endpoints: Arc::new(Mutex::new(HashMap::new())),
            event_log: Arc::new(Mutex::new(Vec::new())),
            health_check_handles: Arc::new(Mutex::new(HashMap::new())),
            performance_metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start monitoring an SSE endpoint
    pub async fn start_monitoring(&self, endpoint_url: &str) -> Result<()> {
        info!("Starting SSE monitoring for endpoint: {}", endpoint_url);
        
        // Initialize metrics for the endpoint
        let metrics = SSEEndpointMetrics {
            endpoint_url: endpoint_url.to_string(),
            health_status: SSEEndpointHealth::Unknown,
            ..Default::default()
        };
        
        {
            let mut endpoints = self.monitored_endpoints.lock().unwrap();
            endpoints.insert(endpoint_url.to_string(), metrics);
        }
        
        // Start health checking
        self.start_health_check(endpoint_url).await?;
        
        // Start performance monitoring
        if self.config.enable_performance_metrics {
            self.start_performance_monitoring(endpoint_url).await?;
        }
        
        // Log monitoring start
        self.log_event(SSEEvent {
            timestamp: SystemTime::now(),
            event_type: "monitor".to_string(),
            data: format!("Started monitoring for endpoint: {}", endpoint_url),
            source: endpoint_url.to_string(),
            severity: SSEEventSeverity::Info,
        });
        
        info!("SSE monitoring started for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Stop monitoring an SSE endpoint
    pub async fn stop_monitoring(&self, endpoint_url: &str) -> Result<()> {
        info!("Stopping SSE monitoring for endpoint: {}", endpoint_url);
        
        // Stop health checking
        self.stop_health_check(endpoint_url).await?;
        
        // Stop performance monitoring
        if self.config.enable_performance_metrics {
            self.stop_performance_monitoring(endpoint_url).await?;
        }
        
        // Remove from monitored endpoints
        {
            let mut endpoints = self.monitored_endpoints.lock().unwrap();
            endpoints.remove(endpoint_url);
        }
        
        // Log monitoring stop
        self.log_event(SSEEvent {
            timestamp: SystemTime::now(),
            event_type: "monitor".to_string(),
            data: format!("Stopped monitoring for endpoint: {}", endpoint_url),
            source: endpoint_url.to_string(),
            severity: SSEEventSeverity::Info,
        });
        
        info!("SSE monitoring stopped for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Get metrics for a monitored endpoint
    pub fn get_endpoint_metrics(&self, endpoint_url: &str) -> Option<SSEEndpointMetrics> {
        let endpoints = self.monitored_endpoints.lock().unwrap();
        endpoints.get(endpoint_url).cloned()
    }

    /// Get all monitored endpoints
    pub fn get_monitored_endpoints(&self) -> Vec<String> {
        let endpoints = self.monitored_endpoints.lock().unwrap();
        endpoints.keys().cloned().collect()
    }

    /// Update endpoint metrics
    pub fn update_metrics(&self, endpoint_url: &str, update_fn: impl FnOnce(&mut SSEEndpointMetrics)) -> Result<()> {
        let mut endpoints = self.monitored_endpoints.lock().unwrap();
        if let Some(metrics) = endpoints.get_mut(endpoint_url) {
            update_fn(metrics);
            info!("Updated metrics for endpoint: {}", endpoint_url);
        } else {
            warn!("Endpoint {} not found in monitored endpoints", endpoint_url);
        }
        Ok(())
    }

    /// Get performance metrics for an endpoint
    pub fn get_performance_metrics(&self, endpoint_url: &str) -> Option<SSEPerformanceMetrics> {
        let metrics = self.performance_metrics.lock().unwrap();
        metrics.get(endpoint_url).cloned()
    }

    /// Get recent events for an endpoint
    pub fn get_recent_events(&self, endpoint_url: &str, limit: Option<usize>) -> Vec<SSEEvent> {
        let event_log = self.event_log.lock().unwrap();
        let events = event_log.iter()
            .filter(|e| e.source == endpoint_url)
            .rev()
            .take(limit.unwrap_or(100));
        
        events
    }

    /// Get system-wide monitoring statistics
    pub fn get_system_stats(&self) -> SSEMonitoringStats {
        let endpoints = self.monitored_endpoints.lock().unwrap();
        let event_log = self.event_log.lock().unwrap();
        let performance_metrics = self.performance_metrics.lock().unwrap();
        
        let total_endpoints = endpoints.len();
        let healthy_endpoints = endpoints.values()
            .filter(|m| m.health_status.is_healthy())
            .count();
        
        let total_events = event_log.len();
        let error_events = event_log.iter()
            .filter(|e| matches!(e.severity, SSEEventSeverity::Error | SSEEventSeverity::Critical))
            .count();
        
        SSEMonitoringStats {
            total_endpoints,
            healthy_endpoints,
            total_events,
            error_events,
            avg_response_time: performance_metrics.values()
                .map(|m| m.avg_response_time_ms)
                .sum_or(0.0) / performance_metrics.len().max(1) as f64,
            avg_throughput: performance_metrics.values()
                .map(|m| m.data_throughput_bps)
                .sum_or(0.0) / performance_metrics.len().max(1) as f64,
        }
    }

    /// Start health checking for an endpoint
    async fn start_health_check(&self, endpoint_url: &str) -> Result<()> {
        info!("Starting health checks for endpoint: {}", endpoint_url);
        
        let endpoint_url = endpoint_url.to_string();
        let sse_validator = self.sse_validator.as_ref();
        let config = self.config.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.health_check_interval_secs));
            
            loop {
                interval.tick().await;
                
                // Perform health check
                match sse_validator.validate_endpoint(&endpoint_url).await {
                    Ok(result) => {
                        let new_health = if result.is_valid {
                            SSEEndpointHealth::Healthy
                        } else {
                            SSEEndpointHealth::Unhealthy
                        };
                        
                        // Update metrics
                        let mut endpoints = self.monitored_endpoints.lock().unwrap();
                        if let Some(metrics) = endpoints.get_mut(&endpoint_url) {
                            metrics.health_status = new_health;
                            if result.is_valid {
                                metrics.last_success_time = Some(SystemTime::now());
                                metrics.successful_connections += 1;
                            } else {
                                metrics.failed_connections += 1;
                                metrics.last_error = Some(format!("Health check failed: {:?}", result.errors));
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Health check failed for endpoint {}: {}", endpoint_url, e);
                        
                        // Update metrics
                        let mut endpoints = self.monitored_endpoints.lock().unwrap();
                        if let Some(metrics) = endpoints.get_mut(&endpoint_url) {
                            metrics.health_status = SSEEndpointHealth::Unhealthy;
                            metrics.failed_connections += 1;
                            metrics.last_error = Some(format!("Health check error: {}", e));
                        }
                    }
                }
            }
        });
        
        {
            let mut handles = self.health_check_handles.lock().unwrap();
            handles.insert(endpoint_url.to_string(), handle);
        }
        
        info!("Health checks started for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Stop health checking for an endpoint
    async fn stop_health_check(&self, endpoint_url: &str) -> Result<()> {
        info!("Stopping health checks for endpoint: {}", endpoint_url);
        
        {
            let mut handles = self.health_check_handles.lock().unwrap();
            if let Some(handle) = handles.remove(&endpoint_url.to_string()) {
                handle.abort();
            }
        }
        
        info!("Health checks stopped for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Start performance monitoring for an endpoint
    async fn start_performance_monitoring(&self, endpoint_url: &str) -> Result<()> {
        info!("Starting performance monitoring for endpoint: {}", endpoint_url);
        
        let endpoint_url = endpoint_url.to_string();
        let config = self.config.clone();
        
        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.monitor_interval_secs));
            let mut start_time = Instant::now();
            
            loop {
                interval.tick().await;
                let elapsed = start_time.elapsed();
                
                // Simulate performance metrics calculation
                let mut metrics = self.performance_metrics.lock().unwrap();
                if let Some(current_metrics) = metrics.get_mut(&endpoint_url) {
                    // Update response time (simulated)
                    let new_response_time = elapsed.as_millis() as f64;
                    current_metrics.avg_response_time_ms = 
                        (current_metrics.avg_response_time_ms * 0.9) + (new_response_time * 0.1);
                    
                    // Update throughput (simulated)
                    let bytes_per_second = 1024.0; // Simulated
                    current_metrics.data_throughput_bps = 
                        (current_metrics.data_throughput_bps * 0.9) + (bytes_per_second * 0.1);
                    
                    // Update success rate
                    let success_rate = if elapsed.as_secs() > 0 {
                        current_metrics.successful_connections as f64 / elapsed.as_secs() as f64
                    } else {
                        current_metrics.successful_connections
                    };
                    
                    current_metrics.connection_success_rate = 
                        (current_metrics.connection_success_rate * 0.9) + (success_rate * 0.1);
                    
                    // Update error rate
                    let error_rate = if elapsed.as_secs() > 0 {
                        (current_metrics.failed_connections as f64 / elapsed.as_secs() as f64) - success_rate
                    } else {
                        current_metrics.error_rate
                    };
                    
                    current_metrics.error_rate = 
                        (current_metrics.error_rate * 0.9) + (error_rate * 0.1);
                    
                    // Update CPU and memory usage (simulated)
                    current_metrics.cpu_usage_percent = 
                        (current_metrics.cpu_usage_percent * 0.9) + (std::cmp::max(50.0, elapsed.as_secs() as f64 / 60.0) * 10.0);
                    
                    current_metrics.memory_usage_mb = 
                        (current_metrics.memory_usage_mb * 0.9) + (std::cmp::max(10.0, elapsed.as_secs() as f64 / 60.0) * 5.0);
                }
            }
        });
        
        {
            let mut handles = self.performance_metrics.lock().unwrap();
            handles.insert(endpoint_url.to_string(), handle);
        }
        
        info!("Performance monitoring started for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Stop performance monitoring for an endpoint
    async fn stop_performance_monitoring(&self, endpoint_url: &str) -> Result<()> {
        info!("Stopping performance monitoring for endpoint: {}", endpoint_url);
        
        {
            let mut handles = self.performance_metrics.lock().unwrap();
            if let Some(handle) = handles.remove(&endpoint_url.to_string()) {
                handle.abort();
            }
        }
        
        info!("Performance monitoring stopped for endpoint: {}", endpoint_url);
        Ok(())
    }

    /// Log an SSE event
    pub fn log_event(&self, event: SSEEvent) {
        let mut event_log = self.event_log.lock().unwrap();
        
        // Limit event log size to prevent memory issues
        if event_log.len() >= 10000 {
            event_log.remove(0);
            warn!("Event log cleared due to size limit");
        }
        
        event_log.push(event);
        
        if self.config.enable_detailed_logging {
            info!("SSE Event: {} - {} - {}", event.severity.as_str(), event.event_type, event.data);
        }
    }

    /// Perform automatic recovery for a failed endpoint
    pub async fn auto_recover_endpoint(&self, endpoint_url: &str) -> Result<()> {
        if !self.config.enable_auto_recovery {
            return Ok(());
        }
        
        info!("Attempting automatic recovery for endpoint: {}", endpoint_url);
        
        // Try to validate and restart monitoring
        match self.sse_validator.validate_endpoint(endpoint_url).await {
            Ok(_) => {
                info!("Endpoint validation successful, restarting monitoring");
                self.stop_monitoring(endpoint_url).await?;
                sleep(Duration::from_secs(1)).await;
                self.start_monitoring(endpoint_url).await?;
            }
            Err(e) => {
                error!("Automatic recovery failed for endpoint {}: {}", endpoint_url, e);
                
                // Log recovery attempt
                self.log_event(SSEEvent {
                    timestamp: SystemTime::now(),
                    event_type: "recovery".to_string(),
                    data: format!("Recovery attempt failed: {}", e),
                    source: endpoint_url.to_string(),
                    severity: SSEEventSeverity::Warning,
                });
            }
        }
        
        Ok(())
    }

    /// Cleanup old monitoring data
    pub fn cleanup(&self) -> Result<()> {
        info!("Cleaning up old monitoring data");
        
        // Clean up old events (keep last 1000)
        {
            let mut event_log = self.event_log.lock().unwrap();
            if event_log.len() > 1000 {
                let keep_events = event_log.split_off(event_log.len() - 1000).to_vec();
                event_log = keep_events;
            }
        }
        
        // Clean up old performance metrics (keep last 100 entries per endpoint)
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            let mut keys_to_remove = Vec::new();
            
            for (endpoint_url, _) in metrics.iter() {
                if !self.get_monitored_endpoints().contains(&endpoint_url) {
                    keys_to_remove.push(endpoint_url.clone());
                }
            }
            
            for key in keys_to_remove {
                metrics.remove(&key);
            }
        }
        
        info!("Monitoring cleanup completed");
        Ok(())
    }
}

/// SSE monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEMonitoringStats {
    /// Total number of monitored endpoints
    pub total_endpoints: usize,
    /// Number of healthy endpoints
    pub healthy_endpoints: usize,
    /// Total number of events logged
    pub total_events: usize,
    /// Number of error events
    pub error_events: usize,
    /// Average response time across all endpoints
    pub avg_response_time: f64,
    /// Average data throughput across all endpoints
    pub avg_throughput: f64,
    /// Average error rate across all endpoints
    pub avg_error_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_sse_monitor() {
        let config = SSEMonitoringConfig::default();
        let port_allocator = Arc::new(PortAllocator::new());
        let sse_validator = Arc::new(SSEValidator::new());
        let sse_manager = Arc::new(SSEManager::new());
        
        let monitor = SSEMonitor::new(
            config,
            port_allocator,
            sse_validator,
            sse_manager,
        );
        
        // Test starting monitoring
        monitor.start_monitoring("http://localhost:8080/sse").await.unwrap();
        
        // Test metrics update
        monitor.update_metrics("http://localhost:8080/sse", |metrics| {
            metrics.successful_connections += 1;
        }).await.unwrap();
        
        let metrics = monitor.get_endpoint_metrics("http://localhost:8080/sse").await.unwrap();
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().successful_connections, 1);
        
        // Test stopping monitoring
        monitor.stop_monitoring("http://localhost:8080/sse").await.unwrap();
        
        let metrics = monitor.get_endpoint_metrics("http://localhost:8080/sse").await.unwrap();
        assert!(metrics.is_some());
        
        // Test event logging
        monitor.log_event(SSEEvent {
            timestamp: SystemTime::now(),
            event_type: "test".to_string(),
            data: "test event".to_string(),
            source: "http://localhost:8080/sse".to_string(),
            severity: SSEEventSeverity::Info,
        });
        
        let events = monitor.get_recent_events("http://localhost:8080/sse", Some(1)).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "test");
        
        // Test system stats
        let stats = monitor.get_system_stats().await;
        assert_eq!(stats.total_endpoints, 1);
        assert_eq!(stats.healthy_endpoints, 0); // Not healthy anymore
        assert_eq!(stats.total_events, 1);
    }

    #[test]
    async fn test_sse_event_severity() {
        assert_eq!(SSEEventSeverity::Info.as_str(), "info");
        assert_eq!(SSEEventSeverity::Warning.as_str(), "warning");
        assert_eq!(SSEEventSeverity::Error.as_str(), "error");
        assert_eq!(SSEEventSeverity::Critical.as_str(), "critical");
    }

    #[test]
    async fn test_sse_endpoint_metrics() {
        let metrics = SSEEndpointMetrics::default();
        assert_eq!(metrics.endpoint_url, "");
        assert_eq!(metrics.health_status, SSEEndpointHealth::Unknown);
        assert_eq!(metrics.successful_connections, 0);
        assert_eq!(metrics.failed_connections, 0);
    }

    #[test]
    fn test_sse_performance_metrics() {
        let metrics = SSEPerformanceMetrics::default();
        assert_eq!(metrics.avg_response_time_ms, 0.0);
        assert_eq!(metrics.connection_success_rate, 0.0);
        assert_eq!(metrics.data_throughput_bps, 0.0);
        assert_eq!(metrics.error_rate, 0.0);
        assert_eq!(metrics.cpu_usage_percent, 0.0);
        assert_eq!(metrics.memory_usage_mb, 0.0);
    }
}