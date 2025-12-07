//! Comprehensive SSE Endpoint Management System
//!
//! This module provides a complete management system for SSE (Server-Sent Events)
//! endpoints used in agent communication. It combines port allocation, validation,
//! monitoring, and lifecycle management.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use tokio::sync::RwLock;
use crate::port_allocator::{PortAllocator, PortAllocationConfig, PortAllocation};
use crate::sse_validator::{SSEValidator, SSEValidationConfig, SSEValidationResult, SSEHealthMonitor, SSEHealthStatus};

/// SSE endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEndpointConfig {
    /// Unique identifier for the endpoint
    pub id: String,
    /// Agent name that owns this endpoint
    pub agent_name: String,
    /// Port number for the endpoint
    pub port: u16,
    /// Endpoint path (e.g., "/sse")
    pub path: String,
    /// Full endpoint URL
    pub url: String,
    /// Whether the endpoint is enabled
    pub enabled: bool,
    /// Endpoint description
    pub description: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl SSEEndpointConfig {
    /// Create a new SSE endpoint configuration
    pub fn new(id: String, agent_name: String, port: u16, path: String) -> Self {
        let url = format!("http://localhost:{}{}", port, path);
        Self {
            id,
            agent_name,
            port,
            path,
            url,
            enabled: true,
            description: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the full endpoint URL
    pub fn get_url(&self) -> String {
        self.url.clone()
    }

    /// Check if the endpoint is currently active
    pub fn is_active(&self) -> bool {
        self.enabled && !self.url.is_empty()
    }
}

/// SSE endpoint status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEndpointStatus {
    /// Endpoint configuration
    pub config: SSEEndpointConfig,
    /// Current status of the endpoint
    pub status: SSEEndpointState,
    /// Last validation result
    pub last_validation: Option<SSEValidationResult>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last activity timestamp
    pub last_activity: Option<SystemTime>,
    /// Number of connections handled
    pub connection_count: u64,
    /// Error count
    pub error_count: u64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// SSE endpoint state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SSEEndpointState {
    /// Endpoint is initialized but not started
    Initialized,
    /// Endpoint is starting up
    Starting,
    /// Endpoint is running and accepting connections
    Running,
    /// Endpoint is stopping
    Stopping,
    /// Endpoint is stopped
    Stopped,
    /// Endpoint encountered an error
    Error(String),
    /// Endpoint health is degraded
    Degraded,
}

impl SSEEndpointState {
    /// Check if the endpoint is in an active state
    pub fn is_active(&self) -> bool {
        matches!(self, SSEEndpointState::Running | SSEEndpointState::Starting)
    }

    /// Check if the endpoint is in a healthy state
    pub fn is_healthy(&self) -> bool {
        matches!(self, SSEEndpointState::Running)
    }

    /// Check if the endpoint is in an error state
    pub fn is_error(&self) -> bool {
        matches!(self, SSEEndpointState::Error(_))
    }
}

/// SSE manager configuration
#[derive(Debug, Clone)]
pub struct SSEManagerConfig {
    /// Port allocation configuration
    pub port_config: PortAllocationConfig,
    /// SSE validation configuration
    pub validation_config: SSEValidationConfig,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum number of endpoints per agent
    pub max_endpoints_per_agent: usize,
    /// Whether to enable automatic health monitoring
    pub enable_health_monitoring: bool,
    /// Whether to enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Endpoint cleanup timeout
    pub cleanup_timeout: Duration,
}

impl Default for SSEManagerConfig {
    fn default() -> Self {
        Self {
            port_config: PortAllocationConfig::default(),
            validation_config: SSEValidationConfig::default(),
            health_check_interval: Duration::from_secs(30),
            max_endpoints_per_agent: 5,
            enable_health_monitoring: true,
            enable_auto_recovery: true,
            cleanup_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Comprehensive SSE endpoint manager
pub struct SSEManager {
    /// Manager configuration
    config: SSEManagerConfig,
    /// Port allocator for managing ports
    port_allocator: PortAllocator,
    /// SSE validator for endpoint validation
    validator: SSEValidator,
    /// Health monitor for endpoint monitoring
    health_monitor: Arc<RwLock<SSEHealthMonitor>>,
    /// Managed endpoints
    endpoints: Arc<RwLock<HashMap<String, SSEEndpointStatus>>>,
    /// Background task handles
    background_tasks: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl SSEManager {
    /// Create a new SSE manager with default configuration
    pub fn new() -> Self {
        Self::with_config(SSEManagerConfig::default())
    }

    /// Create a new SSE manager with custom configuration
    pub fn with_config(config: SSEManagerConfig) -> Self {
        let port_allocator = PortAllocator::with_config(config.port_config.clone());
        let validator = SSEValidator::with_config(config.validation_config.clone());
        let health_monitor = Arc::new(RwLock::new(SSEHealthMonitor::new()));

        Self {
            config,
            port_allocator,
            validator,
            health_monitor,
            endpoints: Arc::new(RwLock::new(HashMap::new())),
            background_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new SSE endpoint for an agent
    pub async fn create_endpoint(
        &self,
        agent_name: &str,
        path: &str,
        preferred_port: Option<u16>,
        description: Option<String>,
    ) -> Result<String> {
        info!("Creating SSE endpoint for agent: {}, path: {}", agent_name, path);

        // Check agent endpoint limit
        let agent_endpoints = self.get_agent_endpoints(agent_name).await?;
        if agent_endpoints.len() >= self.config.max_endpoints_per_agent {
            return Err(anyhow!(
                "Agent {} already has maximum number of endpoints ({})",
                agent_name,
                self.config.max_endpoints_per_agent
            ));
        }

        // Allocate a port
        let port = self.port_allocator
            .allocate_port(agent_name, preferred_port)
            .await?;

        // Create endpoint configuration
        let endpoint_id = format!("{}_{}_{}", agent_name, path.replace('/', "_"), port);
        let config = SSEEndpointConfig {
            id: endpoint_id.clone(),
            agent_name: agent_name.to_string(),
            port,
            path: path.to_string(),
            url: format!("http://localhost:{}{}", port, path),
            enabled: true,
            description,
            metadata: HashMap::new(),
        };

        // Create endpoint status
        let status = SSEEndpointStatus {
            config: config.clone(),
            status: SSEEndpointState::Initialized,
            last_validation: None,
            created_at: SystemTime::now(),
            last_activity: None,
            connection_count: 0,
            error_count: 0,
            uptime_seconds: 0,
        };

        // Register endpoint
        {
            let mut endpoints = self.endpoints.write().await;
            endpoints.insert(endpoint_id.clone(), status);
        }

        // Add to health monitoring
        if self.config.enable_health_monitoring {
            let mut monitor = self.health_monitor.write().await;
            monitor.add_endpoint(config.get_url());
        }

        info!("Created SSE endpoint: {} -> {}", endpoint_id, config.get_url());
        Ok(endpoint_id)
    }

    /// Start an SSE endpoint
    pub async fn start_endpoint(&self, endpoint_id: &str) -> Result<()> {
        info!("Starting SSE endpoint: {}", endpoint_id);

        let mut endpoints = self.endpoints.write().await;
        if let Some(status) = endpoints.get_mut(endpoint_id) {
            if status.status.is_active() {
                warn!("Endpoint {} is already active", endpoint_id);
                return Ok(());
            }

            status.status = SSEEndpointState::Starting;

            // Validate endpoint before starting
            let validation_result = self.validator.validate_endpoint(&status.config.get_url()).await;
            match validation_result {
                Ok(result) if result.is_valid => {
                    status.status = SSEEndpointState::Running;
                    status.last_validation = Some(result);
                    status.last_activity = Some(SystemTime::now());

                    // Start background monitoring task
                    if self.config.enable_health_monitoring {
                        self.start_health_monitoring_task(endpoint_id.to_string()).await;
                    }

                    info!("Successfully started SSE endpoint: {}", endpoint_id);
                    Ok(())
                }
                Ok(result) => {
                    status.status = SSEEndpointState::Error(format!("Validation failed: {:?}", result.errors));
                    status.last_validation = Some(result);
                    Err(anyhow!("Endpoint validation failed"))
                }
                Err(e) => {
                    status.status = SSEEndpointState::Error(e.to_string());
                    Err(anyhow!("Failed to validate endpoint: {}", e))
                }
            }
        } else {
            Err(anyhow!("Endpoint {} not found", endpoint_id))
        }
    }

    /// Stop an SSE endpoint
    pub async fn stop_endpoint(&self, endpoint_id: &str) -> Result<()> {
        info!("Stopping SSE endpoint: {}", endpoint_id);

        let mut endpoints = self.endpoints.write().await;
        if let Some(status) = endpoints.get_mut(endpoint_id) {
            status.status = SSEEndpointState::Stopping;

            // Stop background monitoring task
            let mut tasks = self.background_tasks.lock().unwrap();
            if let Some(task) = tasks.remove(endpoint_id) {
                task.abort();
            }

            status.status = SSEEndpointState::Stopped;
            info!("Successfully stopped SSE endpoint: {}", endpoint_id);
            Ok(())
        } else {
            Err(anyhow!("Endpoint {} not found", endpoint_id))
        }
    }

    /// Delete an SSE endpoint
    pub async fn delete_endpoint(&self, endpoint_id: &str) -> Result<()> {
        info!("Deleting SSE endpoint: {}", endpoint_id);

        // Stop endpoint first
        if let Err(e) = self.stop_endpoint(endpoint_id).await {
            warn!("Failed to stop endpoint before deletion: {}", e);
        }

        // Get endpoint info for cleanup
        let endpoint_info = {
            let endpoints = self.endpoints.read().await;
            endpoints.get(endpoint_id).cloned()
        };

        if let Some(info) = endpoint_info {
            // Release allocated port
            if let Err(e) = self.port_allocator.release_port(info.config.port) {
                warn!("Failed to release port {}: {}", info.config.port, e);
            }

            // Remove from health monitoring
            if self.config.enable_health_monitoring {
                let mut monitor = self.health_monitor.write().await;
                monitor.remove_endpoint(&info.config.get_url());
            }
        }

        // Remove from endpoints
        let mut endpoints = self.endpoints.write().await;
        endpoints.remove(endpoint_id);

        info!("Successfully deleted SSE endpoint: {}", endpoint_id);
        Ok(())
    }

    /// Get endpoint status
    pub async fn get_endpoint_status(&self, endpoint_id: &str) -> Option<SSEEndpointStatus> {
        let endpoints = self.endpoints.read().await;
        endpoints.get(endpoint_id).cloned()
    }

    /// Get all endpoints for an agent
    pub async fn get_agent_endpoints(&self, agent_name: &str) -> Result<Vec<SSEEndpointStatus>> {
        let endpoints = self.endpoints.read().await;
        let agent_endpoints: Vec<SSEEndpointStatus> = endpoints
            .values()
            .filter(|status| status.config.agent_name == agent_name)
            .cloned()
            .collect();

        Ok(agent_endpoints)
    }

    /// Get all managed endpoints
    pub async fn get_all_endpoints(&self) -> Vec<SSEEndpointStatus> {
        let endpoints = self.endpoints.read().await;
        endpoints.values().cloned().collect()
    }

    /// Validate an endpoint
    pub async fn validate_endpoint(&self, endpoint_id: &str) -> Result<SSEValidationResult> {
        let endpoints = self.endpoints.read().await;
        if let Some(status) = endpoints.get(endpoint_id) {
            let result = self.validator.validate_endpoint(&status.config.get_url()).await?;
            
            // Update validation result
            drop(endpoints);
            let mut endpoints = self.endpoints.write().await;
            if let Some(status) = endpoints.get_mut(endpoint_id) {
                status.last_validation = Some(result.clone());
                
                // Update status based on validation
                if result.is_valid {
                    if matches!(status.status, SSEEndpointState::Error(_)) {
                        status.status = SSEEndpointState::Degraded;
                    }
                } else {
                    status.status = SSEEndpointState::Error("Validation failed".to_string());
                }
            }
            
            Ok(result)
        } else {
            Err(anyhow!("Endpoint {} not found", endpoint_id))
        }
    }

    /// Test endpoint connectivity
    pub async fn test_connectivity(&self, endpoint_id: &str) -> Result<bool> {
        let endpoints = self.endpoints.read().await;
        if let Some(status) = endpoints.get(endpoint_id) {
            self.validator.test_connectivity(&status.config.get_url()).await
        } else {
            Err(anyhow!("Endpoint {} not found", endpoint_id))
        }
    }

    /// Get available ports for allocation
    pub async fn get_available_ports(&self) -> Vec<u16> {
        let stats = self.port_allocator.get_stats();
        let mut available = Vec::new();
        
        for port in stats.available_range {
            if self.port_allocator.is_port_available(port).await {
                available.push(port);
            }
        }
        
        available
    }

    /// Get suggested ports for an agent
    pub fn get_suggested_ports(&self, agent_name: &str) -> Vec<u16> {
        self.port_allocator.get_suggested_ports(agent_name)
    }

    /// Get manager statistics
    pub async fn get_stats(&self) -> SSEManagerStats {
        let endpoints = self.endpoints.read().await;
        let total_endpoints = endpoints.len();
        let running_endpoints = endpoints.values()
            .filter(|status| status.status.is_active())
            .count();
        let healthy_endpoints = endpoints.values()
            .filter(|status| status.status.is_healthy())
            .count();
        let error_endpoints = endpoints.values()
            .filter(|status| status.status.is_error())
            .count();

        let mut agent_counts: HashMap<String, usize> = HashMap::new();
        for status in endpoints.values() {
            *agent_counts.entry(status.config.agent_name.clone()).or_insert(0) += 1;
        }

        let port_stats = self.port_allocator.get_stats();

        SSEManagerStats {
            total_endpoints,
            running_endpoints,
            healthy_endpoints,
            error_endpoints,
            agent_counts,
            port_stats,
        }
    }

    /// Cleanup expired endpoints and allocations
    pub async fn cleanup(&self) -> Result<Vec<String>> {
        let mut cleaned_endpoints = Vec::new();
        
        // Cleanup expired port allocations
        let expired_ports = self.port_allocator.cleanup_expired(self.config.cleanup_timeout);
        for port in expired_ports {
            warn!("Cleaned up expired port allocation: {}", port);
        }

        // Cleanup inactive endpoints
        let mut endpoints = self.endpoints.write().await;
        let mut to_remove = Vec::new();

        for (endpoint_id, status) in endpoints.iter() {
            let should_remove = match status.status {
                SSEEndpointState::Stopped | SSEEndpointState::Error(_) => {
                    // Remove if inactive for cleanup timeout
                    status.last_activity
                        .map(|time| time.elapsed().unwrap_or_default() > self.config.cleanup_timeout)
                        .unwrap_or(true)
                }
                _ => false,
            };

            if should_remove {
                to_remove.push(endpoint_id.clone());
            }
        }

        for endpoint_id in to_remove {
            if let Some(status) = endpoints.remove(&endpoint_id) {
                // Release port
                if let Err(e) = self.port_allocator.release_port(status.config.port) {
                    warn!("Failed to release port during cleanup: {}", e);
                }

                // Remove from health monitoring
                if self.config.enable_health_monitoring {
                    let mut monitor = self.health_monitor.write().await;
                    monitor.remove_endpoint(&status.config.get_url());
                }

                cleaned_endpoints.push(endpoint_id.clone());
                info!("Cleaned up inactive endpoint: {}", endpoint_id);
            }
        }

        Ok(cleaned_endpoints)
    }

    /// Start background health monitoring task
    async fn start_health_monitoring_task(&self, endpoint_id: String) {
        let health_monitor = self.health_monitor.clone();
        let endpoints = self.endpoints.clone();
        let interval = self.config.health_check_interval;
        let endpoint_id_clone = endpoint_id.clone();

        let task = tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;

                // Get endpoint URL
                let endpoint_url = {
                    let endpoints_read = endpoints.read().await;
                    endpoints_read.get(&endpoint_id)
                        .map(|status| status.config.get_url())
                };

                if let Some(url) = endpoint_url {
                    // Check health
                    let mut monitor = health_monitor.write().await;
                    monitor.check_endpoint_health(&url).await;
                } else {
                    // Endpoint no longer exists, exit task
                    break;
                }
            }
        });

        let mut tasks = self.background_tasks.lock().unwrap();
        tasks.insert(endpoint_id_clone, task);
    }
}

/// SSE manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEManagerStats {
    /// Total number of managed endpoints
    pub total_endpoints: usize,
    /// Number of running endpoints
    pub running_endpoints: usize,
    /// Number of healthy endpoints
    pub healthy_endpoints: usize,
    /// Number of endpoints with errors
    pub error_endpoints: usize,
    /// Endpoint count per agent
    pub agent_counts: HashMap<String, usize>,
    /// Port allocation statistics
    pub port_stats: crate::port_allocator::PortStats,
}

impl Drop for SSEManager {
    fn drop(&mut self) {
        // Abort all background tasks
        let mut tasks = self.background_tasks.lock().unwrap();
        for (_, task) in tasks.drain() {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_sse_manager_creation() {
        let manager = SSEManager::new();
        let stats = manager.get_stats().await;
        assert_eq!(stats.total_endpoints, 0);
    }

    #[tokio::test]
    async fn test_endpoint_creation() {
        let manager = SSEManager::new();
        
        let endpoint_id = manager.create_endpoint(
            "test_agent",
            "/sse",
            None,
            Some("Test endpoint".to_string()),
        ).await;

        assert!(endpoint_id.is_ok());
        
        let endpoint_id = endpoint_id.unwrap();
        let status = manager.get_endpoint_status(&endpoint_id).await;
        assert!(status.is_some());
        
        let status = status.unwrap();
        assert_eq!(status.config.agent_name, "test_agent");
        assert_eq!(status.config.path, "/sse");
        assert_eq!(status.config.description, Some("Test endpoint".to_string()));
    }

    #[tokio::test]
    async fn test_endpoint_lifecycle() {
        let manager = SSEManager::new();
        
        // Create endpoint
        let endpoint_id = manager.create_endpoint("test_agent", "/sse", None, None).await.unwrap();
        
        // Start endpoint
        assert!(manager.start_endpoint(&endpoint_id).await.is_ok());
        
        let status = manager.get_endpoint_status(&endpoint_id).await.unwrap();
        assert!(status.status.is_active());
        
        // Stop endpoint
        assert!(manager.stop_endpoint(&endpoint_id).await.is_ok());
        
        let status = manager.get_endpoint_status(&endpoint_id).await.unwrap();
        assert_eq!(status.status, SSEEndpointState::Stopped);
        
        // Delete endpoint
        assert!(manager.delete_endpoint(&endpoint_id).await.is_ok());
        
        let status = manager.get_endpoint_status(&endpoint_id).await;
        assert!(status.is_none());
    }

    #[test]
    fn test_endpoint_state() {
        assert!(SSEEndpointState::Running.is_active());
        assert!(SSEEndpointState::Starting.is_active());
        assert!(!SSEEndpointState::Stopped.is_active());
        
        assert!(SSEEndpointState::Running.is_healthy());
        assert!(!SSEEndpointState::Error("test".to_string()).is_healthy());
        
        assert!(SSEEndpointState::Error("test".to_string()).is_error());
        assert!(!SSEEndpointState::Running.is_error());
    }

    #[test]
    fn test_endpoint_config() {
        let config = SSEEndpointConfig::new(
            "test_id".to_string(),
            "test_agent".to_string(),
            8080,
            "/sse".to_string(),
        );
        
        assert_eq!(config.id, "test_id");
        assert_eq!(config.agent_name, "test_agent");
        assert_eq!(config.port, 8080);
        assert_eq!(config.path, "/sse");
        assert_eq!(config.get_url(), "http://localhost:8080/sse");
        assert!(config.is_active());
    }
}