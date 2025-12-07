//! Port Management System for Agent Builder
//!
//! This module provides comprehensive port allocation and management functionality
//! for SSE endpoints and agent services. It includes automatic port discovery,
//! conflict detection, and resolution mechanisms.

use std::collections::{HashMap, HashSet};
use std::net::{TcpListener, SocketAddr};
use std::sync::{Arc, Mutex};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener as TokioTcpListener;
use tracing::{info, warn, error, debug};

/// Port allocation configuration
#[derive(Debug, Clone)]
pub struct PortAllocationConfig {
    /// Port range to use for automatic allocation
    pub port_range: std::ops::Range<u16>,
    /// Preferred ports to try first
    pub preferred_ports: Vec<u16>,
    /// Reserved ports that should not be used
    pub reserved_ports: HashSet<u16>,
    /// Whether to allow port reuse for same agent
    pub allow_reuse: bool,
}

impl Default for PortAllocationConfig {
    fn default() -> Self {
        Self {
            port_range: 3000..9000,
            preferred_ports: vec![8080, 8081, 8082, 3000, 3001, 5000, 5001, 9000, 9001],
            reserved_ports: {
                let mut reserved = HashSet::new();
                // Common system ports
                reserved.extend([20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]);
                // Database ports
                reserved.extend([3306, 5432, 6379, 27017]);
                // Development tools
                reserved.extend([3000, 4200, 8000, 8080]);
                reserved
            },
            allow_reuse: true,
        }
    }
}

/// Port allocation result
#[derive(Debug, Clone)]
pub struct PortAllocation {
    /// The allocated port number
    pub port: u16,
    /// Whether this is a preferred port
    pub is_preferred: bool,
    /// Allocation timestamp
    pub allocated_at: std::time::SystemTime,
    /// Agent that owns this port
    pub owner: String,
}

impl PortAllocation {
    /// Create a new port allocation
    pub fn new(port: u16, is_preferred: bool, owner: String) -> Self {
        Self {
            port,
            is_preferred,
            allocated_at: std::time::SystemTime::now(),
            owner,
        }
    }

    /// Check if the allocation is expired
    pub fn is_expired(&self, timeout: std::time::Duration) -> bool {
        self.allocated_at.elapsed().unwrap_or_default() > timeout
    }
}

/// Port validator for checking port availability and conflicts
pub struct PortValidator {
    config: PortAllocationConfig,
}

impl PortValidator {
    /// Create a new port validator
    pub fn new(config: PortAllocationConfig) -> Self {
        Self { config }
    }

    /// Check if a port is available
    pub async fn is_port_available(&self, port: u16) -> bool {
        // Check if port is in valid range
        if !self.config.port_range.contains(&port) {
            debug!("Port {} is outside valid range {:?}", port, self.config.port_range);
            return false;
        }

        // Check if port is reserved
        if self.config.reserved_ports.contains(&port) {
            debug!("Port {} is reserved", port);
            return false;
        }

        // Try to bind to the port to check if it's actually available
        match TokioTcpListener::bind(format!("127.0.0.1:{}", port)).await {
            Ok(_) => {
                debug!("Port {} is available", port);
                true
            }
            Err(e) => {
                debug!("Port {} is not available: {}", port, e);
                false
            }
        }
    }

    /// Check if a port is in the preferred list
    pub fn is_preferred_port(&self, port: u16) -> bool {
        self.config.preferred_ports.contains(&port)
    }

    /// Get a list of suggested ports for an agent
    pub fn get_suggested_ports(&self, agent_name: &str) -> Vec<u16> {
        let mut suggested = Vec::new();
        
        // Add preferred ports first
        for &port in &self.config.preferred_ports {
            suggested.push(port);
        }

        // Add some ports based on agent name hash for consistency
        let hash = agent_name.chars().map(|c| c as u32).sum::<u32>();
        let base_port = 3000 + (hash % 1000) as u16;
        
        for offset in 0..10 {
            let port = base_port + offset;
            if self.config.port_range.contains(&port) && !suggested.contains(&port) {
                suggested.push(port);
            }
        }

        suggested
    }

    /// Validate port configuration
    pub fn validate_port_config(&self, port: u16) -> Result<()> {
        if port == 0 {
            return Err(anyhow!("Port cannot be 0"));
        }

        if port < 1024 {
            warn!("Port {} may require administrator privileges", port);
        }

        if !self.config.port_range.contains(&port) {
            return Err(anyhow!(
                "Port {} is outside the allowed range {:?}",
                port,
                self.config.port_range
            ));
        }

        if self.config.reserved_ports.contains(&port) {
            return Err(anyhow!("Port {} is reserved for system use", port));
        }

        Ok(())
    }
}

/// Port registry for tracking allocated ports
pub struct PortRegistry {
    allocations: Arc<Mutex<HashMap<u16, PortAllocation>>>,
    config: PortAllocationConfig,
}

impl PortRegistry {
    /// Create a new port registry
    pub fn new(config: PortAllocationConfig) -> Self {
        Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }

    /// Register a port allocation
    pub fn register_port(&self, allocation: PortAllocation) -> Result<()> {
        let mut allocations = self.allocations.lock().unwrap();
        let port = allocation.port; // Clone the port before moving allocation
        
        if allocations.contains_key(&port) {
            return Err(anyhow!("Port {} is already allocated", port));
        }

        allocations.insert(port, allocation);
        info!("Registered port allocation for port {}", port);
        Ok(())
    }

    /// Unregister a port allocation
    pub fn unregister_port(&self, port: u16) -> Result<()> {
        let mut allocations = self.allocations.lock().unwrap();
        
        if let Some(allocation) = allocations.remove(&port) {
            info!("Unregistered port allocation for port {} (owner: {})", port, allocation.owner);
            Ok(())
        } else {
            warn!("Attempted to unregister unallocated port {}", port);
            Err(anyhow!("Port {} was not allocated", port))
        }
    }

    /// Check if a port is allocated
    pub fn is_port_allocated(&self, port: u16) -> bool {
        let allocations = self.allocations.lock().unwrap();
        allocations.contains_key(&port)
    }

    /// Get allocation for a port
    pub fn get_allocation(&self, port: u16) -> Option<PortAllocation> {
        let allocations = self.allocations.lock().unwrap();
        allocations.get(&port).cloned()
    }

    /// Get all allocations for an agent
    pub fn get_agent_allocations(&self, agent_name: &str) -> Vec<PortAllocation> {
        let allocations = self.allocations.lock().unwrap();
        allocations
            .values()
            .filter(|alloc| alloc.owner == agent_name)
            .cloned()
            .collect()
    }

    /// Clean up expired allocations
    pub fn cleanup_expired(&self, timeout: std::time::Duration) -> Vec<u16> {
        let mut allocations = self.allocations.lock().unwrap();
        let mut expired_ports = Vec::new();

        allocations.retain(|&port, allocation| {
            if allocation.is_expired(timeout) {
                warn!("Cleaning up expired port allocation for port {} (owner: {})", port, allocation.owner);
                expired_ports.push(port);
                false
            } else {
                true
            }
        });

        expired_ports
    }

    /// Get all current allocations
    pub fn get_all_allocations(&self) -> Vec<PortAllocation> {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().cloned().collect()
    }

    /// Get allocation statistics
    pub fn get_stats(&self) -> PortStats {
        let allocations = self.allocations.lock().unwrap();
        let total = allocations.len();
        let preferred = allocations.values().filter(|a| a.is_preferred).count();
        let unique_agents: HashSet<_> = allocations.values().map(|a| &a.owner).collect();

        PortStats {
            total_allocated: total,
            preferred_allocated: preferred,
            unique_agents: unique_agents.len(),
            available_range: self.config.port_range.clone(),
        }
    }
}

/// Port allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortStats {
    pub total_allocated: usize,
    pub preferred_allocated: usize,
    pub unique_agents: usize,
    pub available_range: std::ops::Range<u16>,
}

/// Port conflict resolver for handling port conflicts
pub struct PortConflictResolver {
    validator: PortValidator,
    registry: PortRegistry,
    max_attempts: usize,
}

impl PortConflictResolver {
    /// Create a new port conflict resolver
    pub fn new(config: PortAllocationConfig) -> Self {
        let validator = PortValidator::new(config.clone());
        let registry = PortRegistry::new(config.clone());
        
        Self {
            validator,
            registry,
            max_attempts: 50,
        }
    }

    /// Resolve a port conflict by finding an alternative port
    pub async fn resolve_conflict(&self, requested_port: u16, agent_name: &str) -> Result<u16> {
        info!("Resolving port conflict for agent {} requesting port {}", agent_name, requested_port);

        // First, try the requested port if it's available
        if self.validator.is_port_available(requested_port).await && !self.registry.is_port_allocated(requested_port) {
            info!("Requested port {} is available for agent {}", requested_port, agent_name);
            return Ok(requested_port);
        }

        // Try preferred ports
        for &port in &self.validator.config.preferred_ports {
            if self.validator.is_port_available(port).await && !self.registry.is_port_allocated(port) {
                info!("Resolved conflict: allocated preferred port {} for agent {}", port, agent_name);
                return Ok(port);
            }
        }

        // Try suggested ports for this agent
        let suggested_ports = self.validator.get_suggested_ports(agent_name);
        for &port in &suggested_ports {
            if self.validator.is_port_available(port).await && !self.registry.is_port_allocated(port) {
                info!("Resolved conflict: allocated suggested port {} for agent {}", port, agent_name);
                return Ok(port);
            }
        }

        // Try sequential ports in the range
        for attempt in 0..self.max_attempts {
            let port = self.validator.config.port_range.start + (attempt as u16 % (self.validator.config.port_range.end - self.validator.config.port_range.start));
            
            if self.validator.is_port_available(port).await && !self.registry.is_port_allocated(port) {
                info!("Resolved conflict: allocated sequential port {} for agent {} after {} attempts", port, agent_name, attempt + 1);
                return Ok(port);
            }
        }

        Err(anyhow!("Failed to resolve port conflict for agent {} after {} attempts", agent_name, self.max_attempts))
    }

    /// Allocate and register a port for an agent
    pub async fn allocate_port_for_agent(&self, agent_name: &str, preferred_port: Option<u16>) -> Result<u16> {
        let port = if let Some(preferred) = preferred_port {
            self.resolve_conflict(preferred, agent_name).await?
        } else {
            // Find any available port
            let suggested_ports = self.validator.get_suggested_ports(agent_name);
            let mut allocated = None;

            for &port in &suggested_ports {
                if self.validator.is_port_available(port).await && !self.registry.is_port_allocated(port) {
                    allocated = Some(port);
                    break;
                }
            }

            if allocated.is_none() {
                allocated = Some(self.resolve_conflict(0, agent_name).await?);
            }

            allocated.unwrap()
        };

        // Register the allocation
        let allocation = PortAllocation::new(port, self.validator.is_preferred_port(port), agent_name.to_string());
        self.registry.register_port(allocation)?;

        Ok(port)
    }

    /// Release a port allocation
    pub fn release_port(&self, port: u16) -> Result<()> {
        self.registry.unregister_port(port)
    }

    /// Get the validator
    pub fn validator(&self) -> &PortValidator {
        &self.validator
    }

    /// Get the registry
    pub fn registry(&self) -> &PortRegistry {
        &self.registry
    }
}

/// Main port allocator that combines all functionality
pub struct PortAllocator {
    resolver: PortConflictResolver,
}

impl PortAllocator {
    /// Create a new port allocator with default configuration
    pub fn new() -> Self {
        Self::with_config(PortAllocationConfig::default())
    }

    /// Create a new port allocator with custom configuration
    pub fn with_config(config: PortAllocationConfig) -> Self {
        let resolver = PortConflictResolver::new(config);
        Self { resolver }
    }

    /// Allocate a port for an agent
    pub async fn allocate_port(&self, agent_name: &str, preferred_port: Option<u16>) -> Result<u16> {
        self.resolver.allocate_port_for_agent(agent_name, preferred_port).await
    }

    /// Release a port allocation
    pub fn release_port(&self, port: u16) -> Result<()> {
        self.resolver.release_port(port)
    }

    /// Check if a port is available
    pub async fn is_port_available(&self, port: u16) -> bool {
        self.resolver.validator.is_port_available(port).await
    }

    /// Validate a port configuration
    pub fn validate_port(&self, port: u16) -> Result<()> {
        self.resolver.validator.validate_port_config(port)
    }

    /// Get suggested ports for an agent
    pub fn get_suggested_ports(&self, agent_name: &str) -> Vec<u16> {
        self.resolver.validator.get_suggested_ports(agent_name)
    }

    /// Get allocation statistics
    pub fn get_stats(&self) -> PortStats {
        self.resolver.registry.get_stats()
    }

    /// Clean up expired allocations
    pub fn cleanup_expired(&self, timeout: std::time::Duration) -> Vec<u16> {
        self.resolver.registry.cleanup_expired(timeout)
    }

    /// Get all current allocations
    pub fn get_all_allocations(&self) -> Vec<PortAllocation> {
        self.resolver.registry.get_all_allocations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_port_availability() {
        let config = PortAllocationConfig::default();
        let validator = PortValidator::new(config);
        
        // Test with a commonly available port
        let available = validator.is_port_available(12345).await;
        assert!(available, "Port 12345 should be available");
    }

    #[test]
    fn test_port_validation() {
        let config = PortAllocationConfig::default();
        let validator = PortValidator::new(config);
        
        // Test valid port
        assert!(validator.validate_port(8080).is_ok());
        
        // Test invalid port (0)
        assert!(validator.validate_port(0).is_err());
        
        // Test reserved port
        assert!(validator.validate_port(80).is_err());
    }

    #[test]
    fn test_port_registry() {
        let config = PortAllocationConfig::default();
        let registry = PortRegistry::new(config);
        
        let allocation = PortAllocation::new(8080, true, "test_agent".to_string());
        
        // Test registration
        assert!(registry.register_port(allocation).is_ok());
        assert!(registry.is_port_allocated(8080));
        
        // Test retrieval
        let retrieved = registry.get_allocation(8080);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().owner, "test_agent");
        
        // Test unregistration
        assert!(registry.unregister_port(8080).is_ok());
        assert!(!registry.is_port_allocated(8080));
    }

    #[tokio::test]
    async fn test_port_allocator() {
        let allocator = PortAllocator::new();
        
        // Test allocation
        let port = allocator.allocate_port("test_agent", None).await;
        assert!(port.is_ok());
        
        let allocated_port = port.unwrap();
        assert!(allocator.validate_port(allocated_port).is_ok());
        
        // Test release
        assert!(allocator.release_port(allocated_port).is_ok());
    }

    #[test]
    fn test_suggested_ports() {
        let config = PortAllocationConfig::default();
        let validator = PortValidator::new(config);
        
        let suggested = validator.get_suggested_ports("test_agent");
        assert!(!suggested.is_empty());
        assert!(suggested.len() > 10); // Should have preferred + hash-based suggestions
    }
}