//! SSE Endpoint Validator and Testing System
//!
//! This module provides comprehensive validation and testing functionality
//! for SSE (Server-Sent Events) endpoints used in agent communication.

use std::time::{Duration, Instant};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use reqwest::{Client, Response};
use tokio::time::timeout;
// Removed unused import: futures_util::StreamExt

/// SSE validation configuration
#[derive(Debug, Clone)]
pub struct SSEValidationConfig {
    /// Timeout for connection attempts
    pub connection_timeout: Duration,
    /// Timeout for response validation
    pub response_timeout: Duration,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Delay between retry attempts
    pub retry_delay: Duration,
    /// Whether to perform deep validation
    pub deep_validation: bool,
}

impl Default for SSEValidationConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(10),
            response_timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            deep_validation: true,
        }
    }
}

/// SSE endpoint validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEValidationResult {
    /// Whether the endpoint is valid
    pub is_valid: bool,
    /// Endpoint URL that was tested
    pub endpoint_url: String,
    /// Connection time in milliseconds
    pub connection_time_ms: u64,
    /// Response time in milliseconds
    pub response_time_ms: Option<u64>,
    /// HTTP status code received
    pub status_code: Option<u16>,
    /// Content type received
    pub content_type: Option<String>,
    /// Whether SSE protocol was detected
    pub sse_detected: bool,
    /// Number of events received during validation
    pub events_received: usize,
    /// Validation errors encountered
    pub errors: Vec<String>,
    /// Validation warnings encountered
    pub warnings: Vec<String>,
    /// Additional metadata
    pub metadata: SSEValidationMetadata,
}

/// SSE validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEValidationMetadata {
    /// Server software (if available)
    pub server: Option<String>,
    /// Connection headers received
    pub headers: std::collections::HashMap<String, String>,
    /// Supported features detected
    pub features: Vec<String>,
    /// Validation timestamp
    pub validated_at: String,
}

/// SSE event structure for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEEvent {
    /// Event type (optional)
    pub event_type: Option<String>,
    /// Event data
    pub data: String,
    /// Event ID (optional)
    pub id: Option<String>,
    /// Retry interval (optional)
    pub retry: Option<u32>,
}

/// SSE endpoint validator
pub struct SSEValidator {
    client: Client,
    config: SSEValidationConfig,
}

impl SSEValidator {
    /// Create a new SSE validator with default configuration
    pub fn new() -> Self {
        Self::with_config(SSEValidationConfig::default())
    }

    /// Create a new SSE validator with custom configuration
    pub fn with_config(config: SSEValidationConfig) -> Self {
        let client = Client::builder()
            .timeout(config.connection_timeout)
            .user_agent("AgentBuilder-SSEValidator/1.0")
            .build()
            .unwrap_or_else(|_| Client::new());

        Self { client, config }
    }

    /// Validate an SSE endpoint
    pub async fn validate_endpoint(&self, endpoint_url: &str) -> Result<SSEValidationResult> {
        info!("Validating SSE endpoint: {}", endpoint_url);

        let start_time = Instant::now();
        let mut result = SSEValidationResult {
            is_valid: false,
            endpoint_url: endpoint_url.to_string(),
            connection_time_ms: 0,
            response_time_ms: None,
            status_code: None,
            content_type: None,
            sse_detected: false,
            events_received: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: SSEValidationMetadata {
                server: None,
                headers: std::collections::HashMap::new(),
                features: Vec::new(),
                validated_at: chrono::Utc::now().to_rfc3339(),
            },
        };

        // Perform validation with retries
        let mut last_error = None;
        for attempt in 1..=self.config.max_retries {
            if attempt > 1 {
                debug!("Retry attempt {} for endpoint {}", attempt, endpoint_url);
                tokio::time::sleep(self.config.retry_delay).await;
            }

            match self.validate_endpoint_once(endpoint_url, &mut result).await {
                Ok(_) => {
                    result.connection_time_ms = start_time.elapsed().as_millis() as u64;
                    info!("SSE endpoint validation successful for {}", endpoint_url);
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(anyhow::anyhow!("{}", e));
                    warn!("Validation attempt {} failed for {}: {}", attempt, endpoint_url, e);
                    result.errors.push(format!("Attempt {} failed: {}", attempt, e));
                }
            }
        }

        // All attempts failed
        result.connection_time_ms = start_time.elapsed().as_millis() as u64;
        if let Some(error) = last_error {
            result.errors.push(format!("All validation attempts failed: {}", error));
        }

        error!("SSE endpoint validation failed for {}", endpoint_url);
        Ok(result)
    }

    /// Perform a single validation attempt
    async fn validate_endpoint_once(&self, endpoint_url: &str, result: &mut SSEValidationResult) -> Result<()> {
        // Make initial connection
        let response_start = Instant::now();
        let response = self.client
            .get(endpoint_url)
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .send()
            .await?;

        let response_time = response_start.elapsed().as_millis() as u64;
        result.response_time_ms = Some(response_time);

        // Check status code
        let status = response.status();
        result.status_code = Some(status.as_u16());

        if !status.is_success() {
            return Err(anyhow!("HTTP error: {}", status));
        }

        // Check content type
        if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
            let content_type_str = content_type.to_str().unwrap_or("").to_lowercase();
            result.content_type = Some(content_type_str.clone());

            if content_type_str.contains("text/event-stream") {
                result.sse_detected = true;
                result.metadata.features.push("sse".to_string());
            } else {
                result.warnings.push("Content-Type is not text/event-stream".to_string());
            }
        } else {
            result.warnings.push("No Content-Type header found".to_string());
        }

        // Extract server info and headers
        if let Some(server) = response.headers().get(reqwest::header::SERVER) {
            result.metadata.server = Some(server.to_str().unwrap_or("").to_string());
        }

        for (name, value) in response.headers() {
            result.metadata.headers.insert(
                name.to_string(),
                value.to_str().unwrap_or("").to_string(),
            );
        }

        // If deep validation is enabled, try to read some events
        if self.config.deep_validation {
            self.validate_sse_stream(response, result).await?;
        }

        result.is_valid = true;
        Ok(())
    }

    /// Validate SSE stream by reading events
    async fn validate_sse_stream(&self, mut response: Response, result: &mut SSEValidationResult) -> Result<()> {
        let mut buffer = String::new();
        let mut event_count = 0;
        let validation_start = Instant::now();

        // Read events for a limited time
        let validation_duration = Duration::from_secs(5);
        while validation_start.elapsed() < validation_duration {
            match timeout(Duration::from_millis(100), response.chunk()).await {
                Ok(Ok(Some(chunk))) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    buffer.push_str(&chunk_str);

                    // Process complete events
                    while let Some(event_end) = buffer.find("\n\n") {
                        let event_data = buffer[..event_end].trim();
                        if !event_data.is_empty() {
                            if let Ok(event) = self.parse_sse_event(event_data) {
                                event_count += 1;
                                debug!("Received SSE event: {:?}", event);
                                
                                // Detect features from events
                                if event.event_type.is_some() {
                                    result.metadata.features.push("event-types".to_string());
                                }
                                if event.id.is_some() {
                                    result.metadata.features.push("event-ids".to_string());
                                }
                                if event.retry.is_some() {
                                    result.metadata.features.push("retry".to_string());
                                }
                            }
                        }
                        buffer = buffer[event_end + 2..].to_string();
                    }
                }
                Ok(Ok(None)) => {
                    break; // Stream ended
                }
                Ok(Err(e)) => {
                    return Err(anyhow!("Stream error: {}", e));
                }
                Err(_) => {
                    break; // Timeout, continue to next iteration
                }
            }

            if event_count >= 5 {
                // Got enough events for validation
                break;
            }
        }

        result.events_received = event_count;

        if event_count == 0 {
            result.warnings.push("No events received during validation".to_string());
        }

        Ok(())
    }

    /// Parse a single SSE event
    fn parse_sse_event(&self, event_data: &str) -> Result<SSEEvent> {
        let mut event = SSEEvent {
            event_type: None,
            data: String::new(),
            id: None,
            retry: None,
        };

        for line in event_data.lines() {
            if line.is_empty() {
                continue;
            }

            if let Some(colon_pos) = line.find(':') {
                let field = &line[..colon_pos];
                let value = &line[colon_pos + 1..].trim_start();

                match field {
                    "event" => event.event_type = Some(value.to_string()),
                    "data" => {
                        if !event.data.is_empty() {
                            event.data.push('\n');
                        }
                        event.data.push_str(value);
                    }
                    "id" => event.id = Some(value.to_string()),
                    "retry" => event.retry = value.parse().ok(),
                    _ => {} // Ignore unknown fields
                }
            }
        }

        Ok(event)
    }

    /// Test SSE endpoint connectivity (basic check)
    pub async fn test_connectivity(&self, endpoint_url: &str) -> Result<bool> {
        debug!("Testing connectivity to SSE endpoint: {}", endpoint_url);

        match timeout(
            self.config.connection_timeout,
            self.client
                .get(endpoint_url)
                .header("Accept", "text/event-stream")
                .send()
        ).await {
            Ok(Ok(response)) => Ok(response.status().is_success()),
            Ok(Err(_)) => Ok(false),
            Err(_) => Ok(false), // Timeout
        }
    }

    /// Get SSE endpoint information without full validation
    pub async fn get_endpoint_info(&self, endpoint_url: &str) -> Result<SSEValidationResult> {
        let mut result = SSEValidationResult {
            is_valid: false,
            endpoint_url: endpoint_url.to_string(),
            connection_time_ms: 0,
            response_time_ms: None,
            status_code: None,
            content_type: None,
            sse_detected: false,
            events_received: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: SSEValidationMetadata {
                server: None,
                headers: std::collections::HashMap::new(),
                features: Vec::new(),
                validated_at: chrono::Utc::now().to_rfc3339(),
            },
        };

        let start_time = Instant::now();
        
        match self.client
            .get(endpoint_url)
            .header("Accept", "text/event-stream")
            .send()
            .await {
            Ok(response) => {
                result.connection_time_ms = start_time.elapsed().as_millis() as u64;
                result.status_code = Some(response.status().as_u16());

                if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
                    result.content_type = Some(content_type.to_str().unwrap_or("").to_string());
                }

                if let Some(server) = response.headers().get(reqwest::header::SERVER) {
                    result.metadata.server = Some(server.to_str().unwrap_or("").to_string());
                }

                for (name, value) in response.headers() {
                    result.metadata.headers.insert(
                        name.to_string(),
                        value.to_str().unwrap_or("").to_string(),
                    );
                }

                result.is_valid = response.status().is_success();
            }
            Err(e) => {
                result.connection_time_ms = start_time.elapsed().as_millis() as u64;
                result.errors.push(format!("Connection failed: {}", e));
            }
        }

        Ok(result)
    }

    /// Validate multiple SSE endpoints in parallel
    pub async fn validate_endpoints(&self, endpoint_urls: &[String]) -> Vec<SSEValidationResult> {
        let tasks: Vec<_> = endpoint_urls
            .iter()
            .map(|url| self.validate_endpoint(url))
            .collect();

        futures_util::future::join_all(tasks).await
            .into_iter()
            .filter_map(|result| result.ok())
            .collect()
    }

    /// Get the current configuration
    pub fn config(&self) -> &SSEValidationConfig {
        &self.config
    }

    /// Update the configuration
    pub fn update_config(&mut self, config: SSEValidationConfig) {
        self.config = config;
        self.client = Client::builder()
            .timeout(self.config.connection_timeout)
            .user_agent("AgentBuilder-SSEValidator/1.0")
            .build()
            .unwrap_or_else(|_| Client::new());
    }
}

/// SSE endpoint health monitor
pub struct SSEHealthMonitor {
    validator: SSEValidator,
    monitored_endpoints: std::collections::HashMap<String, SSEHealthStatus>,
}

/// SSE endpoint health status
#[derive(Debug, Clone)]
pub struct SSEHealthStatus {
    /// Endpoint URL
    pub endpoint_url: String,
    /// Current health status
    pub status: SSEHealthStatusType,
    /// Last check timestamp
    pub last_check: std::time::SystemTime,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Total number of checks performed
    pub total_checks: u32,
    /// Last validation result
    pub last_result: Option<SSEValidationResult>,
}

/// SSE health status types
#[derive(Debug, Clone, PartialEq)]
pub enum SSEHealthStatusType {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl SSEHealthMonitor {
    /// Create a new SSE health monitor
    pub fn new() -> Self {
        Self {
            validator: SSEValidator::new(),
            monitored_endpoints: std::collections::HashMap::new(),
        }
    }

    /// Add an endpoint to monitor
    pub fn add_endpoint(&mut self, endpoint_url: String) {
        let status = SSEHealthStatus {
            endpoint_url: endpoint_url.clone(),
            status: SSEHealthStatusType::Unknown,
            last_check: std::time::SystemTime::UNIX_EPOCH,
            consecutive_failures: 0,
            avg_response_time_ms: 0.0,
            total_checks: 0,
            last_result: None,
        };

        self.monitored_endpoints.insert(endpoint_url, status);
        info!("Added endpoint to health monitoring");
    }

    /// Check health of all monitored endpoints
    pub async fn check_all_health(&mut self) -> Vec<&SSEHealthStatus> {
        let endpoints: Vec<String> = self.monitored_endpoints.keys().cloned().collect();
        
        for endpoint in endpoints {
            self.check_endpoint_health(&endpoint).await;
        }

        self.monitored_endpoints.values().collect()
    }

    /// Check health of a specific endpoint
    pub async fn check_endpoint_health(&mut self, endpoint_url: &str) -> &SSEHealthStatus {
        let validation_result = self.validator.validate_endpoint(endpoint_url).await;
        
        if let Some(status) = self.monitored_endpoints.get_mut(endpoint_url) {
            status.last_check = std::time::SystemTime::now();
            status.total_checks += 1;
            
            // Store the validation result for later reference
            let is_valid = match &validation_result {
                Ok(result) => result.is_valid,
                Err(_) => false,
            };
            let response_time = match &validation_result {
                Ok(result) => result.response_time_ms,
                Err(_) => None,
            };
            
            // Create a simplified version for storage
            status.last_result = match validation_result {
                Ok(result) => Some(result),
                Err(e) => Some(SSEValidationResult {
                    is_valid: false,
                    endpoint_url: endpoint_url.to_string(),
                    connection_time_ms: 0,
                    response_time_ms: None,
                    status_code: None,
                    content_type: None,
                    sse_detected: false,
                    events_received: 0,
                    errors: vec![e.to_string()],
                    warnings: vec![],
                    metadata: SSEValidationMetadata {
                        server: None,
                        headers: std::collections::HashMap::new(),
                        features: vec![],
                        validated_at: chrono::Utc::now().to_rfc3339(),
                    },
                }),
            };

            if is_valid {
                status.consecutive_failures = 0;
                status.status = SSEHealthStatusType::Healthy;
                
                if let Some(response_time) = response_time {
                    status.avg_response_time_ms =
                        (status.avg_response_time_ms * (status.total_checks - 1) as f64 + response_time as f64)
                        / status.total_checks as f64;
                }
            } else {
                status.consecutive_failures += 1;
                if status.consecutive_failures >= 3 {
                    status.status = SSEHealthStatusType::Unhealthy;
                } else {
                    status.status = SSEHealthStatusType::Degraded;
                }
            }
        }

        self.monitored_endpoints.get(endpoint_url).unwrap()
    }

    /// Get health status for all endpoints
    pub fn get_all_status(&self) -> Vec<&SSEHealthStatus> {
        self.monitored_endpoints.values().collect()
    }

    /// Get health status for a specific endpoint
    pub fn get_status(&self, endpoint_url: &str) -> Option<&SSEHealthStatus> {
        self.monitored_endpoints.get(endpoint_url)
    }

    /// Remove an endpoint from monitoring
    pub fn remove_endpoint(&mut self, endpoint_url: &str) -> Option<SSEHealthStatus> {
        self.monitored_endpoints.remove(endpoint_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_sse_validator_creation() {
        let validator = SSEValidator::new();
        assert!(validator.config().max_retries == 3);
        assert!(validator.config().deep_validation == true);
    }

    #[tokio::test]
    async fn test_sse_event_parsing() {
        let validator = SSEValidator::new();
        
        let event_data = "event: update\ndata: Hello World\nid: 123\nretry: 5000\n\n";
        let event = validator.parse_sse_event(event_data).unwrap();
        
        assert_eq!(event.event_type, Some("update".to_string()));
        assert_eq!(event.data, "Hello World");
        assert_eq!(event.id, Some("123".to_string()));
        assert_eq!(event.retry, Some(5000));
    }

    #[tokio::test]
    async fn test_sse_health_monitor() {
        let mut monitor = SSEHealthMonitor::new();
        
        monitor.add_endpoint("http://localhost:8080/sse".to_string());
        assert_eq!(monitor.get_all_status().len(), 1);
        
        let status = monitor.get_status("http://localhost:8080/sse").unwrap();
        assert_eq!(status.status, SSEHealthStatusType::Unknown);
    }

    #[test]
    fn test_validation_config() {
        let config = SSEValidationConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.connection_timeout, Duration::from_secs(10));
        assert!(config.deep_validation);
    }
}