//! Agent Builder GUI Types
//!
//! This module contains comprehensive data structures for the agent builder GUI application.
//! It includes configuration structures, GUI state management, validation helpers, and conversion functions.

use anyhow::{Result, anyhow};
use egui::{Color32, RichText};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import AgentRunner module for type compatibility
use super::AgentRunner;

// ============================================================================
// Core Configuration Structures
// ============================================================================

/// LLM Provider enumeration for type-safe provider selection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LLMProvider {
    Gemini,
    OpenRouter,
    Ollama,
    LMStudio,
}

impl LLMProvider {
    /// Get the string representation of the provider
    pub fn as_str(&self) -> &'static str {
        match self {
            LLMProvider::Gemini => "gemini",
            LLMProvider::OpenRouter => "openrouter",
            LLMProvider::Ollama => "ollama",
            LLMProvider::LMStudio => "lmstudio",
        }
    }

    /// Get all available providers
    pub fn all() -> Vec<Self> {
        vec![
            LLMProvider::Gemini,
            LLMProvider::OpenRouter,
            LLMProvider::Ollama,
            LLMProvider::LMStudio,
        ]
    }

    /// Check if the provider requires an API key
    pub fn requires_api_key(&self) -> bool {
        !matches!(self, LLMProvider::LMStudio)
    }

    /// Get the default base URL for the provider
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            LLMProvider::Gemini => Some("https://generativelanguage.googleapis.com/v1beta/openai"),
            LLMProvider::OpenRouter => Some("https://openrouter.ai/api/v1"),
            LLMProvider::Ollama => Some("http://localhost:11434/v1"),
            LLMProvider::LMStudio => Some("http://localhost:1234/v1"),
        }
    }

    /// Get default model for the provider
    pub fn default_model(&self) -> &'static str {
        match self {
            LLMProvider::Gemini => "gemini-pro",
            LLMProvider::OpenRouter => "anthropic/claude-3.5-sonnet",
            LLMProvider::Ollama => "llama2",
            LLMProvider::LMStudio => "local-model",
        }
    }

    /// Get the default API key for the provider (if any)
    pub fn default_api_key(&self) -> Option<&'static str> {
        match self {
            LLMProvider::Ollama => Some("ollama"),
            _ => None,
        }
    }
}

/// Tool transport type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolTransportType {
    Stdio,
    SSE,
    Docker,
}

impl ToolTransportType {
    /// Get the string representation of the transport type
    pub fn as_str(&self) -> &'static str {
        match self {
            ToolTransportType::Stdio => "stdio",
            ToolTransportType::SSE => "sse",
            ToolTransportType::Docker => "docker",
        }
    }

    /// Get all available transport types
    pub fn all() -> Vec<Self> {
        vec![
            ToolTransportType::Stdio,
            ToolTransportType::SSE,
            ToolTransportType::Docker,
        ]
    }
}

/// Response format enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseFormat {
    Json,
    Text,
    Markdown,
}

impl ResponseFormat {
    /// Get the string representation of the response format
    pub fn as_str(&self) -> &'static str {
        match self {
            ResponseFormat::Json => "json",
            ResponseFormat::Text => "text",
            ResponseFormat::Markdown => "markdown",
        }
    }

    /// Get all available response formats
    pub fn all() -> Vec<Self> {
        vec![
            ResponseFormat::Json,
            ResponseFormat::Text,
            ResponseFormat::Markdown,
        ]
    }
}

/// LLM Configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: LLMProvider,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: String,
    pub embed_model: Option<String>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            provider: LLMProvider::OpenRouter,
            api_key: None,
            base_url: Some(
                LLMProvider::OpenRouter
                    .default_base_url()
                    .unwrap()
                    .to_string(),
            ),
            model: LLMProvider::OpenRouter.default_model().to_string(),
            embed_model: Some("text-embedding-3-small".to_string()),
        }
    }
}

impl LLMConfig {
    /// Create a new LLMConfig with provider-specific defaults
    pub fn with_provider(provider: LLMProvider) -> Self {
        Self {
            api_key: provider.default_api_key().map(|s| s.to_string()),
            base_url: provider.default_base_url().map(|s| s.to_string()),
            model: provider.default_model().to_string(),
            provider,
            embed_model: Some("text-embedding-3-small".to_string()),
        }
    }
}

/// Stdio Tool Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdioToolConfig {
    pub command: String,
    pub args: Vec<String>,
    pub environment: Option<HashMap<String, String>>,
}

impl Default for StdioToolConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            args: Vec::new(),
            environment: Some(HashMap::new()),
        }
    }
}

/// Enhanced SSE Tool Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEToolConfig {
    pub port: u16,
    pub endpoint: String,
    /// Whether to use automatic port allocation
    pub auto_port: bool,
    /// Connection timeout in seconds
    pub timeout_seconds: Option<u64>,
    /// Whether to enable health monitoring
    pub health_monitoring: bool,
    /// Retry attempts for failed connections
    pub retry_attempts: Option<u32>,
    /// Additional headers to send with requests
    pub headers: Option<std::collections::HashMap<String, String>>,
    /// Whether to validate SSL certificates
    pub validate_ssl: Option<bool>,
    /// Connection keep-alive setting
    pub keep_alive: Option<bool>,
}

impl Default for SSEToolConfig {
    fn default() -> Self {
        Self {
            port: 3001,
            endpoint: "http://localhost:{port}/sse".to_string(),
            auto_port: true,
            timeout_seconds: Some(30),
            health_monitoring: true,
            retry_attempts: Some(3),
            headers: None,
            validate_ssl: Some(true),
            keep_alive: Some(true),
        }
    }
}

/// Tool Configuration (unified for all tool types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    pub name: String,
    pub enabled: bool,
    pub transport_type: ToolTransportType,
    pub stdio_config: Option<StdioToolConfig>,
    pub sse_config: Option<SSEToolConfig>,
    pub description: Option<String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            enabled: true,
            transport_type: ToolTransportType::Stdio,
            stdio_config: Some(StdioToolConfig::default()),
            sse_config: None,
            description: None,
        }
    }
}

/// Server Configuration for SSE server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub port: u16,
    pub bind_address: String,
    pub sse_path: String,
    pub post_path: String,
    pub keep_alive: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            bind_address: "127.0.0.1".to_string(),
            sse_path: "/sse".to_string(),
            post_path: "/message".to_string(),
            keep_alive: true,
        }
    }
}

/// Agent Behavior Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorConfig {
    pub retry_limit: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub max_tokens: Option<usize>,
    pub response_format: Option<ResponseFormat>,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            retry_limit: Some(3),
            timeout_seconds: Some(60),
            max_tokens: Some(4096),
            response_format: Some(ResponseFormat::Json),
        }
    }
}

/// Main Agent Configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub system_prompt: String,
    pub server: ServerConfig,
    pub llm: LLMConfig,
    pub tools: Vec<ToolConfig>,
    pub behavior: BehaviorConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "NewAgent".to_string(),
            description: Some("AI agent created with Agent Builder".to_string()),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            server: ServerConfig::default(),
            llm: LLMConfig::default(),
            tools: Vec::new(),
            behavior: BehaviorConfig::default(),
        }
    }
}

// ============================================================================
// GUI State Management Structures
// ============================================================================

/// Configuration tab enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigTab {
    Basic,
    LLM,
    Tools,
    Behavior,
    Build,
}

impl ConfigTab {
    /// Get the display name for the tab
    pub fn display_name(&self) -> &'static str {
        match self {
            ConfigTab::Basic => "Basic Settings",
            ConfigTab::LLM => "LLM Configuration",
            ConfigTab::Tools => "Tools",
            ConfigTab::Behavior => "Behavior Settings",
            ConfigTab::Build => "Build",
        }
    }

    /// Get all tabs in order
    pub fn all() -> Vec<Self> {
        vec![
            ConfigTab::Basic,
            ConfigTab::LLM,
            ConfigTab::Tools,
            ConfigTab::Behavior,
            ConfigTab::Build,
        ]
    }
}

/// Validation state for individual fields
#[derive(Debug, Clone)]
pub struct FieldValidation {
    pub is_valid: bool,
    pub error_message: Option<String>,
    pub warning_message: Option<String>,
}

impl FieldValidation {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            error_message: None,
            warning_message: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            is_valid: false,
            error_message: Some(message),
            warning_message: None,
        }
    }

    pub fn warning(message: String) -> Self {
        Self {
            is_valid: true,
            error_message: None,
            warning_message: Some(message),
        }
    }
}

/// Overall validation state
#[derive(Debug, Clone)]
pub struct ValidationState {
    pub field_validations: HashMap<String, FieldValidation>,
    pub is_valid: bool,
    pub global_errors: Vec<String>,
    pub global_warnings: Vec<String>,
}

impl ValidationState {
    pub fn new() -> Self {
        Self {
            field_validations: HashMap::new(),
            is_valid: true,
            global_errors: Vec::new(),
            global_warnings: Vec::new(),
        }
    }

    pub fn set_field_validation(&mut self, field: &str, validation: FieldValidation) {
        self.field_validations.insert(field.to_string(), validation);
        self.recalculate_validity();
    }

    pub fn get_field_validation(&self, field: &str) -> Option<&FieldValidation> {
        self.field_validations.get(field)
    }

    pub fn add_global_error(&mut self, error: String) {
        self.global_errors.push(error);
        self.recalculate_validity();
    }

    pub fn add_global_warning(&mut self, warning: String) {
        self.global_warnings.push(warning);
    }

    pub fn clear(&mut self) {
        self.field_validations.clear();
        self.global_errors.clear();
        self.global_warnings.clear();
        self.is_valid = true;
    }

    fn recalculate_validity(&mut self) {
        self.is_valid =
            self.global_errors.is_empty() && self.field_validations.values().all(|v| v.is_valid);
    }
}

/// Build process state
#[derive(Debug, Clone)]
pub enum BuildState {
    Idle,
    Validating,
    Generating,
    Building,
    Running,
    Completed,
    Failed(String),
}

impl BuildState {
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            BuildState::Validating | BuildState::Generating | BuildState::Building
        )
    }

    pub fn can_build(&self) -> bool {
        matches!(
            self,
            BuildState::Idle | BuildState::Completed | BuildState::Failed(_)
        )
    }

    pub fn can_run(&self) -> bool {
        matches!(self, BuildState::Completed)
    }
}

/// Build configuration options
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub target_platform: TargetPlatform,
    pub optimization_level: OptimizationLevel,
    pub output_directory: String,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            target_platform: TargetPlatform::Current,
            optimization_level: OptimizationLevel::Release,
            output_directory: "./output".to_string(),
        }
    }
}

/// Target platform enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetPlatform {
    Current,
    Windows,
    Linux,
    MacOS,
}

impl TargetPlatform {
    pub fn display_name(&self) -> &'static str {
        match self {
            TargetPlatform::Current => "Current Platform",
            TargetPlatform::Windows => "Windows",
            TargetPlatform::Linux => "Linux",
            TargetPlatform::MacOS => "macOS",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![
            TargetPlatform::Current,
            TargetPlatform::Windows,
            TargetPlatform::Linux,
            TargetPlatform::MacOS,
        ]
    }
}

/// Optimization level enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationLevel {
    Debug,
    Release,
}

impl OptimizationLevel {
    pub fn display_name(&self) -> &'static str {
        match self {
            OptimizationLevel::Debug => "Debug",
            OptimizationLevel::Release => "Release",
        }
    }

    pub fn all() -> Vec<Self> {
        vec![OptimizationLevel::Debug, OptimizationLevel::Release]
    }
}

/// UI-specific state
#[derive(Debug, Clone)]
pub struct UIState {
    pub active_tab: ConfigTab,
    pub expanded_sections: HashMap<String, bool>,
    pub selected_tool_index: Option<usize>,
    pub show_advanced_options: bool,
    pub show_generated_code: bool,
    pub show_generated_json: bool,
    pub window_size: Option<egui::Vec2>,
    pub scroll_position: Option<egui::Vec2>,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            active_tab: ConfigTab::Basic,
            expanded_sections: HashMap::new(),
            selected_tool_index: None,
            show_advanced_options: false,
            show_generated_code: false,
            show_generated_json: false,
            window_size: None,
            scroll_position: None,
        }
    }
}

/// Main application state
#[derive(Debug, Clone)]
pub struct AgentBuilderState {
    /// Configuration data
    pub config: AgentConfig,

    /// Validation state
    pub validation: ValidationState,

    /// Build state
    pub build_state: BuildState,
    pub build_options: BuildOptions,
    pub build_output: String,
    pub build_progress: f32,

    /// Generated content
    pub generated_json: String,
    pub generated_code: String,

    /// UI state
    pub ui: UIState,

    /// Application state
    pub has_unsaved_changes: bool,
    pub current_file_path: Option<String>,
    pub recent_files: Vec<String>,
}

impl Default for AgentBuilderState {
    fn default() -> Self {
        Self {
            config: AgentConfig::default(),
            validation: ValidationState::new(),
            build_state: BuildState::Idle,
            build_options: BuildOptions::default(),
            build_output: String::new(),
            build_progress: 0.0,
            generated_json: String::new(),
            generated_code: String::new(),
            ui: UIState::default(),
            has_unsaved_changes: false,
            current_file_path: None,
            recent_files: Vec::new(),
        }
    }
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validate agent name
pub fn validate_agent_name(name: &str) -> FieldValidation {
    if name.is_empty() {
        return FieldValidation::error("Agent name is required".to_string());
    }

    if name.len() > 100 {
        return FieldValidation::error("Agent name must be less than 100 characters".to_string());
    }

    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return FieldValidation::error(
            "Agent name can only contain letters, numbers, underscores, and hyphens".to_string(),
        );
    }

    FieldValidation::valid()
}

/// Validate system prompt
pub fn validate_system_prompt(prompt: &str) -> FieldValidation {
    if prompt.is_empty() {
        return FieldValidation::error("System prompt is required".to_string());
    }

    if prompt.len() > 10000 {
        return FieldValidation::error(
            "System prompt must be less than 10,000 characters".to_string(),
        );
    }

    FieldValidation::valid()
}

/// Validate port number
pub fn validate_port(port: u16) -> FieldValidation {
    if port == 0 {
        return FieldValidation::error("Port cannot be 0".to_string());
    }

    if port < 1024 {
        return FieldValidation::warning(
            "Ports below 1024 may require administrator privileges".to_string(),
        );
    }

    FieldValidation::valid()
}

/// Validate bind address
pub fn validate_bind_address(address: &str) -> FieldValidation {
    if address.is_empty() {
        return FieldValidation::error("Bind address is required".to_string());
    }

    // Basic IPv4 validation
    if address == "localhost" {
        return FieldValidation::valid(); // localhost is valid
    }

    let parts: Vec<&str> = address.split('.').collect();
    if parts.len() != 4 {
        return FieldValidation::error("Invalid IPv4 address format".to_string());
    }

    for part in parts {
        if let Ok(num) = part.parse::<u8>() {
            // Valid octet
        } else {
            return FieldValidation::error("Invalid IPv4 address format".to_string());
        }
    }

    FieldValidation::valid()
}

/// Validate API key
pub fn validate_api_key(api_key: &Option<String>, provider: &LLMProvider) -> FieldValidation {
    if provider.requires_api_key() {
        match api_key {
            Some(key) if key.is_empty() => {
                FieldValidation::error("API key is required for this provider".to_string())
            }
            Some(key) if key.len() < 10 => {
                FieldValidation::error("API key appears to be too short".to_string())
            }
            None => FieldValidation::error("API key is required for this provider".to_string()),
            _ => FieldValidation::valid(),
        }
    } else {
        FieldValidation::valid()
    }
}

/// Validate base URL
pub fn validate_base_url(url: &Option<String>, provider: &LLMProvider) -> FieldValidation {
    match url {
        Some(url_str) if url_str.is_empty() => {
            if provider.default_base_url().is_some() {
                FieldValidation::warning("Using default base URL".to_string())
            } else {
                FieldValidation::error("Base URL is required for this provider".to_string())
            }
        }
        Some(url_str) => {
            if url_str.starts_with("http://") || url_str.starts_with("https://") {
                FieldValidation::valid()
            } else {
                FieldValidation::error("Base URL must start with http:// or https://".to_string())
            }
        }
        None => {
            if provider.default_base_url().is_some() {
                FieldValidation::valid()
            } else {
                FieldValidation::error("Base URL is required for this provider".to_string())
            }
        }
    }
}

/// Validate model name
pub fn validate_model_name(model: &str) -> FieldValidation {
    if model.is_empty() {
        return FieldValidation::error("Model name is required".to_string());
    }

    if model.len() > 100 {
        return FieldValidation::error("Model name must be less than 100 characters".to_string());
    }

    FieldValidation::valid()
}

/// Validate tool configuration
pub fn validate_tool_config(tool: &ToolConfig) -> FieldValidation {
    if tool.name.is_empty() {
        return FieldValidation::error("Tool name is required".to_string());
    }

    match tool.transport_type {
        ToolTransportType::Stdio => {
            if let Some(stdio_config) = &tool.stdio_config {
                if stdio_config.command.is_empty() {
                    return FieldValidation::error(
                        "Command is required for stdio tools".to_string(),
                    );
                }
            } else {
                return FieldValidation::error("Stdio configuration is required".to_string());
            }
        }
        ToolTransportType::SSE => {
            if let Some(sse_config) = &tool.sse_config {
                if sse_config.endpoint.is_empty() {
                    return FieldValidation::error(
                        "Endpoint is required for SSE tools".to_string(),
                    );
                }

                // Validate port
                if !sse_config.auto_port && sse_config.port == 0 {
                    return FieldValidation::error(
                        "Port must be specified when auto_port is disabled".to_string(),
                    );
                }

                // Validate timeout
                if let Some(timeout) = sse_config.timeout_seconds {
                    if timeout == 0 {
                        return FieldValidation::error(
                            "Timeout must be greater than 0".to_string(),
                        );
                    }
                    if timeout > 300 {
                        return FieldValidation::warning(
                            "Timeout over 5 minutes may be excessive".to_string(),
                        );
                    }
                }

                // Validate retry attempts
                if let Some(retries) = sse_config.retry_attempts {
                    if retries == 0 {
                        return FieldValidation::error(
                            "Retry attempts must be greater than 0".to_string(),
                        );
                    }
                    if retries > 10 {
                        return FieldValidation::warning(
                            "High retry count may cause long delays".to_string(),
                        );
                    }
                }
            } else {
                return FieldValidation::error("SSE configuration is required".to_string());
            }
        }
        ToolTransportType::Docker => {
            // Docker tools would need specific validation
            if let Some(stdio_config) = &tool.stdio_config {
                if !stdio_config.command.contains("docker") {
                    return FieldValidation::warning(
                        "Docker tools typically use 'docker' command".to_string(),
                    );
                }
            }
        }
    }

    FieldValidation::valid()
}

/// Validate SSE endpoint configuration
pub fn validate_sse_endpoint_config(config: &SSEToolConfig) -> FieldValidation {
    if config.endpoint.is_empty() {
        return FieldValidation::error("SSE endpoint URL is required".to_string());
    }

    // Check if endpoint URL is valid
    if !config.endpoint.starts_with("http://") && !config.endpoint.starts_with("https://") {
        return FieldValidation::error(
            "SSE endpoint must start with http:// or https://".to_string(),
        );
    }

    // Validate port if not using auto allocation
    if !config.auto_port {
        if config.port == 0 {
            return FieldValidation::error(
                "Port must be specified when auto_port is disabled".to_string(),
            );
        }

        if config.port < 1024 {
            return FieldValidation::warning(
                "Ports below 1024 may require administrator privileges".to_string(),
            );
        }
    }

    // Validate timeout
    if let Some(timeout) = config.timeout_seconds {
        if timeout == 0 {
            return FieldValidation::error("Timeout must be greater than 0".to_string());
        }
        if timeout > 300 {
            return FieldValidation::warning("Timeout over 5 minutes may be excessive".to_string());
        }
    }

    // Validate retry attempts
    if let Some(retries) = config.retry_attempts {
        if retries == 0 {
            return FieldValidation::error("Retry attempts must be greater than 0".to_string());
        }
        if retries > 10 {
            return FieldValidation::warning("High retry count may cause long delays".to_string());
        }
    }

    FieldValidation::valid()
}

/// Get SSE configuration suggestions
pub fn get_sse_config_suggestions() -> Vec<(String, SSEToolConfig)> {
    vec![
        (
            "Local MCP Service".to_string(),
            SSEToolConfig {
                port: 3001,
                endpoint: "http://localhost:{port}/sse".to_string(),
                auto_port: true,
                timeout_seconds: Some(30),
                health_monitoring: true,
                retry_attempts: Some(3),
                headers: None,
                validate_ssl: Some(true),
                keep_alive: Some(true),
            },
        ),
        (
            "Remote MCP Service".to_string(),
            SSEToolConfig {
                port: 8080,
                endpoint: "https://remote-service.com/sse".to_string(),
                auto_port: false,
                timeout_seconds: Some(60),
                health_monitoring: true,
                retry_attempts: Some(5),
                headers: Some({
                    let mut headers = std::collections::HashMap::new();
                    headers.insert("Authorization".to_string(), "Bearer {token}".to_string());
                    headers
                }),
                validate_ssl: Some(true),
                keep_alive: Some(true),
            },
        ),
        (
            "Development SSE Tool".to_string(),
            SSEToolConfig {
                port: 3002,
                endpoint: "http://localhost:{port}/events".to_string(),
                auto_port: true,
                timeout_seconds: Some(10),
                health_monitoring: false,
                retry_attempts: Some(1),
                headers: None,
                validate_ssl: Some(false),
                keep_alive: Some(false),
            },
        ),
    ]
}

/// Validate retry limit
pub fn validate_retry_limit(limit: Option<u32>) -> FieldValidation {
    match limit {
        Some(0) => FieldValidation::error("Retry limit must be greater than 0".to_string()),
        Some(limit) if limit > 100 => {
            FieldValidation::warning("High retry limit may cause long delays".to_string())
        }
        _ => FieldValidation::valid(),
    }
}

/// Validate timeout
pub fn validate_timeout(timeout: Option<u64>) -> FieldValidation {
    match timeout {
        Some(0) => FieldValidation::error("Timeout must be greater than 0".to_string()),
        Some(timeout) if timeout > 3600 => {
            FieldValidation::warning("Timeout over 1 hour may be excessive".to_string())
        }
        _ => FieldValidation::valid(),
    }
}

/// Validate max tokens
pub fn validate_max_tokens(max_tokens: Option<usize>) -> FieldValidation {
    match max_tokens {
        Some(0) => FieldValidation::error("Max tokens must be greater than 0".to_string()),
        Some(tokens) if tokens > 32768 => {
            FieldValidation::warning("Very high token limit may be expensive".to_string())
        }
        _ => FieldValidation::valid(),
    }
}

/// Validate entire agent configuration
pub fn validate_agent_config(config: &AgentConfig) -> ValidationState {
    let mut validation = ValidationState::new();

    // Validate basic fields
    validation.set_field_validation("name", validate_agent_name(&config.name));
    validation.set_field_validation(
        "system_prompt",
        validate_system_prompt(&config.system_prompt),
    );
    validation.set_field_validation("port", validate_port(config.server.port));
    validation.set_field_validation(
        "bind_address",
        validate_bind_address(&config.server.bind_address),
    );

    // Validate LLM configuration
    validation.set_field_validation("llm_provider", FieldValidation::valid());
    validation.set_field_validation(
        "api_key",
        validate_api_key(&config.llm.api_key, &config.llm.provider),
    );
    validation.set_field_validation(
        "base_url",
        validate_base_url(&config.llm.base_url, &config.llm.provider),
    );
    validation.set_field_validation("model", validate_model_name(&config.llm.model));

    // Validate tools
    for (index, tool) in config.tools.iter().enumerate() {
        let field_name = format!("tool_{}", index);
        validation.set_field_validation(&field_name, validate_tool_config(tool));
    }

    // Validate behavior settings
    validation.set_field_validation(
        "retry_limit",
        validate_retry_limit(config.behavior.retry_limit),
    );
    validation.set_field_validation(
        "timeout_seconds",
        validate_timeout(config.behavior.timeout_seconds),
    );
    validation.set_field_validation(
        "max_tokens",
        validate_max_tokens(config.behavior.max_tokens),
    );

    // Check for duplicate tool names
    let mut tool_names = std::collections::HashSet::new();
    for tool in &config.tools {
        if tool_names.contains(&tool.name) {
            validation.add_global_error(format!("Duplicate tool name: {}", tool.name));
        }
        tool_names.insert(tool.name.clone());
    }

    validation
}

// ============================================================================
// Conversion Functions
// ============================================================================

/// Convert GUI state to legacy AgentConfig format (compatible with AgentRunner.rs)
pub fn to_legacy_agent_config(config: &AgentConfig) -> Result<AgentRunner::AgentConfig> {
    use std::collections::HashMap;

    // Convert tools to legacy format
    let mut stdio_tools = HashMap::new();
    let mut sse_tools = HashMap::new();

    for tool in &config.tools {
        if tool.enabled {
            match tool.transport_type {
                ToolTransportType::Stdio | ToolTransportType::Docker => {
                    if let Some(stdio_config) = &tool.stdio_config {
                        stdio_tools.insert(
                            tool.name.clone(),
                            AgentRunner::StdioToolConfig {
                                command: stdio_config.command.clone(),
                                args: stdio_config.args.clone(),
                                environment: stdio_config.environment.clone(),
                            },
                        );
                    }
                }
                ToolTransportType::SSE => {
                    if let Some(sse_config) = &tool.sse_config {
                        sse_tools.insert(
                            tool.name.clone(),
                            AgentRunner::SSEToolConfig {
                                port: sse_config.port,
                                endpoint: sse_config.endpoint.clone(),
                            },
                        );
                    }
                }
            }
        }
    }

    Ok(AgentRunner::AgentConfig {
        name: config.name.clone(),
        description: config.description.clone(),
        system_prompt: config.system_prompt.clone(),
        port: config.server.port,
        bind_address: config.server.bind_address.clone(),
        llm: AgentRunner::LLMConfig {
            provider: config.llm.provider.as_str().to_string(),
            api_key: config.llm.api_key.clone(),
            base_url: config.llm.base_url.clone(),
            model: config.llm.model.clone(),
            embed_model: config.llm.embed_model.clone(),
        },
        stdio_tools,
        sse_tools,
        retry_limit: config.behavior.retry_limit,
        timeout_seconds: config.behavior.timeout_seconds,
        max_tokens: config.behavior.max_tokens,
        response_format: config
            .behavior
            .response_format
            .as_ref()
            .map(|f| f.as_str().to_string()),
        sse_path: Some(config.server.sse_path.clone()),
        post_path: Some(config.server.post_path.clone()),
        keep_alive: Some(config.server.keep_alive),
    })
}

/// Convert legacy AgentConfig to GUI state format
pub fn from_legacy_agent_config(legacy: &AgentRunner::AgentConfig) -> AgentConfig {
    use std::collections::HashMap;

    // Convert tools from legacy format
    let mut tools = Vec::new();

    // Convert stdio tools
    for (name, stdio_config) in &legacy.stdio_tools {
        tools.push(ToolConfig {
            name: name.clone(),
            enabled: true,
            transport_type: if stdio_config.command.starts_with("docker") {
                ToolTransportType::Docker
            } else {
                ToolTransportType::Stdio
            },
            stdio_config: Some(StdioToolConfig {
                command: stdio_config.command.clone(),
                args: stdio_config.args.clone(),
                environment: stdio_config.environment.clone(),
            }),
            sse_config: None,
            description: None,
        });
    }

    // Convert SSE tools
    for (name, sse_config) in &legacy.sse_tools {
        tools.push(ToolConfig {
            name: name.clone(),
            enabled: true,
            transport_type: ToolTransportType::SSE,
            stdio_config: None,
            sse_config: Some(SSEToolConfig {
                port: sse_config.port,
                endpoint: sse_config.endpoint.clone(),
                auto_port: true,           // Default to automatic port allocation
                timeout_seconds: Some(30), // Default 30 second timeout
                health_monitoring: true,   // Enable health monitoring by default
                retry_attempts: Some(3),   // Default 3 retry attempts
                headers: None,             // No custom headers by default
                validate_ssl: Some(true),  // Validate SSL by default
                keep_alive: Some(true),    // Enable keep-alive by default
            }),
            description: None,
        });
    }

    // Parse LLM provider
    let provider = match legacy.llm.provider.as_str() {
        "gemini" => LLMProvider::Gemini,
        "openrouter" => LLMProvider::OpenRouter,
        "ollama" => LLMProvider::Ollama,
        "lmstudio" => LLMProvider::LMStudio,
        _ => LLMProvider::OpenRouter, // Default fallback
    };

    // Parse response format
    let response_format = legacy
        .response_format
        .as_ref()
        .and_then(|f| match f.as_str() {
            "json" => Some(ResponseFormat::Json),
            "text" => Some(ResponseFormat::Text),
            "markdown" => Some(ResponseFormat::Markdown),
            _ => None,
        });

    AgentConfig {
        name: legacy.name.clone(),
        description: legacy.description.clone(),
        system_prompt: legacy.system_prompt.clone(),
        server: ServerConfig {
            port: legacy.port,
            bind_address: legacy.bind_address.clone(),
            sse_path: legacy
                .sse_path
                .clone()
                .unwrap_or_else(|| "/sse".to_string()),
            post_path: legacy
                .post_path
                .clone()
                .unwrap_or_else(|| "/message".to_string()),
            keep_alive: legacy.keep_alive.unwrap_or(true),
        },
        llm: LLMConfig {
            provider: provider.clone(),
            api_key: legacy
                .llm
                .api_key
                .clone()
                .or_else(|| provider.default_api_key().map(|s| s.to_string())),
            base_url: legacy.llm.base_url.clone(),
            model: legacy.llm.model.clone(),
            embed_model: legacy.llm.embed_model.clone(),
        },
        tools,
        behavior: BehaviorConfig {
            retry_limit: legacy.retry_limit,
            timeout_seconds: legacy.timeout_seconds,
            max_tokens: legacy.max_tokens,
            response_format,
        },
    }
}

/// Generate JSON configuration from GUI state
pub fn generate_json_config(config: &AgentConfig) -> Result<String> {
    let legacy_config = to_legacy_agent_config(config)?;
    serde_json::to_string_pretty(&legacy_config)
        .map_err(|e| anyhow!("Failed to generate JSON: {}", e))
}

/// Parse JSON configuration into GUI state
pub fn parse_json_config(json: &str) -> Result<AgentConfig> {
    let legacy_config: AgentRunner::AgentConfig =
        serde_json::from_str(json).map_err(|e| anyhow!("Failed to parse JSON: {}", e))?;
    Ok(from_legacy_agent_config(&legacy_config))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a new tool with default configuration based on transport type
pub fn create_default_tool(transport_type: ToolTransportType) -> ToolConfig {
    match transport_type {
        ToolTransportType::Stdio => ToolConfig {
            name: "new_stdio_tool".to_string(),
            enabled: true,
            transport_type,
            stdio_config: Some(StdioToolConfig::default()),
            sse_config: None,
            description: None,
        },
        ToolTransportType::SSE => ToolConfig {
            name: "new_sse_tool".to_string(),
            enabled: true,
            transport_type,
            stdio_config: None,
            sse_config: Some(SSEToolConfig::default()),
            description: None,
        },
        ToolTransportType::Docker => ToolConfig {
            name: "new_docker_tool".to_string(),
            enabled: true,
            transport_type,
            stdio_config: Some(StdioToolConfig {
                command: "docker".to_string(),
                args: vec![
                    "run".to_string(),
                    "-i".to_string(),
                    "--rm".to_string(),
                    "image-name".to_string(),
                ],
                environment: Some(HashMap::new()),
            }),
            sse_config: None,
            description: None,
        },
    }
}

/// Get predefined tool templates
pub fn get_tool_templates() -> Vec<(String, ToolConfig)> {
    vec![
        (
            "Neo4j Database".to_string(),
            ToolConfig {
                name: "neo4j-cypher".to_string(),
                enabled: true,
                transport_type: ToolTransportType::Docker,
                stdio_config: Some(StdioToolConfig {
                    command: "docker".to_string(),
                    args: vec![
                        "run".to_string(),
                        "-i".to_string(),
                        "--rm".to_string(),
                        "-e".to_string(),
                        "NEO4J_URL=bolt://host.docker.internal:7687".to_string(),
                        "-e".to_string(),
                        "NEO4J_USERNAME=neo4j".to_string(),
                        "-e".to_string(),
                        "NEO4J_PASSWORD=password".to_string(),
                        "-e".to_string(),
                        "NEO4J_DATABASE=neo4j".to_string(),
                        "-e".to_string(),
                        "NEO4J_NAMESPACE=local".to_string(),
                        "-e".to_string(),
                        "NEO4J_TRANSPORT=stdio".to_string(),
                        "mcp/neo4j-cypher:latest".to_string(),
                    ],
                    environment: Some({
                        let mut env = HashMap::new();
                        env.insert(
                            "NEO4J_URL".to_string(),
                            "bolt://host.docker.internal:7687".to_string(),
                        );
                        env.insert("NEO4J_USERNAME".to_string(), "neo4j".to_string());
                        env.insert("NEO4J_PASSWORD".to_string(), "password".to_string());
                        env.insert("NEO4J_DATABASE".to_string(), "neo4j".to_string());
                        env.insert("NEO4J_NAMESPACE".to_string(), "local".to_string());
                        env.insert("NEO4J_TRANSPORT".to_string(), "stdio".to_string());
                        env
                    }),
                }),
                sse_config: None,
                description: Some("Neo4j Cypher query tool via Docker".to_string()),
            },
        ),
        (
            "Qdrant Vector Database".to_string(),
            ToolConfig {
                name: "qdrant".to_string(),
                enabled: true,
                transport_type: ToolTransportType::Stdio,
                stdio_config: Some(StdioToolConfig {
                    command: "uvx".to_string(),
                    args: vec!["mcp-server-qdrant".to_string()],
                    environment: Some({
                        let mut env = HashMap::new();
                        env.insert(
                            "QDRANT_URL".to_string(),
                            "http://localhost:6336".to_string(),
                        );
                        env.insert("COLLECTION_NAME".to_string(), "documents".to_string());
                        env.insert(
                            "EMBEDDING_MODEL".to_string(),
                            "BAAI/bge-small-en-v1.5".to_string(),
                        );
                        env
                    }),
                }),
                sse_config: None,
                description: Some("Qdrant vector database tool".to_string()),
            },
        ),
        (
            "Remote MCP Service".to_string(),
            ToolConfig {
                name: "remote-mcp-service".to_string(),
                enabled: true,
                transport_type: ToolTransportType::SSE,
                stdio_config: None,
                sse_config: Some(SSEToolConfig {
                    port: 3001,
                    endpoint: "http://localhost:{port}/sse".to_string(),
                    auto_port: true,
                    timeout_seconds: Some(30),
                    health_monitoring: true,
                    retry_attempts: Some(3),
                    headers: None,
                    validate_ssl: Some(true),
                    keep_alive: Some(true),
                }),
                description: Some("Remote MCP service via SSE".to_string()),
            },
        ),
    ]
}

/// Generate a unique tool name
pub fn generate_unique_tool_name(base_name: &str, existing_tools: &[ToolConfig]) -> String {
    let mut counter = 1;
    let mut candidate = format!("{}_{}", base_name, counter);

    while existing_tools.iter().any(|tool| tool.name == candidate) {
        counter += 1;
        candidate = format!("{}_{}", base_name, counter);
    }

    candidate
}

/// Check if a port is available (basic check)
pub fn is_port_available(port: u16) -> bool {
    // This is a simplified check - in a real implementation, you would
    // actually try to bind to the port to see if it's available
    port != 0 && port < 65535
}

/// Get suggested ports for common services
pub fn get_suggested_ports() -> Vec<u16> {
    vec![8080, 8081, 8082, 3000, 3001, 5000, 5001, 9000, 9001]
}

/// Format build progress as percentage
pub fn format_build_progress(progress: f32) -> String {
    format!("{:.0}%", progress * 100.0)
}

/// Get color for validation status
pub fn validation_color(is_valid: bool) -> Color32 {
    if is_valid {
        Color32::from_rgb(0, 128, 0) // Green
    } else {
        Color32::from_rgb(200, 0, 0) // Red
    }
}

/// Get rich text for validation status
pub fn validation_rich_text(is_valid: bool, text: &str) -> RichText {
    RichText::new(text).color(validation_color(is_valid))
}

/// Check if the configuration has been modified
pub fn has_unsaved_changes(state: &AgentBuilderState) -> bool {
    state.has_unsaved_changes
}

/// Mark the configuration as modified
pub fn mark_as_modified(state: &mut AgentBuilderState) {
    state.has_unsaved_changes = true;
}

/// Mark the configuration as saved
pub fn mark_as_saved(state: &mut AgentBuilderState, file_path: Option<String>) {
    state.has_unsaved_changes = false;
    state.current_file_path = file_path;

    // Update recent files
    if let Some(path) = &state.current_file_path {
        if !state.recent_files.contains(path) {
            state.recent_files.insert(0, path.clone());
            // Keep only the last 10 files
            state.recent_files.truncate(10);
        }
    }
}

/// Get a display name for the current configuration
pub fn get_display_name(state: &AgentBuilderState) -> String {
    match &state.current_file_path {
        Some(path) => std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Untitled")
            .to_string(),
        None => {
            if state.config.name.is_empty() {
                "Untitled".to_string()
            } else {
                state.config.name.clone()
            }
        }
    }
}

/// Reset the application state to defaults
pub fn reset_to_defaults(state: &mut AgentBuilderState) {
    *state = AgentBuilderState::default();
}

/// Create a backup of the current configuration
pub fn create_backup(state: &AgentBuilderState) -> String {
    serde_json::to_string(&state.config).unwrap_or_default()
}

/// Restore configuration from backup
pub fn restore_from_backup(state: &mut AgentBuilderState, backup: &str) -> Result<()> {
    let config: AgentConfig = serde_json::from_str(backup)
        .map_err(|e| anyhow!("Failed to restore from backup: {}", e))?;
    state.config = config;
    state.has_unsaved_changes = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_provider_defaults() {
        let provider = LLMProvider::OpenRouter;
        assert_eq!(provider.as_str(), "openrouter");
        assert!(provider.requires_api_key());
        assert_eq!(
            provider.default_base_url(),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(provider.default_model(), "anthropic/claude-3.5-sonnet");
    }

    #[test]
    fn test_validation() {
        assert!(!validate_agent_name("").is_valid);
        assert!(!validate_agent_name("name with spaces").is_valid);
        assert!(validate_agent_name("valid_name-123").is_valid);

        assert!(!validate_port(0).is_valid);
        assert!(validate_port(8080).is_valid);

        assert!(!validate_bind_address("").is_valid);
        assert!(validate_bind_address("127.0.0.1").is_valid);
        assert!(validate_bind_address("localhost").is_valid);
    }

    #[test]
    fn test_config_conversion() {
        let config = AgentConfig::default();
        let legacy = to_legacy_agent_config(&config).unwrap();
        let restored = from_legacy_agent_config(&legacy);

        assert_eq!(config.name, restored.name);
        assert_eq!(config.system_prompt, restored.system_prompt);
    }

    #[test]
    fn test_tool_creation() {
        let tool = create_default_tool(ToolTransportType::Stdio);
        assert_eq!(tool.transport_type, ToolTransportType::Stdio);
        assert!(tool.stdio_config.is_some());

        let tool = create_default_tool(ToolTransportType::SSE);
        assert_eq!(tool.transport_type, ToolTransportType::SSE);
        assert!(tool.sse_config.is_some());
    }

    #[test]
    fn test_unique_name_generation() {
        let existing_tools = vec![
            ToolConfig {
                name: "tool_1".to_string(),
                ..Default::default()
            },
            ToolConfig {
                name: "tool_2".to_string(),
                ..Default::default()
            },
        ];

        let unique_name = generate_unique_tool_name("tool", &existing_tools);
        assert_eq!(unique_name, "tool_1");

        let mut more_tools = existing_tools;
        more_tools.push(ToolConfig {
            name: "tool_1".to_string(),
            ..Default::default()
        });

        let unique_name = generate_unique_tool_name("tool", &more_tools);
        assert_eq!(unique_name, "tool_2");
    }
}
