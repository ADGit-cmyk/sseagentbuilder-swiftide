# Agent Builder GUI Design for Swiftide

## Overview

This document outlines the design for a comprehensive GUI application that allows users to create and build AI agents as executable files with SSE endpoints. The system uses a generic runner approach where configuration is loaded from JSON files.

## Architecture Overview

### Generic Runner Pattern

The core architecture follows a pattern where:
1. A generic executable (`GenericRunner.exe`) loads configuration from a JSON file
2. The JSON file name matches the executable name (e.g., `MarketingAgent.exe` loads `MarketingAgent.json`)
3. The runner dynamically creates agents based on the configuration

### GenericRunner.rs Implementation

```rust
use std::env;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use tracing::{info, error};
use tokio;
use rmcp::{
    ServerHandler,
    tool_handler,
    tool_router,
    schemars,
    tool,
    handler::server::{
        router::tool::ToolRouter,
        tool::Parameters,
    },
    model::{ServerCapabilities, ServerInfo, CallToolResult, ErrorData, ErrorCode, Content},
    transport::{sse_server::{SseServer, SseServerConfig}},
};
use rmcp::{ServiceExt as _, model::{ClientInfo, Implementation}, transport::{ConfigureCommandExt as _, TokioChildProcess},};
use swiftide::agents::tools::mcp::McpToolbox;
use swiftide_agents::Agent;
use swiftide_integrations::{gemini::Gemini, open_router::OpenRouter, ollama::Ollama, lmstudio::LMStudio, groq::Groq};
use swiftide_core::chat_completion::traits::ChatCompletion;
use tokio_util::sync::CancellationToken;
use serde_json::json;
use sqlx::types::chrono::Utc;
use reqwest;
use rmcp::transport::sse_client::{SseClientTransport, SseClientConfig};

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct AgentProcessRequest {
    #[schemars(description = "The content to process with the agent.")]
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LLMConfig {
    pub provider: String, // "gemini", "openrouter", "ollama", "lmstudio", "groq"
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: String,
    pub embed_model: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MCPServerConfig {
    pub name: String,
    pub enabled: bool,
    pub transport_type: String, // "docker", "sse", "stdio"
    pub config: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub system_prompt: String,
    pub port: u16,
    pub bind_address: String,
    
    // LLM Configuration
    pub llm: LLMConfig,
    
    // Unlimited MCP Servers Configuration
    pub mcp_servers: Vec<MCPServerConfig>,
    
    // Behavior Settings
    pub retry_limit: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub max_tokens: Option<usize>,
    
    // Response Formatting
    pub response_format: Option<String>, // "json", "text", "markdown"
    
    // SSE Server Settings
    pub sse_path: Option<String>,
    pub post_path: Option<String>,
    pub keep_alive: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct GenericAgentServer {
    config: AgentConfig,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl GenericAgentServer {
    pub fn new(config: AgentConfig) -> Self {
        Self { 
            config,
            tool_router: Self::tool_router()
        }
    }

    #[tool(description = "Process content using the configured agent")]
    async fn process_content(
        &self,
        Parameters(request): Parameters<AgentProcessRequest>
    ) -> Result<CallToolResult, ErrorData> {
        info!("Processing content with agent {}: {}", self.config.name, request.content);

        // Create agent based on configuration
        let agent_result = self.create_agent().await;
        let mut agent = match agent_result {
            Ok(agent) => agent,
            Err(e) => {
                error!("Failed to create agent: {}", e);
                return Err(ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None));
            }
        };

        // Execute the agent
        match agent.query(request.content).await {
            Ok(_) => {
                // Get the response from the agent's context
                let history = agent.context().history().await
                    .map_err(|e| ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))?;

                // Find the last assistant message
                let mut response_content = None;
                for msg in history.iter().rev() {
                    if let swiftide_core::chat_completion::ChatMessage::Assistant(content, _) = msg {
                        response_content = content.clone();
                        break;
                    }
                }

                info!("Agent query result: {:?}", response_content);

                // Format response into OpenAI-compatible JSON
                let openai_response = json!({
                    "id": "chatcmpl-generated",
                    "object": "chat.completion",
                    "created": Utc::now().timestamp(),
                    "model": self.config.llm.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content.unwrap_or_else(|| "No response from agent".to_string()),
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                });

                Ok(CallToolResult::success(vec![Content::text(
                    serde_json::to_string(&openai_response).map_err(|e|
                        ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None)
                    )?
                )]))
            }
            Err(e) => {
                error!("Agent query failed: {}", e);
                Err(ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None))
            }
        }
    }

    async fn create_agent(&self) -> Result<Agent> {
        // Create LLM client based on configuration
        let client: Box<dyn ChatCompletion> = match self.config.llm.provider.as_str() {
            "gemini" => {
                let config = swiftide_integrations::gemini::config::GeminiConfig::default();
                let gemini_client = Gemini::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(gemini_client)
            }
            "openrouter" => {
                let config = swiftide_integrations::open_router::config::OpenRouterConfig::builder()
                    .api_base(self.config.llm.base_url.as_ref().unwrap_or(&"https://openrouter.ai/api/v1".to_string()))
                    .api_key(self.config.llm.api_key.as_ref().unwrap_or(&"".to_string()))
                    .build()?;
                let openrouter_client = swiftide_integrations::open_router::OpenRouter::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(openrouter_client)
            }
            "ollama" => {
                let config = swiftide_integrations::ollama::config::OllamaConfig::builder()
                    .api_base(self.config.llm.base_url.as_ref().unwrap_or(&"http://localhost:11434/v1".to_string()))
                    .api_key(self.config.llm.api_key.as_ref().unwrap_or(&"ollama".to_string()))
                    .build()?;
                let ollama_client = Ollama::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(ollama_client)
            }
            "lmstudio" => {
                let config = swiftide_integrations::lmstudio::config::LMStudioConfig::builder()
                    .api_base(self.config.llm.base_url.as_ref().unwrap_or(&"http://localhost:1234/v1".to_string()))
                    .api_key(self.config.llm.api_key.as_ref().unwrap_or(&"".to_string()))
                    .build()?;
                let lmstudio_client = LMStudio::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(lmstudio_client)
            }
            "groq" => {
                let config = swiftide_integrations::groq::config::GroqConfig::default();
                let groq_client = Groq::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(groq_client)
            }
            _ => return Err(anyhow::anyhow!("Unsupported LLM provider: {}", self.config.llm.provider)),
        };

        // Create agent builder
        let mut agent_builder = Agent::builder()
            .llm(&client)
            .system_prompt(&self.config.system_prompt)
            .name(&self.config.name);

        // Add retry limit if specified
        if let Some(retry_limit) = self.config.retry_limit {
            agent_builder = agent_builder.limit(retry_limit);
        }

        // Add toolboxes based on configuration - support unlimited number of tools
        let mut toolboxes = Vec::new();

        // Process all configured MCP servers
        for mcp_server in &self.config.mcp_servers {
            if mcp_server.enabled {
                let mcp_toolbox = self.create_mcp_toolbox(mcp_server).await?;
                toolboxes.push(mcp_toolbox);
            }
        }

        // Add all toolboxes to agent
        for toolbox in toolboxes {
            agent_builder = agent_builder.add_toolbox(toolbox);
        }

        // Build and return the agent
        agent_builder.build()
    }

    async fn create_mcp_toolbox(&self, config: &MCPServerConfig) -> Result<McpToolbox> {
        match config.transport_type.as_str() {
            "docker" | "stdio" => {
                let client_info = ClientInfo {
                    client_info: Implementation {
                        name: "generic-agent".into(),
                        version: env!("CARGO_PKG_VERSION").into(),
                    },
                    ..Default::default()
                };

                // Parse Docker configuration
                let docker_config = &config.config;
                let image = docker_config.get("image").and_then(|v| v.as_str()).unwrap_or(&config.name);
                let env_vars = docker_config.get("env").and_then(|v| v.as_object()).unwrap_or(&serde_json::Map::new());

                let mut cmd = tokio::process::Command::new("docker");
                cmd.args(["run", "-i", "--rm"]);

                // Add environment variables
                for (key, value) in env_vars {
                    if let Some(str_val) = value.as_str() {
                        cmd.args(["-e", &format!("{}={}", key, str_val)]);
                    }
                }

                cmd.args([image]);

                let child_process = TokioChildProcess::new(cmd)?;
                let running_service = client_info.serve(child_process).await?;
                Ok(McpToolbox::from_running_service(running_service))
            }
            "sse" => {
                let sse_endpoint = config.config.get("endpoint")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("SSE endpoint not specified"))?;

                let client = reqwest::Client::new();
                let sse_config = SseClientConfig {
                    sse_endpoint: sse_endpoint.to_string(),
                    ..Default::default()
                };

                let transport = SseClientTransport::start_with_client(client, sse_config).await?;
                Ok(McpToolbox::try_from_transport(transport).await?)
            }
            _ => Err(anyhow::anyhow!("Unsupported MCP transport type: {}", config.transport_type)),
        }
    }
}

#[tool_handler]
impl ServerHandler for GenericAgentServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(format!("Generic agent server: {}", self.config.description.as_ref().unwrap_or(&"Configurable AI agent".to_string()))),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing_subscriber::fmt::init();
    
    // Get path of currently running executable
    let current_exe_path = env::current_exe().context("Failed to get current exe path")?;
    
    // Extract filename without extension (e.g., "MarketingAgent")
    let agent_name = current_exe_path
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?
        .to_string_lossy()
        .to_string();

    info!("🚀 Starting Agent: {}", agent_name);

    // Construct expected config filename (e.g., "MarketingAgent.json")
    let config_path = current_exe_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Invalid executable path"))?
        .join(format!("{}.json", agent_name));

    // Load and Parse JSON
    if !config_path.exists() {
        error!("❌ Error: Could not find configuration file: {:?}", config_path);
        return Err(anyhow::anyhow!("Configuration file not found: {:?}", config_path));
    }

    let config_content = fs::read_to_string(&config_path)
        .context("Failed to read config file")?;
    
    let config: AgentConfig = serde_json::from_str(&config_content)
        .context("Invalid JSON format in configuration file")?;

    info!("✅ Configuration Loaded: {:?}", config.name);

    // Start Server using config
    start_server(config).await
}

async fn start_server(config: AgentConfig) -> Result<()> {
    // Configure SSE server
    let sse_config = SseServerConfig {
        bind: format!("{}:{}", config.bind_address, config.port).parse()?,
        sse_path: config.sse_path.unwrap_or_else(|| "/sse".to_string()),
        post_path: config.post_path.unwrap_or_else(|| "/message".to_string()),
        ct: CancellationToken::new(),
        sse_keep_alive: config.keep_alive.map(|v| if v { Some(60) } else { None }),
    };

    info!("Server configured to bind to {}:{}", config.bind_address, config.port);

    // Create SSE server
    let (sse_server, router) = SseServer::new(sse_config);

    let listener = tokio::net::TcpListener::bind(sse_server.config.bind).await
        .context("Failed to bind TCP listener")?;
    info!("TCP listener bound to {}", listener.local_addr()?);
    let ct = sse_server.config.ct.child_token();

    // Start server in separate task
    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        ct.cancelled().await;
        info!("Agent server cancelled");
    });

    tokio::spawn(async move {
        info!("Server task started");
        if let Err(e) = server.await {
            error!("Server error: {}", e);
        }
    });

    // Attach the service
    let ct = sse_server.with_service(|| GenericAgentServer::new(config.clone()));

    info!("Agent server listening on {}:{}", config.bind_address, config.port);

    // Wait for cancellation signal
    tokio::signal::ctrl_c().await
        .context("Failed to listen for Ctrl+C signal")?;
    ct.cancel();

    Ok(())
}
```

## GUI Design Specification

### Main Application Structure

The GUI will be built using egui and will consist of the following main components:

1. **Main Window**: Tabbed interface with different configuration sections
2. **Configuration Forms**: Forms for each configuration category
3. **Code Generation Panel**: Shows generated Rust code and JSON configuration
4. **Build System**: Compiles the agent and creates executable
5. **Status/Feedback Panel**: Shows build progress and errors

### GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ Agent Builder - Swiftide                                    │
├─────────────────────────────────────────────────────────────────┤
│ [Basic Settings] [LLM Config] [Tools] [Behavior] [Build]   │
├─────────────────────────────────────────────────────────────────┤
│                                                         │
│  Configuration Panel (changes based on selected tab)         │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────────────┤
│ [Generate Code] [Build Agent] [Run Agent] [Save Config]   │
├─────────────────────────────────────────────────────────────────┤
│ Status: Ready                                            │
│ Progress: ████████████████████████████████████ 100%        │
│ Output: Agent built successfully                            │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Categories

#### 1. Basic Settings Tab
- Agent Name (text input)
- Description (textarea)
- Bind Address (text input, default: "127.0.0.1")
- Port (number input, default: 8080)
- SSE Path (text input, default: "/sse")
- Post Path (text input, default: "/message")

#### 2. LLM Configuration Tab
- Provider Selection (dropdown: Gemini, OpenRouter, Ollama, LMStudio, Groq)
- API Key (password field, optional for some providers)
- Base URL (text input, optional for some providers)
- Model Selection (text input with autocomplete)
- Embedding Model (text input, optional)
- Test Connection button

#### 3. Tool Configuration Tab
- MCP Servers Management:
  - List of configured MCP servers with add/edit/delete functionality
  - Each server configuration includes:
    - Name (text input)
    - Transport Type (dropdown: Docker, SSE, Stdio)
    - Configuration (JSON editor for environment variables and settings)
    - Enabled checkbox
  - Pre-configured templates for common tools:
    - Neo4j Database (with default Docker image and environment)
    - Qdrant Vector Database (with default settings)
    - File System Tools
    - Web Search Tools
    - Custom MCP Servers

#### 4. Behavior Settings Tab
- System Prompt (textarea with syntax highlighting)
- Retry Limit (number input)
- Timeout Seconds (number input)
- Max Tokens (number input)
- Response Format (dropdown: JSON, Text, Markdown)

#### 5. Build Tab
- Generated JSON Configuration (read-only JSON editor)
- Generated Rust Code (read-only code editor)
- Build Options:
  - Target Platform (dropdown: Windows, Linux, macOS)
  - Optimization Level (dropdown: Debug, Release)
- Build Button
- Run Button (after successful build)

### User Interaction Flow

1. **Agent Configuration**:
   - User fills in configuration forms across different tabs
   - Real-time validation of all fields
   - Preview of generated JSON and code

2. **Code Generation**:
   - User clicks "Generate Code" button
   - GUI generates:
     - JSON configuration file
     - Rust code for custom agent (if needed)
   - Shows preview in dedicated panels

3. **Build Process**:
   - User clicks "Build Agent" button
   - GUI:
     - Validates all configuration
     - Creates temporary project directory
     - Copies GenericRunner.rs
     - Writes JSON configuration
     - Runs cargo build command
     - Shows progress and any errors
     - Copies resulting executable to output directory

4. **Running Agent**:
   - After successful build, user can click "Run Agent"
   - GUI starts the executable and shows output
   - Provides interface to test the agent

### Form Validation Strategy

#### Real-time Validation
- Required field checking
- Format validation (URLs, ports, JSON)
- Range validation (ports, timeouts)
- Dependency validation (e.g., API key required for certain providers)

#### Validation Feedback
- Field-level error messages
- Tab-level validation status indicators
- Global validation status in status bar
- Prevent build if validation fails

### Integration with Backend Systems

#### Code Generation
- Template-based code generation
- JSON schema validation
- Rust code compilation check

#### Build System Integration
- Cargo project creation
- Dependency management
- Cross-compilation support
- Executable packaging

#### Docker Integration
- Docker image validation
- Container management (optional)
- Environment variable handling

### Component Organization

#### Main Application Structure
```rust
struct AgentBuilderApp {
    // Configuration state
    config: AgentConfig,
    
    // UI state
    active_tab: ConfigTab,
    validation_errors: HashMap<String, String>,
    
    // Build state
    build_state: BuildState,
    build_output: String,
    
    // Generated code
    generated_json: String,
    generated_code: String,
}

enum ConfigTab {
    Basic,
    LLM,
    Tools,
    Behavior,
    Build,
}

enum BuildState {
    Idle,
    Validating,
    Generating,
    Building,
    Running,
    Completed,
    Failed(String),
}
```

#### Component Hierarchy
- Main Application
  - Tab Panel
    - Basic Settings Panel
    - LLM Configuration Panel
    - Tools Configuration Panel
    - Behavior Settings Panel
    - Build Panel
  - Action Buttons Panel
  - Status Panel
  - Progress Dialog

### Feedback System Design

#### Progress Indicators
- Tab-level validation status
- Build progress bar
- Real-time build output
- Error highlighting

#### Status Messages
- Success notifications
- Error messages with details
- Warning messages
- Information updates

#### Error Handling
- Field-level error display
- Build error parsing and display
- Recovery suggestions
- Help links for common issues

### JSON Configuration Schema

The complete JSON configuration schema will include:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Agent Configuration",
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Agent name"
    },
    "description": {
      "type": "string",
      "description": "Agent description"
    },
    "system_prompt": {
      "type": "string",
      "description": "System prompt for the agent"
    },
    "port": {
      "type": "integer",
      "minimum": 1,
      "maximum": 65535,
      "description": "Port for SSE server"
    },
    "bind_address": {
      "type": "string",
      "format": "ipv4",
      "default": "127.0.0.1",
      "description": "Bind address for SSE server"
    },
    "llm": {
      "$ref": "#/definitions/LLMConfig"
    },
    "mcp_servers": {
      "type": "array",
      "items": {
        "$ref": "#/definitions/MCPServerConfig"
      }
    },
    "retry_limit": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "description": "Maximum number of retries"
    },
    "timeout_seconds": {
      "type": "integer",
      "minimum": 1,
      "maximum": 3600,
      "description": "Timeout in seconds"
    },
    "max_tokens": {
      "type": "integer",
      "minimum": 1,
      "maximum": 32768,
      "description": "Maximum tokens for response"
    },
    "response_format": {
      "type": "string",
      "enum": ["json", "text", "markdown"],
      "description": "Response format"
    }
  },
  "required": ["name", "system_prompt", "port", "bind_address", "llm"],
  "definitions": {
    "LLMConfig": {
      "type": "object",
      "properties": {
        "provider": {
          "type": "string",
          "enum": ["gemini", "openrouter", "ollama", "lmstudio", "groq"]
        },
        "api_key": {
          "type": "string",
          "description": "API key for the provider"
        },
        "base_url": {
          "type": "string",
          "format": "uri",
          "description": "Base URL for the provider"
        },
        "model": {
          "type": "string",
          "description": "Model name"
        },
        "embed_model": {
          "type": "string",
          "description": "Embedding model name"
        }
      },
      "required": ["provider", "model"]
    },
    "MCPServerConfig": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string"
        },
        "enabled": {
          "type": "boolean"
        },
        "transport_type": {
          "type": "string",
          "enum": ["docker", "sse", "stdio"]
        },
        "config": {
          "type": "object"
        }
      },
      "required": ["name", "enabled", "transport_type"]
    }
  }
}
```

## Implementation Roadmap

### Phase 1: Core Infrastructure
1. Create GenericRunner.rs with dynamic configuration loading
2. Implement basic egui application structure
3. Create configuration data structures
4. Implement basic form components

### Phase 2: Configuration UI
1. Implement Basic Settings tab
2. Implement LLM Configuration tab
3. Implement Tools Configuration tab
4. Implement Behavior Settings tab

### Phase 3: Code Generation
1. Implement JSON generation from configuration
2. Implement Rust code generation (if needed)
3. Add preview panels
4. Implement validation logic

### Phase 4: Build System
1. Implement cargo project creation
2. Implement build process with progress tracking
3. Add error handling and recovery
4. Implement executable packaging

### Phase 5: Testing and Polish
1. Add comprehensive testing
2. Improve error messages
3. Add help documentation
4. Performance optimization

## Technical Considerations

### Performance
- Efficient form validation
- Responsive UI during build process
- Memory-efficient code generation

### Security
- Secure API key handling
- Safe temporary file creation
- Input sanitization

### Extensibility
- Plugin architecture for custom tools
- Template system for code generation
- Configurable build targets

### Cross-Platform Support
- Platform-specific build configurations
- Docker availability detection
- Path handling for different OS

This design provides a comprehensive, user-friendly interface for creating and building AI agents while maintaining flexibility and extensibility for future enhancements.