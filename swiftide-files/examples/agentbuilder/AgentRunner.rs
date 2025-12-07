use anyhow::{Context, Result, anyhow};
use reqwest;
use rmcp::transport::sse_client::{SseClientConfig, SseClientTransport};
use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::{CallToolResult, Content, ErrorCode, ErrorData, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::sse_server::{SseServer, SseServerConfig},
};
use rmcp::{
    ServiceExt as _,
    model::{ClientInfo, Implementation},
    transport::{ConfigureCommandExt as _, TokioChildProcess},
};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::types::chrono::Utc;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::future::Future;
use std::path::Path;
use swiftide::agents::tools::mcp::McpToolbox;
use swiftide_agents::Agent;
use swiftide_core::chat_completion::traits::ChatCompletion;
use swiftide_integrations::{
    gemini::Gemini, lmstudio::LMStudio, ollama::Ollama, open_router::OpenRouter,
};
use tokio;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ContentProcessRequest {
    #[schemars(description = "The content to process with the agent.")]
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LLMConfig {
    pub provider: String, // "gemini", "openrouter", "ollama", "lmstudio"
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub model: String,
    pub embed_model: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StdioToolConfig {
    pub command: String,
    pub args: Vec<String>,
    pub environment: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SSEToolConfig {
    pub port: u16,
    pub endpoint: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AgentConfig {
    pub name: String,
    pub description: Option<String>,
    pub system_prompt: String,
    pub port: u16,
    pub bind_address: String,

    // LLM Configuration
    pub llm: LLMConfig,

    // Generic Tool Configurations
    pub stdio_tools: HashMap<String, StdioToolConfig>,
    pub sse_tools: HashMap<String, SSEToolConfig>,

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
pub struct AgentRunnerServer {
    config: AgentConfig,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl AgentRunnerServer {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Process content using the configured agent")]
    async fn start_unified_agent(
        &self,
        Parameters(request): Parameters<ContentProcessRequest>,
    ) -> Result<CallToolResult, ErrorData> {
        info!(
            "Processing content with agent {}: {}",
            self.config.name, request.content
        );

        // Create agent based on configuration
        let agent_result = self.create_agent().await;
        let mut agent = match agent_result {
            Ok(agent) => agent,
            Err(e) => {
                error!("Failed to create agent: {}", e);
                return Err(ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    e.to_string(),
                    None,
                ));
            }
        };

        // Execute the agent
        match agent.query(request.content).await {
            Ok(_) => {
                // Get the response from the agent's context
                let history =
                    agent.context().history().await.map_err(|e| {
                        ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None)
                    })?;

                // Find the last assistant message
                let mut response_content = None;
                for msg in history.iter().rev() {
                    if let swiftide_core::chat_completion::ChatMessage::Assistant(content, _) = msg
                    {
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
                    serde_json::to_string(&openai_response).map_err(|e| {
                        ErrorData::new(ErrorCode::INTERNAL_ERROR, e.to_string(), None)
                    })?,
                )]))
            }
            Err(e) => {
                error!("Agent query failed: {}", e);
                Err(ErrorData::new(
                    ErrorCode::INTERNAL_ERROR,
                    e.to_string(),
                    None,
                ))
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
                let config =
                    swiftide_integrations::open_router::config::OpenRouterConfig::builder()
                        .api_base(
                            self.config
                                .llm
                                .base_url
                                .as_ref()
                                .unwrap_or(&"https://openrouter.ai/api/v1".to_string())
                                .clone(),
                        )
                        .api_key(
                            self.config
                                .llm
                                .api_key
                                .as_ref()
                                .unwrap_or(&"".to_string())
                                .clone(),
                        )
                        .build()?;
                let openrouter_client = swiftide_integrations::open_router::OpenRouter::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(openrouter_client)
            }
            "ollama" => {
                let config = swiftide_integrations::ollama::config::OllamaConfig::builder()
                    .api_base(
                        self.config
                            .llm
                            .base_url
                            .as_ref()
                            .unwrap_or(&"http://localhost:11434/v1".to_string())
                            .clone(),
                    )
                    .api_key(
                        self.config
                            .llm
                            .api_key
                            .as_ref()
                            .unwrap_or(&"ollama".to_string())
                            .clone()
                            .into(),
                    )
                    .build()?;
                let ollama_client = Ollama::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(ollama_client)
            }
            "lmstudio" => {
                let config = swiftide_integrations::lmstudio::config::LMStudioConfig::builder()
                    .api_base(
                        self.config
                            .llm
                            .base_url
                            .as_ref()
                            .unwrap_or(&"http://localhost:1234/v1".to_string())
                            .clone(),
                    )
                    .api_key(
                        self.config
                            .llm
                            .api_key
                            .as_ref()
                            .unwrap_or(&"".to_string())
                            .clone()
                            .into(),
                    )
                    .build()?;
                let lmstudio_client = LMStudio::builder()
                    .client(async_openai::Client::with_config(config))
                    .default_prompt_model(&self.config.llm.model)
                    .build()?;
                Box::new(lmstudio_client)
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported LLM provider: {}",
                    self.config.llm.provider
                ));
            }
        };

        // Create agent builder
        let mut agent_builder = Agent::builder();
        let mut agent_builder = agent_builder
            .llm(&client)
            .system_prompt(self.config.system_prompt.clone())
            .name(&self.config.name);

        // Add retry limit if specified
        if let Some(retry_limit) = self.config.retry_limit {
            agent_builder = agent_builder.limit(retry_limit as usize);
        }

        // Add toolboxes based on configuration - support unlimited number of tools
        let mut toolboxes = Vec::new();

        // Process stdio tools
        for (tool_name, stdio_config) in &self.config.stdio_tools {
            info!("Creating stdio toolbox for tool: {}", tool_name);
            let stdio_toolbox = self.create_stdio_toolbox(tool_name, stdio_config).await?;
            toolboxes.push(stdio_toolbox);
        }

        // Process SSE tools
        for (tool_name, sse_config) in &self.config.sse_tools {
            info!("Creating SSE toolbox for tool: {}", tool_name);
            let sse_toolbox = self.create_sse_toolbox(tool_name, sse_config).await?;
            toolboxes.push(sse_toolbox);
        }

        // Add all toolboxes to agent
        for toolbox in toolboxes {
            agent_builder = agent_builder.add_toolbox(toolbox);
        }

        // Build and return the agent
        Ok(agent_builder.build()?)
    }

    async fn create_stdio_toolbox(
        &self,
        tool_name: &str,
        config: &StdioToolConfig,
    ) -> Result<McpToolbox> {
        let client_info = ClientInfo {
            client_info: Implementation {
                name: "agent-runner".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            ..Default::default()
        };

        info!(
            "Creating stdio process: {} with args: {:?}",
            config.command, config.args
        );

        // Build command with arguments using configure method
        let child_process = TokioChildProcess::new(
            tokio::process::Command::new(&config.command).configure(|cmd| {
                cmd.args(&config.args);

                // Add environment variables if provided
                if let Some(env_vars) = &config.environment {
                    for (key, value) in env_vars {
                        cmd.env(key, value);
                        info!("Setting environment variable: {}={}", key, value);
                    }
                }
            }),
        )?;
        let running_service = client_info.serve(child_process).await?;
        Ok(McpToolbox::from_running_service(running_service))
    }

    async fn create_sse_toolbox(
        &self,
        tool_name: &str,
        config: &SSEToolConfig,
    ) -> Result<McpToolbox> {
        // Replace {port} placeholder with actual port
        let endpoint = config.endpoint.replace("{port}", &config.port.to_string());
        info!("Creating SSE connection to: {}", endpoint);

        // Create client with basic configuration
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30)) // Default timeout
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        // Configure SSE client
        let sse_config = SseClientConfig {
            sse_endpoint: endpoint.into(),
            ..Default::default()
        };

        let transport = SseClientTransport::start_with_client(client, sse_config).await?;
        Ok(McpToolbox::try_from_transport(transport).await?)
    }
}

#[tool_handler]
impl ServerHandler for AgentRunnerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(format!(
                "Agent runner server: {}",
                self.config
                    .description
                    .as_ref()
                    .unwrap_or(&"Configurable AI agent".to_string())
            )),
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
        error!(
            "❌ Error: Could not find configuration file: {:?}",
            config_path
        );
        return Err(anyhow::anyhow!(
            "Configuration file not found: {:?}",
            config_path
        ));
    }

    let config_content = fs::read_to_string(&config_path).context("Failed to read config file")?;

    let config: AgentConfig = serde_json::from_str(&config_content)
        .context("Invalid JSON format in configuration file")?;

    info!("✅ Configuration Loaded: {:?}", config.name);
    info!(
        "📋 Found {} stdio tools and {} SSE tools",
        config.stdio_tools.len(),
        config.sse_tools.len()
    );

    // Start Server using config
    start_server(config).await
}

async fn start_server(config: AgentConfig) -> Result<()> {
    // Configure SSE server
    let sse_config = SseServerConfig {
        bind: format!("{}:{}", config.bind_address, config.port).parse()?,
        sse_path: "/sse".into(),
        post_path: "/message".to_string(),
        ct: CancellationToken::new(),
        sse_keep_alive: if config.keep_alive.unwrap_or(false) {
            Some(std::time::Duration::from_secs(60))
        } else {
            None
        },
    };

    info!(
        "Server configured to bind to {}:{}",
        config.bind_address, config.port
    );

    // Create SSE server
    let (sse_server, router) = SseServer::new(sse_config);

    let listener = tokio::net::TcpListener::bind(sse_server.config.bind)
        .await
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
    let config_clone = config.clone();
    let ct = sse_server.with_service(move || AgentRunnerServer::new(config.clone()));

    info!(
        "Agent server listening on {}:{}",
        config_clone.bind_address, config_clone.port
    );

    // Wait for cancellation signal
    tokio::signal::ctrl_c()
        .await
        .context("Failed to listen for Ctrl+C signal")?;
    ct.cancel();

    Ok(())
}
