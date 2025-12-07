//! Enhanced Agent Build System with SSE Management
//!
//! This module provides a comprehensive build system that handles:
//! 1. Creating JSON configuration files from user input
//! 2. Copying and renaming the AgentRunner.exe file to match the agent name
//! 3. SSE endpoint management and validation
//! 4. Port allocation and conflict resolution
//! 5. No compilation, no cargo, no build process - just file operations
//! 6. Enhanced egui-based GUI for agent configuration and building

use crate::agent_builder_types::{
    AgentConfig, ConfigTab, LLMProvider, ResponseFormat, SSEToolConfig, ToolTransportType,
    to_legacy_agent_config, validate_sse_endpoint_config,
};
use crate::port_allocator::PortAllocator;
use crate::sse_manager::SSEManager;
use crate::sse_validator::SSEValidator;
use anyhow::{Context, Result, anyhow};
use eframe::egui;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Simplified build system that only handles file operations
pub struct SimpleAgentBuilder {
    /// Output directory for generated agents
    output_directory: PathBuf,
    /// Path to the AgentRunner.exe template
    default_agent_path: PathBuf,
}

impl SimpleAgentBuilder {
    /// Create a new SimpleAgentBuilder
    ///
    /// # Arguments
    /// * `output_directory` - Directory where agents will be created
    /// * `default_agent_path` - Path to the AgentRunner.exe template file
    pub fn new<P: AsRef<Path>>(output_directory: P, default_agent_path: P) -> Self {
        Self {
            output_directory: output_directory.as_ref().to_path_buf(),
            default_agent_path: default_agent_path.as_ref().to_path_buf(),
        }
    }

    /// Build an agent from configuration
    ///
    /// This method:
    /// 1. Validates the configuration
    /// 2. Generates a JSON configuration file
    /// 3. Copies and renames AgentRunner.exe to match the agent name
    ///
    /// # Arguments
    /// * `config` - Agent configuration from the GUI
    ///
    /// # Returns
    /// * `Result<PathBuf>` - Path to the generated agent executable
    pub async fn build_agent(&self, config: &AgentConfig) -> Result<PathBuf> {
        // Validate configuration
        self.validate_config(config)?;

        // Ensure output directory exists
        fs::create_dir_all(&self.output_directory).with_context(|| {
            format!(
                "Failed to create output directory: {:?}",
                self.output_directory
            )
        })?;

        // Generate JSON configuration file
        let config_path = self.generate_config_file(config).await?;

        // Copy and rename AgentRunner.exe
        let agent_exe_path = self.copy_agent_executable(config).await?;

        Ok(agent_exe_path)
    }

    /// Validate the agent configuration
    fn validate_config(&self, config: &AgentConfig) -> Result<()> {
        // Basic validation
        if config.name.is_empty() {
            return Err(anyhow!("Agent name cannot be empty"));
        }

        if config.name.len() > 100 {
            return Err(anyhow!("Agent name must be less than 100 characters"));
        }

        // Check for invalid characters in name (for filename)
        if !config
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
        {
            return Err(anyhow!(
                "Agent name can only contain letters, numbers, underscores, and hyphens"
            ));
        }

        if config.system_prompt.is_empty() {
            return Err(anyhow!("System prompt cannot be empty"));
        }

        // Validate LLM configuration
        if config.llm.model.is_empty() {
            return Err(anyhow!("LLM model cannot be empty"));
        }

        // Check if AgentRunner.exe exists
        if !self.default_agent_path.exists() {
            return Err(anyhow!(
                "AgentRunner.exe not found at: {:?}",
                self.default_agent_path
            ));
        }

        Ok(())
    }

    /// Generate JSON configuration file
    async fn generate_config_file(&self, config: &AgentConfig) -> Result<PathBuf> {
        // Convert to legacy format for compatibility with AgentRunner.rs
        let legacy_config = to_legacy_agent_config(config)
            .context("Failed to convert configuration to legacy format")?;

        // Serialize to JSON
        let json_content = serde_json::to_string_pretty(&legacy_config)
            .context("Failed to serialize configuration to JSON")?;

        // Create config file path
        let config_path = self.output_directory.join(format!("{}.json", config.name));

        // Write configuration file
        fs::write(&config_path, json_content)
            .with_context(|| format!("Failed to write configuration file: {:?}", config_path))?;

        println!("✅ Generated configuration file: {:?}", config_path);

        Ok(config_path)
    }

    /// Copy and rename AgentRunner.exe to match the agent name
    async fn copy_agent_executable(&self, config: &AgentConfig) -> Result<PathBuf> {
        let agent_exe_path = self.output_directory.join(format!("{}.exe", config.name));

        // Check if target already exists
        if agent_exe_path.exists() {
            return Err(anyhow!(
                "Agent executable already exists: {:?}",
                agent_exe_path
            ));
        }

        // Copy the AgentRunner.exe file
        fs::copy(&self.default_agent_path, &agent_exe_path).with_context(|| {
            format!(
                "Failed to copy AgentRunner.exe from {:?} to {:?}",
                self.default_agent_path, agent_exe_path
            )
        })?;

        println!("✅ Created agent executable: {:?}", agent_exe_path);

        Ok(agent_exe_path)
    }

    /// Get the output directory
    pub fn output_directory(&self) -> &Path {
        &self.output_directory
    }

    /// Set the output directory
    pub fn set_output_directory<P: AsRef<Path>>(&mut self, directory: P) {
        self.output_directory = directory.as_ref().to_path_buf();
    }

    /// Get the default agent path
    pub fn default_agent_path(&self) -> &Path {
        &self.default_agent_path
    }

    /// Set the default agent path
    pub fn set_default_agent_path<P: AsRef<Path>>(&mut self, path: P) {
        self.default_agent_path = path.as_ref().to_path_buf();
    }

    /// List all existing agents in the output directory
    pub fn list_agents(&self) -> Result<Vec<String>> {
        if !self.output_directory.exists() {
            return Ok(Vec::new());
        }

        let mut agents = Vec::new();

        for entry in fs::read_dir(&self.output_directory).with_context(|| {
            format!(
                "Failed to read output directory: {:?}",
                self.output_directory
            )
        })? {
            let entry = entry.with_context(|| "Failed to read directory entry")?;

            let path = entry.path();

            // Look for .exe files
            if let Some(extension) = path.extension() {
                if extension == "exe" {
                    if let Some(stem) = path.file_stem() {
                        if let Some(name) = stem.to_str() {
                            agents.push(name.to_string());
                        }
                    }
                }
            }
        }

        agents.sort();
        Ok(agents)
    }

    /// Check if an agent already exists
    pub fn agent_exists(&self, agent_name: &str) -> bool {
        let agent_exe_path = self.output_directory.join(format!("{}.exe", agent_name));
        let agent_config_path = self.output_directory.join(format!("{}.json", agent_name));

        agent_exe_path.exists() && agent_config_path.exists()
    }

    /// Delete an agent (both executable and config file)
    pub fn delete_agent(&self, agent_name: &str) -> Result<()> {
        let agent_exe_path = self.output_directory.join(format!("{}.exe", agent_name));
        let agent_config_path = self.output_directory.join(format!("{}.json", agent_name));

        let mut deleted_files = Vec::new();

        if agent_exe_path.exists() {
            fs::remove_file(&agent_exe_path).with_context(|| {
                format!("Failed to remove agent executable: {:?}", agent_exe_path)
            })?;
            deleted_files.push(agent_exe_path);
        }

        if agent_config_path.exists() {
            fs::remove_file(&agent_config_path).with_context(|| {
                format!("Failed to remove agent config: {:?}", agent_config_path)
            })?;
            deleted_files.push(agent_config_path);
        }

        if deleted_files.is_empty() {
            return Err(anyhow!("Agent '{}' not found", agent_name));
        }

        println!("🗑️  Deleted agent '{}': {:?}", agent_name, deleted_files);
        Ok(())
    }

    /// Get the path to an agent's executable
    pub fn get_agent_executable_path(&self, agent_name: &str) -> PathBuf {
        self.output_directory.join(format!("{}.exe", agent_name))
    }

    /// Get the path to an agent's configuration file
    pub fn get_agent_config_path(&self, agent_name: &str) -> PathBuf {
        self.output_directory.join(format!("{}.json", agent_name))
    }
}

/// Create a SimpleAgentBuilder with default settings
///
/// This function creates a builder with:
/// - Output directory: "./agents"
/// - AgentRunner.exe path: "./AgentRunner.exe"
///
/// # Returns
/// * `SimpleAgentBuilder` - Configured builder instance
pub fn create_default_builder() -> SimpleAgentBuilder {
    SimpleAgentBuilder::new("./agents", "./AgentRunner.exe")
}

/// Convenience function to build an agent with default settings
///
/// # Arguments
/// * `config` - Agent configuration
///
/// # Returns
/// * `Result<PathBuf>` - Path to the generated agent executable
pub async fn build_agent_with_defaults(config: &AgentConfig) -> Result<PathBuf> {
    let builder = create_default_builder();
    builder.build_agent(config).await
}

/// GUI Application for Enhanced Agent Builder with SSE Management
pub struct AgentBuilderApp {
    builder: SimpleAgentBuilder,
    config: AgentConfig,
    active_tab: ConfigTab,
    status_message: String,
    show_success: bool,
    show_error: bool,
    existing_agents: Vec<String>,
    selected_agent: Option<String>,

    // SSE Management components
    port_allocator: Arc<PortAllocator>,
    sse_validator: Arc<SSEValidator>,
    sse_manager: Arc<SSEManager>,

    // SSE UI state
    show_sse_diagnostics: bool,
    sse_test_results: std::collections::HashMap<String, String>,
    testing_sse_endpoint: Option<String>,

    // Tool Management UI state
    show_stdio_popup: bool,
    show_sse_popup: bool,
    show_add_tool_menu: bool,
    editing_tool_index: Option<usize>,

    // Temporary storage for popup input data
    temp_stdio_config: String, // JSON configuration for stdio tools
    temp_sse_port: String,     // Port number for SSE tools
    temp_tool_name: String,    // Tool name for both types
}

impl AgentBuilderApp {
    /// Create a new agent builder app with SSE management
    pub fn new() -> Self {
        let builder = create_default_builder();
        let existing_agents = builder.list_agents().unwrap_or_default();
        let port_allocator = Arc::new(PortAllocator::new());
        let sse_validator = Arc::new(SSEValidator::new());
        let sse_manager = Arc::new(SSEManager::new());

        Self {
            builder,
            config: AgentConfig::default(),
            active_tab: ConfigTab::Basic,
            status_message: String::new(),
            show_success: false,
            show_error: false,
            existing_agents,
            selected_agent: None,
            port_allocator,
            sse_validator,
            sse_manager,
            show_sse_diagnostics: false,
            sse_test_results: std::collections::HashMap::new(),
            testing_sse_endpoint: None,

            // Tool Management UI state
            show_stdio_popup: false,
            show_sse_popup: false,
            show_add_tool_menu: false,
            editing_tool_index: None,

            // Temporary storage for popup input data
            temp_stdio_config: String::new(),
            temp_sse_port: String::new(),
            temp_tool_name: String::new(),
        }
    }

    /// Create a new agent builder app with custom paths and SSE management
    pub fn new_with_paths<P: AsRef<Path>>(output_dir: P, default_agent_path: P) -> Self {
        let builder = SimpleAgentBuilder::new(output_dir, default_agent_path);
        let existing_agents = builder.list_agents().unwrap_or_default();
        let port_allocator = Arc::new(PortAllocator::new());
        let sse_validator = Arc::new(SSEValidator::new());
        let sse_manager = Arc::new(SSEManager::new());

        Self {
            builder,
            config: AgentConfig::default(),
            active_tab: ConfigTab::Basic,
            status_message: String::new(),
            show_success: false,
            show_error: false,
            existing_agents,
            selected_agent: None,
            port_allocator,
            sse_validator,
            sse_manager,
            show_sse_diagnostics: false,
            sse_test_results: std::collections::HashMap::new(),
            testing_sse_endpoint: None,

            // Tool Management UI state
            show_stdio_popup: false,
            show_sse_popup: false,
            show_add_tool_menu: false,
            editing_tool_index: None,

            // Temporary storage for popup input data
            temp_stdio_config: String::new(),
            temp_sse_port: String::new(),
            temp_tool_name: String::new(),
        }
    }

    /// Show main GUI
    pub fn show(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🤖 Simple Agent Builder");
            ui.separator();

            // Show status message if any
            if !self.status_message.is_empty() {
                let color = if self.show_success {
                    egui::Color32::GREEN
                } else if self.show_error {
                    egui::Color32::RED
                } else {
                    egui::Color32::GRAY
                };

                ui.label(egui::RichText::new(&self.status_message).color(color));
                ui.separator();
            }

            // Create tabs for different sections using a simple approach
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, ConfigTab::Basic, "Configuration");
                ui.selectable_value(&mut self.active_tab, ConfigTab::LLM, "Existing Agents");
                ui.selectable_value(&mut self.active_tab, ConfigTab::Tools, "SSE Diagnostics");
                ui.selectable_value(&mut self.active_tab, ConfigTab::Behavior, "Settings");
            });

            ui.separator();

            // Make the main content area scrollable
            egui::ScrollArea::vertical().show(ui, |ui| {
                match self.active_tab {
                    ConfigTab::Basic => self.show_configuration_tab(ui),
                    ConfigTab::LLM => self.show_existing_agents_tab(ui),
                    ConfigTab::Tools => self.show_sse_diagnostics_tab(ui),
                    ConfigTab::Behavior => self.show_settings_tab(ui),
                    ConfigTab::Build => {} // Not used in this case
                }
            });

            // Show popups if needed (these use the context, not ui)
            self.show_stdio_popup(ctx);
            self.show_sse_popup(ctx);
            self.show_add_tool_menu_popup(ctx);
        });
    }

    /// Show configuration tab
    fn show_configuration_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Agent Configuration");
        ui.separator();

        // Basic Settings
        ui.collapsing("Basic Settings", |ui| {
            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.text_edit_singleline(&mut self.config.name);
            });

            ui.horizontal(|ui| {
                ui.label("Description:");
                ui.text_edit_singleline(self.config.description.get_or_insert_with(String::new));
            });

            ui.label("System Prompt:");
            // Make the system prompt text field scrollable with a fixed height
            egui::ScrollArea::vertical()
                .max_height(200.0) // Set a reasonable default height
                .show(ui, |ui| {
                    ui.text_edit_multiline(&mut self.config.system_prompt);
                });
        });

        // LLM Configuration
        ui.collapsing("LLM Configuration", |ui| {
            ui.horizontal(|ui| {
                ui.label("Provider:");
                // Store the previous provider value before showing the dropdown
                let previous_provider = self.config.llm.provider.clone();

                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.config.llm.provider))
                    .show_ui(ui, |ui| {
                        for provider in LLMProvider::all() {
                            ui.selectable_value(
                                &mut self.config.llm.provider,
                                provider.clone(),
                                format!("{:?}", provider),
                            );
                        }
                    });

                // Check if the provider has changed and update base_url accordingly
                if previous_provider != self.config.llm.provider {
                    // Update the base_url to the new provider's default URL
                    if let Some(default_url) = self.config.llm.provider.default_base_url() {
                        self.config.llm.base_url = Some(default_url.to_string());
                    } else {
                        // For providers without a default URL (like Gemini), clear the base_url
                        self.config.llm.base_url = None;
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("API Key:");
                ui.text_edit_singleline(self.config.llm.api_key.get_or_insert_with(String::new));
            });

            ui.horizontal(|ui| {
                ui.label("Base URL:");
                ui.text_edit_singleline(self.config.llm.base_url.get_or_insert_with(String::new));
            });

            ui.horizontal(|ui| {
                ui.label("Model:");
                ui.text_edit_singleline(&mut self.config.llm.model);
            });

            ui.horizontal(|ui| {
                ui.label("Embed Model:");
                ui.text_edit_singleline(
                    self.config.llm.embed_model.get_or_insert_with(String::new),
                );
            });
        });

        // Server Configuration
        ui.collapsing("Server Configuration", |ui| {
            ui.horizontal(|ui| {
                ui.label("Port:");
                ui.add(egui::DragValue::new(&mut self.config.server.port).range(1024..=65535));
            });

            ui.horizontal(|ui| {
                ui.label("Bind Address:");
                ui.text_edit_singleline(&mut self.config.server.bind_address);
            });

            ui.horizontal(|ui| {
                ui.label("SSE Path:");
                ui.text_edit_singleline(&mut self.config.server.sse_path);
            });

            ui.horizontal(|ui| {
                ui.label("POST Path:");
                ui.text_edit_singleline(&mut self.config.server.post_path);
            });

            ui.checkbox(&mut self.config.server.keep_alive, "Keep Alive");
        });

        // Behavior Configuration
        ui.collapsing("Behavior Settings", |ui| {
            ui.horizontal(|ui| {
                ui.label("Retry Limit:");
                if self.config.behavior.retry_limit.is_none() {
                    self.config.behavior.retry_limit = Some(3);
                }
                ui.add(
                    egui::DragValue::new(self.config.behavior.retry_limit.as_mut().unwrap())
                        .range(1..=10),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Timeout (seconds):");
                if self.config.behavior.timeout_seconds.is_none() {
                    self.config.behavior.timeout_seconds = Some(60);
                }
                ui.add(
                    egui::DragValue::new(self.config.behavior.timeout_seconds.as_mut().unwrap())
                        .range(10..=300),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Max Tokens:");
                if self.config.behavior.max_tokens.is_none() {
                    self.config.behavior.max_tokens = Some(4096);
                }
                ui.add(
                    egui::DragValue::new(self.config.behavior.max_tokens.as_mut().unwrap())
                        .range(100..=32768),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Response Format:");
                if self.config.behavior.response_format.is_none() {
                    self.config.behavior.response_format = Some(ResponseFormat::Json);
                }
                egui::ComboBox::from_label("")
                    .selected_text(format!(
                        "{:?}",
                        self.config.behavior.response_format.as_ref().unwrap()
                    ))
                    .show_ui(ui, |ui| {
                        for format in ResponseFormat::all() {
                            ui.selectable_value(
                                self.config.behavior.response_format.as_mut().unwrap(),
                                format.clone(),
                                format!("{:?}", format),
                            );
                        }
                    });
            });
        });

        // Tools Configuration
        ui.collapsing("Tools Configuration", |ui| {
            self.show_tools_configuration(ui);
        });

        ui.separator();

        // Build button
        if ui.button("🔨 Build Agent").clicked() {
            self.build_agent();
        }
    }

    /// Show existing agents tab
    fn show_existing_agents_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Existing Agents");
        ui.separator();

        // Refresh button
        if ui.button("🔄 Refresh").clicked() {
            self.existing_agents = self.builder.list_agents().unwrap_or_default();
        }

        ui.label(format!("Found {} agents:", self.existing_agents.len()));

        // Collect agent names to avoid borrowing issues
        let agent_names: Vec<String> = self.existing_agents.iter().cloned().collect();
        let mut agent_to_delete: Option<String> = None;

        // List existing agents
        for agent_name in &agent_names {
            ui.horizontal(|ui| {
                ui.label(&*agent_name);

                if ui.button("🗑️ Delete").clicked() {
                    agent_to_delete = Some(agent_name.clone());
                }
            });
        }

        // Handle deletion outside the loop to avoid borrowing conflicts
        if let Some(agent_name) = agent_to_delete {
            if let Err(e) = self.builder.delete_agent(&agent_name) {
                self.show_error_message(&format!("Failed to delete agent: {}", e));
            } else {
                self.show_success_message(&format!("Successfully deleted agent: {}", agent_name));
                self.existing_agents = self.builder.list_agents().unwrap_or_default();
            }
        }
    }

    /// Show SSE diagnostics tab
    fn show_sse_diagnostics_tab(&mut self, ui: &mut egui::Ui) {
        self.show_sse_diagnostics_panel(ui);
    }

    /// Show settings tab
    fn show_settings_tab(&mut self, ui: &mut egui::Ui) {
        ui.heading("Builder Settings");
        ui.separator();

        ui.label(format!(
            "Output Directory: {:?}",
            self.builder.output_directory()
        ));
        ui.label(format!(
            "Default Agent Path: {:?}",
            self.builder.default_agent_path()
        ));

        ui.separator();

        // Check if AgentRunner.exe exists
        if self.builder.default_agent_path().exists() {
            ui.label("✅ AgentRunner.exe found");
        } else {
            ui.label("❌ AgentRunner.exe not found");
            ui.label("Please ensure AgentRunner.exe exists in specified path");
        }

        ui.separator();

        // SSE Management Settings
        ui.heading("SSE Management Settings");
        ui.checkbox(&mut self.show_sse_diagnostics, "Show SSE Diagnostics");

        ui.separator();

        // Port allocation settings
        ui.collapsing("Port Allocation Settings", |ui| {
            let stats = self.port_allocator.as_ref().get_stats();
            ui.label(format!("Port Range: {:?}", stats.available_range));
            ui.label(format!("Total Allocated: {}", stats.total_allocated));
            ui.label(format!(
                "Preferred Allocated: {}",
                stats.preferred_allocated
            ));
            ui.label(format!("Unique Agents: {}", stats.unique_agents));
        });
    }

    /// Build agent with current configuration
    fn build_agent(&mut self) {
        let rt = tokio::runtime::Runtime::new().unwrap();

        match rt.block_on(self.builder.build_agent(&self.config)) {
            Ok(agent_path) => {
                self.show_success_message(&format!(
                    "Agent successfully created at: {:?}",
                    agent_path
                ));
                self.existing_agents = self.builder.list_agents().unwrap_or_default();
            }
            Err(e) => {
                self.show_error_message(&format!("Failed to build agent: {}", e));
            }
        }
    }

    /// Show a success message
    fn show_success_message(&mut self, message: &str) {
        self.status_message = message.to_string();
        self.show_success = true;
        self.show_error = false;
    }

    /// Show an error message
    fn show_error_message(&mut self, message: &str) {
        self.status_message = message.to_string();
        self.show_success = false;
        self.show_error = true;
    }

    /// Test SSE endpoint connectivity
    async fn test_sse_endpoint(&mut self, endpoint_url: &str) {
        self.testing_sse_endpoint = Some(endpoint_url.to_string());
        self.status_message = format!("Testing SSE endpoint: {}", endpoint_url);
        self.show_success = false;
        self.show_error = false;

        let validator = self.sse_validator.as_ref();
        match validator.validate_endpoint(endpoint_url).await {
            Ok(result) => {
                let status = if result.is_valid {
                    "✅ Valid"
                } else {
                    "❌ Invalid"
                };
                let details = format!(
                    "Status: {}\nConnection: {}ms\nEvents: {}\nErrors: {}",
                    status,
                    result.connection_time_ms,
                    result.events_received,
                    result.errors.join(", ")
                );
                self.sse_test_results
                    .insert(endpoint_url.to_string(), details);
                self.show_success_message(&format!("SSE endpoint test completed: {}", status));
            }
            Err(e) => {
                self.sse_test_results
                    .insert(endpoint_url.to_string(), format!("Test failed: {}", e));
                self.show_error_message(&format!("SSE endpoint test failed: {}", e));
            }
        }
        self.testing_sse_endpoint = None;
    }

    /// Get available ports for SSE endpoints
    async fn get_available_ports(&self) -> Vec<u16> {
        let allocator = self.port_allocator.as_ref();
        let mut available = Vec::new();

        // Check common ports
        let common_ports = vec![8080, 8081, 8082, 3000, 3001, 3002, 5000, 5001];
        for &port in &common_ports {
            if allocator.is_port_available(port).await {
                available.push(port);
            }
        }

        available
    }

    /// Auto-assign port for SSE tool
    async fn auto_assign_port(&mut self, tool_index: usize) -> Option<u16> {
        if let Some(tool) = self.config.tools.get_mut(tool_index) {
            if tool.transport_type == ToolTransportType::SSE {
                if let Some(sse_config) = &mut tool.sse_config {
                    if sse_config.auto_port {
                        let allocator = self.port_allocator.as_ref();
                        let agent_name = &self.config.name;

                        match allocator
                            .allocate_port(agent_name, Some(sse_config.port))
                            .await
                        {
                            Ok(port) => {
                                sse_config.port = port;
                                // Update endpoint URL with new port
                                sse_config.endpoint =
                                    sse_config.endpoint.replace("{port}", &port.to_string());
                                self.show_success_message(&format!(
                                    "Auto-assigned port {} for SSE tool",
                                    port
                                ));
                                Some(port)
                            }
                            Err(e) => {
                                self.show_error_message(&format!(
                                    "Failed to auto-assign port: {}",
                                    e
                                ));
                                None
                            }
                        }
                    } else {
                        Some(sse_config.port)
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Validate all SSE endpoints in current configuration
    async fn validate_all_sse_endpoints(&mut self) {
        let mut valid_count = 0;
        let mut total_count = 0;
        let mut error_messages = Vec::new();

        for tool in &self.config.tools {
            if tool.transport_type == ToolTransportType::SSE {
                total_count += 1;
                if let Some(sse_config) = &tool.sse_config {
                    let validation = validate_sse_endpoint_config(sse_config);
                    if validation.is_valid {
                        valid_count += 1;
                    } else {
                        if let Some(error) = &validation.error_message {
                            error_messages.push(format!(
                                "SSE tool '{}' validation failed: {}",
                                tool.name, error
                            ));
                        }
                    }
                }
            }
        }

        // Display all error messages first
        for error in error_messages {
            self.show_error_message(&error);
        }

        if total_count > 0 {
            let message = format!("SSE endpoints: {}/{} valid", valid_count, total_count);
            if valid_count == total_count {
                self.show_success_message(&message);
            } else {
                self.show_error_message(&message);
            }
        }
    }

    /// Show SSE diagnostics panel
    fn show_sse_diagnostics_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("🔌 SSE Endpoint Diagnostics");
        ui.separator();

        // Port allocation status
        ui.collapsing("Port Allocation", |ui| {
            let stats = self.port_allocator.as_ref().get_stats();
            ui.label(format!("Total allocated: {}", stats.total_allocated));
            ui.label(format!(
                "Preferred allocated: {}",
                stats.preferred_allocated
            ));
            ui.label(format!("Unique agents: {}", stats.unique_agents));

            if ui.button("🔄 Refresh Port Status").clicked() {
                // Port stats are updated in real-time
                self.show_success_message("Port allocation status refreshed");
            }
        });

        ui.separator();

        // SSE test results
        if !self.sse_test_results.is_empty() {
            ui.collapsing("SSE Test Results", |ui| {
                for (endpoint, result) in &self.sse_test_results {
                    ui.group(|ui| {
                        ui.label(format!("Endpoint: {}", endpoint));
                        ui.label(result);
                    });
                }

                if ui.button("🗑️ Clear Results").clicked() {
                    self.sse_test_results.clear();
                    self.show_success_message("SSE test results cleared");
                }
            });
        }

        ui.separator();

        // SSE endpoint management
        ui.collapsing("SSE Endpoint Management", |ui| {
            let sse_tools: Vec<_> = self
                .config
                .tools
                .iter()
                .filter(|t| t.transport_type == ToolTransportType::SSE)
                .collect();

            if sse_tools.is_empty() {
                ui.label("No SSE tools configured");
            } else {
                // Collect test and auto-assign operations to avoid borrowing conflicts
                let mut test_endpoint: Option<(String, String)> = None;
                let mut auto_assign_port: Option<usize> = None;

                for (index, tool) in sse_tools.iter().enumerate() {
                    ui.group(|ui| {
                        ui.label(format!("Tool: {}", tool.name));
                        ui.label(format!("Transport: {}", tool.transport_type.as_str()));

                        if let Some(sse_config) = &tool.sse_config {
                            ui.label(format!("Port: {}", sse_config.port));
                            ui.label(format!("Endpoint: {}", sse_config.endpoint));
                            ui.label(format!("Auto Port: {}", sse_config.auto_port));

                            // Test button
                            if ui.button(format!("🧪 Test {}", tool.name)).clicked() {
                                let endpoint_url = sse_config
                                    .endpoint
                                    .replace("{port}", &sse_config.port.to_string());
                                test_endpoint = Some((tool.name.clone(), endpoint_url));
                            }

                            // Auto-assign port button
                            if sse_config.auto_port && ui.button("🔄 Auto-Assign Port").clicked()
                            {
                                auto_assign_port = Some(index);
                            }
                        }
                    });
                }

                // Handle test endpoint outside the loop to avoid borrowing conflicts
                if let Some((tool_name, endpoint_url)) = test_endpoint {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(self.test_sse_endpoint(&endpoint_url));
                }

                // Handle auto-assign port outside the loop to avoid borrowing conflicts
                if let Some(index) = auto_assign_port {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(self.auto_assign_port(index));
                }
            }

            ui.separator();

            // Validate all button
            if ui.button("✅ Validate All SSE Endpoints").clicked() {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.validate_all_sse_endpoints());
            }
        });

        ui.separator();

        // Available ports
        ui.collapsing("Available Ports", |ui| {
            ui.label("Scanning for available ports...");

            let available_ports = {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.get_available_ports())
            };

            if available_ports.is_empty() {
                ui.label("No available ports found in common range");
            } else {
                ui.label(format!("Found {} available ports:", available_ports.len()));
                for port in &available_ports {
                    ui.label(format!("  📡 {}", port));
                }
            }

            if ui.button("🔄 Refresh Ports").clicked() {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(self.get_available_ports());
                self.show_success_message("Port scan completed");
            }
        });
    }

    /// Show tools configuration section
    fn show_tools_configuration(&mut self, ui: &mut egui::Ui) {
        ui.heading("Tools");
        ui.separator();

        // Add Tool button
        if ui.button("➕ Add Tool").clicked() {
            self.show_add_tool_menu = true;
        }

        ui.separator();

        // Display existing tools
        if self.config.tools.is_empty() {
            ui.label("No tools configured. Click 'Add Tool' to get started.");
        } else {
            // Collect actions to avoid borrowing issues
            let mut edit_tool: Option<usize> = None;
            let mut delete_tool: Option<usize> = None;

            for (index, tool) in self.config.tools.iter().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(&tool.name);
                        ui.label(format!("({})", tool.transport_type.as_str()));

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("🗑️").clicked() {
                                delete_tool = Some(index);
                            }
                            if ui.button("✏️").clicked() {
                                edit_tool = Some(index);
                            }

                            let status = if tool.enabled { "✅" } else { "❌" };
                            ui.label(status);
                        });
                    });

                    if let Some(description) = &tool.description {
                        ui.label(format!("Description: {}", description));
                    }
                });
            }

            // Handle tool actions outside the loop
            if let Some(index) = edit_tool {
                self.edit_tool(index);
            }
            if let Some(index) = delete_tool {
                self.delete_tool(index);
            }
        }
    }

    /// Show add tool menu popup
    fn show_add_tool_menu_popup(&mut self, ctx: &egui::Context) {
        let mut should_close = false;

        if self.show_add_tool_menu {
            egui::Window::new("Select Tool Type")
                .collapsible(false)
                .resizable(false)
                .default_size([300.0, 150.0])
                .show(ctx, |ui| {
                    ui.heading("Select Tool Type");
                    ui.separator();

                    if ui.button("Stdio Tool").clicked() {
                        self.show_stdio_popup = true;
                        self.editing_tool_index = None;
                        self.temp_tool_name = String::new();
                        self.temp_stdio_config = serde_json::to_string_pretty(
                            &crate::agent_builder_types::StdioToolConfig::default(),
                        )
                        .unwrap_or_default();
                        should_close = true;
                    }

                    if ui.button("SSE Tool").clicked() {
                        self.show_sse_popup = true;
                        self.editing_tool_index = None;
                        self.temp_tool_name = String::new();
                        self.temp_sse_port = "3001".to_string();
                        should_close = true;
                    }
                });
        }

        if should_close {
            self.show_add_tool_menu = false;
        }
    }

    /// Show stdio tool configuration popup
    fn show_stdio_popup(&mut self, ctx: &egui::Context) {
        let mut should_close = false;

        if self.show_stdio_popup {
            egui::Window::new("Stdio Tool Configuration")
                .collapsible(false)
                .resizable(true)
                .default_size([600.0, 400.0])
                .show(ctx, |ui| {
                    ui.heading("Configure Stdio Tool");
                    ui.separator();

                    // Tool name
                    ui.horizontal(|ui| {
                        ui.label("Tool Name:");
                        ui.text_edit_singleline(&mut self.temp_tool_name);
                    });

                    ui.separator();

                    // JSON configuration
                    ui.label("JSON Configuration:");
                    let available_height = ui.available_height() - 80.0; // Leave space for buttons
                    egui::ScrollArea::vertical()
                        .max_height(available_height)
                        .show(ui, |ui| {
                            ui.add_sized(
                                [ui.available_width(), available_height],
                                egui::TextEdit::multiline(&mut self.temp_stdio_config)
                                    .font(egui::FontId::monospace(14.0)),
                            );
                        });

                    ui.separator();

                    // Buttons
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save").clicked() {
                            if self.save_stdio_tool() {
                                should_close = true;
                            }
                        }

                        if ui.button("❌ Cancel").clicked() {
                            should_close = true;
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("📋 Example").clicked() {
                                self.temp_stdio_config = serde_json::to_string_pretty(
                                    &crate::agent_builder_types::StdioToolConfig {
                                        command: "uvx".to_string(),
                                        args: vec!["mcp-server-neo4j".to_string()],
                                        environment: Some({
                                            let mut env = std::collections::HashMap::new();
                                            env.insert(
                                                "NEO4J_URL".to_string(),
                                                "bolt://localhost:7687".to_string(),
                                            );
                                            env.insert(
                                                "NEO4J_USERNAME".to_string(),
                                                "neo4j".to_string(),
                                            );
                                            env.insert(
                                                "NEO4J_PASSWORD".to_string(),
                                                "password".to_string(),
                                            );
                                            env
                                        }),
                                    },
                                )
                                .unwrap_or_default();
                            }
                        });
                    });
                });
        }

        if should_close {
            self.show_stdio_popup = false;
            self.editing_tool_index = None;
            self.temp_tool_name.clear();
            self.temp_stdio_config.clear();
        }
    }

    /// Show SSE tool configuration popup
    fn show_sse_popup(&mut self, ctx: &egui::Context) {
        let mut should_close = false;

        if self.show_sse_popup {
            egui::Window::new("SSE Tool Configuration")
                .collapsible(false)
                .resizable(false)
                .default_size([400.0, 200.0])
                .show(ctx, |ui| {
                    ui.heading("Configure SSE Tool");
                    ui.separator();

                    // Tool name
                    ui.horizontal(|ui| {
                        ui.label("Tool Name:");
                        ui.text_edit_singleline(&mut self.temp_tool_name);
                    });

                    // Port number
                    ui.horizontal(|ui| {
                        ui.label("Port:");
                        ui.text_edit_singleline(&mut self.temp_sse_port);
                    });

                    ui.separator();

                    // Buttons
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save").clicked() {
                            if self.save_sse_tool() {
                                should_close = true;
                            }
                        }

                        if ui.button("❌ Cancel").clicked() {
                            should_close = true;
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label("Port for SSE endpoint (e.g., 3001)");
                        });
                    });
                });
        }

        if should_close {
            self.show_sse_popup = false;
            self.editing_tool_index = None;
            self.temp_tool_name.clear();
            self.temp_sse_port.clear();
        }
    }

    /// Save stdio tool configuration
    fn save_stdio_tool(&mut self) -> bool {
        if self.temp_tool_name.trim().is_empty() {
            self.show_error_message("Tool name cannot be empty");
            return false;
        }

        // Parse JSON configuration
        let stdio_config: crate::agent_builder_types::StdioToolConfig =
            match serde_json::from_str(&self.temp_stdio_config) {
                Ok(config) => config,
                Err(e) => {
                    self.show_error_message(&format!("Invalid JSON configuration: {}", e));
                    return false;
                }
            };

        // Create or update tool
        let tool = crate::agent_builder_types::ToolConfig {
            name: self.temp_tool_name.clone(),
            enabled: true,
            transport_type: crate::agent_builder_types::ToolTransportType::Stdio,
            stdio_config: Some(stdio_config),
            sse_config: None,
            description: None,
        };

        if let Some(index) = self.editing_tool_index {
            // Update existing tool
            if index < self.config.tools.len() {
                self.config.tools[index] = tool;
                self.show_success_message("Stdio tool updated successfully");
            }
        } else {
            // Add new tool
            self.config.tools.push(tool);
            self.show_success_message("Stdio tool added successfully");
        }

        true
    }

    /// Save SSE tool configuration
    fn save_sse_tool(&mut self) -> bool {
        if self.temp_tool_name.trim().is_empty() {
            self.show_error_message("Tool name cannot be empty");
            return false;
        }

        // Parse port number
        let port: u16 = match self.temp_sse_port.parse() {
            Ok(port) => port,
            Err(_) => {
                self.show_error_message("Invalid port number");
                return false;
            }
        };

        // Create SSE configuration
        let sse_config = crate::agent_builder_types::SSEToolConfig {
            port,
            endpoint: format!("http://localhost:{{port}}/sse"),
            auto_port: true,
            timeout_seconds: Some(30),
            health_monitoring: true,
            retry_attempts: Some(3),
            headers: None,
            validate_ssl: Some(true),
            keep_alive: Some(true),
        };

        // Create or update tool
        let tool = crate::agent_builder_types::ToolConfig {
            name: self.temp_tool_name.clone(),
            enabled: true,
            transport_type: crate::agent_builder_types::ToolTransportType::SSE,
            stdio_config: None,
            sse_config: Some(sse_config),
            description: None,
        };

        if let Some(index) = self.editing_tool_index {
            // Update existing tool
            if index < self.config.tools.len() {
                self.config.tools[index] = tool;
                self.show_success_message("SSE tool updated successfully");
            }
        } else {
            // Add new tool
            self.config.tools.push(tool);
            self.show_success_message("SSE tool added successfully");
        }

        true
    }

    /// Edit existing tool
    fn edit_tool(&mut self, index: usize) {
        if index >= self.config.tools.len() {
            return;
        }

        let tool = &self.config.tools[index];
        self.editing_tool_index = Some(index);
        self.temp_tool_name = tool.name.clone();

        match tool.transport_type {
            crate::agent_builder_types::ToolTransportType::Stdio => {
                if let Some(stdio_config) = &tool.stdio_config {
                    self.temp_stdio_config =
                        serde_json::to_string_pretty(stdio_config).unwrap_or_default();
                } else {
                    self.temp_stdio_config = serde_json::to_string_pretty(
                        &crate::agent_builder_types::StdioToolConfig::default(),
                    )
                    .unwrap_or_default();
                }
                self.show_stdio_popup = true;
            }
            crate::agent_builder_types::ToolTransportType::SSE => {
                if let Some(sse_config) = &tool.sse_config {
                    self.temp_sse_port = sse_config.port.to_string();
                } else {
                    self.temp_sse_port = "3001".to_string();
                }
                self.show_sse_popup = true;
            }
            crate::agent_builder_types::ToolTransportType::Docker => {
                // For Docker tools, show stdio popup with Docker configuration
                if let Some(stdio_config) = &tool.stdio_config {
                    self.temp_stdio_config =
                        serde_json::to_string_pretty(stdio_config).unwrap_or_default();
                } else {
                    self.temp_stdio_config = serde_json::to_string_pretty(
                        &crate::agent_builder_types::StdioToolConfig::default(),
                    )
                    .unwrap_or_default();
                }
                self.show_stdio_popup = true;
            }
        }
    }

    /// Delete tool
    fn delete_tool(&mut self, index: usize) {
        if index >= self.config.tools.len() {
            return;
        }

        let tool_name = self.config.tools[index].name.clone();
        self.config.tools.remove(index);
        self.show_success_message(&format!("Tool '{}' deleted successfully", tool_name));
    }
}

impl eframe::App for AgentBuilderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.show(ctx);
    }
}

/// Run agent builder GUI application
pub fn run_gui() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Simple Agent Builder",
        options,
        Box::new(|_cc| Ok(Box::new(AgentBuilderApp::new()))),
    )
    .map_err(|e| anyhow!("Failed to run GUI: {}", e))
}

/// Run agent builder GUI application with custom paths
pub fn run_gui_with_paths<P: AsRef<Path>>(output_dir: P, default_agent_path: P) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Simple Agent Builder",
        options,
        Box::new(|_cc| {
            Ok(Box::new(AgentBuilderApp::new_with_paths(
                output_dir,
                default_agent_path,
            )))
        }),
    )
    .map_err(|e| anyhow!("Failed to run GUI: {}", e))
}
