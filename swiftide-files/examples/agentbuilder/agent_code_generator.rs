//! Simple Agent Code Generator
//!
//! This module provides a simplified code generation system for creating agents
//! by copying a base executable and generating configuration files.
//!
//! ## Approach
//!
//! Instead of generating complete Rust projects, this system:
//! 1. Copies a base `AgentRunner.exe` to a new name
//! 2. Generates a corresponding JSON configuration file
//! 3. The executable dynamically adapts based on the JSON config
//!
//! ## Features
//!
//! - Simple configuration file generation
//! - Executable copying and renaming
//! - Validation of agent configurations
//! - Easy deployment and management

use agent_builder_types::{AgentConfig, to_legacy_agent_config};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

/// Simple agent generator for exe + json deployment
#[derive(Debug, Clone)]
pub struct SimpleAgentGenerator {
    /// Path to the base executable template
    base_exe_path: PathBuf,
    /// Output directory for generated agents
    output_directory: PathBuf,
}

impl SimpleAgentGenerator {
    /// Create a new simple agent generator
    ///
    /// # Arguments
    ///
    /// * `base_exe_path` - Path to the base AgentRunner.exe
    /// * `output_directory` - Directory where agents will be created
    ///
    /// # Example
    ///
    /// ```rust
    /// let generator = SimpleAgentGenerator::new(
    ///     "path/to/AgentRunner.exe",
    ///     "output/agents"
    /// )?;
    /// ```
    pub fn new(base_exe_path: &str, output_directory: &str) -> Result<Self> {
        let base_path = PathBuf::from(base_exe_path);
        let output_path = PathBuf::from(output_directory);

        // Validate base executable exists
        if !base_path.exists() {
            return Err(anyhow!(
                "Base executable not found: {:?}. Please ensure AgentRunner.exe exists.",
                base_path
            ));
        }

        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_path)
            .with_context(|| format!("Failed to create output directory: {:?}", output_path))?;

        info!(
            "SimpleAgentGenerator initialized with base exe: {:?}",
            base_path
        );
        info!("Output directory: {:?}", output_path);

        Ok(Self {
            base_exe_path: base_path,
            output_directory: output_path,
        })
    }

    /// Generate a new agent from configuration
    ///
    /// # Arguments
    ///
    /// * `config` - The agent configuration
    ///
    /// # Returns
    ///
    /// * `Result<AgentDeployment>` - Information about the generated agent
    ///
    /// # Example
    ///
    /// ```rust
    /// let deployment = generator.generate_agent(&agent_config).await?;
    /// println!("Agent created at: {:?}", deployment.exe_path);
    /// ```
    pub async fn generate_agent(&self, config: &AgentConfig) -> Result<AgentDeployment> {
        info!("🚀 Generating agent: {}", config.name);

        // Validate configuration
        self.validate_config(config)?;

        // Generate deployment paths
        let deployment = self.create_deployment_paths(config)?;

        // Copy and rename executable
        self.copy_executable(&deployment).await?;

        // Generate configuration file
        self.generate_config_file(config, &deployment).await?;

        info!("✅ Agent generated successfully:");
        info!("   Executable: {:?}", deployment.exe_path);
        info!("   Config: {:?}", deployment.config_path);

        Ok(deployment)
    }

    /// Generate multiple agents from configurations
    ///
    /// # Arguments
    ///
    /// * `configs` - Vector of agent configurations
    ///
    /// # Returns
    ///
    /// * `Result<Vec<AgentDeployment>>` - Information about all generated agents
    pub async fn generate_agents(&self, configs: &[AgentConfig]) -> Result<Vec<AgentDeployment>> {
        let mut deployments = Vec::new();

        for config in configs {
            match self.generate_agent(config).await {
                Ok(deployment) => deployments.push(deployment),
                Err(e) => {
                    error!("Failed to generate agent {}: {}", config.name, e);
                    return Err(e);
                }
            }
        }

        info!("🎉 Generated {} agents successfully", deployments.len());
        Ok(deployments)
    }

    /// Validate agent configuration
    fn validate_config(&self, config: &AgentConfig) -> Result<()> {
        if config.name.is_empty() {
            return Err(anyhow!("Agent name cannot be empty"));
        }

        if config.name.len() > 50 {
            return Err(anyhow!("Agent name cannot exceed 50 characters"));
        }

        if !config
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow!(
                "Agent name can only contain letters, numbers, hyphens, and underscores"
            ));
        }

        if config.system_prompt.is_empty() {
            return Err(anyhow!("System prompt cannot be empty"));
        }

        if config.llm.model.is_empty() {
            return Err(anyhow!("LLM model cannot be empty"));
        }

        // Validate API key for providers that require it
        if config.llm.provider.requires_api_key() {
            if config.llm.api_key.as_ref().map_or(true, |k| k.is_empty()) {
                warn!(
                    "API key not provided for {} provider",
                    config.llm.provider.as_str()
                );
            }
        }

        // Check if agent already exists
        let exe_path = self.output_directory.join(format!("{}.exe", config.name));
        if exe_path.exists() {
            warn!("Agent executable already exists: {:?}", exe_path);
        }

        let config_path = self.output_directory.join(format!("{}.json", config.name));
        if config_path.exists() {
            warn!("Agent config already exists: {:?}", config_path);
        }

        Ok(())
    }

    /// Create deployment paths for the agent
    fn create_deployment_paths(&self, config: &AgentConfig) -> Result<AgentDeployment> {
        let exe_name = format!("{}.exe", config.name);
        let config_name = format!("{}.json", config.name);

        let exe_path = self.output_directory.join(&exe_name);
        let config_path = self.output_directory.join(&config_name);

        Ok(AgentDeployment {
            name: config.name.clone(),
            exe_path,
            config_path,
            base_exe_path: self.base_exe_path.clone(),
        })
    }

    /// Copy and rename the base executable
    async fn copy_executable(&self, deployment: &AgentDeployment) -> Result<()> {
        info!(
            "Copying executable: {:?} -> {:?}",
            deployment.base_exe_path, deployment.exe_path
        );

        fs::copy(&deployment.base_exe_path, &deployment.exe_path).with_context(|| {
            format!(
                "Failed to copy executable from {:?} to {:?}",
                deployment.base_exe_path, deployment.exe_path
            )
        })?;

        debug!("Executable copied successfully");
        Ok(())
    }

    /// Generate the JSON configuration file
    async fn generate_config_file(
        &self,
        config: &AgentConfig,
        deployment: &AgentDeployment,
    ) -> Result<()> {
        info!(
            "Generating configuration file: {:?}",
            deployment.config_path
        );

        // Convert to legacy format for compatibility with AgentRunner
        let legacy_config = to_legacy_agent_config(config)?;

        // Serialize to pretty JSON
        let json_content = serde_json::to_string_pretty(&legacy_config)
            .with_context(|| "Failed to serialize configuration to JSON")?;

        // Write to file
        fs::write(&deployment.config_path, json_content).with_context(|| {
            format!(
                "Failed to write configuration file: {:?}",
                deployment.config_path
            )
        })?;

        debug!("Configuration file generated successfully");
        Ok(())
    }

    /// List all existing agents in the output directory
    pub fn list_agents(&self) -> Result<Vec<AgentInfo>> {
        let mut agents = Vec::new();

        if !self.output_directory.exists() {
            return Ok(agents);
        }

        let entries = fs::read_dir(&self.output_directory).with_context(|| {
            format!(
                "Failed to read output directory: {:?}",
                self.output_directory
            )
        })?;

        for entry in entries {
            let entry = entry.with_context(|| "Failed to read directory entry")?;
            let path = entry.path();

            // Look for .exe files
            if path.extension().and_then(|s| s.to_str()) == Some("exe") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    let config_path = self.output_directory.join(format!("{}.json", name));
                    let has_config = config_path.exists();

                    agents.push(AgentInfo {
                        name: name.to_string(),
                        exe_path: path,
                        config_path,
                        has_config,
                    });
                }
            }
        }

        agents.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(agents)
    }

    /// Delete an agent (both exe and config)
    pub fn delete_agent(&self, agent_name: &str) -> Result<()> {
        let exe_path = self.output_directory.join(format!("{}.exe", agent_name));
        let config_path = self.output_directory.join(format!("{}.json", agent_name));

        let mut deleted_files = Vec::new();

        if exe_path.exists() {
            fs::remove_file(&exe_path)
                .with_context(|| format!("Failed to remove executable: {:?}", exe_path))?;
            deleted_files.push(exe_path);
        }

        if config_path.exists() {
            fs::remove_file(&config_path)
                .with_context(|| format!("Failed to remove config: {:?}", config_path))?;
            deleted_files.push(config_path);
        }

        if deleted_files.is_empty() {
            warn!("No files found for agent: {}", agent_name);
        } else {
            info!(
                "Deleted agent {} and removed {} files",
                agent_name,
                deleted_files.len()
            );
        }

        Ok(())
    }

    /// Update an agent's configuration
    pub async fn update_agent_config(&self, config: &AgentConfig) -> Result<()> {
        info!("Updating configuration for agent: {}", config.name);

        let config_path = self.output_directory.join(format!("{}.json", config.name));

        if !config_path.exists() {
            return Err(anyhow!(
                "Agent configuration not found: {:?}. Use generate_agent first.",
                config_path
            ));
        }

        // Validate configuration
        self.validate_config(config)?;

        // Generate new configuration file
        let legacy_config = to_legacy_agent_config(config)?;
        let json_content = serde_json::to_string_pretty(&legacy_config)
            .with_context(|| "Failed to serialize configuration to JSON")?;

        fs::write(&config_path, json_content).with_context(|| {
            format!(
                "Failed to write updated configuration file: {:?}",
                config_path
            )
        })?;

        info!("✅ Configuration updated for agent: {}", config.name);
        Ok(())
    }

    /// Get the base executable path
    pub fn base_exe_path(&self) -> &Path {
        &self.base_exe_path
    }

    /// Get the output directory
    pub fn output_directory(&self) -> &Path {
        &self.output_directory
    }
}

/// Information about a generated agent deployment
#[derive(Debug, Clone)]
pub struct AgentDeployment {
    /// Agent name
    pub name: String,
    /// Path to the generated executable
    pub exe_path: PathBuf,
    /// Path to the generated configuration file
    pub config_path: PathBuf,
    /// Path to the base executable that was copied
    pub base_exe_path: PathBuf,
}

impl AgentDeployment {
    /// Check if both exe and config files exist
    pub fn is_complete(&self) -> bool {
        self.exe_path.exists() && self.config_path.exists()
    }

    /// Get the size of the executable in bytes
    pub fn exe_size(&self) -> Result<u64> {
        Ok(fs::metadata(&self.exe_path)?.len())
    }

    /// Get the size of the config file in bytes
    pub fn config_size(&self) -> Result<u64> {
        Ok(fs::metadata(&self.config_path)?.len())
    }
}

/// Information about an existing agent
#[derive(Debug, Clone)]
pub struct AgentInfo {
    /// Agent name
    pub name: String,
    /// Path to the executable
    pub exe_path: PathBuf,
    /// Path to the configuration file
    pub config_path: PathBuf,
    /// Whether the config file exists
    pub has_config: bool,
}

impl AgentInfo {
    /// Check if the agent is properly configured
    pub fn is_configured(&self) -> bool {
        self.exe_path.exists() && self.has_config
    }

    /// Load the agent's configuration
    pub fn load_config(&self) -> Result<AgentConfig> {
        if !self.has_config {
            return Err(anyhow!(
                "No configuration file found for agent: {}",
                self.name
            ));
        }

        let config_content = fs::read_to_string(&self.config_path)
            .with_context(|| format!("Failed to read config file: {:?}", self.config_path))?;

        let legacy_config: crate::AgentConfig = serde_json::from_str(&config_content)
            .with_context(|| "Failed to parse configuration JSON")?;

        agent_builder_types::from_legacy_agent_config(&legacy_config)
    }
}

/// Utility functions for agent management
pub mod utils {
    use super::*;
    use std::process::Command;

    /// Sanitize an agent name for file system use
    pub fn sanitize_agent_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim_matches('_')
            .to_string()
    }

    /// Validate that an agent name is valid
    pub fn validate_agent_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(anyhow!("Agent name cannot be empty"));
        }

        if name.len() > 50 {
            return Err(anyhow!("Agent name cannot exceed 50 characters"));
        }

        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
        {
            return Err(anyhow!(
                "Agent name can only contain letters, numbers, hyphens, and underscores"
            ));
        }

        Ok(())
    }

    /// Check if a port is available on the local system
    pub fn is_port_available(port: u16) -> bool {
        // Simple check - in a real implementation, you might want to actually try binding
        port > 0 && port < 65536 && port != 80 && port != 443
    }

    /// Get suggested ports for agents
    pub fn get_suggested_ports() -> Vec<u16> {
        vec![8080, 8081, 8082, 3000, 3001, 5000, 5001, 9000, 9001]
    }

    /// Run an agent executable
    pub fn run_agent(exe_path: &Path) -> Result<std::process::Child> {
        info!("Starting agent: {:?}", exe_path);

        let child = Command::new(exe_path)
            .spawn()
            .with_context(|| format!("Failed to start agent: {:?}", exe_path))?;

        Ok(child)
    }

    /// Check if an agent is running (by checking if the process exists)
    pub fn is_agent_running(agent_name: &str) -> Result<bool> {
        // This is a simplified check - in a real implementation,
        // you might want to check for running processes or use a PID file
        let output = Command::new("tasklist")
            .args(&["/FI", "IMAGENAME eq", &format!("{}.exe", agent_name)])
            .output();

        match output {
            Ok(result) => {
                let output_str = String::from_utf8_lossy(&result.stdout);
                Ok(output_str.contains(agent_name))
            }
            Err(_) => {
                // Fallback for non-Windows systems
                Ok(false)
            }
        }
    }

    /// Generate a unique agent name
    pub fn generate_unique_name(base_name: &str, existing_names: &[String]) -> String {
        let mut counter = 1;
        let mut candidate = format!("{}_{}", base_name, counter);

        while existing_names.contains(&candidate) {
            counter += 1;
            candidate = format!("{}_{}", base_name, counter);
        }

        candidate
    }

    /// Create a backup of an agent's configuration
    pub fn backup_config(config_path: &Path) -> Result<PathBuf> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let backup_name = format!(
            "{}.backup.{}.json",
            config_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("config"),
            timestamp
        );

        let backup_path = config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(backup_name);

        fs::copy(config_path, &backup_path)
            .with_context(|| format!("Failed to backup config: {:?}", config_path))?;

        info!("Configuration backed up to: {:?}", backup_path);
        Ok(backup_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_builder_types::*;
    use tempfile::TempDir;

    #[test]
    fn test_sanitize_agent_name() {
        assert_eq!(utils::sanitize_agent_name("test-agent"), "test-agent");
        assert_eq!(utils::sanitize_agent_name("test agent"), "test_agent");
        assert_eq!(utils::sanitize_agent_name("test@agent"), "test_agent");
        assert_eq!(utils::sanitize_agent_name("123agent"), "123agent");
    }

    #[test]
    fn test_validate_agent_name() {
        assert!(utils::validate_agent_name("valid-name").is_ok());
        assert!(utils::validate_agent_name("invalid name").is_err());
        assert!(utils::validate_agent_name("").is_err());
        assert!(utils::validate_agent_name("a".repeat(51).as_str()).is_err());
    }

    #[test]
    fn test_generate_unique_name() {
        let existing = vec!["agent_1".to_string(), "agent_2".to_string()];
        let unique = utils::generate_unique_name("agent", &existing);
        assert_eq!(unique, "agent_3");
    }

    #[tokio::test]
    async fn test_simple_agent_generator() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_exe_path = temp_dir.path().join("AgentRunner.exe");

        // Create a dummy base executable
        fs::write(&base_exe_path, "dummy exe content")?;

        let generator = SimpleAgentGenerator::new(
            base_exe_path.to_str().unwrap(),
            temp_dir.path().to_str().unwrap(),
        )?;

        let config = AgentConfig {
            name: "TestAgent".to_string(),
            system_prompt: "You are a test agent".to_string(),
            llm: LLMConfig {
                provider: LLMProvider::OpenRouter,
                model: "test-model".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };

        let deployment = generator.generate_agent(&config).await?;

        assert!(deployment.exe_path.exists());
        assert!(deployment.config_path.exists());
        assert!(deployment.is_complete());

        Ok(())
    }
}
