//! Main entry point for the Simple Agent Builder GUI application
//!
//! Run with: cargo run

mod agent_build_system;
mod agent_builder_types;
mod AgentRunner;
mod port_allocator;
mod sse_validator;
mod sse_manager;

use agent_build_system::run_gui;
use anyhow::Result;

fn main() -> Result<()> {
    // Run the GUI application
    run_gui()
}