//! This module provides integration with `Ollama`'s API, enabling the use of language models and
//! embeddings within the Swiftide project. It includes the `Ollama` struct for managing API clients
//! and default options for embedding and prompt models. The module is conditionally compiled based
//! on the "ollama" feature flag.

use config::LMStudioConfig;

use crate::openai;

pub mod config;

/// The `Ollama` struct encapsulates an `Ollama` client and default options for embedding and prompt
/// models. It uses the `Builder` pattern for flexible and customizable instantiation.
///
/// By default it will look for a `OLLAMA_API_KEY` environment variable. Note that either a prompt
/// model or embedding model always need to be set, either with
/// [`Ollama::with_default_prompt_model`] or [`Ollama::with_default_embed_model`] or via the
/// builder. You can find available models in the Ollama documentation.
///
/// Under the hood it uses [`async_openai`], with the Ollama openai mapping. This means
/// some features might not work as expected. See the Ollama documentation for details.
pub type LMStudio = openai::GenericOpenAI<LMStudioConfig>;
pub type LMStudioBuilder = openai::GenericOpenAIBuilder<LMStudioConfig>;
pub type LMStudioBuilderError = openai::GenericOpenAIBuilderError;
pub use openai::{Options, OptionsBuilder, OptionsBuilderError};

impl LMStudio {
    /// Build a new `Ollama` instance
    pub fn builder() -> LMStudioBuilder {
        LMStudioBuilder::default()
    }
}
impl Default for LMStudio {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_default_prompt_model() {
        let openai = LMStudio::builder()
            .default_prompt_model("qwen2.5-coder-7b-instruct")
            .build()
            .unwrap();
        assert_eq!(
            openai.default_options.prompt_model,
            Some("qwen2.5-coder-7b-instruct".to_string())
        );
    }

    #[test]
    fn test_default_embed_model() {
        let lmstudio = LMStudio::builder()
            .default_embed_model("mxbai-embed-large")
            .build()
            .unwrap();
        assert_eq!(
            lmstudio.default_options.embed_model,
            Some("mxbai-embed-large".to_string())
        );
    }

    #[test]
    fn test_default_models() {
        let lmstudio = LMStudio::builder()
            .default_embed_model("mxbai-embed-large")
            .default_prompt_model("qwen2.5-coder-7b-instruct")
            .build()
            .unwrap();
        assert_eq!(
            lmstudio.default_options.embed_model,
            Some("mxbai-embed-large".to_string())
        );
        assert_eq!(
            lmstudio.default_options.prompt_model,
            Some("qwen2.5-coder-7b-instruct".to_string())
        );
    }

    #[test]
    fn test_building_via_default_prompt_model() {
        let mut client = LMStudio::default();

        assert!(client.default_options.prompt_model.is_none());

        client.with_default_prompt_model("qwen2.5-coder-7b-instruct");
        assert_eq!(
            client.default_options.prompt_model,
            Some("qwen2.5-coder-7b-instruct".to_string())
        );
    }

    #[test]
    fn test_building_via_default_embed_model() {
        let mut client = LMStudio::default();

        assert!(client.default_options.embed_model.is_none());

        client.with_default_embed_model("mxbai-embed-large");
        assert_eq!(
            client.default_options.embed_model,
            Some("mxbai-embed-large".to_string())
        );
    }

    #[test]
    fn test_building_via_default_models() {
        let mut client = LMStudio::default();

        assert!(client.default_options.embed_model.is_none());

        client.with_default_prompt_model("qwen2.5-coder-7b-instruct");
        client.with_default_embed_model("mxbai-embed-large");
        assert_eq!(
            client.default_options.prompt_model,
            Some("qwen2.5-coder-7b-instruct".to_string())
        );
        assert_eq!(
            client.default_options.embed_model,
            Some("mxbai-embed-large".to_string())
        );
    }
}
