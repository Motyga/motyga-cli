//! OSS provider utilities shared between TUI and exec.

pub mod discovery;

use std::collections::HashMap;

use motyga_config::config_toml::ConfigToml;
use motyga_core::config::Config;
use motyga_model_provider_info::LMSTUDIO_OSS_PROVIDER_ID;
use motyga_model_provider_info::ModelProviderInfo;
use motyga_model_provider_info::OLLAMA_OSS_PROVIDER_ID;
use motyga_model_provider_info::built_in_model_providers;
use motyga_model_provider_info::merge_configured_model_providers;

use crate::discovery::ResponsesApiSupport;
use crate::discovery::probe_local_server;

/// Returns the default model for a given OSS provider.
pub fn get_default_model_for_oss_provider(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => Some(motyga_lmstudio::DEFAULT_OSS_MODEL),
        OLLAMA_OSS_PROVIDER_ID => Some(motyga_ollama::DEFAULT_OSS_MODEL),
        _ => None,
    }
}

/// Providers as `Config` will see them: built-ins plus `[model_providers]`.
fn resolved_providers(config_toml: &ConfigToml) -> HashMap<String, ModelProviderInfo> {
    let built_in = built_in_model_providers(/*openai_base_url*/ None);
    merge_configured_model_providers(built_in.clone(), config_toml.model_providers.clone())
        .unwrap_or(built_in)
}

fn provider_api_key(provider: &ModelProviderInfo) -> Option<String> {
    provider
        .env_key
        .as_deref()
        .and_then(|var| std::env::var(var).ok())
        .filter(|key| !key.trim().is_empty())
        .or_else(|| provider.experimental_bearer_token.clone())
}

/// What an `--oss` session starts with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OssSessionModel {
    pub model: String,
    /// Context window the server runs `model` with, when it reports one.
    pub context_window: Option<i64>,
}

/// Picks the model an `--oss` session starts with.
///
/// `requested` (`-m`) wins, then the saved `oss_model` when it belongs to the
/// selected provider. Otherwise the first model the server already has is used
/// (the provider's default when it is among them), so nothing large is
/// downloaded behind the user's back. Only a server with no models at all falls
/// back to the provider default, which Ollama and LM Studio then fetch. `None`
/// means there is no model to run.
pub async fn pick_oss_session_model(
    provider_id: &str,
    requested: Option<&str>,
    config_toml: &ConfigToml,
) -> Option<OssSessionModel> {
    let providers = resolved_providers(config_toml);
    let served = match providers
        .get(provider_id)
        .and_then(|provider| Some((provider.base_url.clone()?, provider_api_key(provider))))
    {
        Some((base_url, api_key)) => probe_local_server(&base_url, api_key.as_deref())
            .await
            .map(|report| report.models)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let saved = (config_toml.oss_provider.as_deref() == Some(provider_id))
        .then(|| config_toml.oss_model.clone())
        .flatten();
    let default_model = get_default_model_for_oss_provider(provider_id);
    let model = requested
        .map(str::to_string)
        .or(saved)
        .or_else(|| {
            served
                .iter()
                .find(|model| Some(model.id.as_str()) == default_model)
                .or_else(|| served.first())
                .map(|model| model.id.clone())
        })
        .or_else(|| default_model.map(str::to_string))?;
    let context_window = served
        .iter()
        .find(|served_model| served_model.id == model)
        .and_then(|served_model| served_model.context_window);
    Some(OssSessionModel {
        model,
        context_window,
    })
}

/// For `--oss` without a configured provider in a non-interactive run: the one
/// running server among the built-in Ollama and LM Studio providers.
pub async fn detect_running_oss_provider(config_toml: &ConfigToml) -> Result<String, String> {
    let providers = resolved_providers(config_toml);
    let candidates = [OLLAMA_OSS_PROVIDER_ID, LMSTUDIO_OSS_PROVIDER_ID];
    let probes = futures::future::join_all(candidates.iter().map(|id| {
        let provider = providers.get(*id).cloned();
        async move {
            let provider = provider?;
            let base_url = provider.base_url.clone()?;
            probe_local_server(&base_url, provider_api_key(&provider).as_deref())
                .await
                .ok()
                .filter(|report| report.responses_api != ResponsesApiSupport::Missing)
        }
    }))
    .await;
    let running: Vec<&str> = candidates
        .iter()
        .zip(probes)
        .filter_map(|(id, probe)| probe.map(|_| *id))
        .collect();
    match running.as_slice() {
        [provider] => Ok((*provider).to_string()),
        [] => Err(
            "No local model server found. Start Ollama (`ollama serve`) or the LM Studio server, or pass --local-provider <id> (see `motyga local`)."
                .to_string(),
        ),
        _ => Err(format!(
            "Both Ollama and LM Studio are running; choose one with --local-provider {OLLAMA_OSS_PROVIDER_ID}|{LMSTUDIO_OSS_PROVIDER_ID} or `motyga local add`."
        )),
    }
}

/// Ensures the specified OSS provider is ready (models downloaded, service reachable).
///
/// Returns warnings to show the user, such as a context window too small for an agent.
pub async fn ensure_oss_provider_ready(
    provider_id: &str,
    config: &Config,
) -> Result<Vec<String>, std::io::Error> {
    let downloads_models = match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => {
            motyga_lmstudio::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
            true
        }
        OLLAMA_OSS_PROVIDER_ID => {
            motyga_ollama::ensure_responses_supported(&config.model_provider).await?;
            motyga_ollama::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
            true
        }
        _ => false,
    };

    let provider = &config.model_provider;
    let Some(base_url) = provider.base_url.as_deref() else {
        return Ok(Vec::new());
    };
    let report = match probe_local_server(base_url, provider_api_key(provider).as_deref()).await
    {
        Ok(report) => report,
        // Ollama and LM Studio were just checked above with their own clients.
        Err(_) if downloads_models => return Ok(Vec::new()),
        Err(err) => {
            return Err(std::io::Error::other(format!(
                "Local provider `{provider_id}` at {base_url} is not usable: {err}. Check it with `motyga local`."
            )));
        }
    };
    let model = config.model.as_deref();
    if report.responses_api == ResponsesApiSupport::Missing {
        return Err(std::io::Error::other(
            report.warnings_for(model).join("\n"),
        ));
    }
    if let Some(model) = model
        && !report.models.is_empty()
        && report.model(model).is_none()
    {
        let available: Vec<&str> = report.models.iter().map(|model| model.id.as_str()).collect();
        return Err(std::io::Error::other(format!(
            "{} does not serve `{model}`. Available: {}",
            report.base_url,
            available.join(", ")
        )));
    }
    Ok(report.warnings_for(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_get_default_model_for_provider_lmstudio() {
        let result = get_default_model_for_oss_provider(LMSTUDIO_OSS_PROVIDER_ID);
        assert_eq!(result, Some(motyga_lmstudio::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_ollama() {
        let result = get_default_model_for_oss_provider(OLLAMA_OSS_PROVIDER_ID);
        assert_eq!(result, Some(motyga_ollama::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_unknown() {
        let result = get_default_model_for_oss_provider("unknown-provider");
        assert_eq!(result, None);
    }

    fn config_toml_with_local_provider(base_url: &str) -> ConfigToml {
        toml::from_str(&format!(
            "[model_providers.llamacpp]\nname = \"llama.cpp\"\nbase_url = \"{base_url}\"\n"
        ))
        .expect("valid config")
    }

    #[tokio::test]
    async fn saved_oss_model_wins_only_for_its_provider() {
        let config_toml: ConfigToml = toml::from_str(
            "oss_provider = \"vllm-box\"\noss_model = \"qwen3:8b\"\n\
             [model_providers.vllm-box]\nname = \"vLLM\"\nbase_url = \"http://127.0.0.1:9/v1\"\n",
        )
        .expect("valid config");

        assert_eq!(
            [
                pick_oss_session_model("vllm-box", /*requested*/ None, &config_toml).await,
                pick_oss_session_model("vllm-box", Some("other"), &config_toml).await,
                // The saved model belongs to `vllm-box`; an unknown provider has nothing.
                pick_oss_session_model("missing-provider", /*requested*/ None, &config_toml)
                    .await,
            ],
            [
                Some(OssSessionModel {
                    model: "qwen3:8b".to_string(),
                    context_window: None,
                }),
                Some(OssSessionModel {
                    model: "other".to_string(),
                    context_window: None,
                }),
                None,
            ]
        );
    }

    #[tokio::test]
    async fn session_model_comes_from_the_running_server() {
        if std::env::var(motyga_core::spawn::MOTYGA_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok() {
            return;
        }
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/v1/models"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"data": [{"id": "qwen3-coder-30b"}, {"id": "gemma-3-12b"}]}),
            ))
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/props"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"default_generation_settings": {"n_ctx": 32768}}),
            ))
            .mount(&server)
            .await;
        let config_toml = config_toml_with_local_provider(&format!("{}/v1", server.uri()));

        assert_eq!(
            pick_oss_session_model("llamacpp", /*requested*/ None, &config_toml).await,
            Some(OssSessionModel {
                model: "qwen3-coder-30b".to_string(),
                context_window: Some(32768),
            })
        );
    }
}
