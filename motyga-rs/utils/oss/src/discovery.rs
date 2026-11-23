//! Probing of local model servers (Ollama, LM Studio, llama.cpp, vLLM, or any
//! OpenAI-compatible server) so `motyga local` and `--oss` startup can tell the
//! user what is running, which models it serves, and whether it can drive an
//! agent at all.

use std::time::Duration;

use motyga_model_provider_info::is_local_base_url;
use serde_json::Value as JsonValue;

const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

/// Smallest context window that still holds the base instructions, tool
/// definitions and a few turns of history. Below it every turn either fails or,
/// worse, the server silently drops the start of the prompt.
pub const MIN_AGENT_CONTEXT_WINDOW: i64 = 16_384;

/// Which server software answered on a base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalServerKind {
    Ollama,
    LmStudio,
    LlamaCpp,
    Vllm,
    OpenAiCompatible,
}

impl LocalServerKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::LmStudio => "LM Studio",
            Self::LlamaCpp => "llama.cpp",
            Self::Vllm => "vLLM",
            Self::OpenAiCompatible => "OpenAI-compatible server",
        }
    }

    /// How to give a model a larger context window on this server.
    pub fn context_window_hint(self) -> &'static str {
        match self {
            Self::Ollama => {
                "set OLLAMA_CONTEXT_LENGTH=32768 (or more) and restart `ollama serve`"
            }
            Self::LmStudio => {
                "reload the model with a larger context length, e.g. `lms load <model> --context-length 32768`"
            }
            Self::LlamaCpp => "restart `llama-server` with `-c 32768` (or more)",
            Self::Vllm => "restart vLLM with `--max-model-len 32768` (or more)",
            Self::OpenAiCompatible => "configure the server to load the model with at least 32768 tokens of context",
        }
    }
}

/// A local server that Motyga looks for without being told about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WellKnownServer {
    /// Provider id that `motyga local` suggests for this server.
    pub provider_id: &'static str,
    pub kind: LocalServerKind,
    pub base_url: &'static str,
}

/// Default endpoints of the popular local servers. Ollama and LM Studio are
/// also built-in providers; llama.cpp and vLLM need `motyga local add`.
pub const WELL_KNOWN_SERVERS: [WellKnownServer; 4] = [
    WellKnownServer {
        provider_id: "ollama",
        kind: LocalServerKind::Ollama,
        base_url: "http://localhost:11434/v1",
    },
    WellKnownServer {
        provider_id: "lmstudio",
        kind: LocalServerKind::LmStudio,
        base_url: "http://localhost:1234/v1",
    },
    WellKnownServer {
        provider_id: "llamacpp",
        kind: LocalServerKind::LlamaCpp,
        base_url: "http://localhost:8080/v1",
    },
    WellKnownServer {
        provider_id: "vllm",
        kind: LocalServerKind::Vllm,
        base_url: "http://localhost:8000/v1",
    },
];

/// A model served by a local server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModel {
    pub id: String,
    /// Context window the server actually runs the model with, when it says.
    pub context_window: Option<i64>,
}

/// Whether the server exposes `POST {base_url}/responses`, the only wire API
/// Motyga speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesApiSupport {
    Supported,
    Missing,
    Unknown,
}

/// What a probe learned about one base URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalServerReport {
    pub base_url: String,
    pub kind: LocalServerKind,
    pub version: Option<String>,
    pub responses_api: ResponsesApiSupport,
    pub models: Vec<LocalModel>,
}

impl LocalServerReport {
    pub fn model(&self, id: &str) -> Option<&LocalModel> {
        self.models.iter().find(|model| model.id == id)
    }

    /// Human-readable problems that would keep `model` (or, without one, every
    /// listed model) from working as an agent.
    pub fn warnings_for(&self, model: Option<&str>) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.responses_api == ResponsesApiSupport::Missing {
            warnings.push(format!(
                "{} at {} does not serve POST /responses (the OpenAI Responses API). Update the server: Ollama 0.13.4+, LM Studio 0.3.29+, a recent llama.cpp or vLLM.",
                self.kind.display_name(),
                self.base_url
            ));
        }
        let small_context = self
            .models
            .iter()
            .filter(|candidate| model.is_none_or(|model| candidate.id == model))
            .filter_map(|candidate| {
                candidate
                    .context_window
                    .filter(|window| *window < MIN_AGENT_CONTEXT_WINDOW)
                    .map(|window| (candidate.id.as_str(), window))
            });
        for (id, window) in small_context {
            warnings.push(format!(
                "{id} runs with a {window}-token context, too small for an agent (needs {MIN_AGENT_CONTEXT_WINDOW}+): {}.",
                self.kind.context_window_hint()
            ));
        }
        warnings
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    /// Nothing answered, or the server refused the request.
    Unreachable(String),
    /// Something answered, but not with an OpenAI-style `/models` listing.
    NotOpenAiCompatible(String),
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable(reason) | Self::NotOpenAiCompatible(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Probe an OpenAI-compatible base URL (for example `http://localhost:11434/v1`).
///
/// Only `GET` requests and an empty-bodied `POST /responses` are sent, so no
/// inference runs and nothing is downloaded or loaded.
pub async fn probe_local_server(
    base_url: &str,
    api_key: Option<&str>,
) -> Result<LocalServerReport, ProbeError> {
    let base_url = base_url.trim_end_matches('/').to_string();
    let client = probe_client(&base_url)?;
    let root = server_root(&base_url);

    let listing = get_json(&client, &format!("{base_url}/models"), api_key)
        .await
        .map_err(ProbeError::Unreachable)?
        .ok_or_else(|| ProbeError::Unreachable(format!("{base_url}/models did not answer")))?;
    let entries = listing
        .get("data")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            ProbeError::NotOpenAiCompatible(format!(
                "{base_url}/models did not return an OpenAI-style model list"
            ))
        })?;
    let mut models: Vec<LocalModel> = entries
        .iter()
        .filter_map(|entry| {
            let id = entry.get("id").and_then(JsonValue::as_str)?;
            (!is_embedding_model(id)).then(|| LocalModel {
                id: id.to_string(),
                context_window: entry.get("max_model_len").and_then(JsonValue::as_i64),
            })
        })
        .collect();

    let owned_by_vllm = entries
        .iter()
        .any(|entry| entry.get("owned_by").and_then(JsonValue::as_str) == Some("vllm"));
    let (kind, version) = if owned_by_vllm {
        (LocalServerKind::Vllm, None)
    } else if let Ok(Some(version)) = get_json(&client, &format!("{root}/api/version"), api_key).await
        && let Some(version) = version.get("version").and_then(JsonValue::as_str)
    {
        if let Ok(Some(running)) = get_json(&client, &format!("{root}/api/ps"), api_key).await {
            apply_ollama_running_contexts(&mut models, &running);
        }
        (LocalServerKind::Ollama, Some(version.to_string()))
    } else if let Ok(Some(native)) = get_json(&client, &format!("{root}/api/v0/models"), api_key).await
        && native.get("data").and_then(JsonValue::as_array).is_some()
    {
        apply_lmstudio_contexts(&mut models, &native);
        (LocalServerKind::LmStudio, None)
    } else if let Ok(Some(props)) = get_json(&client, &format!("{root}/props"), api_key).await
        && let Some(n_ctx) = props
            .get("default_generation_settings")
            .and_then(|settings| settings.get("n_ctx"))
            .and_then(JsonValue::as_i64)
    {
        for model in &mut models {
            model.context_window.get_or_insert(n_ctx);
        }
        (LocalServerKind::LlamaCpp, None)
    } else {
        (LocalServerKind::OpenAiCompatible, None)
    };

    let responses_api = probe_responses_route(&client, &base_url, api_key).await;
    Ok(LocalServerReport {
        base_url,
        kind,
        version,
        responses_api,
        models,
    })
}

fn probe_client(base_url: &str) -> Result<reqwest::Client, ProbeError> {
    let builder = reqwest::Client::builder().timeout(PROBE_TIMEOUT);
    // A system proxy cannot reach a server on this machine or the LAN, and some
    // corporate proxies answer for any host, which would fake a running server.
    let builder = if is_local_base_url(base_url) {
        builder.no_proxy()
    } else {
        builder
    };
    builder
        .build()
        .map_err(|err| ProbeError::Unreachable(format!("failed to build HTTP client: {err}")))
}

/// `http://host:port/v1` -> `http://host:port`; native endpoints of Ollama,
/// LM Studio and llama.cpp live next to the OpenAI-compatible prefix.
fn server_root(base_url: &str) -> String {
    base_url
        .strip_suffix("/v1")
        .unwrap_or(base_url)
        .trim_end_matches('/')
        .to_string()
}

/// Embedding models show up in `/models` on Ollama and LM Studio, but cannot chat.
fn is_embedding_model(id: &str) -> bool {
    id.to_ascii_lowercase().contains("embed")
}

/// `Ok(None)` means the server answered with a non-success status.
async fn get_json(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<Option<JsonValue>, String> {
    let mut request = client.get(url);
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    let response = request
        .send()
        .await
        .map_err(|err| format!("could not reach {url}: {err}"))?;
    if !response.status().is_success() {
        return Ok(None);
    }
    response
        .json::<JsonValue>()
        .await
        .map(Some)
        .map_err(|err| format!("{url} did not return JSON: {err}"))
}

/// An empty body is rejected by every server that has the route (400/422, or
/// 500 on llama.cpp), while a server without it answers 404 or 405. An auth
/// failure says nothing about the route.
async fn probe_responses_route(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
) -> ResponsesApiSupport {
    let mut request = client
        .post(format!("{base_url}/responses"))
        .json(&serde_json::json!({}));
    if let Some(api_key) = api_key {
        request = request.bearer_auth(api_key);
    }
    match request.send().await {
        Ok(response) => match response.status().as_u16() {
            404 | 405 => ResponsesApiSupport::Missing,
            401 | 403 => ResponsesApiSupport::Unknown,
            _ => ResponsesApiSupport::Supported,
        },
        Err(_) => ResponsesApiSupport::Unknown,
    }
}

/// Ollama only reports the context of models that are loaded right now, and
/// that is the value it will keep using for them.
fn apply_ollama_running_contexts(models: &mut [LocalModel], running: &JsonValue) {
    let Some(running) = running.get("models").and_then(JsonValue::as_array) else {
        return;
    };
    for entry in running {
        let name = entry
            .get("name")
            .or_else(|| entry.get("model"))
            .and_then(JsonValue::as_str);
        let context = entry.get("context_length").and_then(JsonValue::as_i64);
        if let (Some(name), Some(context)) = (name, context)
            && let Some(model) = models.iter_mut().find(|model| model.id == name)
        {
            model.context_window = Some(context);
        }
    }
}

/// LM Studio reports the loaded context when a model is loaded and the
/// trained maximum otherwise; the loaded one is what requests get.
fn apply_lmstudio_contexts(models: &mut [LocalModel], native: &JsonValue) {
    let Some(entries) = native.get("data").and_then(JsonValue::as_array) else {
        return;
    };
    for entry in entries {
        let Some(id) = entry.get("id").and_then(JsonValue::as_str) else {
            continue;
        };
        let context = entry
            .get("loaded_context_length")
            .and_then(JsonValue::as_i64)
            .or_else(|| entry.get("max_context_length").and_then(JsonValue::as_i64));
        if let Some(model) = models.iter_mut().find(|model| model.id == id) {
            model.context_window = context;
        }
    }
}

#[cfg(test)]
#[path = "discovery_tests.rs"]
mod tests;
