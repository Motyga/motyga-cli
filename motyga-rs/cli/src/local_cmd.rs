//! `motyga local`: find local model servers and register them as providers.

use std::collections::BTreeSet;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use motyga_core::config::Config;
use motyga_core::config::edit::ConfigEdit;
use motyga_core::config::edit::ConfigEditsBuilder;
use motyga_core::config::find_motyga_home;
use motyga_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use motyga_model_provider_info::LMSTUDIO_OSS_PROVIDER_ID;
use motyga_model_provider_info::MOTYGA_PROVIDER_ID;
use motyga_model_provider_info::ModelProviderInfo;
use motyga_model_provider_info::OLLAMA_OSS_PROVIDER_ID;
use motyga_model_provider_info::OPENAI_PROVIDER_ID;
use motyga_utils_cli::CliConfigOverrides;
use motyga_utils_oss::discovery::LocalServerKind;
use motyga_utils_oss::discovery::LocalServerReport;
use motyga_utils_oss::discovery::ProbeError;
use motyga_utils_oss::discovery::ResponsesApiSupport;
use motyga_utils_oss::discovery::WELL_KNOWN_SERVERS;
use motyga_utils_oss::discovery::probe_local_server;
use toml_edit::value;

/// Use models running on this machine or your LAN (Ollama, LM Studio, llama.cpp, vLLM, ...).
#[derive(Debug, clap::Parser)]
pub struct LocalCli {
    #[clap(flatten)]
    pub config_overrides: CliConfigOverrides,

    #[command(subcommand)]
    pub subcommand: Option<LocalSubcommand>,
}

#[derive(Debug, clap::Subcommand)]
pub enum LocalSubcommand {
    /// Look for local model servers and show which models Motyga can use (default).
    Status,

    /// Register a server's OpenAI-compatible API and make it the default for `motyga --oss`.
    Add(AddArgs),

    /// Remove a server registered with `motyga local add`.
    Remove(RemoveArgs),
}

#[derive(Debug, clap::Parser)]
pub struct AddArgs {
    /// Server address: a base URL (`http://192.168.1.20:8080/v1`), `host:port`, or just a port.
    pub url: String,

    /// Provider id to register; defaults to the detected server type (`llamacpp`, `vllm`, ...).
    #[arg(long)]
    pub name: Option<String>,

    /// Model to start `motyga --oss` with; defaults to the first model the server lists.
    #[arg(long, short = 'm')]
    pub model: Option<String>,

    /// Environment variable that holds the server's API key, if it requires one.
    #[arg(long = "api-key-env", value_name = "ENV_VAR")]
    pub api_key_env: Option<String>,
}

#[derive(Debug, clap::Parser)]
pub struct RemoveArgs {
    /// Provider id given to `motyga local add`.
    pub name: String,
}

pub async fn run(cli: LocalCli) -> Result<()> {
    let LocalCli {
        config_overrides,
        subcommand,
    } = cli;
    match subcommand.unwrap_or(LocalSubcommand::Status) {
        LocalSubcommand::Status => run_status(&config_overrides).await,
        LocalSubcommand::Add(args) => run_add(&config_overrides, args).await,
        LocalSubcommand::Remove(args) => run_remove(&config_overrides, args).await,
    }
}

async fn load_config(config_overrides: &CliConfigOverrides) -> Result<Config> {
    let overrides = config_overrides
        .parse_overrides()
        .map_err(anyhow::Error::msg)?;
    Config::load_with_cli_overrides(overrides)
        .await
        .context("failed to load configuration")
}

/// `oss_provider` / `oss_model` as configured (they live outside `Config`).
fn configured_oss_default(config: &Config) -> (Option<String>, Option<String>) {
    let effective = config.config_layer_stack.effective_config();
    let read = |key: &str| {
        effective
            .get(key)
            .and_then(toml::Value::as_str)
            .map(str::to_string)
    };
    (read("oss_provider"), read("oss_model"))
}

struct Candidate {
    provider_id: String,
    base_url: String,
    api_key: Option<String>,
    registered: bool,
}

#[allow(clippy::print_stdout)]
async fn run_status(config_overrides: &CliConfigOverrides) -> Result<()> {
    let config = load_config(config_overrides).await?;
    let mut candidates: Vec<Candidate> = config
        .model_providers
        .iter()
        .filter(|(_, provider)| provider.is_local_server())
        .filter_map(|(id, provider)| {
            Some(Candidate {
                provider_id: id.clone(),
                base_url: provider.base_url.clone()?,
                api_key: provider_api_key(provider),
                registered: true,
            })
        })
        .collect();
    candidates.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));
    let known_urls: BTreeSet<String> = candidates
        .iter()
        .map(|candidate| normalized(&candidate.base_url))
        .collect();
    candidates.extend(
        WELL_KNOWN_SERVERS
            .iter()
            .filter(|server| !known_urls.contains(&normalized(server.base_url)))
            .map(|server| Candidate {
                provider_id: server.provider_id.to_string(),
                base_url: server.base_url.to_string(),
                api_key: None,
                registered: false,
            }),
    );

    let probes =
        futures::future::join_all(candidates.iter().map(|candidate| {
            probe_local_server(&candidate.base_url, candidate.api_key.as_deref())
        }))
        .await;
    let (oss_provider, oss_model) = configured_oss_default(&config);

    println!("Local model servers:");
    let mut any_running = false;
    for (candidate, probe) in candidates.iter().zip(probes) {
        match probe {
            Ok(report) => {
                any_running = true;
                print_running(candidate, &report, oss_model.as_deref());
            }
            Err(ProbeError::Unreachable(_)) if candidate.registered => {
                println!(
                    "  ○ {}  {}  not running",
                    candidate.provider_id, candidate.base_url
                );
            }
            Err(ProbeError::NotOpenAiCompatible(reason)) if candidate.registered => {
                println!(
                    "  ○ {}  {}  {}",
                    candidate.provider_id,
                    candidate.base_url,
                    terminal_safe(&reason)
                );
            }
            Err(_) => {}
        }
    }
    if !any_running {
        println!("  No running server found.");
        println!(
            "  Start one (for example `ollama serve` or LM Studio's server) and run `motyga local` again,"
        );
        println!("  or register a server elsewhere with `motyga local add <host:port>`.");
    }
    println!();
    match oss_provider {
        Some(provider) => {
            let model = oss_model.unwrap_or_else(|| "first listed model".to_string());
            println!("`motyga --oss` uses: {provider} ({model})");
        }
        None => println!(
            "`motyga --oss` picks a running Ollama or LM Studio server; choose another with `--local-provider <id>`."
        ),
    }
    println!("Switch models in a session with /model, or start with `motyga --oss -m <model>`.");
    Ok(())
}

#[allow(clippy::print_stdout)]
fn print_running(candidate: &Candidate, report: &LocalServerReport, oss_model: Option<&str>) {
    let version = report
        .version
        .as_deref()
        .map(|version| format!(" {}", terminal_safe(version)))
        .unwrap_or_default();
    let responses = match report.responses_api {
        ResponsesApiSupport::Supported => "Responses API ok",
        ResponsesApiSupport::Missing => "no Responses API",
        ResponsesApiSupport::Unknown => "Responses API unknown",
    };
    println!(
        "  ● {}  {}  {}{version}, {responses}",
        candidate.provider_id,
        report.base_url,
        report.kind.display_name()
    );
    if !candidate.registered {
        println!(
            "      not registered yet: motyga local add {} --name {}",
            report.base_url, candidate.provider_id
        );
    }
    if report.models.is_empty() {
        println!("      no models; {}", download_hint(report.kind));
    }
    for model in &report.models {
        let context = model
            .context_window
            .map(|window| format!("  (context {window})"))
            .unwrap_or_default();
        let marker = if oss_model == Some(model.id.as_str()) {
            "  [default]"
        } else {
            ""
        };
        println!("      {}{context}{marker}", terminal_safe(&model.id));
    }
    for warning in report.warnings_for(/*model*/ None) {
        println!("      ! {}", terminal_safe(&warning));
    }
}

fn download_hint(kind: LocalServerKind) -> &'static str {
    match kind {
        LocalServerKind::Ollama => "download one with `ollama pull qwen3-coder:30b` (or any other)",
        LocalServerKind::LmStudio => "download one in LM Studio or with `lms get <model>`",
        LocalServerKind::LlamaCpp | LocalServerKind::Vllm | LocalServerKind::OpenAiCompatible => {
            "start the server with a model loaded"
        }
    }
}

#[allow(clippy::print_stdout)]
async fn run_add(config_overrides: &CliConfigOverrides, args: AddArgs) -> Result<()> {
    let config = load_config(config_overrides).await?;
    let AddArgs {
        url,
        name,
        model,
        api_key_env,
    } = args;
    let base_url = normalize_base_url(&url)?;
    let api_key = match api_key_env.as_deref() {
        Some(var) => Some(std::env::var(var).with_context(|| {
            format!("environment variable {var} (from --api-key-env) is not set")
        })?),
        None => None,
    };

    let report = probe_local_server(&base_url, api_key.as_deref())
        .await
        .map_err(|err| match err {
            ProbeError::Unreachable(reason) | ProbeError::NotOpenAiCompatible(reason) => {
                anyhow::anyhow!("{reason}. Is the server running and is the address right?")
            }
        })?;
    if report.responses_api == ResponsesApiSupport::Missing {
        bail!("{}", report.warnings_for(/*model*/ None).join("\n"));
    }
    if let Some(model) = model.as_deref()
        && !report.models.is_empty()
        && report.model(model).is_none()
    {
        let available: Vec<&str> = report
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect();
        bail!(
            "{} does not serve `{model}`. Available: {}",
            report.base_url,
            available.join(", ")
        );
    }

    let explicit_name = name.is_some();
    let provider_id = name.unwrap_or_else(|| suggested_provider_id(report.kind).to_string());
    validate_provider_id(&provider_id)?;
    let is_builtin_local =
        provider_id == OLLAMA_OSS_PROVIDER_ID || provider_id == LMSTUDIO_OSS_PROVIDER_ID;
    if !explicit_name
        && !is_builtin_local
        && let Some(existing) = config.model_providers.get(&provider_id)
        && existing.base_url.as_deref().map(normalized) != Some(normalized(&report.base_url))
    {
        bail!(
            "`{provider_id}` is already registered for {}. Pick another id with --name.",
            existing.base_url.as_deref().unwrap_or("another server")
        );
    }

    let motyga_home = find_motyga_home().context("failed to resolve MOTYGA_HOME")?;
    let set_provider_field = |field: &str, field_value: &str| ConfigEdit::SetPath {
        segments: vec![
            "model_providers".to_string(),
            provider_id.clone(),
            field.to_string(),
        ],
        value: value(field_value),
    };
    let mut edits = vec![
        ConfigEdit::ClearPath {
            segments: vec!["model_providers".to_string(), provider_id.clone()],
        },
        set_provider_field("name", report.kind.display_name()),
        set_provider_field("base_url", &report.base_url),
    ];
    if let Some(var) = api_key_env.as_deref() {
        edits.push(set_provider_field("env_key", var));
    }
    edits.push(ConfigEdit::SetPath {
        segments: vec!["oss_provider".to_string()],
        value: value(provider_id.as_str()),
    });
    edits.push(match model.as_deref() {
        Some(model) => ConfigEdit::SetPath {
            segments: vec!["oss_model".to_string()],
            value: value(model),
        },
        None => ConfigEdit::ClearPath {
            segments: vec!["oss_model".to_string()],
        },
    });
    ConfigEditsBuilder::new(&motyga_home)
        .with_edits(edits)
        .apply()
        .await
        .with_context(|| format!("failed to update {}", motyga_home.display()))?;

    let replaced_builtin = config.model_providers.contains_key(&provider_id);
    println!(
        "{} {provider_id} -> {} ({}).",
        if replaced_builtin { "Updated" } else { "Added" },
        report.base_url,
        report.kind.display_name()
    );
    if !motyga_model_provider_info::is_local_base_url(&report.base_url) {
        println!(
            "! {} is not on this machine or a private network, so Motyga treats it like a cloud provider (hosted tools stay enabled).",
            report.base_url
        );
    }
    for warning in report.warnings_for(model.as_deref()) {
        println!("! {}", terminal_safe(&warning));
    }
    let model_label = model
        .or_else(|| report.models.first().map(|model| model.id.clone()))
        .unwrap_or_else(|| "the first model it lists".to_string());
    println!(
        "`motyga --oss` now starts with {provider_id} ({}).",
        terminal_safe(&model_label)
    );
    Ok(())
}

#[allow(clippy::print_stdout)]
async fn run_remove(config_overrides: &CliConfigOverrides, args: RemoveArgs) -> Result<()> {
    let config = load_config(config_overrides).await?;
    let RemoveArgs { name } = args;
    validate_provider_id(&name)?;
    let motyga_home = find_motyga_home().context("failed to resolve MOTYGA_HOME")?;

    if config.model_provider_id == name {
        bail!(
            "`{name}` is the default `model_provider` in config.toml; change that first, then remove it."
        );
    }
    let mut edits = vec![ConfigEdit::ClearPath {
        segments: vec!["model_providers".to_string(), name.clone()],
    }];
    let (oss_provider, _) = configured_oss_default(&config);
    if oss_provider.as_deref() == Some(name.as_str()) {
        edits.push(ConfigEdit::ClearPath {
            segments: vec!["oss_provider".to_string()],
        });
        edits.push(ConfigEdit::ClearPath {
            segments: vec!["oss_model".to_string()],
        });
    }
    ConfigEditsBuilder::new(&motyga_home)
        .with_edits(edits)
        .apply()
        .await
        .with_context(|| format!("failed to update {}", motyga_home.display()))?;
    println!("Removed {name}.");
    Ok(())
}

fn provider_api_key(provider: &ModelProviderInfo) -> Option<String> {
    provider
        .env_key
        .as_deref()
        .and_then(|var| std::env::var(var).ok())
        .filter(|key| !key.trim().is_empty())
        .or_else(|| provider.experimental_bearer_token.clone())
}

fn suggested_provider_id(kind: LocalServerKind) -> &'static str {
    match kind {
        LocalServerKind::Ollama => OLLAMA_OSS_PROVIDER_ID,
        LocalServerKind::LmStudio => LMSTUDIO_OSS_PROVIDER_ID,
        LocalServerKind::LlamaCpp => "llamacpp",
        LocalServerKind::Vllm => "vllm",
        LocalServerKind::OpenAiCompatible => "local",
    }
}

fn validate_provider_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        bail!("provider id `{id}` must use only letters, digits, `-` and `_`");
    }
    if [
        MOTYGA_PROVIDER_ID,
        OPENAI_PROVIDER_ID,
        AMAZON_BEDROCK_PROVIDER_ID,
    ]
    .contains(&id)
    {
        bail!("`{id}` is a built-in cloud provider; pick another name with --name");
    }
    Ok(())
}

/// Accept `8080`, `host:8080`, `host:8080/v1` or a full URL; default to `http`
/// and the conventional `/v1` prefix when the address has no path.
fn normalize_base_url(input: &str) -> Result<String> {
    let input = input.trim();
    if input.is_empty() {
        bail!("server address is empty");
    }
    let (scheme, rest) = input.split_once("://").unwrap_or(("http", input));
    if scheme != "http" && scheme != "https" {
        bail!("server address must use http or https, got `{scheme}`");
    }
    let rest = rest.trim_end_matches('/');
    let rest = if !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()) {
        format!("localhost:{rest}")
    } else {
        rest.to_string()
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        bail!("invalid server address `{input}`");
    }
    if authority.contains('@') {
        bail!(
            "put credentials in an environment variable and pass --api-key-env, not in the address"
        );
    }
    Ok(if rest.contains('/') {
        format!("{scheme}://{rest}")
    } else {
        format!("{scheme}://{rest}/v1")
    })
}

/// Server-supplied text (model ids, versions, errors) must not drive the terminal.
fn terminal_safe(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_control()).collect()
}

fn normalized(base_url: &str) -> String {
    base_url
        .trim_end_matches('/')
        .replace("://127.0.0.1", "://localhost")
        .to_ascii_lowercase()
}

#[cfg(test)]
#[path = "local_cmd_tests.rs"]
mod tests;
