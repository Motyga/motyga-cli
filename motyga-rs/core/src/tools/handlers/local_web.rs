//! Web tools the CLI executes itself, for providers that cannot run the hosted
//! `web_search` tool (local model servers such as Ollama or LM Studio).
//!
//! `web_search` queries DuckDuckGo by default and needs no key. Setting
//! `MOTYGA_SEARXNG_URL` (a SearXNG instance with the JSON format enabled) or
//! `BRAVE_SEARCH_API_KEY` switches the backend. `web_fetch` downloads a public
//! page and returns it as text; it refuses this machine and private networks.

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::time::Duration;

use motyga_network_proxy::NetworkProxy;
use motyga_protocol::items::TurnItem;
use motyga_protocol::items::WebSearchItem;
use motyga_protocol::models::WebSearchAction;
use motyga_tools::JsonSchema;
use motyga_tools::ResponsesApiTool;
use motyga_tools::ToolName;
use motyga_tools::ToolSpec;
use motyga_utils_output_truncation::TruncationPolicy;
use motyga_utils_output_truncation::approx_bytes_for_tokens;
use regex_lite::Captures;
use regex_lite::Regex;
use serde::Deserialize;
use url::Host;
use url::Url;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;

pub(crate) const WEB_SEARCH_TOOL_NAME: &str = "web_search";
pub(crate) const WEB_FETCH_TOOL_NAME: &str = "web_fetch";

const SEARXNG_URL_ENV: &str = "MOTYGA_SEARXNG_URL";
const BRAVE_API_KEY_ENV: &str = "BRAVE_SEARCH_API_KEY";
const DUCKDUCKGO_HTML_URL: &str = "https://html.duckduckgo.com/html/";
const BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";

const DEFAULT_MAX_RESULTS: usize = 8;
const MAX_RESULTS_LIMIT: usize = 20;
const DEFAULT_MAX_CHARS: usize = 20_000;
const MIN_MAX_CHARS: usize = 1_000;
const MAX_CHARS_LIMIT: usize = 100_000;
const MAX_BODY_BYTES: usize = 5 * 1024 * 1024;
const MAX_SEARCH_BODY_BYTES: usize = 2 * 1024 * 1024;
/// Room kept for the URL, title and continuation note around the page text,
/// so the whole result stays within the model's tool-output budget.
const RENDER_OVERHEAD_BYTES: usize = 1_024;
const MIN_PAGE_TEXT_BYTES: usize = 2_000;
const MAX_REDIRECTS: usize = 5;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const USER_AGENT: &str = concat!(
    "Mozilla/5.0 (compatible; MotygaCLI/",
    env!("CARGO_PKG_VERSION"),
    "; +https://motyga.com)"
);

pub struct LocalWebSearchHandler;

pub struct LocalWebFetchHandler;

#[derive(Debug, Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default)]
    max_chars: Option<usize>,
    #[serde(default)]
    offset: Option<usize>,
}

fn create_web_search_function_tool() -> ToolSpec {
    let properties = BTreeMap::from([
        (
            "query".to_string(),
            JsonSchema::string(Some("What to search the web for.".to_string())),
        ),
        (
            "max_results".to_string(),
            JsonSchema::integer(Some(format!(
                "How many results to return (1-{MAX_RESULTS_LIMIT}, default {DEFAULT_MAX_RESULTS})."
            ))),
        ),
    ]);
    ToolSpec::Function(ResponsesApiTool {
        name: WEB_SEARCH_TOOL_NAME.to_string(),
        description: "Search the internet. Returns titles, URLs and snippets of matching pages. \
Use it for current events, documentation, error messages and anything that may be newer than \
your training data; then call web_fetch to read a result in full."
            .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            properties,
            Some(vec!["query".to_string()]),
            Some(false.into()),
        ),
        output_schema: None,
    })
}

fn create_web_fetch_function_tool() -> ToolSpec {
    let properties = BTreeMap::from([
        (
            "url".to_string(),
            JsonSchema::string(Some(
                "Absolute http(s) URL of the page to read.".to_string(),
            )),
        ),
        (
            "max_chars".to_string(),
            JsonSchema::integer(Some(format!(
                "Maximum characters of page text to return ({MIN_MAX_CHARS}-{MAX_CHARS_LIMIT}, default {DEFAULT_MAX_CHARS}). Long pages may come back shorter; the result says which offset to continue from."
            ))),
        ),
        (
            "offset".to_string(),
            JsonSchema::integer(Some(
                "Character offset to start from, to continue reading a long page (default 0)."
                    .to_string(),
            )),
        ),
    ]);
    ToolSpec::Function(ResponsesApiTool {
        name: WEB_FETCH_TOOL_NAME.to_string(),
        description: "Download a web page and return its readable text, with links kept as \
markdown. Use it to read search results, documentation or any public URL."
            .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            properties,
            Some(vec!["url".to_string()]),
            Some(false.into()),
        ),
        output_schema: None,
    })
}

impl ToolExecutor<ToolInvocation> for LocalWebSearchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_SEARCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_web_search_function_tool()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle(&self, invocation: ToolInvocation) -> motyga_tools::ToolExecutorFuture<'_> {
        Box::pin(async move {
            let ToolInvocation {
                session,
                turn,
                call_id,
                payload,
                ..
            } = invocation;
            let ToolPayload::Function { arguments } = payload else {
                return Err(FunctionCallError::RespondToModel(format!(
                    "{WEB_SEARCH_TOOL_NAME} handler received unsupported payload"
                )));
            };
            let args: WebSearchArgs = parse_arguments(&arguments)?;
            let query = args.query.trim().to_string();
            if query.is_empty() {
                return Err(FunctionCallError::RespondToModel(
                    "query must not be empty".to_string(),
                ));
            }
            let max_results = args
                .max_results
                .unwrap_or(DEFAULT_MAX_RESULTS)
                .clamp(1, MAX_RESULTS_LIMIT);

            let item = TurnItem::WebSearch(WebSearchItem {
                id: call_id,
                query: query.clone(),
                action: WebSearchAction::Search {
                    query: Some(query.clone()),
                    queries: None,
                },
            });
            session.emit_turn_item_started(turn.as_ref(), &item).await;
            let backend = SearchBackend::from_env();
            let managed_proxy = turn.network.as_ref().map(NetworkProxy::http_addr);
            let outcome = match web_client(ClientKind::Search, managed_proxy) {
                Ok(client) => search(&client, &backend, &query, max_results).await,
                Err(err) => Err(err),
            };
            session.emit_turn_item_completed(turn.as_ref(), item).await;

            let results = outcome.map_err(FunctionCallError::RespondToModel)?;
            Ok(boxed_tool_output(FunctionToolOutput::from_text(
                format_search_results(&query, backend.label(), &results),
                Some(true),
            )))
        })
    }
}

impl CoreToolRuntime for LocalWebSearchHandler {}

impl ToolExecutor<ToolInvocation> for LocalWebFetchHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(WEB_FETCH_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_web_fetch_function_tool()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle(&self, invocation: ToolInvocation) -> motyga_tools::ToolExecutorFuture<'_> {
        Box::pin(async move {
            let ToolInvocation {
                session,
                turn,
                call_id,
                payload,
                ..
            } = invocation;
            let ToolPayload::Function { arguments } = payload else {
                return Err(FunctionCallError::RespondToModel(format!(
                    "{WEB_FETCH_TOOL_NAME} handler received unsupported payload"
                )));
            };
            let args: WebFetchArgs = parse_arguments(&arguments)?;
            let url = parse_fetch_url(&args.url).map_err(FunctionCallError::RespondToModel)?;
            let max_chars = args
                .max_chars
                .unwrap_or(DEFAULT_MAX_CHARS)
                .clamp(MIN_MAX_CHARS, MAX_CHARS_LIMIT);
            let offset = args.offset.unwrap_or(0);

            let item = TurnItem::WebSearch(WebSearchItem {
                id: call_id,
                query: url.to_string(),
                action: WebSearchAction::OpenPage {
                    url: Some(url.to_string()),
                },
            });
            session.emit_turn_item_started(turn.as_ref(), &item).await;
            let managed_proxy = turn.network.as_ref().map(NetworkProxy::http_addr);
            let outcome = match web_client(ClientKind::Fetch, managed_proxy) {
                Ok(client) => fetch_page(&client, url).await,
                Err(err) => Err(err),
            };
            session.emit_turn_item_completed(turn.as_ref(), item).await;

            let page = outcome.map_err(FunctionCallError::RespondToModel)?;
            let max_bytes = page_text_byte_budget(turn.model_info.truncation_policy.into());
            Ok(boxed_tool_output(FunctionToolOutput::from_text(
                page.render(offset, max_chars, max_bytes),
                Some(true),
            )))
        })
    }
}

impl CoreToolRuntime for LocalWebFetchHandler {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClientKind {
    /// Search backends: redirects are followed only within the same host, so
    /// an API key header never travels to another origin.
    Search,
    /// `web_fetch`: no automatic redirects (`fetch_page` checks every hop) and
    /// a resolver that never hands out local or private addresses.
    Fetch,
}

/// Returns the HTTP client for `kind`. When the session runs a managed network
/// proxy, requests go through it so its allowlist applies to these tools too.
fn web_client(
    kind: ClientKind,
    managed_proxy: Option<SocketAddr>,
) -> Result<reqwest::Client, String> {
    static SEARCH: OnceLock<reqwest::Client> = OnceLock::new();
    static FETCH: OnceLock<reqwest::Client> = OnceLock::new();
    if managed_proxy.is_some() {
        return build_web_client(kind, managed_proxy);
    }
    let cell = match kind {
        ClientKind::Search => &SEARCH,
        ClientKind::Fetch => &FETCH,
    };
    if let Some(client) = cell.get() {
        return Ok(client.clone());
    }
    let client = build_web_client(kind, /*managed_proxy*/ None)?;
    Ok(cell.get_or_init(|| client).clone())
}

fn build_web_client(
    kind: ClientKind,
    managed_proxy: Option<SocketAddr>,
) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .referer(false);
    builder = match kind {
        ClientKind::Search => builder.redirect(reqwest::redirect::Policy::custom(|attempt| {
            let same_host = attempt
                .previous()
                .first()
                .is_some_and(|first| first.host_str() == attempt.url().host_str());
            if attempt.previous().len() >= MAX_REDIRECTS {
                attempt.error("too many redirects")
            } else if same_host {
                attempt.follow()
            } else {
                attempt.stop()
            }
        })),
        ClientKind::Fetch => builder
            .redirect(reqwest::redirect::Policy::none())
            .dns_resolver(Arc::new(PublicOnlyResolver::from_env())),
    };
    if let Some(address) = managed_proxy {
        let proxy = reqwest::Proxy::all(format!("http://{address}"))
            .map_err(|err| format!("failed to use the session network proxy: {err}"))?;
        builder = builder.proxy(proxy);
    }
    motyga_client::build_reqwest_client_with_custom_ca(builder)
        .map_err(|err| format!("failed to set up the HTTP client: {err}"))
}

/// DNS resolver for `web_fetch` that drops local and private addresses, so a
/// name cannot be pointed inward between the destination check and the
/// connection. Names of configured proxies are exempt: the request has to
/// reach the proxy, which then resolves the destination itself.
struct PublicOnlyResolver {
    proxy_hosts: HashSet<String>,
}

impl PublicOnlyResolver {
    fn from_env() -> Self {
        let proxy_hosts = PROXY_ENV_VARS
            .iter()
            .filter_map(|name| env_value(name))
            .filter_map(|value| {
                Url::parse(&value)
                    .ok()
                    .or_else(|| Url::parse(&format!("http://{value}")).ok())
            })
            .filter_map(|url| url.host_str().map(normalize_host))
            .collect();
        Self { proxy_hosts }
    }
}

const PROXY_ENV_VARS: [&str; 6] = [
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
];

fn normalize_host(host: &str) -> String {
    host.trim_end_matches('.').to_ascii_lowercase()
}

impl reqwest::dns::Resolve for PublicOnlyResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let host = normalize_host(name.as_str());
        let allow_private = self.proxy_hosts.contains(&host);
        Box::pin(async move {
            let addresses = tokio::net::lookup_host((host.as_str(), 0))
                .await?
                .filter(|address| allow_private || !is_non_public_ip(address.ip()))
                .collect::<Vec<_>>();
            if addresses.is_empty() {
                return Err(format!("{host} resolves only to local or private addresses").into());
            }
            Ok(Box::new(addresses.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

/// Reads at most `limit` bytes of the body; anything beyond is dropped.
async fn read_body_capped(
    mut response: reqwest::Response,
    limit: usize,
    source: &str,
) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| format!("failed to read {source}: {err}"))?
    {
        let room = limit - body.len();
        if chunk.len() >= room {
            body.extend_from_slice(&chunk[..room]);
            break;
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SearchBackend {
    DuckDuckGo,
    SearXng { base_url: String },
    Brave { api_key: String },
}

impl SearchBackend {
    fn from_env() -> Self {
        Self::from_values(env_value(SEARXNG_URL_ENV), env_value(BRAVE_API_KEY_ENV))
    }

    fn from_values(searxng_url: Option<String>, brave_api_key: Option<String>) -> Self {
        if let Some(base_url) = searxng_url {
            return Self::SearXng {
                base_url: base_url.trim_end_matches('/').to_string(),
            };
        }
        if let Some(api_key) = brave_api_key {
            return Self::Brave { api_key };
        }
        Self::DuckDuckGo
    }

    fn label(&self) -> &'static str {
        match self {
            Self::DuckDuckGo => "DuckDuckGo",
            Self::SearXng { .. } => "SearXNG",
            Self::Brave { .. } => "Brave Search",
        }
    }
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

async fn search(
    client: &reqwest::Client,
    backend: &SearchBackend,
    query: &str,
    max_results: usize,
) -> Result<Vec<SearchResult>, String> {
    let label = backend.label();
    let count = max_results.to_string();
    let request = match backend {
        SearchBackend::DuckDuckGo => client.post(DUCKDUCKGO_HTML_URL).form(&[("q", query)]),
        SearchBackend::SearXng { base_url } => client
            .get(format!("{base_url}/search"))
            .query(&[("q", query), ("format", "json")]),
        SearchBackend::Brave { api_key } => client
            .get(BRAVE_SEARCH_URL)
            .query(&[("q", query), ("count", count.as_str())])
            .header(reqwest::header::ACCEPT, "application/json")
            .header("X-Subscription-Token", api_key),
    };
    let response = request
        .send()
        .await
        .map_err(|err| format!("{label} request failed: {err}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "{label} answered HTTP {status}. {}",
            search_backend_hint(backend)
        ));
    }
    let body = read_body_capped(response, MAX_SEARCH_BODY_BYTES, label).await?;
    let body = String::from_utf8_lossy(&body);
    let results = match backend {
        SearchBackend::DuckDuckGo => {
            let results = parse_duckduckgo_html(&body, max_results);
            if results.is_empty() && looks_like_bot_challenge(&body) {
                return Err(format!(
                    "DuckDuckGo refused the automated search. {}",
                    search_backend_hint(backend)
                ));
            }
            results
        }
        SearchBackend::SearXng { .. } => parse_searxng_json(&body, max_results)
            .map_err(|err| format!("SearXNG returned an unexpected response ({err}). Make sure the instance has the JSON format enabled."))?,
        SearchBackend::Brave { .. } => parse_brave_json(&body, max_results)
            .map_err(|err| format!("Brave Search returned an unexpected response ({err})."))?,
    };
    Ok(results)
}

fn search_backend_hint(backend: &SearchBackend) -> String {
    match backend {
        SearchBackend::DuckDuckGo => format!(
            "Ask the user to set {SEARXNG_URL_ENV} (their SearXNG instance) or {BRAVE_API_KEY_ENV} to use another search backend."
        ),
        SearchBackend::SearXng { .. } => format!("Check {SEARXNG_URL_ENV}."),
        SearchBackend::Brave { .. } => format!("Check {BRAVE_API_KEY_ENV}."),
    }
}

fn looks_like_bot_challenge(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    body.contains("anomaly") || body.contains("challenge-form") || body.contains("captcha")
}

static DDG_TITLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<a\b([^>]*\bclass="[^"]*\bresult__a\b[^"]*"[^>]*)>(.*?)</a\s*>"#)
        .unwrap_or_else(|err| panic!("invalid DuckDuckGo title regex: {err}"))
});
static DDG_SNIPPET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?s)<(?:a|div|td|span)\b[^>]*\bclass="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</(?:a|div|td|span)\s*>"#,
    )
    .unwrap_or_else(|err| panic!("invalid DuckDuckGo snippet regex: {err}"))
});
static HREF_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')"#)
        .unwrap_or_else(|err| panic!("invalid href regex: {err}"))
});

fn parse_duckduckgo_html(html: &str, max_results: usize) -> Vec<SearchResult> {
    let titles = DDG_TITLE_RE.captures_iter(html).collect::<Vec<_>>();
    let mut results = Vec::new();
    let mut previous_end = 0;
    for (index, captures) in titles.iter().enumerate() {
        let Some(whole) = captures.get(0) else {
            continue;
        };
        let block_prefix = &html[previous_end..whole.start()];
        previous_end = whole.end();
        if block_prefix.contains("result--ad") {
            continue;
        }
        let attributes = captures.get(1).map_or("", |m| m.as_str());
        let Some(url) = HREF_RE
            .captures(attributes)
            .and_then(|href| href.get(1).or_else(|| href.get(2)))
            .map(|href| decode_html_entities(href.as_str()))
            .and_then(|href| unwrap_duckduckgo_redirect(&href))
        else {
            continue;
        };
        let title = inline_html_to_text(captures.get(2).map_or("", |m| m.as_str()));
        let block_end = titles
            .get(index + 1)
            .and_then(|next| next.get(0))
            .map_or(html.len(), |next| next.start());
        let snippet = DDG_SNIPPET_RE
            .captures(&html[whole.end()..block_end])
            .and_then(|snippet| snippet.get(1))
            .map(|snippet| inline_html_to_text(snippet.as_str()))
            .unwrap_or_default();
        if results.iter().any(|result: &SearchResult| result.url == url) {
            continue;
        }
        results.push(SearchResult {
            title,
            url,
            snippet,
        });
        if results.len() >= max_results {
            break;
        }
    }
    results
}

/// DuckDuckGo sometimes links results through `//duckduckgo.com/l/?uddg=<target>`;
/// returns the real target, or `None` for ads and other non-result links.
fn unwrap_duckduckgo_redirect(href: &str) -> Option<String> {
    let absolute = if href.starts_with("//") {
        format!("https:{href}")
    } else {
        href.to_string()
    };
    let url = Url::parse(&absolute).ok()?;
    let is_duckduckgo = url
        .host_str()
        .is_some_and(|host| host == "duckduckgo.com" || host.ends_with(".duckduckgo.com"));
    if !is_duckduckgo {
        return matches!(url.scheme(), "http" | "https").then(|| url.to_string());
    }
    if url.path() != "/l/" {
        return None;
    }
    let target = url
        .query_pairs()
        .find_map(|(key, value)| (key == "uddg").then(|| value.into_owned()))?;
    let target = Url::parse(&target).ok()?;
    matches!(target.scheme(), "http" | "https").then(|| target.to_string())
}

#[derive(Debug, Deserialize)]
struct SearXngResponse {
    #[serde(default)]
    results: Vec<SearXngResult>,
}

#[derive(Debug, Deserialize)]
struct SearXngResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default)]
    content: Option<String>,
}

fn parse_searxng_json(body: &str, max_results: usize) -> Result<Vec<SearchResult>, String> {
    let response: SearXngResponse =
        serde_json::from_str(body).map_err(|err| err.to_string())?;
    Ok(response
        .results
        .into_iter()
        .take(max_results)
        .map(|result| SearchResult {
            title: inline_html_to_text(&result.title),
            url: result.url,
            snippet: inline_html_to_text(result.content.as_deref().unwrap_or_default()),
        })
        .collect())
}

#[derive(Debug, Deserialize)]
struct BraveResponse {
    #[serde(default)]
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    #[serde(default)]
    results: Vec<BraveResult>,
}

#[derive(Debug, Deserialize)]
struct BraveResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default)]
    description: Option<String>,
}

fn parse_brave_json(body: &str, max_results: usize) -> Result<Vec<SearchResult>, String> {
    let response: BraveResponse = serde_json::from_str(body).map_err(|err| err.to_string())?;
    Ok(response
        .web
        .map(|web| web.results)
        .unwrap_or_default()
        .into_iter()
        .take(max_results)
        .map(|result| SearchResult {
            title: inline_html_to_text(&result.title),
            url: result.url,
            snippet: inline_html_to_text(result.description.as_deref().unwrap_or_default()),
        })
        .collect())
}

fn format_search_results(query: &str, backend_label: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No results for \"{query}\" ({backend_label}). Try different wording.");
    }
    let mut output = format!("Search results for \"{query}\" ({backend_label}):\n");
    for (index, result) in results.iter().enumerate() {
        let title = if result.title.is_empty() {
            result.url.as_str()
        } else {
            result.title.as_str()
        };
        output.push_str(&format!("\n{}. {title}\n   {}\n", index + 1, result.url));
        if !result.snippet.is_empty() {
            output.push_str(&format!("   {}\n", result.snippet));
        }
    }
    output.push_str("\nCall web_fetch with a URL to read the page.");
    output
}

fn parse_fetch_url(raw: &str) -> Result<Url, String> {
    let url = Url::parse(raw.trim()).map_err(|err| format!("invalid URL `{raw}`: {err}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("only http and https URLs can be fetched, got `{raw}`"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("URLs with embedded credentials are not fetched".to_string());
    }
    if let Some(reason) = non_public_host_reason(&url) {
        return Err(format!("refusing to fetch `{raw}`: {reason}"));
    }
    Ok(url)
}

/// Why `url` must not be fetched: it names this machine or a private network.
fn non_public_host_reason(url: &Url) -> Option<String> {
    match url.host()? {
        Host::Ipv4(ip) => is_non_public_ip(IpAddr::V4(ip))
            .then(|| format!("{ip} is a local or private address")),
        Host::Ipv6(ip) => is_non_public_ip(IpAddr::V6(ip))
            .then(|| format!("{ip} is a local or private address")),
        Host::Domain(domain) => {
            let domain = domain.trim_end_matches('.').to_ascii_lowercase();
            let local = domain == "localhost"
                || [".localhost", ".local", ".lan", ".home.arpa", ".internal"]
                    .iter()
                    .any(|suffix| domain.ends_with(suffix));
            local.then(|| format!("{domain} is a local host name"))
        }
    }
}

fn is_non_public_ip(ip: IpAddr) -> bool {
    match ip.to_canonical() {
        IpAddr::V4(ip) => {
            let [first, second, ..] = ip.octets();
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_broadcast()
                || ip.is_multicast()
                // Carrier-grade NAT, 100.64.0.0/10.
                || (first == 100 && (64..128).contains(&second))
        }
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
        }
    }
}

/// Resolves the host and refuses addresses on this machine or a private
/// network, so a public-looking name cannot point the fetch inward. A failed
/// lookup is not an error: a configured proxy may resolve names we cannot.
async fn ensure_public_destination(url: &Url) -> Result<(), String> {
    let Some(Host::Domain(domain)) = url.host() else {
        return Ok(());
    };
    let port = url.port_or_known_default().unwrap_or(443);
    let Ok(addresses) = tokio::net::lookup_host((domain, port)).await else {
        return Ok(());
    };
    for address in addresses {
        if is_non_public_ip(address.ip()) {
            return Err(format!(
                "refusing to fetch {url}: {domain} resolves to the local or private address {}",
                address.ip()
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FetchedPage {
    url: String,
    title: Option<String>,
    text: String,
}

/// How many bytes of page text fit in one `web_fetch` result without the
/// conversation history truncating it, which would break offset paging.
fn page_text_byte_budget(policy: TruncationPolicy) -> usize {
    let limit = match policy {
        TruncationPolicy::Bytes(bytes) => bytes,
        TruncationPolicy::Tokens(tokens) => approx_bytes_for_tokens(tokens),
    };
    limit
        .saturating_sub(RENDER_OVERHEAD_BYTES)
        .max(MIN_PAGE_TEXT_BYTES)
}

impl FetchedPage {
    fn render(&self, offset: usize, max_chars: usize, max_bytes: usize) -> String {
        let total = self.text.chars().count();
        let start = offset.min(total);
        let mut shown = String::new();
        for ch in self.text.chars().skip(start).take(max_chars) {
            if shown.len() + ch.len_utf8() > max_bytes {
                break;
            }
            shown.push(ch);
        }
        let end = start + shown.chars().count();
        let mut output = format!("URL: {}\n", self.url);
        if let Some(title) = self.title.as_deref().filter(|title| !title.is_empty()) {
            output.push_str(&format!("Title: {title}\n"));
        }
        output.push('\n');
        if total == 0 {
            output.push_str("(the page has no readable text)");
        } else if start >= total {
            output.push_str(&format!(
                "(offset {offset} is past the end of the page, which has {total} characters)"
            ));
        } else {
            output.push_str(&shown);
        }
        if end < total {
            output.push_str(&format!(
                "\n\n[Showing characters {start}-{end} of {total}. Call web_fetch again with offset={end} to continue.]"
            ));
        }
        output
    }
}

async fn fetch_page(client: &reqwest::Client, url: Url) -> Result<FetchedPage, String> {
    let mut current = url;
    let mut redirects = 0;
    let response = loop {
        ensure_public_destination(&current).await?;
        let response = client
            .get(current.clone())
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            )
            .send()
            .await
            .map_err(|err| format!("failed to fetch {current}: {err}"))?;
        if !response.status().is_redirection() {
            break response;
        }
        let Some(location) = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
        else {
            break response;
        };
        redirects += 1;
        if redirects > MAX_REDIRECTS {
            return Err(format!("{current} redirected too many times"));
        }
        let next = current
            .join(location)
            .map_err(|err| format!("{current} redirected to an invalid URL: {err}"))?;
        current = parse_fetch_url(next.as_str())?;
    };
    let final_url = response.url().clone();
    let status = response.status();
    if !status.is_success() {
        return Err(format!("{final_url} answered HTTP {status}"));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if let Some(length) = response.content_length()
        && length > MAX_BODY_BYTES as u64
    {
        return Err(format!(
            "{final_url} is too large to read ({length} bytes, limit {MAX_BODY_BYTES})"
        ));
    }
    let body = read_body_capped(response, MAX_BODY_BYTES, final_url.as_str()).await?;
    page_from_body(&final_url, &content_type, &body)
}

fn page_from_body(url: &Url, content_type: &str, body: &[u8]) -> Result<FetchedPage, String> {
    let sniffed_html = content_type.is_empty()
        && body
            .iter()
            .find(|byte| !byte.is_ascii_whitespace())
            .is_some_and(|byte| *byte == b'<');
    let is_html = content_type.contains("html") || sniffed_html;
    let is_text = content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
        || content_type.contains("yaml")
        || content_type.contains("markdown");
    if !is_html && !is_text && !content_type.is_empty() {
        let kind = content_type.split(';').next().unwrap_or(content_type);
        return Err(format!(
            "{url} is `{kind}`, which web_fetch cannot read as text"
        ));
    }
    let decoded = decode_body(body, content_type, is_html);
    if is_html {
        let (title, text) = html_to_text(&decoded, Some(url));
        Ok(FetchedPage {
            url: url.to_string(),
            title,
            text,
        })
    } else {
        Ok(FetchedPage {
            url: url.to_string(),
            title: None,
            text: decoded.trim().to_string(),
        })
    }
}

static META_CHARSET_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)<meta\b[^>]*\bcharset\s*=\s*["']?([a-z0-9_\-]+)"#)
        .unwrap_or_else(|err| panic!("invalid charset regex: {err}"))
});

fn decode_body(body: &[u8], content_type: &str, is_html: bool) -> String {
    let header_charset = content_type
        .split(';')
        .filter_map(|part| part.trim().strip_prefix("charset="))
        .map(|charset| charset.trim_matches(['"', '\'']).to_string())
        .next();
    let meta_charset = || {
        if !is_html {
            return None;
        }
        let head = String::from_utf8_lossy(&body[..body.len().min(4096)]);
        META_CHARSET_RE
            .captures(&head)
            .and_then(|captures| captures.get(1))
            .map(|charset| charset.as_str().to_string())
    };
    let encoding = header_charset
        .or_else(meta_charset)
        .and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
        .unwrap_or(encoding_rs::UTF_8);
    let (text, _, _) = encoding.decode(body);
    text.into_owned()
}

struct TextRegexes {
    title: Regex,
    removed_blocks: Vec<Regex>,
    comment: Regex,
    anchor: Regex,
    heading: Regex,
    list_item: Regex,
    line_break: Regex,
    block: Regex,
    cell: Regex,
    tag: Regex,
    spaces: Regex,
    blank_lines: Regex,
}

static TEXT_REGEXES: LazyLock<TextRegexes> = LazyLock::new(|| {
    let compile = |pattern: &str| {
        Regex::new(pattern).unwrap_or_else(|err| panic!("invalid HTML regex `{pattern}`: {err}"))
    };
    TextRegexes {
        title: compile(r"(?is)<title\b[^>]*>(.*?)</title\s*>"),
        removed_blocks: [
            "head", "script", "style", "noscript", "svg", "template", "iframe", "canvas",
            "object",
        ]
        .iter()
        .map(|tag| compile(&format!(r"(?is)<{tag}\b[^>]*>.*?</{tag}\s*>")))
        .collect(),
        comment: compile(r"(?s)<!--.*?-->"),
        anchor: compile(
            r#"(?is)<a\b[^>]*?\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>(.*?)</a\s*>"#,
        ),
        heading: compile(r"(?i)<h([1-6])\b[^>]*>"),
        list_item: compile(r"(?i)<li\b[^>]*>"),
        line_break: compile(r"(?i)<(?:br|hr)\b[^>]*>"),
        block: compile(
            r"(?i)</?(?:p|div|section|article|header|footer|main|nav|aside|ul|ol|table|thead|tbody|tr|blockquote|pre|h[1-6]|dl|dt|dd|figure|figcaption|form|fieldset|details|summary)\b[^>]*>",
        ),
        cell: compile(r"(?i)<(?:td|th)\b[^>]*>"),
        tag: compile(r"(?s)<[^>]*>"),
        spaces: compile("[ \t\u{a0}]+"),
        blank_lines: compile(r"\n{3,}"),
    }
});

/// Converts an HTML document to readable text: scripts, styles and other
/// non-content blocks are dropped, block elements become line breaks and
/// links are kept as markdown. Returns the page title and the text.
fn html_to_text(html: &str, base_url: Option<&Url>) -> (Option<String>, String) {
    let regexes = &*TEXT_REGEXES;
    let title = regexes
        .title
        .captures(html)
        .and_then(|captures| captures.get(1))
        .map(|title| inline_html_to_text(title.as_str()))
        .filter(|title| !title.is_empty());

    let mut text = regexes.comment.replace_all(html, "").into_owned();
    for removed in &regexes.removed_blocks {
        text = removed.replace_all(&text, "").into_owned();
    }
    text = regexes
        .anchor
        .replace_all(&text, |captures: &Captures<'_>| {
            let href = captures
                .get(1)
                .or_else(|| captures.get(2))
                .map_or("", |href| href.as_str());
            let label = inline_html_to_text(captures.get(3).map_or("", |label| label.as_str()));
            match absolute_link(href, base_url) {
                Some(link) if !label.is_empty() && label != link => format!("[{label}]({link})"),
                Some(_) if label.is_empty() => String::new(),
                _ => label,
            }
        })
        .into_owned();
    text = regexes
        .heading
        .replace_all(&text, |captures: &Captures<'_>| {
            let level = captures
                .get(1)
                .and_then(|level| level.as_str().parse::<usize>().ok())
                .unwrap_or(1);
            format!("\n\n{} ", "#".repeat(level))
        })
        .into_owned();
    text = regexes.list_item.replace_all(&text, "\n- ").into_owned();
    text = regexes.line_break.replace_all(&text, "\n").into_owned();
    text = regexes.block.replace_all(&text, "\n").into_owned();
    text = regexes.cell.replace_all(&text, " ").into_owned();
    text = regexes.tag.replace_all(&text, "").into_owned();
    text = decode_html_entities(&text);
    text = text.replace("\r\n", "\n").replace('\r', "\n");

    let lines = text
        .lines()
        .map(|line| regexes.spaces.replace_all(line, " ").trim().to_string())
        .collect::<Vec<_>>();
    let mut cleaned = Vec::with_capacity(lines.len());
    for line in lines {
        // Drop headings and bullets left with no content.
        if (!line.is_empty() && line.chars().all(|ch| ch == '#')) || line == "-" {
            continue;
        }
        cleaned.push(line);
    }
    let text = regexes
        .blank_lines
        .replace_all(&cleaned.join("\n"), "\n\n")
        .trim()
        .to_string();
    (title, text)
}

fn absolute_link(href: &str, base_url: Option<&Url>) -> Option<String> {
    let href = decode_html_entities(href.trim());
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    let url = match base_url {
        Some(base) => base.join(&href).ok()?,
        None => Url::parse(&href).ok()?,
    };
    matches!(url.scheme(), "http" | "https").then(|| url.to_string())
}

/// Strips tags from a short HTML fragment (a title or snippet) and decodes
/// entities into a single line of text.
fn inline_html_to_text(fragment: &str) -> String {
    let regexes = &*TEXT_REGEXES;
    let without_tags = regexes.tag.replace_all(fragment, "");
    let decoded = decode_html_entities(&without_tags);
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn decode_html_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        output.push_str(&rest[..amp]);
        let candidate = &rest[amp..];
        let decoded = candidate
            .char_indices()
            .take(12)
            .find(|(_, ch)| *ch == ';')
            .and_then(|(semicolon, _)| {
                decode_entity(&candidate[1..semicolon]).map(|ch| (ch, semicolon + 1))
            });
        match decoded {
            Some((ch, consumed)) => {
                output.push(ch);
                rest = &candidate[consumed..];
            }
            None => {
                output.push('&');
                rest = &candidate[1..];
            }
        }
    }
    output.push_str(rest);
    output
}

fn decode_entity(entity: &str) -> Option<char> {
    if let Some(number) = entity.strip_prefix('#') {
        let code = match number.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => number.parse::<u32>().ok()?,
        };
        return char::from_u32(code);
    }
    let ch = match entity {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "ndash" => '\u{2013}',
        "mdash" => '\u{2014}',
        "hellip" => '\u{2026}',
        "laquo" => '\u{ab}',
        "raquo" => '\u{bb}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "bull" => '\u{2022}',
        "middot" => '\u{b7}',
        "copy" => '\u{a9}',
        "reg" => '\u{ae}',
        "trade" => '\u{2122}',
        "deg" => '\u{b0}',
        "times" => '\u{d7}',
        "euro" => '\u{20ac}',
        _ => return None,
    };
    Some(ch)
}

#[cfg(test)]
#[path = "local_web_tests.rs"]
mod tests;
