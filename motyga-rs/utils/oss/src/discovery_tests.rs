use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

fn network_disabled() -> bool {
    std::env::var(motyga_core::spawn::MOTYGA_SANDBOX_NETWORK_DISABLED_ENV_VAR).is_ok()
}

async fn mount_json(server: &MockServer, http_method: &str, route: &str, status: u16, body: JsonValue) {
    Mock::given(method(http_method))
        .and(path(route))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(server)
        .await;
}

#[tokio::test]
async fn probe_detects_ollama_and_its_loaded_context() {
    if network_disabled() {
        return;
    }
    let server = MockServer::start().await;
    mount_json(
        &server,
        "GET",
        "/v1/models",
        200,
        json!({"data": [{"id": "qwen3:8b"}, {"id": "nomic-embed-text:latest"}, {"id": "llama3.2:3b"}]}),
    )
    .await;
    mount_json(&server, "GET", "/api/version", 200, json!({"version": "0.14.2"})).await;
    mount_json(
        &server,
        "GET",
        "/api/ps",
        200,
        json!({"models": [{"name": "qwen3:8b", "context_length": 4096}]}),
    )
    .await;
    mount_json(&server, "POST", "/v1/responses", 400, json!({"error": "model is required"})).await;

    let base_url = format!("{}/v1", server.uri());
    let report = probe_local_server(&base_url, /*api_key*/ None)
        .await
        .expect("probe should succeed");

    assert_eq!(
        report,
        LocalServerReport {
            base_url: base_url.clone(),
            kind: LocalServerKind::Ollama,
            version: Some("0.14.2".to_string()),
            responses_api: ResponsesApiSupport::Supported,
            models: vec![
                LocalModel {
                    id: "qwen3:8b".to_string(),
                    context_window: Some(4096),
                },
                LocalModel {
                    id: "llama3.2:3b".to_string(),
                    context_window: None,
                },
            ],
        }
    );
    assert_eq!(
        report.warnings_for(Some("qwen3:8b")),
        vec![format!(
            "qwen3:8b runs with a 4096-token context, too small for an agent (needs {MIN_AGENT_CONTEXT_WINDOW}+): {}.",
            LocalServerKind::Ollama.context_window_hint()
        )]
    );
    assert_eq!(report.warnings_for(Some("llama3.2:3b")), Vec::<String>::new());
}

#[tokio::test]
async fn probe_detects_lm_studio_and_prefers_the_loaded_context() {
    if network_disabled() {
        return;
    }
    let server = MockServer::start().await;
    mount_json(
        &server,
        "GET",
        "/v1/models",
        200,
        json!({"data": [{"id": "qwen2.5-coder-7b-instruct"}, {"id": "openai/gpt-oss-20b"}]}),
    )
    .await;
    mount_json(
        &server,
        "GET",
        "/api/v0/models",
        200,
        json!({"data": [
            {"id": "qwen2.5-coder-7b-instruct", "max_context_length": 131072, "loaded_context_length": 32768},
            {"id": "openai/gpt-oss-20b", "max_context_length": 131072}
        ]}),
    )
    .await;
    mount_json(&server, "POST", "/v1/responses", 400, json!({"error": "bad request"})).await;

    let base_url = format!("{}/v1", server.uri());
    let report = probe_local_server(&base_url, /*api_key*/ None)
        .await
        .expect("probe should succeed");

    assert_eq!(report.kind, LocalServerKind::LmStudio);
    assert_eq!(
        report.models,
        vec![
            LocalModel {
                id: "qwen2.5-coder-7b-instruct".to_string(),
                context_window: Some(32768),
            },
            LocalModel {
                id: "openai/gpt-oss-20b".to_string(),
                context_window: Some(131072),
            },
        ]
    );
    assert_eq!(report.warnings_for(/*model*/ None), Vec::<String>::new());
}

#[tokio::test]
async fn probe_reads_llama_cpp_slot_context() {
    if network_disabled() {
        return;
    }
    let server = MockServer::start().await;
    mount_json(
        &server,
        "GET",
        "/v1/models",
        200,
        json!({"data": [{"id": "qwen3-coder-30b-a3b", "owned_by": "llamacpp"}]}),
    )
    .await;
    mount_json(
        &server,
        "GET",
        "/props",
        200,
        json!({"default_generation_settings": {"n_ctx": 65536}}),
    )
    .await;
    mount_json(&server, "POST", "/v1/responses", 500, json!({"error": "invalid"})).await;

    let report = probe_local_server(&format!("{}/v1", server.uri()), /*api_key*/ None)
        .await
        .expect("probe should succeed");

    assert_eq!(report.kind, LocalServerKind::LlamaCpp);
    assert_eq!(
        report.models,
        vec![LocalModel {
            id: "qwen3-coder-30b-a3b".to_string(),
            context_window: Some(65536),
        }]
    );
    assert_eq!(report.responses_api, ResponsesApiSupport::Supported);
}

#[tokio::test]
async fn probe_flags_servers_without_the_responses_api() {
    if network_disabled() {
        return;
    }
    let server = MockServer::start().await;
    mount_json(
        &server,
        "GET",
        "/v1/models",
        200,
        json!({"data": [{"id": "mistral-7b", "owned_by": "vllm", "max_model_len": 8192}]}),
    )
    .await;

    let base_url = format!("{}/v1", server.uri());
    let report = probe_local_server(&base_url, /*api_key*/ None)
        .await
        .expect("probe should succeed");

    assert_eq!(report.kind, LocalServerKind::Vllm);
    assert_eq!(report.responses_api, ResponsesApiSupport::Missing);
    assert_eq!(
        report.warnings_for(Some("mistral-7b")),
        vec![
            format!(
                "vLLM at {base_url} does not serve POST /responses (the OpenAI Responses API). Update the server: Ollama 0.13.4+, LM Studio 0.3.29+, a recent llama.cpp or vLLM."
            ),
            format!(
                "mistral-7b runs with a 8192-token context, too small for an agent (needs {MIN_AGENT_CONTEXT_WINDOW}+): {}.",
                LocalServerKind::Vllm.context_window_hint()
            ),
        ]
    );
}

#[tokio::test]
async fn probe_reports_unreachable_when_models_listing_fails() {
    if network_disabled() {
        return;
    }
    let server = MockServer::start().await;

    let err = probe_local_server(&format!("{}/v1", server.uri()), /*api_key*/ None)
        .await
        .expect_err("probe should fail");

    assert_eq!(
        err,
        ProbeError::Unreachable(format!("{}/v1/models did not answer", server.uri()))
    );
}
