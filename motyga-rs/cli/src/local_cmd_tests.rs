use super::*;
use pretty_assertions::assert_eq;

#[test]
fn normalize_base_url_accepts_ports_host_ports_and_urls() {
    let cases = [
        ("8080", "http://localhost:8080/v1"),
        ("192.168.1.20:11434", "http://192.168.1.20:11434/v1"),
        ("localhost:1234/", "http://localhost:1234/v1"),
        ("gpu-box.lan:8000/v1", "http://gpu-box.lan:8000/v1"),
        (
            "https://llm.example.com/openai/v1/",
            "https://llm.example.com/openai/v1",
        ),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/v1"),
    ];

    let normalized: Vec<(&str, String)> = cases
        .iter()
        .map(|(input, _)| (*input, normalize_base_url(input).expect("valid address")))
        .collect();
    let expected: Vec<(&str, String)> = cases
        .iter()
        .map(|(input, url)| (*input, (*url).to_string()))
        .collect();
    assert_eq!(normalized, expected);
}

#[test]
fn normalize_base_url_rejects_other_schemes_and_empty_input() {
    for input in [
        "",
        "ftp://host:21",
        "http://",
        "http://user:secret@10.0.0.2:8000",
    ] {
        assert!(
            normalize_base_url(input).is_err(),
            "{input} should be rejected"
        );
    }
}

#[test]
fn provider_ids_cannot_shadow_cloud_providers() {
    for id in ["motyga", "openai", "amazon-bedrock", "bad id", ""] {
        assert!(validate_provider_id(id).is_err(), "{id} should be rejected");
    }
    for id in ["ollama", "lmstudio", "llamacpp", "gpu_box-2"] {
        assert!(validate_provider_id(id).is_ok(), "{id} should be accepted");
    }
}
