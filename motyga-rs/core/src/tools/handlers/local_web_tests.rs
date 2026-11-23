use super::*;
use pretty_assertions::assert_eq;

const DUCKDUCKGO_FIXTURE: &str = r#"
<div class="result results_links results_links_deep result--ad ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad_domain=ads.example">Sponsored</a>
  </h2>
  <a class="result__snippet" href="https://duckduckgo.com/y.js?ad_domain=ads.example">Buy now</a>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://docs.rs/reqwest/latest/reqwest/">reqwest - Rust &amp; Docs.rs</a>
  </h2>
  <a class="result__url" href="https://docs.rs/reqwest/latest/reqwest/">docs.rs/reqwest</a>
  <a class="result__snippet" href="https://docs.rs/reqwest/latest/reqwest/">The <b>reqwest</b> crate provides a convenient,
     higher-level HTTP Client.</a>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fseanmonstar%2Freqwest&amp;rut=abc" class="result__a">GitHub - seanmonstar/reqwest</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Fseanmonstar%2Freqwest">An easy and powerful Rust HTTP Client</a>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://docs.rs/reqwest/latest/reqwest/">Duplicate</a>
  </h2>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="https://lib.rs/crates/reqwest">Reqwest — Lib.rs</a>
  </h2>
</div>
"#;

#[test]
fn duckduckgo_results_skip_ads_unwrap_redirects_and_dedupe() {
    let results = parse_duckduckgo_html(DUCKDUCKGO_FIXTURE, /*max_results*/ 10);
    assert_eq!(
        results,
        vec![
            SearchResult {
                title: "reqwest - Rust & Docs.rs".to_string(),
                url: "https://docs.rs/reqwest/latest/reqwest/".to_string(),
                snippet: "The reqwest crate provides a convenient, higher-level HTTP Client."
                    .to_string(),
            },
            SearchResult {
                title: "GitHub - seanmonstar/reqwest".to_string(),
                url: "https://github.com/seanmonstar/reqwest".to_string(),
                snippet: "An easy and powerful Rust HTTP Client".to_string(),
            },
            SearchResult {
                title: "Reqwest — Lib.rs".to_string(),
                url: "https://lib.rs/crates/reqwest".to_string(),
                snippet: String::new(),
            },
        ]
    );
}

#[test]
fn duckduckgo_results_respect_max_results() {
    let results = parse_duckduckgo_html(DUCKDUCKGO_FIXTURE, /*max_results*/ 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].url, "https://docs.rs/reqwest/latest/reqwest/");
}

#[test]
fn duckduckgo_redirects_resolve_only_to_web_targets() {
    assert_eq!(
        unwrap_duckduckgo_redirect("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1"),
        Some("https://example.com/a?b=1".to_string())
    );
    assert_eq!(
        unwrap_duckduckgo_redirect("https://duckduckgo.com/y.js?ad_domain=x"),
        None
    );
    assert_eq!(
        unwrap_duckduckgo_redirect("//duckduckgo.com/l/?uddg=javascript%3Aalert(1)"),
        None
    );
    assert_eq!(unwrap_duckduckgo_redirect("ftp://example.com/file"), None);
}

#[test]
fn bot_challenge_pages_are_recognised() {
    assert!(looks_like_bot_challenge(
        "<div class=\"anomaly-modal__title\">Unfortunately, bots use DuckDuckGo too.</div>"
    ));
    assert!(!looks_like_bot_challenge("<html><body>No results.</body></html>"));
}

#[test]
fn searxng_and_brave_responses_are_parsed() {
    let searxng = r#"{"results":[
        {"title":"First &amp; best","url":"https://a.example/","content":"<b>alpha</b> text"},
        {"title":"Second","url":"https://b.example/"}
    ]}"#;
    assert_eq!(
        parse_searxng_json(searxng, /*max_results*/ 5).expect("searxng json"),
        vec![
            SearchResult {
                title: "First & best".to_string(),
                url: "https://a.example/".to_string(),
                snippet: "alpha text".to_string(),
            },
            SearchResult {
                title: "Second".to_string(),
                url: "https://b.example/".to_string(),
                snippet: String::new(),
            },
        ]
    );

    let brave = r#"{"web":{"results":[
        {"title":"Brave","url":"https://c.example/","description":"<strong>found</strong> it"}
    ]}}"#;
    assert_eq!(
        parse_brave_json(brave, /*max_results*/ 5).expect("brave json"),
        vec![SearchResult {
            title: "Brave".to_string(),
            url: "https://c.example/".to_string(),
            snippet: "found it".to_string(),
        }]
    );
    assert_eq!(
        parse_brave_json("{}", /*max_results*/ 5).expect("empty brave json"),
        Vec::new()
    );
    assert!(parse_searxng_json("<html>", /*max_results*/ 5).is_err());
}

#[test]
fn search_backend_prefers_searxng_then_brave_then_duckduckgo() {
    assert_eq!(
        SearchBackend::from_values(
            Some("https://search.example/".to_string()),
            Some("key".to_string())
        ),
        SearchBackend::SearXng {
            base_url: "https://search.example".to_string()
        }
    );
    assert_eq!(
        SearchBackend::from_values(None, Some("key".to_string())),
        SearchBackend::Brave {
            api_key: "key".to_string()
        }
    );
    assert_eq!(
        SearchBackend::from_values(None, None),
        SearchBackend::DuckDuckGo
    );
}

#[test]
fn search_results_are_numbered_with_urls_and_snippets() {
    let output = format_search_results(
        "rust http",
        "DuckDuckGo",
        &[
            SearchResult {
                title: "reqwest".to_string(),
                url: "https://docs.rs/reqwest".to_string(),
                snippet: "HTTP client".to_string(),
            },
            SearchResult {
                title: String::new(),
                url: "https://lib.rs/".to_string(),
                snippet: String::new(),
            },
        ],
    );
    assert_eq!(
        output,
        "Search results for \"rust http\" (DuckDuckGo):\n\
         \n1. reqwest\n   https://docs.rs/reqwest\n   HTTP client\n\
         \n2. https://lib.rs/\n   https://lib.rs/\n\
         \nCall web_fetch with a URL to read the page."
    );
    assert_eq!(
        format_search_results("nothing", "SearXNG", &[]),
        "No results for \"nothing\" (SearXNG). Try different wording."
    );
}

#[test]
fn fetch_urls_must_be_public_http() {
    assert!(parse_fetch_url("https://example.com/docs").is_ok());
    assert!(parse_fetch_url("  http://example.com  ").is_ok());
    for rejected in [
        "file:///etc/passwd",
        "ftp://example.com/",
        "not a url",
        "http://localhost:8080/",
        "http://printer.local/",
        "http://router.lan/admin",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fd00::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://2130706433/",
        "https://user:secret@example.com/",
    ] {
        assert!(
            parse_fetch_url(rejected).is_err(),
            "expected `{rejected}` to be rejected"
        );
    }
}

#[test]
fn public_and_private_ips_are_told_apart() {
    for public in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
        assert!(!is_non_public_ip(public.parse().expect("ip")), "{public}");
    }
    for private in [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.1",
        "169.254.1.1",
        "100.127.255.255",
        "224.0.0.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:192.168.0.1",
    ] {
        assert!(is_non_public_ip(private.parse().expect("ip")), "{private}");
    }
}

#[test]
fn html_is_converted_to_readable_text_with_links() {
    let html = r##"<!doctype html>
<html><head><title>Guide &amp; Notes</title><style>body{color:red}</style>
<script>var secret = "hidden";</script></head>
<body>
<nav><a href="/">Home</a></nav>
<!-- a comment -->
<h1>Getting   started</h1>
<p>Install with <code>cargo add reqwest</code>.<br>Then build.</p>
<ul><li>First</li><li>Second <a href="https://example.org/more">more info</a></li></ul>
<p>Read the <a href="docs/intro.html">intro</a> or <a href="#top">jump</a>.&nbsp;&#169; 2026 &#x2014; done</p>
<noscript>Enable JavaScript</noscript>
</body></html>"##;
    let base = Url::parse("https://site.example/guide/index.html").expect("base url");
    let (title, text) = html_to_text(html, Some(&base));
    assert_eq!(title.as_deref(), Some("Guide & Notes"));
    assert_eq!(
        text,
        "[Home](https://site.example/)\n\
         \n# Getting started\n\
         \nInstall with cargo add reqwest.\nThen build.\n\
         \n- First\n- Second [more info](https://example.org/more)\n\
         \nRead the [intro](https://site.example/guide/docs/intro.html) or jump. © 2026 — done"
    );
    assert!(!text.contains("secret"));
    assert!(!text.contains("color:red"));
    assert!(!text.contains("JavaScript"));
}

#[test]
fn entities_decode_and_unknown_ones_survive() {
    assert_eq!(
        decode_html_entities("a &lt;b&gt; &amp;amp; &#39;c&#39; &#x41; &unknown; & d"),
        "a <b> &amp; 'c' A &unknown; & d"
    );
}

#[test]
fn pages_decode_their_declared_charset() {
    let url = Url::parse("https://ru.example/").expect("url");
    let (body, _, _) = encoding_rs::WINDOWS_1251.encode(
        "<html><head><meta charset=\"windows-1251\"><title>Привет</title></head><body><p>Мир</p></body></html>",
    );
    let page = page_from_body(&url, "text/html", &body).expect("html page");
    assert_eq!(page.title.as_deref(), Some("Привет"));
    assert_eq!(page.text, "Мир");

    let (body, _, _) = encoding_rs::KOI8_R.encode("Текст");
    let page = page_from_body(&url, "text/plain; charset=koi8-r", &body).expect("text page");
    assert_eq!(page.text, "Текст");
}

#[test]
fn non_text_content_is_refused_and_untyped_html_is_sniffed() {
    let url = Url::parse("https://files.example/image").expect("url");
    let error = page_from_body(&url, "image/png", b"\x89PNG").expect_err("binary content");
    assert!(error.contains("image/png"), "{error}");

    let page = page_from_body(&url, "", b"  <html><body><p>Hi</p></body></html>")
        .expect("sniffed html");
    assert_eq!(page.text, "Hi");

    let page = page_from_body(&url, "application/json", br#"{"ok": true}"#).expect("json");
    assert_eq!(page.text, r#"{"ok": true}"#);
}

#[test]
fn long_pages_are_paged_by_offset() {
    let page = FetchedPage {
        url: "https://long.example/".to_string(),
        title: Some("Long".to_string()),
        text: "abcdefghij".to_string(),
    };
    assert_eq!(
        page.render(/*offset*/ 0, /*max_chars*/ 4, /*max_bytes*/ 10_000),
        "URL: https://long.example/\nTitle: Long\n\nabcd\n\n[Showing characters 0-4 of 10. Call web_fetch again with offset=4 to continue.]"
    );
    assert_eq!(
        page.render(/*offset*/ 8, /*max_chars*/ 4, /*max_bytes*/ 10_000),
        "URL: https://long.example/\nTitle: Long\n\nij"
    );
    assert_eq!(
        page.render(/*offset*/ 20, /*max_chars*/ 4, /*max_bytes*/ 10_000),
        "URL: https://long.example/\nTitle: Long\n\n(offset 20 is past the end of the page, which has 10 characters)"
    );
}

#[tokio::test]
async fn names_resolving_to_private_addresses_are_refused() {
    let url = Url::parse("http://localhost.:8080/").expect("url");
    assert!(non_public_host_reason(&url).is_some());

    // The resolution check runs on its own: `localhost` resolves to loopback.
    let url = Url::parse("http://localhost/").expect("url");
    let error = ensure_public_destination(&url)
        .await
        .expect_err("loopback resolution must be refused");
    assert!(error.contains("resolves to the local or private address"), "{error}");

    let literal = Url::parse("https://93.184.215.14/").expect("url");
    assert_eq!(ensure_public_destination(&literal).await, Ok(()));
}

#[test]
fn page_text_fits_the_byte_budget_and_resumes_where_it_stopped() {
    let page = FetchedPage {
        url: "https://ru.example/".to_string(),
        title: None,
        text: "абвгдеёжзи".to_string(),
    };
    // Cyrillic letters take two bytes each: a 7-byte budget fits three of them.
    assert_eq!(
        page.render(/*offset*/ 0, /*max_chars*/ 100, /*max_bytes*/ 7),
        "URL: https://ru.example/\n\nабв\n\n[Showing characters 0-3 of 10. Call web_fetch again with offset=3 to continue.]"
    );
    assert_eq!(
        page.render(/*offset*/ 3, /*max_chars*/ 100, /*max_bytes*/ 7),
        "URL: https://ru.example/\n\nгде\n\n[Showing characters 3-6 of 10. Call web_fetch again with offset=6 to continue.]"
    );
}

#[test]
fn page_budget_follows_the_model_truncation_policy() {
    assert_eq!(page_text_byte_budget(TruncationPolicy::Bytes(10_000)), 8_976);
    assert_eq!(
        page_text_byte_budget(TruncationPolicy::Tokens(10_000)),
        approx_bytes_for_tokens(10_000) - RENDER_OVERHEAD_BYTES
    );
    assert_eq!(page_text_byte_budget(TruncationPolicy::Bytes(500)), MIN_PAGE_TEXT_BYTES);
}

#[test]
fn proxy_hosts_are_exempt_from_the_private_address_filter() {
    let resolver = PublicOnlyResolver {
        proxy_hosts: HashSet::from(["proxy.corp.local".to_string()]),
    };
    assert!(resolver.proxy_hosts.contains(&normalize_host("Proxy.Corp.Local.")));
}

#[tokio::test]
async fn fetch_resolver_refuses_names_that_only_resolve_privately() {
    use reqwest::dns::Resolve as _;

    let resolver = PublicOnlyResolver {
        proxy_hosts: HashSet::new(),
    };
    let name: reqwest::dns::Name = "localhost".parse().expect("dns name");
    let Err(error) = resolver.resolve(name).await else {
        panic!("localhost must not resolve for web_fetch");
    };
    assert!(error.to_string().contains("local or private"), "{error}");

    let resolver = PublicOnlyResolver {
        proxy_hosts: HashSet::from(["localhost".to_string()]),
    };
    let name: reqwest::dns::Name = "localhost".parse().expect("dns name");
    assert!(resolver.resolve(name).await.is_ok());
}
