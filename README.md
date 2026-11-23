<h1 align="center">Motyga CLI</h1>

<p align="center"><strong><code>motyga</code></strong> is a local coding agent that runs on your machine
and talks to <a href="https://motyga.com">Motyga</a> by default. It speaks the <b>Responses API</b>, so it
runs against Motyga with your <code>nb-…</code> key (or any provider you configure) — no third-party account
or subscription needed.</p>

## Quickstart

```shell
npm install -g @motyga/cli
```

> **Supported platforms.** Prebuilt binaries are published for **Windows** (`x64` and `arm64`).
> macOS and Linux builds are produced on request — open an issue and we'll publish the package for your
> OS/arch. Their legs are commented out in the release workflow rather than missing: Linux (x64/arm64) and
> Apple Silicon all built green in 0.1.9, so re-enabling one is a one-line change. Apple Intel is the
> exception — it needs a `macos-13` runner, and that pool no longer picks jobs up. An unpublished platform
> is a clean skip (the payloads are optional dependencies), not a broken install.

Motyga is the **built-in default provider**, so all you need is your API key:

```shell
# macOS / Linux
export MOTYGA_API_KEY=nb-YOUR_KEY

# Windows PowerShell
$env:MOTYGA_API_KEY = "nb-YOUR_KEY"
```

```shell
motyga              # interactive
motyga exec "..."   # headless
```

Get an `nb-…` key at [motyga.com](https://motyga.com).

### Configuration (optional)

`motyga` keeps its config and credentials in `~/.motyga/` — override with the `MOTYGA_HOME` environment
variable. You only need a `~/.motyga/config.toml` to change a default, for example to use the
**RU mirror** or point at another provider:

```toml
[model_providers.motyga]
name = "Motyga"
base_url = "https://ru.motyga.com/v1"   # RU mirror; default is https://api.motyga.com/v1
```

Only `base_url` is taken from this table, and it must be a `https://*.motyga.com` address — your
Motyga key is only ever sent to Motyga hosts. Define any other gateway under its own provider id.

### Local models (Ollama, LM Studio, llama.cpp, vLLM)

Run the agent against a model on your own machine or LAN — no key needed, nothing is billed:

```shell
motyga local                          # find running servers and list their models
motyga --oss                          # start with the running Ollama / LM Studio server
motyga --oss -m qwen3:8b              # pick the model (or switch later with /model)
motyga local add 192.168.1.20:8080    # register llama.cpp, vLLM or an Ollama on another machine
motyga --local-provider llamacpp      # use a registered server
```

The server must speak the OpenAI **Responses API** (`POST /v1/responses`): Ollama 0.13.4+, LM Studio
0.3.29+, a recent `llama-server`, or vLLM. Give the model at least 16K tokens of context (for Ollama,
`OLLAMA_CONTEXT_LENGTH=32768`); `motyga local` warns when it is smaller. Local servers never receive your
Motyga key.

Local models get the same MCP tools as cloud ones (exposed as plain `mcp__<server>__<tool>` functions), and
can use the internet through two tools the CLI runs itself:

- `web_search` — searches DuckDuckGo, no key needed. Set `MOTYGA_SEARXNG_URL` to your SearXNG instance
  (with the JSON format enabled) or `BRAVE_SEARCH_API_KEY` to use those instead.
- `web_fetch` — reads a public page as text. It refuses `localhost` and private-network addresses.

Turn both off with `web_search = "disabled"` in `config.toml`. Image generation needs a cloud model.

## Highlights

- **Default provider is Motyga** — `https://api.motyga.com/v1`, Responses API, auth via `MOTYGA_API_KEY`.
- **API-key auth only** — no third-party login; no identity or telemetry calls.
- **Telemetry is off by default.**
- **Config directory is `~/.motyga`.**

## Docs

- Using Motyga with the CLI: [motyga.com](https://motyga.com)

---

## Notice

Motyga CLI is a fork of [OpenAI Codex CLI](https://github.com/openai/codex), licensed under the
[Apache-2.0 License](LICENSE). The upstream `LICENSE` and `NOTICE` are retained as required.
