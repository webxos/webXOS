// ============================================================
//  RustyClaw v0.6.0 – single‑file TUI with permanent logo
// ============================================================

// ──────────────────────────────────────────────────────────────
//  Imports
// ──────────────────────────────────────────────────────────────
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::{mpsc, RwLock};
use tokio::time;
use tracing::{error, info, warn};
use walkdir::WalkDir;
use regex::Regex;
use warp::Filter;

// ---------- Config ----------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ollama_url: String,
    pub ollama_model: String,
    pub api_port: u16,
    pub root_dir: PathBuf,
    pub bio_file: PathBuf,
    pub heartbeat_log: PathBuf,
    pub memory_sync_interval_secs: u64,
    pub max_log_lines: usize,
    pub git_auto_commit: bool,
}

impl Default for Config {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let root_dir = home.join(".rustyclaw");
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            ollama_model: "qwen2.5:0.5b".to_string(),
            api_port: 3030,
            root_dir: root_dir.clone(),
            bio_file: root_dir.join("bio.md"),
            heartbeat_log: root_dir.join("data/logs/heartbeat.log"),
            memory_sync_interval_secs: 3600,
            max_log_lines: 200,
            git_auto_commit: true,
        }
    }
}

impl Config {
    pub async fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            let raw = fs::read_to_string(path).await?;
            let cfg: Config = serde_yaml::from_str(&raw)?;
            Ok(cfg)
        } else {
            Ok(Config::default())
        }
    }

    pub async fn save(&self, path: &Path) -> Result<()> {
        let yaml = serde_yaml::to_string(self)?;
        fs::write(path, yaml).await?;
        Ok(())
    }
}

// ---------- JsonLogger ----------
struct JsonLogger {
    log_file: PathBuf,
}

impl JsonLogger {
    fn new(log_file: PathBuf) -> Self {
        Self { log_file }
    }

    fn init_global(&self) -> Result<()> {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive(tracing::Level::INFO.into()),
            )
            .with_target(true)
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
            .init();
        info!("JsonLogger initialized → {}", self.log_file.display());
        Ok(())
    }
}

// ---------- Agent ----------
pub struct Agent {
    config: Arc<RwLock<Config>>,
    bio_path: PathBuf,
    heartbeat_log_path: PathBuf,
}

impl Agent {
    pub async fn new(config: Arc<RwLock<Config>>) -> Result<Self> {
        let cfg = config.read().await;
        let bio_path = cfg.bio_file.clone();
        let heartbeat_log_path = cfg.heartbeat_log.clone();
        drop(cfg);

        if let Some(parent) = bio_path.parent() {
            fs::create_dir_all(parent).await?;
        }
        if let Some(parent) = heartbeat_log_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        if !bio_path.exists() {
            let template = format!(
                r#"# BIO.MD – Living Agent Identity
**Last Updated:** {}
**Agent Role:** Local Assistant

## SOUL
Core personality, values, constraints, and behavioral rules.
- You are a local-only agent running on Ollama with no internet access.
- Stay sandboxed and respect the host system's security.
- Personality traits: concise, reflective, self-improving, helpful.

## SKILLS
Reusable capabilities and "how-to" instructions.
- Read and write local files using incremental edits.
- Execute safe, whitelisted shell commands.
- Summarize interactions and distill insights.

## MEMORY
Curated long-term knowledge and history.

## CONTEXT
Current runtime state.
- Operating System: Debian Linux
- Working Directory: {}
- Config: {}

## SESSION TREE
Pointers or summaries of active conversation branches.
"#,
                chrono::Utc::now().to_rfc3339(),
                std::env::current_dir().unwrap_or_default().display(),
                config.read().await.ollama_model
            );
            fs::write(&bio_path, template).await?;
            info!("Created initial bio.md at {}", bio_path.display());
        }

        if !heartbeat_log_path.exists() {
            fs::File::create(&heartbeat_log_path).await?;
            info!("Created heartbeat log at {}", heartbeat_log_path.display());
        }

        Ok(Self {
            config,
            bio_path,
            heartbeat_log_path,
        })
    }

    pub fn bio_path(&self) -> &Path {
        &self.bio_path
    }

    pub async fn read_bio(&self) -> Result<String> {
        fs::read_to_string(&self.bio_path).await.context("Failed to read bio.md")
    }

    pub async fn update_bio_timestamp(&self) -> Result<()> {
        let content = self.read_bio().await?;
        let now = chrono::Utc::now().to_rfc3339();
        let updated = content
            .lines()
            .map(|line| {
                if line.starts_with("**Last Updated:**") {
                    format!("**Last Updated:** {}", now)
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&self.bio_path, updated).await?;
        Ok(())
    }

    pub async fn append_heartbeat(&self, user_msg: &str, assistant_reply: &str) -> Result<()> {
        let timestamp = chrono::Utc::now().to_rfc3339();
        let entry = format!(
            r#"{{"timestamp":"{}","user":{},"assistant":{}}}"#,
            timestamp,
            serde_json::to_string(user_msg)?,
            serde_json::to_string(assistant_reply)?
        );
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.heartbeat_log_path)
            .await?;
        file.write_all(entry.as_bytes()).await?;
        file.write_all(b"\n").await?;
        Ok(())
    }

    pub async fn consolidate_memory(&self) -> Result<()> {
        let heartbeat_content = fs::read_to_string(&self.heartbeat_log_path).await?;
        let entries: Vec<serde_json::Value> = heartbeat_content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();

        if entries.is_empty() {
            return Ok(());
        }

        let mut summary_text = String::new();
        for entry in entries.iter().take(20) {
            let user = entry["user"].as_str().unwrap_or("");
            let assistant = entry["assistant"].as_str().unwrap_or("");
            summary_text.push_str(&format!("User: {}\nAssistant: {}\n\n", user, assistant));
        }

        let cfg = self.config.read().await;
        let prompt = format!(
            "You are a memory summarizer. Please distill the following recent interactions into a concise note for the agent's MEMORY section. Keep it factual and useful.\n\n{}",
            summary_text
        );
        let summary = match ollama_generate(&cfg.ollama_url, &cfg.ollama_model, &prompt).await {
            Ok(s) => s,
            Err(e) => {
                error!("Memory summarization failed: {}", e);
                return Ok(());
            }
        };
        drop(cfg);

        let mut bio = self.read_bio().await?;
        let memory_marker = "## MEMORY";
        if let Some(pos) = bio.find(memory_marker) {
            let insert_pos = pos + memory_marker.len();
            let memory_block = format!(
                "\n### Summary for {}\n{}\n",
                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
                summary
            );
            bio.insert_str(insert_pos, &memory_block);
            fs::write(&self.bio_path, bio).await?;
            info!("Memory consolidated and bio.md updated.");
        } else {
            warn!("MEMORY section not found in bio.md");
        }
        Ok(())
    }

    pub async fn chat(&self, user_msg: &str) -> Result<String> {
        let bio_content = self.read_bio().await?;
        let cfg = self.config.read().await;
        ollama_generate_with_system(
            &cfg.ollama_url,
            &cfg.ollama_model,
            &bio_content,
            user_msg,
        )
        .await
    }
}

// ---------- Ollama functions ----------
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: ChatMessage,
}

async fn ollama_generate_with_system(
    base_url: &str,
    model: &str,
    system: &str,
    user: &str,
) -> Result<String> {
    let client = reqwest::Client::new();
    let req = OllamaChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user.to_string(),
            },
        ],
        stream: false,
    };
    let url = format!("{}/api/chat", base_url);
    let resp = client
        .post(&url)
        .json(&req)
        .timeout(Duration::from_secs(60))
        .send()
        .await
        .context("Failed to contact Ollama")?;
    let body: OllamaChatResponse = resp.json().await.context("Failed to parse Ollama response")?;
    Ok(body.message.content)
}

async fn ollama_generate(base_url: &str, model: &str, prompt: &str) -> Result<String> {
    let client = reqwest::Client::new();
    #[derive(Debug, Serialize)]
    struct GenerateRequest {
        model: String,
        prompt: String,
        stream: bool,
    }
    #[derive(Debug, Deserialize)]
    struct GenerateResponse {
        response: String,
    }
    let req = GenerateRequest {
        model: model.to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };
    let url = format!("{}/api/generate", base_url);
    let resp = client
        .post(&url)
        .json(&req)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .context("Failed to contact Ollama")?;
    let body: GenerateResponse = resp.json().await.context("Failed to parse Ollama response")?;
    Ok(body.response)
}

#[derive(Debug, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
    pub size: u64,
    pub modified_at: String,
}

async fn list_ollama_models(base_url: &str) -> Result<Vec<OllamaModelInfo>> {
    let url = format!("{}/api/tags", base_url);
    let resp = reqwest::get(&url)
        .await
        .context("Failed to fetch models")?
        .json::<OllamaTagsResponse>()
        .await?;
    Ok(resp.models)
}

// ---------- Sandbox ----------
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for comp in path.components() {
        match comp {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::Normal(c) => components.push(c),
            _ => {}
        }
    }
    let mut result = PathBuf::new();
    for comp in components {
        result.push(comp);
    }
    result
}

fn sanitize_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let full = root.join(relative);
    if full.exists() {
        let resolved = full.canonicalize().context("Failed to canonicalize path")?;
        if !resolved.starts_with(root) {
            anyhow::bail!("Access denied: path outside sandbox");
        }
        Ok(resolved)
    } else {
        let normalized = normalize_path(&full);
        if !normalized.starts_with(root) {
            anyhow::bail!("Access denied: path would be outside sandbox");
        }
        Ok(normalized)
    }
}

// ---------- Safe command ----------
async fn run_safe_command(cmd: &str, args: &[&str], cwd: &PathBuf) -> Result<String> {
    let allowed = ["ls", "cat", "echo", "git", "pwd"];
    if !allowed.contains(&cmd) {
        anyhow::bail!("Command not allowed: {}", cmd);
    }
    let output = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .output()
        .await?;
    let stdout = String::from_utf8(output.stdout)?;
    let stderr = String::from_utf8(output.stderr)?;
    if output.status.success() {
        Ok(format!("{}{}", stdout, stderr))
    } else {
        anyhow::bail!("Command failed: {}", stderr)
    }
}

// ---------- AppCommand ----------
#[derive(Debug)]
pub enum AppCommand {
    Chat(String),
    ConsolidateMemory,
    WriteFile { path: String, content: String },
    ReadFile { path: String },
    ListModels,
    SelectModel(String),
    ListDir(String),
    SearchFiles(String),
    RunCommand(String),
    GitStatus,
    GitLog(usize),
    GitCommit(String),
    Quit,
}

// ---------- Command dispatcher ----------
async fn run_command(
    cmd: AppCommand,
    agent: Arc<Agent>,
    config: Arc<RwLock<Config>>,
    log_tx: mpsc::Sender<String>,
) -> Result<()> {
    match cmd {
        AppCommand::Chat(prompt) => {
            let _ = log_tx.send(format!("🤖 Thinking: {}", prompt)).await;
            match agent.chat(&prompt).await {
                Ok(reply) => {
                    let _ = log_tx.send(format!("💬 {}", reply)).await;
                    if let Err(e) = agent.append_heartbeat(&prompt, &reply).await {
                        let _ = log_tx.send(format!("❌ Failed to log heartbeat: {e}")).await;
                    } else if let Err(e) = agent.update_bio_timestamp().await {
                        let _ = log_tx.send(format!("❌ Failed to update bio timestamp: {e}")).await;
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Ollama error: {e}")).await;
                }
            }
        }
        AppCommand::ConsolidateMemory => {
            let _ = log_tx.send("🧠 Consolidating memory...".to_string()).await;
            match agent.consolidate_memory().await {
                Ok(_) => {
                    let _ = log_tx.send("✅ Memory consolidated.".to_string()).await;
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Consolidation error: {e}")).await;
                }
            }
        }
        AppCommand::WriteFile { path, content } => {
            let _ = log_tx.send(format!("✍️ Writing file: {}", path)).await;
            let cfg = config.read().await;
            let data_root = cfg.root_dir.join("data");
            drop(cfg);
            let full_path = sanitize_path(&data_root, &path)?;
            if let Some(parent) = full_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&full_path, content).await?;
            let _ = log_tx.send(format!("✅ File written: {}", full_path.display())).await;

            let auto_commit = config.read().await.git_auto_commit;
            if auto_commit {
                let repo_path = data_root;
                let file_rel = path;
                let msg = format!("Agent write: {}", file_rel);
                let result = run_safe_command("git", &["add", &file_rel], &repo_path).await;
                if let Err(e) = result {
                    let _ = log_tx.send(format!("⚠️ Git add failed: {e}")).await;
                } else {
                    let result = run_safe_command("git", &["commit", "-m", &msg], &repo_path).await;
                    if let Err(e) = result {
                        let _ = log_tx.send(format!("⚠️ Git commit failed: {e}")).await;
                    } else {
                        let _ = log_tx.send(format!("✅ Committed: {}", msg)).await;
                    }
                }
            }
        }
        AppCommand::ReadFile { path } => {
            let _ = log_tx.send(format!("📖 Reading file: {}", path)).await;
            let cfg = config.read().await;
            let data_root = cfg.root_dir.join("data");
            drop(cfg);
            let full_path = sanitize_path(&data_root, &path)?;
            let content = fs::read_to_string(&full_path).await?;
            let _ = log_tx.send(format!("📄 Content of {}:\n{}", full_path.display(), content)).await;
        }
        AppCommand::ListModels => {
            let _ = log_tx.send("📦 Fetching Ollama models...".to_string()).await;
            let cfg = config.read().await;
            match list_ollama_models(&cfg.ollama_url).await {
                Ok(models) => {
                    for m in models {
                        let size_mb = m.size / (1024 * 1024);
                        let _ = log_tx.send(format!("  {} ({} MB, updated {})", m.name, size_mb, m.modified_at)).await;
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Failed to list models: {e}")).await;
                }
            }
        }
        AppCommand::SelectModel(model_name) => {
            let _ = log_tx.send(format!("🔧 Switching to model: {}", model_name)).await;
            let mut cfg = config.write().await;
            cfg.ollama_model = model_name.clone();
            cfg.save(PathBuf::from("config.yaml").as_path()).await?;
            let _ = log_tx.send(format!("✅ Model switched to {}.", model_name)).await;
        }
        AppCommand::ListDir(path) => {
            let cfg = config.read().await;
            let data_root = cfg.root_dir.join("data");
            drop(cfg);
            let target = if path.is_empty() {
                data_root
            } else {
                sanitize_path(&data_root, &path)?
            };
            let _ = log_tx.send(format!("📁 Listing: {}", target.display())).await;
            let entries = WalkDir::new(&target)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .map(|e| {
                    let typ = if e.file_type().is_dir() { "📁" } else { "📄" };
                    format!("{} {}", typ, e.file_name().to_string_lossy())
                })
                .collect::<Vec<_>>();
            if entries.is_empty() {
                let _ = log_tx.send("  (empty)".to_string()).await;
            } else {
                for entry in entries {
                    let _ = log_tx.send(entry).await;
                }
            }
        }
        AppCommand::SearchFiles(query) => {
            let _ = log_tx.send(format!("🔍 Searching for: {}", query)).await;
            let cfg = config.read().await;
            let data_root = cfg.root_dir.join("data");
            drop(cfg);
            let re = Regex::new(&regex::escape(&query)).unwrap();
            let walker = WalkDir::new(&data_root)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file());
            let mut matches = Vec::new();
            for entry in walker {
                if let Ok(content) = std::fs::read_to_string(entry.path()) {
                    if re.is_match(&content) {
                        matches.push(entry.path().strip_prefix(&data_root).unwrap_or(entry.path()).display().to_string());
                    }
                }
            }
            if matches.is_empty() {
                let _ = log_tx.send("  No matches found.".to_string()).await;
            } else {
                for m in matches {
                    let _ = log_tx.send(m).await;
                }
            }
        }
        AppCommand::RunCommand(cmd_line) => {
            let parts: Vec<&str> = cmd_line.split_whitespace().collect();
            if parts.is_empty() {
                let _ = log_tx.send("Usage: /run <command> [args...]".to_string()).await;
                return Ok(());
            }
            let cmd = parts[0];
            let args = &parts[1..];
            let cfg = config.read().await;
            let cwd = cfg.root_dir.join("data");
            drop(cfg);
            let _ = log_tx.send(format!("🖥️ Running: {} {}", cmd, args.join(" "))).await;
            match run_safe_command(cmd, args, &cwd).await {
                Ok(output) => {
                    for line in output.lines() {
                        let _ = log_tx.send(line.to_string()).await;
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Command failed: {e}")).await;
                }
            }
        }
        AppCommand::GitStatus => {
            let cfg = config.read().await;
            let cwd = cfg.root_dir.join("data");
            drop(cfg);
            match run_safe_command("git", &["status", "--short"], &cwd).await {
                Ok(output) => {
                    if output.is_empty() {
                        let _ = log_tx.send("  Working tree clean".to_string()).await;
                    } else {
                        for line in output.lines() {
                            let _ = log_tx.send(line.to_string()).await;
                        }
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Git status failed: {e}")).await;
                }
            }
        }
        AppCommand::GitLog(n) => {
            let cfg = config.read().await;
            let cwd = cfg.root_dir.join("data");
            drop(cfg);
            match run_safe_command("git", &["log", "-n", &n.to_string(), "--oneline"], &cwd).await {
                Ok(output) => {
                    if output.is_empty() {
                        let _ = log_tx.send("  No commits yet".to_string()).await;
                    } else {
                        for line in output.lines() {
                            let _ = log_tx.send(line.to_string()).await;
                        }
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Git log failed: {e}")).await;
                }
            }
        }
        AppCommand::GitCommit(msg) => {
            let cfg = config.read().await;
            let cwd = cfg.root_dir.join("data");
            drop(cfg);
            let _ = log_tx.send(format!("📦 Committing all changes: {}", msg)).await;
            match run_safe_command("git", &["add", "-A"], &cwd).await {
                Ok(_) => {
                    match run_safe_command("git", &["commit", "-m", &msg], &cwd).await {
                        Ok(output) => {
                            let _ = log_tx.send(format!("✅ {}", output.trim())).await;
                        }
                        Err(e) => {
                            let _ = log_tx.send(format!("❌ Commit failed: {e}")).await;
                        }
                    }
                }
                Err(e) => {
                    let _ = log_tx.send(format!("❌ Add failed: {e}")).await;
                }
            }
        }
        AppCommand::Quit => {}
    }
    Ok(())
}

// ---------- AppState (TUI – no blocking) ----------
struct AppState {
    input: String,
    logs: VecDeque<String>,
    config: Arc<RwLock<Config>>,
    bio_path: PathBuf,
    model_name: String,
    max_log_lines: usize,
    cmd_tx: mpsc::Sender<AppCommand>,
    log_rx: mpsc::Receiver<String>,
}

impl AppState {
    fn new(
        config: Arc<RwLock<Config>>,
        cmd_tx: mpsc::Sender<AppCommand>,
        log_rx: mpsc::Receiver<String>,
        bio_path: PathBuf,
        initial_model: String,
    ) -> Self {
        Self {
            input: String::new(),
            logs: VecDeque::new(),
            config,
            bio_path,
            model_name: initial_model,
            max_log_lines: 200,
            cmd_tx,
            log_rx,
        }
    }

    fn push_log(&mut self, line: String) {
        if self.logs.len() >= self.max_log_lines {
            self.logs.pop_front();
        }
        self.logs.push_back(line);
    }

    fn visible_logs(&self, height: usize) -> Vec<String> {
        let skip = if self.logs.len() > height {
            self.logs.len() - height
        } else {
            0
        };
        self.logs.iter().skip(skip).cloned().collect()
    }

    fn drain_logs(&mut self) {
        while let Ok(line) = self.log_rx.try_recv() {
            self.push_log(line);
        }
    }

    // Non‑blocking cache update – ignore any lock error
    fn refresh_model_cache(&mut self) {
        if let Ok(cfg) = self.config.try_read() {
            self.model_name = cfg.ollama_model.clone();
        }
    }

    fn handle_command(&mut self, cmd: &str) -> Option<AppCommand> {
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() {
            return None;
        }
        match parts[0] {
            "/help" => {
                let help_text = r#"Commands:
  /help               – Show this help
  /bio                – Display current bio.md
  /consolidate        – Force memory consolidation
  /write_file <path> <content> – Write content to a file in data/ folder
  /read_file <path>   – Read a file from data/ folder
  /model list         – List all Ollama models
  /model select <name> – Switch to another model
  /list_dir [path]    – List contents of data/ or subfolder
  /search <query>     – Search for text in all files under data/
  /run <command>      – Run a safe command (whitelisted: ls, cat, echo, git, pwd)
  /git status         – Show git status of data/ folder
  /git log [n]        – Show last n commits (default 10)
  /git commit <msg>   – Commit all changes in data/ folder
  /quit or /exit      – Exit RustyClaw"#;
                for line in help_text.lines() {
                    self.push_log(line.to_string());
                }
                None
            }
            "/bio" => {
                match std::fs::read_to_string(&self.bio_path) {
                    Ok(content) => {
                        for line in content.lines() {
                            self.push_log(line.to_string());
                        }
                    }
                    Err(e) => self.push_log(format!("❌ Error reading bio.md: {e}")),
                }
                None
            }
            "/consolidate" => {
                self.push_log("Consolidating memory...".into());
                Some(AppCommand::ConsolidateMemory)
            }
            "/write_file" => {
                if parts.len() < 3 {
                    self.push_log("Usage: /write_file <path> <content>".into());
                    return None;
                }
                let path = parts[1].to_string();
                let content = parts[2..].join(" ");
                Some(AppCommand::WriteFile { path, content })
            }
            "/read_file" => {
                if parts.len() < 2 {
                    self.push_log("Usage: /read_file <path>".into());
                    return None;
                }
                let path = parts[1].to_string();
                Some(AppCommand::ReadFile { path })
            }
            "/model" => {
                if parts.len() < 2 {
                    self.push_log("Usage: /model list | /model select <name>".into());
                    return None;
                }
                match parts[1] {
                    "list" => Some(AppCommand::ListModels),
                    "select" => {
                        if parts.len() < 3 {
                            self.push_log("Usage: /model select <model_name>".into());
                            None
                        } else {
                            Some(AppCommand::SelectModel(parts[2].to_string()))
                        }
                    }
                    _ => {
                        self.push_log("Unknown /model subcommand".into());
                        None
                    }
                }
            }
            "/list_dir" => {
                let path = if parts.len() > 1 { parts[1] } else { "" };
                Some(AppCommand::ListDir(path.to_string()))
            }
            "/search" => {
                if parts.len() < 2 {
                    self.push_log("Usage: /search <query>".into());
                    None
                } else {
                    let query = parts[1..].join(" ");
                    Some(AppCommand::SearchFiles(query))
                }
            }
            "/run" => {
                if parts.len() < 2 {
                    self.push_log("Usage: /run <command> [args...]".into());
                    None
                } else {
                    let full_cmd = parts[1..].join(" ");
                    Some(AppCommand::RunCommand(full_cmd))
                }
            }
            "/git" => {
                if parts.len() < 2 {
                    self.push_log("Usage: /git status | /git log [n] | /git commit <msg>".into());
                    return None;
                }
                match parts[1] {
                    "status" => Some(AppCommand::GitStatus),
                    "log" => {
                        let n = if parts.len() > 2 {
                            parts[2].parse::<usize>().unwrap_or(10)
                        } else {
                            10
                        };
                        Some(AppCommand::GitLog(n))
                    }
                    "commit" => {
                        if parts.len() < 3 {
                            self.push_log("Usage: /git commit <message>".into());
                            None
                        } else {
                            let msg = parts[2..].join(" ");
                            Some(AppCommand::GitCommit(msg))
                        }
                    }
                    _ => {
                        self.push_log("Unknown /git subcommand".into());
                        None
                    }
                }
            }
            "/quit" | "/exit" => Some(AppCommand::Quit),
            _ => Some(AppCommand::Chat(cmd.to_string())),
        }
    }
}

// ---------- Worker ----------
async fn worker(
    agent: Arc<Agent>,
    config: Arc<RwLock<Config>>,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    log_tx: mpsc::Sender<String>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        let result = run_command(cmd, agent.clone(), config.clone(), log_tx.clone()).await;
        if let Err(e) = result {
            let _ = log_tx.send(format!("❌ Command error: {e}")).await;
        }
    }
}

// ---------- REST API ----------
fn build_api(agent: Arc<Agent>) -> impl warp::Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    let bio_get = warp::path!("api" / "bio")
        .and(warp::get())
        .and_then(move || {
            let agent = agent.clone();
            async move {
                match agent.read_bio().await {
                    Ok(content) => Ok::<_, warp::Rejection>(warp::reply::json(&serde_json::json!({"bio": content}))),
                    Err(e) => Ok(warp::reply::json(&serde_json::json!({"error": e.to_string()}))),
                }
            }
        });

    let health = warp::path!("health")
        .and(warp::get())
        .map(|| warp::reply::json(&serde_json::json!({"status": "ok"})));

    bio_get.or(health)
}

// ---------- TUI rendering (permanent logo) ----------
fn ui(frame: &mut Frame, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(3),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .split(frame.size());

    let logo_text = r#"    
▄▖    ▗     ▜      
▙▘▌▌▛▘▜▘▌▌▛▘▐ ▀▌▌▌▌
▌▌▙▌▄▌▐▖▙▌▙▖▐▖█▌▚▚▘
        ▄▌         
 🦞 RustyClaw v0.6.0"#;
    let logo = Paragraph::new(logo_text)
        .block(Block::default().borders(Borders::NONE))
        .style(Style::default().fg(Color::Rgb(205, 127, 50)).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);
    frame.render_widget(logo, chunks[0]);

    let header = Paragraph::new(Line::from(vec![
        Span::styled("📄 bio.md active  ", Style::default().fg(Color::Rgb(205, 127, 50)).add_modifier(Modifier::BOLD)),
        Span::styled(format!("Model: {}", app.model_name), Style::default().fg(Color::Cyan)),
    ]))
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(205, 127, 50))))
    .alignment(Alignment::Left);
    frame.render_widget(header, chunks[1]);

    let log_items: Vec<ListItem> = app
        .visible_logs(chunks[2].height.saturating_sub(2) as usize)
        .into_iter()
        .map(|s| ListItem::new(Line::from(s)))
        .collect();
    let logs = List::new(log_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Logs  (ESC to quit · /help for commands)")
            .border_style(Style::default().fg(Color::Rgb(205, 127, 50))),
    );
    frame.render_widget(logs, chunks[2]);

    let input = Paragraph::new(app.input.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Input (Enter to send)")
                .border_style(Style::default().fg(Color::Rgb(205, 127, 50))),
        )
        .style(Style::default().fg(Color::White));
    frame.render_widget(input, chunks[3]);

    frame.set_cursor(
        chunks[3].x + app.input.len() as u16 + 1,
        chunks[3].y + 1,
    );
}

// ---------- Main ----------
#[tokio::main]
async fn main() -> Result<()> {
    let ollama_ok = reqwest::get("http://localhost:11434/api/tags")
        .await
        .is_ok();
    if !ollama_ok {
        eprintln!("⚠️  Ollama not detected at http://localhost:11434");
        eprintln!("   Start Ollama with: ollama serve");
        eprintln!("   Pull a model: ollama pull qwen2.5:0.5b");
    }

    let git_ok = tokio::process::Command::new("git")
        .arg("--version")
        .output()
        .await
        .is_ok();
    if !git_ok {
        eprintln!("⚠️  Git not found in PATH. Git commands will fail.");
    }

    let config = Config::load(Path::new("config.yaml")).await.unwrap_or_default();
    let config = Arc::new(RwLock::new(config));

    let log_file = {
        let cfg = config.read().await;
        cfg.root_dir.join("data/logs/app.log")
    };
    let logger = JsonLogger::new(log_file);
    logger.init_global()?;
    info!("RustyClaw v0.6.0 starting up");

    let agent = Arc::new(Agent::new(config.clone()).await?);
    info!("Agent initialized, bio.md at {}", agent.bio_path().display());

    let data_repo_path = {
        let cfg = config.read().await;
        cfg.root_dir.join("data")
    };
    if git_ok && !data_repo_path.join(".git").exists() {
        let result = run_safe_command("git", &["init"], &data_repo_path).await;
        if let Err(e) = result {
            warn!("Git init failed: {}", e);
        } else {
            info!("Git repo initialized at {}", data_repo_path.display());
        }
    }

    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(32);
    let (log_tx, log_rx) = mpsc::channel::<String>(256);

    let worker_agent = agent.clone();
    let worker_config = config.clone();
    tokio::spawn(async move {
        worker(worker_agent, worker_config, cmd_rx, log_tx).await;
    });

    let interval = {
        let cfg = config.read().await;
        Duration::from_secs(cfg.memory_sync_interval_secs)
    };
    let timer_cmd_tx = cmd_tx.clone();
    tokio::spawn(async move {
        let mut interval = time::interval(interval);
        loop {
            interval.tick().await;
            let _ = timer_cmd_tx.send(AppCommand::ConsolidateMemory).await;
        }
    });

    let api_agent = agent.clone();
    let api_port = {
        let cfg = config.read().await;
        cfg.api_port
    };
    tokio::spawn(async move {
        let api = build_api(api_agent);
        info!("REST API listening on :{}", api_port);
        warp::serve(api).run(([127, 0, 0, 1], api_port)).await;
    });

    crossterm::terminal::enable_raw_mode().unwrap();
    let backend = ratatui::backend::CrosstermBackend::new(std::io::stdout());
    let mut terminal = ratatui::Terminal::new(backend).unwrap();
    crossterm::execute!(
        std::io::stderr(),
        crossterm::terminal::EnterAlternateScreen,
        crossterm::event::EnableMouseCapture,
    )
    .ok();

    let (bio_path, initial_model) = {
        let cfg = config.read().await;
        (cfg.bio_file.clone(), cfg.ollama_model.clone())
    };
    let mut app = AppState::new(config.clone(), cmd_tx.clone(), log_rx, bio_path, initial_model);
    app.push_log("🦀 Welcome to RustyClaw v0.6.0!".into());
    app.push_log("📄 bio.md loaded as persistent memory.".into());
    if !ollama_ok {
        app.push_log("⚠️  Ollama not running! Please start it: ollama serve".into());
    } else {
        app.push_log("✅ Ollama detected.".into());
    }
    if !git_ok {
        app.push_log("⚠️  Git not found. Git commands will fail.".into());
    } else {
        app.push_log("✅ Git detected.".into());
    }
    app.push_log(format!("🌐 REST API: http://127.0.0.1:{}/api/bio", api_port));
    app.push_log("📁 Data folder is a Git repo (auto‑commits after writes)".into());
    app.push_log("💡 Try /help to see all commands.".into());

    loop {
        app.drain_logs();
        app.refresh_model_cache();

        terminal.draw(|f| ui(f, &app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Esc => {
                            let _ = cmd_tx.send(AppCommand::Quit).await;
                            break;
                        }
                        KeyCode::Enter => {
                            let cmd = app.input.trim().to_string();
                            app.input.clear();
                            if cmd.is_empty() {
                                continue;
                            }
                            app.push_log(format!("> {}", cmd));
                            if let Some(worker_cmd) = app.handle_command(&cmd) {
                                match worker_cmd {
                                    AppCommand::Quit => {
                                        let _ = cmd_tx.send(AppCommand::Quit).await;
                                        break;
                                    }
                                    other => {
                                        let _ = cmd_tx.send(other).await;
                                    }
                                }
                            }
                        }
                        KeyCode::Char(c) => app.input.push(c),
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    crossterm::terminal::disable_raw_mode().unwrap();
    crossterm::execute!(
        std::io::stderr(),
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::event::DisableMouseCapture,
    )
    .ok();

    info!("RustyClaw shut down cleanly");
    Ok(())
}
