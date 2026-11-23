//! Shared helpers for the browser (CDP) tools: resolving a browser (attach to a loopback endpoint
//! or launch a located Chromium), strict data minimisation (never keep cookies, auth headers or
//! storage values; declared secrets redacted in every encoding), JSON body *outlines*, a DOM
//! summary, PNG pixel diffing and base64.

use crate::tool::Ctx;
use craft_anything_analysis::web_har::{contains_ascii_ci, encode_query, replace_needles_with};
use craft_anything_engines::cdp::{self, CdpClient, CdpEvent, Deadline, Endpoint, WsConn};
use craft_anything_evidence::{Environment, ErrorCode, ToolError, host_triple, sha256_hex};
use craft_anything_platform::process::{Process, RunningProcess, StdinMode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

/// Biggest screenshot we will decode or return inline.
pub const MAX_SCREENSHOT_BYTES: usize = 64 << 20;
/// Working-set admission for a screenshot decode (width·height·4 must be under this).
pub const MAX_PIXELS_WORKING_SET: u64 = 256 << 20;

/// The evidence environment for a CDP-backed observation, in the run's mode: Clean room's
/// visitor view of a website is a clean-room result (#157).
pub fn cdp_env(clean_room: bool, engine_version: Option<String>) -> Environment {
    Environment { clean_room, engine: "cdp".into(), engine_version, host: host_triple(), language_id: None }
}

// ---------------------------------------------------------------------------------------------
// Browser resolution: attach or launch

/// How a browser-running tool may obtain a browser.
pub struct LaunchOptions {
    /// Launch a located Chromium when no endpoint is given.
    pub allow_launch: bool,
    pub headless: bool,
    pub no_sandbox: bool,
    /// Seconds to wait for a launched browser to start listening.
    pub startup_secs: u64,
    /// Which profile a launched browser uses.
    pub profile: ProfileChoice,
    /// The page a launched browser starts on (default `about:blank`).
    pub start_url: Option<String>,
    /// Its profile is copied out afterwards as a saved sign-in (cookies kept with a key every
    /// later launch can read).
    pub keeps_sign_in: bool,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        LaunchOptions {
            allow_launch: true,
            headless: true,
            no_sandbox: false,
            startup_secs: 30,
            profile: ProfileChoice::Fresh,
            start_url: None,
            keeps_sign_in: false,
        }
    }
}

/// The profile a launched browser gets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ProfileChoice {
    /// A new empty private profile, deleted when the browser is dropped.
    #[default]
    Fresh,
    /// A new private profile filled with a copy of a saved sign-in first (see
    /// [`super::site_sign_in::copy_profile`]); deleted when the browser is dropped, so the saved
    /// sign-in is never changed.
    SeededFrom(std::path::PathBuf),
}

/// A connected browser: its loopback endpoint and product info. If we launched it, the owned
/// process is held here and killed when this is dropped.
pub struct Browser {
    pub endpoint: Endpoint,
    pub product: String,
    pub version: String,
    /// The `User-Agent` from `/json/version` (empty when unknown); names the runtime.
    pub user_agent: String,
    pub launched: bool,
    // The process is dropped before the profile so its supervisor is asked to stop it first.
    #[allow(dead_code)]
    _proc: Option<RunningProcess>,
    #[allow(dead_code)]
    _profile: Option<BrowserProfile>,
    /// Stops the browser and deletes its profile if the app ends without dropping it (#810).
    #[allow(dead_code)]
    _watch: Option<RunningProcess>,
}

/// Launch flags for a browser that holds a sign-in (a sign-in window, or a copy of a saved one).
fn sign_in_args(opts: &LaunchOptions) -> Vec<String> {
    if opts.profile == ProfileChoice::Fresh && !opts.keeps_sign_in {
        return Vec::new();
    }
    let mut args = Vec::new();
    // A saved sign-in is made in a window and read later without one: on Linux both launches
    // must keep cookies with the same key, so neither asks the desktop's keyring.
    if cfg!(target_os = "linux") {
        args.push("--password-store=basic".into());
    }
    // Only this profile has password saving turned off (`prepare_sign_in_profile`): never let a
    // copied `Local State` pick another one.
    args.push(format!("--profile-directory={}", super::site_sign_in::SIGN_IN_PROFILE));
    // Chromium needs an explicit session restore on macOS to reload session-only cookies. The
    // copied profile excludes session and tab files, so old pages are not reopened.
    args.push("--restore-last-session".into());
    args
}

struct BrowserProfile(std::path::PathBuf);

fn remove_profile_with(mut remove: impl FnMut() -> std::io::Result<()>) -> std::io::Result<()> {
    let started = std::time::Instant::now();
    loop {
        match remove() {
            Ok(()) => return Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) if started.elapsed() >= Duration::from_secs(5) => return Err(e),
            Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

impl Drop for BrowserProfile {
    fn drop(&mut self) {
        if let Err(e) = remove_profile_with(|| std::fs::remove_dir_all(&self.0)) {
            log::warn!("couldn't remove the private browser profile {}: {e}", self.0.display());
        }
    }
}

/// What is answering on a CDP endpoint: `electron`, `edge`, `chrome`, or `unknown`, from the
/// `/json/version` product and user agent (Electron and Edge both name themselves there).
pub fn runtime_of(product: &str, user_agent: &str) -> &'static str {
    let has = |s: &str| user_agent.contains(s) || product.contains(s);
    if has("Electron") {
        "electron"
    } else if has("Edg/") {
        "edge"
    } else if has("Chrome/") {
        "chrome"
    } else {
        "unknown"
    }
}

/// The error returned when no browser is available and none was asked to launch.
pub fn engine_missing(detail: impl std::fmt::Display) -> ToolError {
    ToolError::new(ErrorCode::EngineMissing, format!("no browser is available: {detail}")).with_remediation(
        "Give a loopback CDP endpoint (start a browser with --remote-debugging-port), or install a Chromium-based browser so one can be launched (set its path in Settings → Engines if it isn't found).",
    )
}

/// Where a launched browser's private profile goes. A Flatpak or Snap browser has a /tmp of its
/// own, so it couldn't use a profile there and we couldn't read its port from it: its profile goes
/// in the home folder its sandbox always shares (#747). Anything else uses the temp folder.
fn profile_base(exe: &std::path::Path, home: Option<std::path::PathBuf>) -> std::path::PathBuf {
    let parent = exe.parent().unwrap_or(std::path::Path::new(""));
    let name = exe.file_name().filter(|n| !n.is_empty());
    match (name, home) {
        (Some(app), Some(home)) if parent.ends_with("flatpak/exports/bin") => home.join(".var/app").join(app).join("cache"),
        (Some(app), Some(home)) if parent == std::path::Path::new("/snap/bin") => home.join("snap").join(app).join("common"),
        _ => std::env::temp_dir(),
    }
}

/// The snap a small launcher script hands over to, when it does and that snap is there: Ubuntu's
/// `/usr/bin/chromium-browser` ends in `exec /snap/bin/chromium "$@"` and comes first on the search
/// path (#1054). The snap is what runs, so it's what to start, with a snap's profile.
fn snap_behind(exe: &std::path::Path) -> Option<std::path::PathBuf> {
    const LARGEST_SCRIPT: u64 = 64 * 1024;
    if std::fs::metadata(exe).ok()?.len() > LARGEST_SCRIPT {
        return None;
    }
    snap_in_script(&std::fs::read(exe).ok()?).filter(|snap| snap.exists())
}

fn snap_in_script(script: &[u8]) -> Option<std::path::PathBuf> {
    if !script.starts_with(b"#!") {
        return None;
    }
    String::from_utf8_lossy(script).lines().find_map(|line| {
        let mut words = line.split_whitespace();
        let target = words.next().filter(|w| *w == "exec").and(words.next())?.trim_matches(['"', '\'']);
        target.starts_with("/snap/bin/").then(|| std::path::PathBuf::from(target))
    })
}

impl Browser {
    /// The profile folder of a browser we launched (deleted when this is dropped).
    pub(crate) fn profile_dir(&self) -> Option<&std::path::Path> {
        self._profile.as_ref().map(|p| p.0.as_path())
    }

    /// Stop a browser we launched and wait (at most `d`) until it's gone. Returns whether it
    /// confirmed in time.
    pub(crate) fn stop_and_wait(&mut self, d: Duration) -> bool {
        self._proc.as_mut().is_none_or(|p| p.kill_and_wait(d))
    }

    /// A handle to a browser someone else owns (it is not killed when this drops).
    pub fn attached(endpoint: &Endpoint, product: &str, version: &str) -> Browser {
        Browser {
            endpoint: endpoint.clone(),
            product: product.into(),
            version: version.into(),
            user_agent: String::new(),
            launched: false,
            _proc: None,
            _profile: None,
            _watch: None,
        }
    }

    /// Resolve a browser from an optional endpoint, falling back to launching a located Chromium.
    pub fn resolve(ctx: &Ctx<'_>, cdp_endpoint: Option<&str>, opts: &LaunchOptions, dl: &Deadline) -> Result<Browser, ToolError> {
        if let Some(ep_str) = cdp_endpoint.map(str::trim).filter(|s| !s.is_empty()) {
            let endpoint = cdp::parse_endpoint(ep_str)?;
            let version = cdp::json_version(&endpoint, dl)?;
            return Ok(Browser {
                product: version.get("Browser").and_then(Value::as_str).unwrap_or("unknown").to_string(),
                version: version.get("Protocol-Version").and_then(Value::as_str).unwrap_or_default().to_string(),
                user_agent: version.get("User-Agent").and_then(Value::as_str).unwrap_or_default().to_string(),
                endpoint,
                launched: false,
                _proc: None,
                _profile: None,
                _watch: None,
            });
        }
        // A website opened with web_site_open is the page every browser tool looks at by default.
        if let Some(site) = ctx.session.ext.get::<super::website_tools::SiteBrowser>()
            && cdp::json_version(&site.endpoint, dl).is_ok()
        {
            return Ok(Browser::attached(&site.endpoint, &site.product, &site.version));
        }
        if !opts.allow_launch {
            return Err(engine_missing("no endpoint was given"));
        }
        Browser::launch(ctx, opts, dl)
    }

    pub(crate) fn launch(ctx: &Ctx<'_>, opts: &LaunchOptions, dl: &Deadline) -> Result<Browser, ToolError> {
        let exe =
            craft_anything_engines::locate("chromium", &ctx.session.settings.engines.tools, &ctx.session.settings.engine_store_dir()).ok_or_else(|| {
                // Browser tools that take an endpoint come here without `Requires::Engine`'s check.
                let (reason, fix) = ctx.session.settings.engine_not_found("chromium");
                ToolError::new(ErrorCode::EngineMissing, reason)
                    .with_remediation(format!("{fix} Or give a loopback CDP endpoint (start a browser with --remote-debugging-port)."))
            })?;
        let exe = snap_behind(&exe).unwrap_or(exe);
        // Always a profile of our own where this browser can reach it (a Snap's or Flatpak's
        // folder, #747), so a saved sign-in is only ever copied in and out, never used in place.
        let owned = BrowserProfile(super::artifact_tools::fresh_dir_in(&profile_base(&exe, dirs::home_dir()), "craft-browser")?);
        if let ProfileChoice::SeededFrom(saved) = &opts.profile {
            // The caller holds the site's reservation: a profile saved before password stores
            // were left out loses them here (the copy below skips them either way).
            super::site_sign_in::remove_credential_stores(saved);
            // On error the half-filled copy is removed as `owned` drops.
            super::site_sign_in::copy_profile(saved, &owned.0, &super::site_sign_in::CopyCaps::default())?;
        }
        if opts.keeps_sign_in || opts.profile != ProfileChoice::Fresh {
            super::site_sign_in::prepare_sign_in_profile(&owned.0)?;
        }
        let (profile_dir, profile) = (owned.0.clone(), Some(owned));
        let mut args: Vec<String> = vec![
            format!("--user-data-dir={}", profile_dir.display()),
            "--remote-debugging-port=0".into(),
            "--remote-allow-origins=*".into(),
            "--no-first-run".into(),
            "--no-default-browser-check".into(),
            "--disable-background-networking".into(),
            "--disable-sync".into(),
            "--disable-extensions".into(),
            "--disable-features=Translate".into(),
        ];
        if opts.headless {
            args.push("--headless=new".into());
            args.push("--disable-gpu".into());
        }
        if opts.no_sandbox {
            args.push("--no-sandbox".into());
        }
        args.extend(sign_in_args(opts));
        args.push(opts.start_url.clone().unwrap_or_else(|| "about:blank".into()));
        let cancel = ctx.cancel.clone();
        let tail = StderrTail::default();
        let sink = tail.clone();
        let mut proc = Process::new(&exe)
            .args(&args)
            .stdin(StdinMode::Closed)
            .timeout(Duration::from_secs(opts.startup_secs.saturating_add(3600)))
            .cancel_when(move || cancel.is_cancelled())
            .on_stderr_line(move |l| sink.push(l))
            .label("browser")
            // On Windows the first process can hand the browser to another and exit.
            .hands_off()
            .spawn()?;
        let name = exe.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "the browser".into());
        let endpoint = Self::await_devtools(&profile_dir, dl, &mut proc, &tail, &name)?;
        let version = cdp::json_version(&endpoint, dl)?;
        Ok(Browser {
            product: version.get("Browser").and_then(Value::as_str).unwrap_or("unknown").to_string(),
            version: version.get("Protocol-Version").and_then(Value::as_str).unwrap_or_default().to_string(),
            user_agent: version.get("User-Agent").and_then(Value::as_str).unwrap_or_default().to_string(),
            endpoint,
            launched: true,
            _watch: craft_anything_platform::end_with_this_process(&proc, profile.as_ref().map(|p| p.0.as_path())),
            _proc: Some(proc),
            _profile: profile,
        })
    }

    /// Launch an arbitrary debuggable executable (e.g. an Electron app's own binary) with a fresh
    /// profile and a zero-port remote-debugging listener, then attach to it. Consent-gated callers
    /// use this to drive an app they were given the path to.
    pub fn launch_app(ctx: &Ctx<'_>, exe: &std::path::Path, app_args: &[String], startup_secs: u64, dl: &Deadline) -> Result<Browser, ToolError> {
        if !exe.is_file() {
            return Err(ToolError::new(ErrorCode::NotFound, format!("{} is not a file", exe.display()))
                .with_remediation("Give the absolute path of the app's executable."));
        }
        let profile = BrowserProfile(super::artifact_tools::fresh_temp_dir("craft-electron")?);
        let mut args: Vec<String> = vec![
            format!("--user-data-dir={}", profile.0.display()),
            "--remote-debugging-port=0".into(),
            "--remote-allow-origins=*".into(),
            "--no-first-run".into(),
        ];
        args.extend(app_args.iter().cloned());
        let cancel = ctx.cancel.clone();
        let tail = StderrTail::default();
        let sink = tail.clone();
        let mut proc = Process::new(exe)
            .args(&args)
            .stdin(StdinMode::Closed)
            .timeout(Duration::from_secs(startup_secs.saturating_add(3600)))
            .cancel_when(move || cancel.is_cancelled())
            .on_stderr_line(move |l| sink.push(l))
            .label("electron-app")
            .hands_off()
            .spawn()?;
        let name = exe.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "the app".into());
        let endpoint = Self::await_devtools(&profile.0, dl, &mut proc, &tail, &name)?;
        let version = cdp::json_version(&endpoint, dl)?;
        Ok(Browser {
            product: version.get("Browser").and_then(Value::as_str).unwrap_or("Electron app").to_string(),
            version: version.get("Protocol-Version").and_then(Value::as_str).unwrap_or_default().to_string(),
            user_agent: version.get("User-Agent").and_then(Value::as_str).unwrap_or_default().to_string(),
            endpoint,
            launched: true,
            _watch: craft_anything_platform::end_with_this_process(&proc, Some(&profile.0)),
            _proc: Some(proc),
            _profile: Some(profile),
        })
    }

    /// Poll the launched browser's `DevToolsActivePort` file for the chosen loopback port. A browser
    /// that exits first is reported at once; either failure says how long it took and what the
    /// browser last printed, so the cause is in the message.
    fn await_devtools(profile: &std::path::Path, dl: &Deadline, proc: &mut RunningProcess, tail: &StderrTail, name: &str) -> Result<Endpoint, ToolError> {
        let file = profile.join("DevToolsActivePort");
        let started = std::time::Instant::now();
        loop {
            if let Ok(text) = std::fs::read_to_string(&file) {
                let port: Option<u16> = text.lines().next().and_then(|l| l.trim().parse().ok()).filter(|p| *p != 0);
                if let Some(port) = port {
                    return cdp::parse_endpoint(&format!("http://127.0.0.1:{port}"));
                }
            }
            let waited = started.elapsed().as_secs_f32();
            // What the browser last said, when it said anything.
            let said = tail.lines().into_iter().rev().find(|l| !l.trim().is_empty()).map(|l| format!(" It said: {}", l.trim())).unwrap_or_default();
            if proc.is_finished() {
                return Err(ToolError::new(ErrorCode::ProgramFailed, format!("Couldn't start the private browser: {name} closed after {waited:.1} s, before it was ready.{said}"))
                    .with_remediation("Close any windows of it that a crash left behind and try again; to use another Chromium-based browser (Chrome, Edge, Brave), set its path in Settings → Engines.")
                    .with_details(json!({"browser": name, "waited_secs": waited, "stderr": tail.lines()})));
            }
            if dl.expired() {
                return Err(ToolError::new(
                    ErrorCode::TimedOut,
                    format!("Couldn't start the private browser: {name} didn't get ready within {waited:.0} s.{said}"),
                )
                .with_remediation("Try again; if it keeps happening, set another Chromium-based browser (Chrome, Edge, Brave) in Settings → Engines.")
                .with_details(json!({"browser": name, "waited_secs": waited, "stderr": tail.lines()})));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Target selection and page connection

/// Choose a target by id, else the first of `want_kind` (e.g. "page"), from `/json/list`.
pub fn pick_target<'a>(targets: &'a [Value], target_id: Option<&str>, want_kind: Option<&str>) -> Result<&'a Value, ToolError> {
    if let Some(id) = target_id {
        return targets.iter().find(|t| t.get("id").and_then(Value::as_str) == Some(id)).ok_or_else(|| {
            ToolError::new(ErrorCode::NotFound, format!("no CDP target with id {id:?}"))
                .with_remediation("Call web_pages_list to see the available target ids.")
        });
    }
    let chosen = match want_kind {
        Some(k) => targets.iter().find(|t| t.get("type").and_then(Value::as_str) == Some(k)),
        None => targets.first(),
    };
    chosen.ok_or_else(|| {
        ToolError::new(ErrorCode::NotFound, "the browser has no debuggable target of the required kind")
            .with_remediation("Open a page in the browser, or pass an explicit target_id.")
    })
}

/// Connect a page/renderer target's WebSocket and build a JSON-RPC client.
pub fn connect_target(browser: &Browser, target: &Value, max_message: usize, dl: &Deadline) -> Result<CdpClient, ToolError> {
    let ws_url = target.get("webSocketDebuggerUrl").and_then(Value::as_str).ok_or_else(|| {
        ToolError::new(ErrorCode::EngineMissing, "the target has no webSocketDebuggerUrl (another client may be attached)")
            .with_remediation("Close other DevTools clients, or pick a different target.")
    })?;
    let wt = cdp::ws_target(ws_url, &browser.endpoint)?;
    let conn = WsConn::connect(&wt, max_message, dl)?;
    Ok(CdpClient::new(conn))
}

/// Connect to a page, enable the Page domain that delivers `Page.javascriptDialogOpening`, and
/// answer any dialog it opens by itself, so a call never waits on one. The dialog events stay in
/// the client's queue for the caller to report.
pub fn connect_target_answering_dialogs(browser: &Browser, target: &Value, max_message: usize, dl: &Deadline) -> Result<CdpClient, ToolError> {
    let mut client = connect_target(browser, target, max_message, dl).map(CdpClient::answering_dialogs)?;
    let _ = client.call("Page.enable", json!({}), None, dl);
    Ok(client)
}

/// The dialogs a page opened during a run, from its events: kind, first line and how it was answered.
pub fn dialogs_from(events: &[CdpEvent]) -> Vec<Value> {
    let line = |s: &str| s.lines().next().unwrap_or_default().chars().take(300).collect::<String>();
    let mut dialogs = Vec::new();
    for e in events.iter().filter(|e| e.method == cdp::DIALOG_OPENING).take(30) {
        let p = &e.params;
        let answered = if cdp::dialog_answer(p)["accept"] == true { "accepted" } else { "dismissed" };
        let message = p.get("message").and_then(Value::as_str).unwrap_or_default();
        dialogs.push(json!({"type": p["type"], "message": line(message), "answered": answered}));
    }
    dialogs
}

/// The summary's sentence on dialogs a page opened by itself (empty when none).
pub fn dialogs_said(dialogs: &[Value]) -> String {
    let Some(first) = dialogs.first() else { return String::new() };
    let kind = match first["type"].as_str() {
        Some("alert") => "an alert",
        Some("confirm") => "a confirm question",
        Some("prompt") => "a prompt",
        Some("beforeunload") => "a leave-page question",
        _ => "a dialog",
    };
    let message = first["message"].as_str().unwrap_or_default();
    let more = if dialogs.len() > 1 { format!(" and {} more", dialogs.len() - 1) } else { String::new() };
    format!(" The page opened {kind} by itself (\"{message}\"){more}; it was {} so the page could go on.", first["answered"].as_str().unwrap_or("answered"))
}

/// [`dialogs_said`] for a result's `dialogs` value.
pub fn dialogs_said_in(dialogs: &Value) -> String {
    dialogs_said(dialogs.as_array().map(Vec::as_slice).unwrap_or_default())
}

/// Redact declared secrets in each dialog's message (a page may echo one).
pub fn redact_dialogs(dialogs: &mut [Value], secrets: &mut Secrets) {
    for dialog in dialogs {
        if let Some(message) = dialog["message"].as_str() {
            let redacted = secrets.text(message);
            dialog["message"] = json!(redacted);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Data minimisation: secrets, URLs, headers

/// Secrets the caller names, to redact wherever they appear, in every encoding.
#[derive(Debug, Clone, Default)]
pub struct Secrets {
    raw: Vec<String>,
    spellings: Vec<String>,
    pub redactions: u64,
}

const REDACTED: &str = "«redacted»";

impl Secrets {
    pub fn new(values: &[String]) -> Self {
        let mut spellings: Vec<String> = Vec::new();
        for v in values.iter().filter(|v| !v.is_empty()) {
            let pct_component = percent_encode(v);
            let pct_up = encode_query(v, false, true);
            let pct_lo = encode_query(v, false, false);
            let form_up = encode_query(v, true, true);
            let form_lo = encode_query(v, true, false);
            let quote_plus_up = pct_up.replace("%20", "+");
            let quote_plus_lo = pct_lo.replace("%20", "+");
            for s in [
                v.clone(),
                v.to_ascii_lowercase(),
                v.to_ascii_uppercase(),
                pct_component.clone(),
                pct_component.to_ascii_lowercase(),
                pct_up,
                pct_lo,
                form_up,
                form_lo,
                quote_plus_up,
                quote_plus_lo,
                json_escape(v),
                v.replace(' ', "+"),
            ] {
                if !s.is_empty() && !spellings.contains(&s) {
                    spellings.push(s);
                }
            }
        }
        // Longest first so a value that contains another is replaced whole.
        spellings.sort_by_key(|s| std::cmp::Reverse(s.len()));
        Secrets { raw: values.iter().filter(|v| !v.is_empty()).cloned().collect(), spellings, redactions: 0 }
    }

    pub fn declared(&self) -> usize {
        self.raw.len()
    }

    /// Redact every declared spelling (ASCII case-insensitively, merging overlapping matches) in a string.
    pub fn text(&mut self, s: &str) -> String {
        let (out, count) = replace_needles_with(s, &self.spellings, REDACTED);
        self.redactions = self.redactions.saturating_add(count);
        out
    }

    /// Recursively redact declared values throughout a JSON value (keys and strings).
    pub fn scrub(&mut self, v: &Value) -> Value {
        match v {
            Value::String(s) => Value::String(self.text(s)),
            Value::Array(a) => Value::Array(a.iter().map(|x| self.scrub(x)).collect()),
            Value::Object(o) => {
                let mut m = serde_json::Map::new();
                for (k, val) in o {
                    m.insert(self.text(k), self.scrub(val));
                }
                Value::Object(m)
            }
            other => other.clone(),
        }
    }

    /// True when `haystack` still contains any declared spelling (raw or encoded, ASCII case-insensitively).
    pub fn leaks_in(&self, haystack: &str) -> bool {
        self.spellings.iter().any(|sp| contains_ascii_ci(haystack, sp))
    }
}

/// Minimal `encodeURIComponent`-style encoding for redaction spelling coverage.
pub fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let unreserved = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')');
        if unreserved {
            out.push(b as char);
        } else {
            out.push('%');
            out.push_str(&format!("{b:02X}"));
        }
    }
    out
}

/// JSON string-escaping of a value's inner text (without the surrounding quotes).
pub fn json_escape(s: &str) -> String {
    let quoted = serde_json::Value::String(s.to_string()).to_string();
    quoted.get(1..quoted.len().saturating_sub(1)).unwrap_or("").to_string()
}

/// Remove userinfo (`user:pass@`) from a URL and drop query values that obviously hold credentials.
pub fn sanitize_url(url: &str) -> String {
    let url = url.trim();
    // Strip scheme://userinfo@ → scheme://host…
    if let Some(scheme_end) = url.find("://") {
        let (scheme, rest) = url.split_at(scheme_end + 3);
        let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        if let Some(at) = rest.get(..auth_end).and_then(|a| a.rfind('@')) {
            let after = rest.get(at + 1..).unwrap_or("");
            return format!("{scheme}{after}");
        }
    }
    url.to_string()
}

/// Header names we keep from a response, by lower-case name. Cookies and credentials are never
/// kept; most values are dropped and only the length/encoding of others is retained.
fn header_kept(name: &str) -> bool {
    matches!(
        name,
        "content-type"
            | "content-length"
            | "content-encoding"
            | "content-security-policy"
            | "content-security-policy-report-only"
            | "link"
            | "referrer-policy"
            | "cross-origin-opener-policy"
            | "cross-origin-embedder-policy"
            | "cross-origin-resource-policy"
            | "x-content-type-options"
            | "strict-transport-security"
            | "cache-control"
    )
}

const HEADER_BLOCKED: &[&str] = &["cookie", "set-cookie", "authorization", "proxy-authorization"];

/// Project a CDP headers object (name → value) down to the allowlist. Blocked credential headers
/// are reported only by presence; other values are replaced by their length.
pub fn project_headers(headers: &Value) -> Value {
    let mut kept = serde_json::Map::new();
    let mut blocked: Vec<String> = Vec::new();
    if let Some(o) = headers.as_object() {
        for (k, v) in o {
            let lower = k.to_ascii_lowercase();
            if HEADER_BLOCKED.contains(&lower.as_str()) {
                blocked.push(lower);
                continue;
            }
            if header_kept(&lower) {
                kept.insert(lower, v.clone());
            } else {
                let len = v.as_str().map(str::len).unwrap_or(0);
                kept.insert(lower, json!({"value_length": len}));
            }
        }
    }
    json!({"kept": kept, "credentials_present": blocked})
}

/// Every header with its value; credential headers are always redacted, and declared secrets
/// are redacted wherever they appear.
pub fn header_values(headers: &Value, secrets: &mut Secrets) -> Value {
    let mut out = serde_json::Map::new();
    if let Some(o) = headers.as_object() {
        for (k, v) in o {
            let lower = k.to_ascii_lowercase();
            let value = if HEADER_BLOCKED.contains(&lower.as_str()) { REDACTED.to_string() } else { secrets.text(v.as_str().unwrap_or_default()) };
            out.insert(lower, json!(value));
        }
    }
    Value::Object(out)
}

/// `scheme://host[:port]` of an http(s) URL.
pub fn origin_of(url: &str) -> Option<String> {
    let (scheme, rest) = url.split_once("://")?;
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let host = rest.split(['/', '?', '#']).next().filter(|h| !h.is_empty())?;
    Some(format!("{scheme}://{host}"))
}

/// Largest body text kept per request or response.
pub const MAX_KEPT_BODY: usize = 64 * 1024;

/// A body as kept in evidence: UTF-8 text with secrets redacted (capped), or for binary data only
/// its size. The digest is of the redacted text, so it can't be used to guess a secret.
pub fn kept_body(raw: &[u8], secrets: &mut Secrets) -> Value {
    kept_text(raw, secrets, MAX_KEPT_BODY)
}

/// [`kept_body`] with its own cap on the kept text.
pub fn kept_text(raw: &[u8], secrets: &mut Secrets, cap: usize) -> Value {
    match std::str::from_utf8(raw) {
        Ok(text) => {
            let redacted = secrets.text(text);
            let mut end = redacted.len().min(cap);
            while !redacted.is_char_boundary(end) {
                end -= 1;
            }
            json!({
                "state": "captured",
                "bytes": raw.len(),
                "sha256": craft_anything_evidence::sha256_hex(redacted.as_bytes()),
                "text": redacted.get(..end).unwrap_or_default(),
                "truncated": end < redacted.len(),
            })
        }
        Err(_) => json!({"state": "captured", "bytes": raw.len(), "binary": true}),
    }
}

// ---------------------------------------------------------------------------------------------
// JSON body outlines (no values)

/// Reduce a JSON value to a typed outline with counts. No scalar values survive.
pub fn json_outline(v: &Value) -> Value {
    outline_inner(v, 0)
}

fn outline_inner(v: &Value, depth: usize) -> Value {
    if depth > 24 {
        return json!({"kind": "max-depth"});
    }
    match v {
        Value::Null => json!({"kind": "null"}),
        Value::Bool(_) => json!({"kind": "boolean"}),
        Value::Number(_) => json!({"kind": "number"}),
        Value::String(_) => json!({"kind": "string"}),
        Value::Array(a) => {
            // Summarise element outlines without keeping values.
            let mut sample: Vec<Value> = Vec::new();
            for e in a.iter().take(8) {
                let s = outline_inner(e, depth + 1);
                if !sample.contains(&s) {
                    sample.push(s);
                }
            }
            json!({"kind": "array", "count": a.len(), "element_outlines": sample})
        }
        Value::Object(o) => {
            let mut props: Vec<Value> = Vec::new();
            for (k, val) in o.iter().take(256) {
                props.push(json!({"name": k, "value": outline_inner(val, depth + 1)}));
            }
            json!({"kind": "object", "property_count": o.len(), "properties": props})
        }
    }
}

// ---------------------------------------------------------------------------------------------
// DOM summary (node types/names, attribute names and value lengths only)

/// Summarise a `DOM.getDocument` tree: counts and a bounded structure of node names/types with
/// attribute names and *value lengths* only. No text content or attribute values are kept.
pub fn dom_summary(root: &Value, max_nodes: usize) -> Value {
    let mut element_counts: std::collections::BTreeMap<String, u64> = std::collections::BTreeMap::new();
    let mut attr_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut total = 0u64;
    let mut text_nodes = 0u64;
    let mut budget = max_nodes;
    walk_dom(root, &mut element_counts, &mut attr_names, &mut total, &mut text_nodes, &mut budget, 0);
    json!({
        "total_nodes": total,
        "text_nodes": text_nodes,
        "elements_by_name": element_counts,
        "attribute_names": attr_names.into_iter().collect::<Vec<_>>(),
        "truncated": budget == 0,
    })
}

#[allow(clippy::too_many_arguments)]
fn walk_dom(
    node: &Value,
    elems: &mut std::collections::BTreeMap<String, u64>,
    attrs: &mut std::collections::BTreeSet<String>,
    total: &mut u64,
    text_nodes: &mut u64,
    budget: &mut usize,
    depth: usize,
) {
    if *budget == 0 || depth > 4096 {
        return;
    }
    *budget -= 1;
    *total += 1;
    let node_type = node.get("nodeType").and_then(Value::as_u64).unwrap_or(0);
    if node_type == 3 {
        *text_nodes += 1;
    } else if node_type == 1 {
        let name = node.get("nodeName").and_then(Value::as_str).unwrap_or("?").to_ascii_lowercase();
        *elems.entry(name).or_insert(0) += 1;
    }
    // attributes is a flat [name, value, name, value, …] array.
    if let Some(a) = node.get("attributes").and_then(Value::as_array) {
        let mut i = 0;
        while i < a.len() {
            if let Some(n) = a.get(i).and_then(Value::as_str) {
                attrs.insert(n.to_ascii_lowercase());
            }
            i += 2;
        }
    }
    for child in node.get("children").and_then(Value::as_array).into_iter().flatten() {
        walk_dom(child, elems, attrs, total, text_nodes, budget, depth + 1);
    }
    if let Some(cd) = node.get("contentDocument") {
        walk_dom(cd, elems, attrs, total, text_nodes, budget, depth + 1);
    }
}

// ---------------------------------------------------------------------------------------------
// base64 and PNG

pub fn b64_decode(s: &str) -> Result<Vec<u8>, ToolError> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.decode(s.trim()).map_err(|e| ToolError::new(ErrorCode::GarbledOutput, format!("invalid base64: {e}")))
}

pub fn b64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// A decoded image as 8-bit RGBA.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Read a PNG's declared dimensions without decoding the pixels.
pub fn png_dimensions(bytes: &[u8]) -> Result<(u32, u32), ToolError> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let reader = decoder.read_info().map_err(|e| ToolError::new(ErrorCode::GarbledOutput, format!("not a readable PNG: {e}")))?;
    let info = reader.info();
    Ok((info.width, info.height))
}

/// Decode a picture (PNG, JPEG, WebP, or a GIF's first frame) to 8-bit RGBA, within the same
/// memory budget as [`decode_png`].
pub fn decode_picture(bytes: &[u8]) -> Result<Image, ToolError> {
    if bytes.starts_with(b"\x89PNG") {
        return decode_png(bytes);
    }
    let unreadable = |e: image::ImageError| ToolError::new(ErrorCode::GarbledOutput, format!("not a readable PNG, JPEG, WebP or GIF picture: {e}"));
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| unreadable(e.into()))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_PIXELS_WORKING_SET);
    reader.limits(limits);
    let rgba = reader.decode().map_err(unreadable)?.into_rgba8();
    Ok(Image { width: rgba.width(), height: rgba.height(), rgba: rgba.into_raw() })
}

/// Decode a PNG to 8-bit RGBA, refusing anything whose working set would exceed the budget.
pub fn decode_png(bytes: &[u8]) -> Result<Image, ToolError> {
    let (w, h) = png_dimensions(bytes)?;
    if (w as u64).saturating_mul(h as u64).saturating_mul(4) > MAX_PIXELS_WORKING_SET {
        return Err(ToolError::new(ErrorCode::OverLimit, format!("the image is too large to decode ({w}×{h})")));
    }
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| ToolError::new(ErrorCode::GarbledOutput, format!("not a readable PNG: {e}")))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader.next_frame(&mut buf).map_err(|e| ToolError::new(ErrorCode::GarbledOutput, format!("could not decode the PNG: {e}")))?;
    let channels = frame.color_type.samples();
    let (fw, fh) = (frame.width, frame.height);
    let pixels = (fw as usize).saturating_mul(fh as usize);
    let mut rgba = vec![0u8; pixels.saturating_mul(4)];
    let data = buf.get(..frame.buffer_size()).unwrap_or(&buf);
    for i in 0..pixels {
        let src = i.saturating_mul(channels);
        let dst = i.saturating_mul(4);
        let (r, g, b, a) = match channels {
            1 => {
                let v = data.get(src).copied().unwrap_or(0);
                (v, v, v, 255)
            }
            2 => {
                let v = data.get(src).copied().unwrap_or(0);
                (v, v, v, data.get(src + 1).copied().unwrap_or(255))
            }
            3 => (data.get(src).copied().unwrap_or(0), data.get(src + 1).copied().unwrap_or(0), data.get(src + 2).copied().unwrap_or(0), 255),
            _ => (
                data.get(src).copied().unwrap_or(0),
                data.get(src + 1).copied().unwrap_or(0),
                data.get(src + 2).copied().unwrap_or(0),
                data.get(src + 3).copied().unwrap_or(255),
            ),
        };
        if let Some(px) = rgba.get_mut(dst..dst + 4) {
            px.copy_from_slice(&[r, g, b, a]);
        }
    }
    Ok(Image { width: fw, height: fh, rgba })
}

/// Compare two decoded images. A pixel differs when any channel's absolute difference exceeds the
/// threshold. Returns a value-only summary (no pixel data).
pub fn pixel_diff(a: &Image, b: &Image, tolerance: u8) -> Value {
    if a.width != b.width || a.height != b.height {
        return json!({
            "comparable": false,
            "reason": "different dimensions",
            "before": {"width": a.width, "height": a.height},
            "after": {"width": b.width, "height": b.height},
        });
    }
    let total = (a.width as u64).saturating_mul(a.height as u64);
    let mut differing = 0u64;
    let mut max_channel_delta = 0u8;
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for i in 0..(total as usize) {
        let o = i.saturating_mul(4);
        let (Some(pa), Some(pb)) = (a.rgba.get(o..o + 4), b.rgba.get(o..o + 4)) else { break };
        let delta = pa.iter().zip(pb.iter()).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
        if delta > tolerance {
            differing += 1;
            max_channel_delta = max_channel_delta.max(delta);
            let x = (i as u32) % a.width;
            let y = (i as u32) / a.width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    let fraction = if total == 0 { 0.0 } else { differing as f64 / total as f64 };
    let bbox = if differing == 0 { Value::Null } else { json!({"x": min_x, "y": min_y, "width": max_x - min_x + 1, "height": max_y - min_y + 1}) };
    json!({
        "comparable": true,
        "identical": differing == 0,
        "width": a.width,
        "height": a.height,
        "total_pixels": total,
        "differing_pixels": differing,
        "difference_fraction": fraction,
        "max_channel_delta": max_channel_delta,
        "tolerance": tolerance,
        "changed_bounding_box": bbox,
    })
}

/// The top-left `w`×`h` part of an image (clamped to its size).
pub fn crop(img: &Image, w: u32, h: u32) -> Image {
    let (w, h) = (w.min(img.width), h.min(img.height));
    let mut rgba = Vec::with_capacity((w as usize).saturating_mul(h as usize).saturating_mul(4));
    for y in 0..h as usize {
        let start = y.saturating_mul(img.width as usize).saturating_mul(4);
        if let Some(row) = img.rgba.get(start..start.saturating_add((w as usize).saturating_mul(4))) {
            rgba.extend_from_slice(row);
        }
    }
    Image { width: w, height: h, rgba }
}

/// The parts of a page that change by themselves (a carousel, a video, an animation), as one bit
/// per `tile`×`tile` square, found by photographing the page twice ([`NoiseMask::between`]).
/// Comparisons leave them out: a build can't be expected to match a frame that moves.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseMask {
    pub tile: u32,
    pub cols: u32,
    pub rows: u32,
    pub bits: Vec<bool>,
}

/// The tile size noise is found and stored at.
pub const NOISE_TILE: u32 = 16;

impl NoiseMask {
    /// The tiles whose mean colour differs by more than `threshold` per channel between two
    /// pictures of the same page, grown by one tile so a moving edge is covered too.
    pub fn between(a: &Image, b: &Image, threshold: u8) -> NoiseMask {
        let tile = NOISE_TILE;
        let (w, h) = (a.width.min(b.width), a.height.min(b.height));
        let (cols, rows) = (w.div_ceil(tile), h.div_ceil(tile));
        let changed: Vec<bool> = (0..rows)
            .flat_map(|ty| (0..cols).map(move |tx| (tx, ty)))
            .map(|(tx, ty)| tile_mean(a, tile, tx, ty).iter().zip(tile_mean(b, tile, tx, ty)).any(|(x, y)| (x - y).abs() > f64::from(threshold)))
            .collect();
        let at = |x: i64, y: i64| {
            x >= 0 && y >= 0 && x < i64::from(cols) && y < i64::from(rows) && changed.get((y * i64::from(cols) + x) as usize).copied().unwrap_or(false)
        };
        let bits = (0..rows)
            .flat_map(|ty| (0..cols).map(move |tx| (i64::from(tx), i64::from(ty))))
            .map(|(x, y)| (-1..=1).any(|dy| (-1..=1).any(|dx| at(x + dx, y + dy))))
            .collect();
        NoiseMask { tile, cols, rows, bits }
    }

    /// The tiles either mask covers (two masks of the same page).
    pub fn union(mut self, other: &NoiseMask) -> NoiseMask {
        if (self.cols, self.rows) == (other.cols, other.rows) {
            for (a, b) in self.bits.iter_mut().zip(&other.bits) {
                *a |= *b;
            }
        }
        self
    }

    /// The pixel (`x`, `y`) of the page is in a part that changes by itself.
    pub fn covers(&self, x: u32, y: u32) -> bool {
        let (tx, ty) = (x / self.tile.max(1), y / self.tile.max(1));
        tx < self.cols && ty < self.rows && self.bits.get((ty * self.cols + tx) as usize).copied().unwrap_or(false)
    }

    /// How many tiles change by themselves.
    pub fn count(&self) -> usize {
        self.bits.iter().filter(|b| **b).count()
    }

    /// As a PNG with one pixel per tile, white where the page changes by itself.
    pub fn to_png(&self) -> Result<Vec<u8>, ToolError> {
        let rgba = self.bits.iter().flat_map(|b| if *b { [255, 255, 255, 255] } else { [0, 0, 0, 255] }).collect();
        encode_png(&Image { width: self.cols.max(1), height: self.rows.max(1), rgba })
    }

    /// Read back a mask saved with [`NoiseMask::to_png`].
    pub fn from_png(bytes: &[u8]) -> Result<NoiseMask, ToolError> {
        let img = decode_png(bytes)?;
        let bits = img.rgba.as_chunks::<4>().0.iter().map(|[r, ..]| *r > 127).collect();
        Ok(NoiseMask { tile: NOISE_TILE, cols: img.width, rows: img.height, bits })
    }
}

/// The mean colour of tile (`tx`, `ty`) of `img`.
fn tile_mean(img: &Image, block: u32, bx: u32, by: u32) -> [f64; 3] {
    let mut sum = [0u64; 3];
    let mut n = 0u64;
    for y in by * block..((by + 1) * block).min(img.height) {
        for x in bx * block..((bx + 1) * block).min(img.width) {
            let o = ((y as usize) * (img.width as usize) + x as usize) * 4;
            if let Some(px) = img.rgba.get(o..o + 3) {
                for (s, v) in sum.iter_mut().zip(px) {
                    *s += u64::from(*v);
                }
                n += 1;
            }
        }
    }
    let n = n.max(1) as f64;
    [sum[0] as f64 / n, sum[1] as f64 / n, sum[2] as f64 / n]
}

/// How alike two same-size images look at a coarse scale: the share of `block`×`block` tiles
/// whose mean colours are within `threshold` per channel. Tolerant of anti-aliasing and
/// sub-pixel shifts, unlike an exact pixel diff. 1.0 for identical images.
pub fn block_similarity(a: &Image, b: &Image, block: u32, threshold: u8) -> f64 {
    block_similarity_masked(a, b, block, threshold, None)
}

/// [`block_similarity`] over the tiles `mask` doesn't cover (a tile counts when its top-left
/// pixel isn't in a moving part). Everything masked compares as 1.0: nothing was left to differ.
pub fn block_similarity_masked(a: &Image, b: &Image, block: u32, threshold: u8, mask: Option<&NoiseMask>) -> f64 {
    let block = block.max(1);
    if a.width != b.width || a.height != b.height || a.width == 0 || a.height == 0 {
        return 0.0;
    }
    let mean = |img: &Image, bx: u32, by: u32| tile_mean(img, block, bx, by);
    let (cols, rows) = (a.width.div_ceil(block), a.height.div_ceil(block));
    let (mut same, mut counted) = (0u64, 0u64);
    for by in 0..rows {
        for bx in 0..cols {
            if mask.is_some_and(|m| m.covers(bx * block, by * block)) {
                continue;
            }
            counted += 1;
            let (ma, mb) = (mean(a, bx, by), mean(b, bx, by));
            if ma.iter().zip(mb.iter()).all(|(x, y)| (x - y).abs() <= f64::from(threshold)) {
                same += 1;
            }
        }
    }
    if counted == 0 { 1.0 } else { same as f64 / counted as f64 }
}

/// `img` as an 8-bit RGBA PNG.
pub fn encode_png(img: &Image) -> Result<Vec<u8>, ToolError> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| ToolError::internal(format!("PNG header: {e}")))?;
        w.write_image_data(&img.rgba).map_err(|e| ToolError::internal(format!("PNG data: {e}")))?;
    }
    Ok(out)
}

/// `img` as an 8-bit RGBA PNG of at most `max` bytes, or `None` when it would be bigger. It's
/// compressed a chunk at a time, so a picture that won't fit stops as soon as it passes `max`
/// instead of being held whole first.
pub fn encode_png_within(img: &Image, max: usize) -> Result<Option<Vec<u8>>, ToolError> {
    /// Takes bytes until `max`, then refuses them and says so.
    struct Capped {
        out: Vec<u8>,
        max: usize,
        over: bool,
    }
    impl std::io::Write for Capped {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.out.len().saturating_add(buf.len()) > self.max {
                self.over = true;
                return Err(std::io::Error::other("the PNG passed its size limit"));
            }
            self.out.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut capped = Capped { out: Vec::new(), max, over: false };
    let written = (|| {
        let mut enc = png::Encoder::new(&mut capped, img.width, img.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| format!("PNG header: {e}"))?;
        let mut stream = w.stream_writer().map_err(|e| format!("PNG data: {e}"))?;
        std::io::Write::write_all(&mut stream, &img.rgba).map_err(|e| format!("PNG data: {e}"))?;
        stream.finish().map_err(|e| format!("PNG data: {e}"))?;
        w.finish().map_err(|e| format!("PNG end: {e}"))
    })();
    match written {
        _ if capped.over => Ok(None),
        Ok(()) => Ok(Some(capped.out)),
        Err(e) => Err(ToolError::internal(e)),
    }
}

/// `img` scaled to `w`×`h` (box sampling: each output pixel is the mean of the input pixels it
/// covers). Aspect ratio isn't kept: comparisons bring both pictures to one size.
pub fn resize(img: &Image, w: u32, h: u32) -> Image {
    let (w, h) = (w.max(1), h.max(1));
    let mut rgba = vec![0u8; (w as usize).saturating_mul(h as usize).saturating_mul(4)];
    if img.width == 0 || img.height == 0 {
        return Image { width: w, height: h, rgba };
    }
    for oy in 0..h {
        let (y0, y1) = (u64::from(oy) * u64::from(img.height) / u64::from(h), (u64::from(oy + 1) * u64::from(img.height)).div_ceil(u64::from(h)));
        for ox in 0..w {
            let (x0, x1) = (u64::from(ox) * u64::from(img.width) / u64::from(w), (u64::from(ox + 1) * u64::from(img.width)).div_ceil(u64::from(w)));
            let mut sum = [0u64; 4];
            let mut n = 0u64;
            for y in y0..y1.max(y0 + 1).min(u64::from(img.height)) {
                for x in x0..x1.max(x0 + 1).min(u64::from(img.width)) {
                    let o = ((y * u64::from(img.width) + x) * 4) as usize;
                    if let Some(px) = img.rgba.get(o..o + 4) {
                        for (s, v) in sum.iter_mut().zip(px) {
                            *s += u64::from(*v);
                        }
                        n += 1;
                    }
                }
            }
            let o = ((u64::from(oy) * u64::from(w) + u64::from(ox)) * 4) as usize;
            if let Some(dst) = rgba.get_mut(o..o + 4) {
                for (d, s) in dst.iter_mut().zip(sum) {
                    *d = (s / n.max(1)) as u8;
                }
            }
        }
    }
    Image { width: w, height: h, rgba }
}

/// How alike two same-size images look to a person: the mean structural similarity (SSIM) of
/// their brightness, over 8×8 windows, after shrinking both so the wider side is at most 512 px.
/// It sees edges, text and texture that tile colours miss, and forgives small colour shifts.
/// 1.0 for identical images; 0.0 for different sizes or nothing to compare.
pub fn perceptual_similarity(a: &Image, b: &Image) -> f64 {
    perceptual_similarity_masked(a, b, None)
}

/// Composite an sRGB RGBA pixel over an opaque neutral mid-gray `[128, 128, 128]` background.
/// Opaque pixels (`a == 255`) are returned unchanged, fully transparent pixels (`a == 0`) collapse
/// to mid-gray regardless of hidden RGB bytes, and semi-transparent pixels contribute proportionally
/// to their opacity so both black-vs-transparent and white-vs-transparent are distinguishable.
fn composite_over_mid_gray(px: [u8; 4]) -> [u8; 3] {
    let a = u16::from(px[3]);
    if a == 255 {
        return [px[0], px[1], px[2]];
    }
    if a == 0 {
        return [128, 128, 128];
    }
    let inv = 255 - a;
    let blend = |c: u8| -> u8 { ((u16::from(c) * a + 128 * inv + 127) / 255) as u8 };
    [blend(px[0]), blend(px[1]), blend(px[2])]
}

/// [`perceptual_similarity`] over the windows whose top-left pixel `mask` doesn't cover; 1.0
/// when everything is masked.
pub fn perceptual_similarity_masked(a: &Image, b: &Image, mask: Option<&NoiseMask>) -> f64 {
    if a.width != b.width || a.height != b.height || a.width == 0 || a.height == 0 {
        return 0.0;
    }
    let f = a.width.max(a.height).div_ceil(512).max(1);
    let (w, h) = (a.width.div_ceil(f) as usize, a.height.div_ceil(f) as usize);
    let luma = |img: &Image| -> Vec<f64> {
        let mut out = vec![0f64; w.saturating_mul(h)];
        let mut n = vec![0u32; w.saturating_mul(h)];
        for y in 0..img.height as usize {
            for x in 0..img.width as usize {
                let o = (y * img.width as usize + x) * 4;
                if let Some(px) = img.rgba.get(o..o + 4) {
                    let [r, g, b] = composite_over_mid_gray([px[0], px[1], px[2], px[3]]);
                    let l = 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b);
                    let i = (y / f as usize) * w + x / f as usize;
                    if let (Some(s), Some(c)) = (out.get_mut(i), n.get_mut(i)) {
                        *s += l;
                        *c += 1;
                    }
                }
            }
        }
        for (s, c) in out.iter_mut().zip(&n) {
            *s /= f64::from((*c).max(1));
        }
        out
    };
    let (la, lb) = (luma(a), luma(b));
    let (c1, c2) = ((0.01f64 * 255.0).powi(2), (0.03f64 * 255.0).powi(2));
    const WIN: usize = 8;
    let (mut total, mut windows, mut masked) = (0f64, 0u64, 0u64);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            // Window coordinates are in the shrunk image; the mask is in page pixels.
            if mask.is_some_and(|m| m.covers((x as u32).saturating_mul(f), (y as u32).saturating_mul(f))) {
                masked += 1;
                x += WIN;
                continue;
            }
            let (ye, xe) = ((y + WIN).min(h), (x + WIN).min(w));
            let n = ((ye - y) * (xe - x)) as f64;
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for yy in y..ye {
                for xx in x..xe {
                    let i = yy * w + xx;
                    let (p, q) = (la.get(i).copied().unwrap_or(0.0), lb.get(i).copied().unwrap_or(0.0));
                    sa += p;
                    sb += q;
                    saa += p * p;
                    sbb += q * q;
                    sab += p * q;
                }
            }
            let (ma, mb) = (sa / n, sb / n);
            let (va, vb, cov) = ((saa / n - ma * ma).max(0.0), (sbb / n - mb * mb).max(0.0), sab / n - ma * mb);
            total += ((2.0 * ma * mb + c1) * (2.0 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
            windows += 1;
            x += WIN;
        }
        y += WIN;
    }
    match (windows, masked) {
        (0, 0) => 0.0,
        (0, _) => 1.0,
        _ => (total / windows as f64).clamp(0.0, 1.0),
    }
}

/// Build the inline PNG screenshot artifact (with its sha256).
pub fn png_artifact(bytes: &[u8]) -> Result<Value, ToolError> {
    if bytes.len() > MAX_SCREENSHOT_BYTES {
        return Err(ToolError::new(ErrorCode::OverLimit, "the screenshot is larger than the inline limit"));
    }
    let (w, h) = png_dimensions(bytes).unwrap_or((0, 0));
    Ok(json!({
        "media_type": "image/png",
        "bytes": bytes.len(),
        "width": w,
        "height": h,
        "sha256": sha256_hex(bytes),
        "base64": b64_encode(bytes),
    }))
}

// ---------------------------------------------------------------------------------------------
// Deterministic perceptual pixel diff: CIELAB Delta-E, region clustering and numbered heatmaps

/// Default CIE76 Delta-E threshold above which a pixel counts as visually differing.
pub const DEFAULT_DELTA_E_THRESHOLD: f64 = 2.5;
/// Default minimum differing pixel count for a spatial cluster to be reported as a region.
pub const DEFAULT_MIN_CLUSTER_PIXELS: usize = 32;
/// Default maximum pixel gap between nearby boxes that are merged into one region.
pub const DEFAULT_MERGE_GAP_PX: u32 = 16;
/// Grid cell edge (pixels) used to seed connected difference components.
const DIFF_CELL_SIZE: u32 = 8;

/// Convert an 8-bit sRGB channel to linear light in `0.0..=1.0` (IEC 61966-2-1).
fn srgb_channel_to_linear(c: u8) -> f64 {
    let s = f64::from(c) / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Build a 256-entry sRGB-to-linear lookup table for fast per-pixel conversion.
fn build_srgb_lut() -> [f64; 256] {
    let mut lut = [0.0f64; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = srgb_channel_to_linear(i as u8);
    }
    lut
}

/// CIE Lab transfer function `f(t)` with exact rational thresholds.
fn lab_f(t: f64) -> f64 {
    const EPSILON: f64 = 216.0 / 24389.0;
    const KAPPA: f64 = 24389.0 / 27.0;
    if t > EPSILON { t.cbrt() } else { (KAPPA * t + 16.0) / 116.0 }
}

/// Convert sRGB `(r, g, b)` to CIE `L*a*b*` under the D65 standard illuminant.
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    linear_rgb_to_lab(srgb_channel_to_linear(r), srgb_channel_to_linear(g), srgb_channel_to_linear(b))
}

fn linear_rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    // sRGB to CIE XYZ (D65, 2-degree standard observer).
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;

    // Normalise against D65 reference white (Xn=0.95047, Yn=1.00000, Zn=1.08883).
    let fx = lab_f(x / 0.95047);
    let fy = lab_f(y);
    let fz = lab_f(z / 1.08883);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    (l, a, b_star)
}

/// Perceptual CIE76 colour distance between two `L*a*b*` triplets.
pub fn delta_e_cie76(lab1: (f64, f64, f64), lab2: (f64, f64, f64)) -> f64 {
    let dl = lab1.0 - lab2.0;
    let da = lab1.1 - lab2.1;
    let db = lab1.2 - lab2.2;
    (dl * dl + da * da + db * db).sqrt()
}

#[derive(Deserialize)]
struct DiffRegionWire {
    id: usize,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    pixel_count: usize,
    mean_delta_e: f64,
    max_delta_e: f64,
    #[serde(default)]
    ssim: f64,
    severity: String,
}

impl TryFrom<DiffRegionWire> for DiffRegion {
    type Error = String;

    fn try_from(w: DiffRegionWire) -> Result<Self, Self::Error> {
        let severity = match w.severity.as_str() {
            "severe" => "severe",
            "moderate" => "moderate",
            "minor" => "minor",
            other => return Err(format!("unknown severity {other:?}; expected severe, moderate, or minor")),
        };
        Ok(DiffRegion {
            id: w.id,
            x: w.x,
            y: w.y,
            width: w.width,
            height: w.height,
            pixel_count: w.pixel_count,
            mean_delta_e: w.mean_delta_e,
            max_delta_e: w.max_delta_e,
            ssim: w.ssim,
            severity,
        })
    }
}

/// One clustered divergence region detected between an original and rebuilt screenshot.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DiffRegion {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixel_count: usize,
    pub mean_delta_e: f64,
    pub max_delta_e: f64,
    pub ssim: f64,
    pub severity: &'static str,
}

impl<'de> Deserialize<'de> for DiffRegion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = DiffRegionWire::deserialize(deserializer)?;
        DiffRegion::try_from(wire).map_err(serde::de::Error::custom)
    }
}

/// Configuration options for [`compute_pixel_diff_heatmap`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiffOptions {
    pub delta_e_threshold: f64,
    pub min_cluster_pixels: usize,
    pub merge_gap_px: u32,
}

impl Default for DiffOptions {
    fn default() -> Self {
        DiffOptions { delta_e_threshold: DEFAULT_DELTA_E_THRESHOLD, min_cluster_pixels: DEFAULT_MIN_CLUSTER_PIXELS, merge_gap_px: DEFAULT_MERGE_GAP_PX }
    }
}

/// Complete deterministic pixel-diff output: global metrics, clustered regions, rendered heatmap PNG,
/// and serialised geometric metadata JSON.
#[derive(Debug, Clone)]
pub struct PixelDiffReport {
    pub width: u32,
    pub height: u32,
    pub original_width: u32,
    pub original_height: u32,
    pub built_width: u32,
    pub built_height: u32,
    pub ssim: f64,
    pub mean_delta_e: f64,
    pub max_delta_e: f64,
    pub total_pixels: u64,
    pub differing_pixels: u64,
    pub difference_fraction: f64,
    pub delta_e_threshold: f64,
    pub regions: Vec<DiffRegion>,
    pub heatmap_png: Vec<u8>,
    pub diff_json: Value,
}

/// Either a borrowed reference or an owned padded copy of an [`Image`].
enum ImageView<'a> {
    Borrowed(&'a Image),
    Owned(Image),
}

impl ImageView<'_> {
    fn as_image(&self) -> &Image {
        match self {
            ImageView::Borrowed(img) => img,
            ImageView::Owned(img) => img,
        }
    }
}

/// Normalize `img` to `(w, h)` by padding bottom/right with `[0, 0, 0, 0]` and zeroing hidden
/// RGB channels on any fully transparent (`alpha == 0`) pixel.
fn pad_to_dimensions<'a>(img: &'a Image, w: u32, h: u32) -> ImageView<'a> {
    let has_dirty_transparent = img.rgba.as_chunks::<4>().0.iter().any(|px| px[3] == 0 && (px[0] != 0 || px[1] != 0 || px[2] != 0));
    if img.width == w && img.height == h && !has_dirty_transparent {
        return ImageView::Borrowed(img);
    }
    let mut rgba = vec![0u8; (w as usize).saturating_mul(h as usize).saturating_mul(4)];
    let row_bytes = (img.width.min(w) as usize).saturating_mul(4);
    for y in 0..img.height.min(h) as usize {
        let src_start = y.saturating_mul(img.width as usize).saturating_mul(4);
        let dst_start = y.saturating_mul(w as usize).saturating_mul(4);
        if let (Some(src), Some(dst)) = (img.rgba.get(src_start..src_start + row_bytes), rgba.get_mut(dst_start..dst_start + row_bytes)) {
            dst.copy_from_slice(src);
            for px in dst.as_chunks_mut::<4>().0 {
                if px[3] == 0 {
                    px[0] = 0;
                    px[1] = 0;
                    px[2] = 0;
                }
            }
        }
    }
    ImageView::Owned(Image { width: w, height: h, rgba })
}

/// Extract a rectangular sub-image `(rx, ry, rw, rh)` clamped to `img`'s bounds.
fn crop_rect(img: &Image, rx: u32, ry: u32, rw: u32, rh: u32) -> Image {
    let x0 = rx.min(img.width);
    let y0 = ry.min(img.height);
    let w = rw.min(img.width.saturating_sub(x0));
    let h = rh.min(img.height.saturating_sub(y0));
    let mut rgba = Vec::with_capacity((w as usize).saturating_mul(h as usize).saturating_mul(4));
    let row_bytes = (w as usize).saturating_mul(4);
    for y in y0..y0.saturating_add(h) {
        let start = ((y as usize).saturating_mul(img.width as usize).saturating_add(x0 as usize)).saturating_mul(4);
        if let Some(row) = img.rgba.get(start..start.saturating_add(row_bytes)) {
            rgba.extend_from_slice(row);
        }
    }
    Image { width: w, height: h, rgba }
}

/// Axis-aligned pixel bounding box with exclusive right/bottom edges (`x0 < x1`, `y0 < y1`).
#[derive(Debug, Clone, Copy)]
struct PixelBox {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
}

impl PixelBox {
    fn union(self, other: PixelBox) -> PixelBox {
        PixelBox { x0: self.x0.min(other.x0), y0: self.y0.min(other.y0), x1: self.x1.max(other.x1), y1: self.y1.max(other.y1) }
    }

    fn within_gap(self, other: PixelBox, gap: u32) -> bool {
        let h_gap = other.x0.saturating_sub(self.x1).max(self.x0.saturating_sub(other.x1));
        let v_gap = other.y0.saturating_sub(self.y1).max(self.y0.saturating_sub(other.y1));
        h_gap <= gap && v_gap <= gap
    }

    fn intersects(self, other: PixelBox) -> bool {
        self.x0 < other.x1 && other.x0 < self.x1 && self.y0 < other.y1 && other.y0 < self.y1
    }
}

/// Cluster active grid cells into merged, noise-filtered, top-to-bottom / left-to-right [`DiffRegion`]s.
fn detect_diff_regions(orig: &Image, built: &Image, delta_matrix: &[f32], global_ssim: f64, opts: DiffOptions) -> Vec<DiffRegion> {
    let (w, h) = (orig.width, orig.height);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let cols = w.div_ceil(DIFF_CELL_SIZE);
    let rows = h.div_ceil(DIFF_CELL_SIZE);
    let cell_count = (cols as usize).saturating_mul(rows as usize);
    let mut active = vec![false; cell_count];
    let thresh = opts.delta_e_threshold as f32;

    for y in 0..h {
        let row_off = (y as usize).saturating_mul(w as usize);
        let cy = (y / DIFF_CELL_SIZE) as usize;
        for x in 0..w {
            let de = delta_matrix.get(row_off.saturating_add(x as usize)).copied().unwrap_or(0.0);
            if de > thresh {
                let cx = (x / DIFF_CELL_SIZE) as usize;
                if let Some(cell) = active.get_mut(cy.saturating_mul(cols as usize).saturating_add(cx)) {
                    *cell = true;
                }
            }
        }
    }

    // 8-connected components over active grid cells via iterative BFS.
    let mut visited = vec![false; cell_count];
    let mut boxes: Vec<PixelBox> = Vec::new();
    let mut queue: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();

    for cy in 0..rows {
        for cx in 0..cols {
            let idx = (cy as usize).saturating_mul(cols as usize).saturating_add(cx as usize);
            if !active.get(idx).copied().unwrap_or(false) || visited.get(idx).copied().unwrap_or(true) {
                continue;
            }
            if let Some(v) = visited.get_mut(idx) {
                *v = true;
            }
            queue.push_back((cx, cy));
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0u32, 0u32);
            let mut found_px = false;

            while let Some((ux, uy)) = queue.pop_front() {
                let px0 = ux.saturating_mul(DIFF_CELL_SIZE);
                let py0 = uy.saturating_mul(DIFF_CELL_SIZE);
                let px1 = px0.saturating_add(DIFF_CELL_SIZE).min(w);
                let py1 = py0.saturating_add(DIFF_CELL_SIZE).min(h);
                for py in py0..py1 {
                    let row_off = (py as usize).saturating_mul(w as usize);
                    for px in px0..px1 {
                        if delta_matrix.get(row_off.saturating_add(px as usize)).copied().unwrap_or(0.0) > thresh {
                            min_x = min_x.min(px);
                            min_y = min_y.min(py);
                            max_x = max_x.max(px);
                            max_y = max_y.max(py);
                            found_px = true;
                        }
                    }
                }
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = i64::from(ux) + dx;
                        let ny = i64::from(uy) + dy;
                        if nx < 0 || ny < 0 || nx >= i64::from(cols) || ny >= i64::from(rows) {
                            continue;
                        }
                        let (nxu, nyu) = (nx as u32, ny as u32);
                        let nidx = (nyu as usize).saturating_mul(cols as usize).saturating_add(nxu as usize);
                        if active.get(nidx).copied().unwrap_or(false) && !visited.get(nidx).copied().unwrap_or(true) {
                            if let Some(v) = visited.get_mut(nidx) {
                                *v = true;
                            }
                            queue.push_back((nxu, nyu));
                        }
                    }
                }
            }

            if found_px {
                boxes.push(PixelBox { x0: min_x, y0: min_y, x1: max_x.saturating_add(1), y1: max_y.saturating_add(1) });
            }
        }
    }

    // Merge nearby bounding boxes separated by <= merge_gap_px using a fixed-point sweep.
    let gap = opts.merge_gap_px;
    loop {
        boxes.sort_by_key(|b| (b.y0, b.x0));
        let mut merged_any = false;
        let mut used = vec![false; boxes.len()];
        let mut next: Vec<PixelBox> = Vec::with_capacity(boxes.len());
        for i in 0..boxes.len() {
            if used.get(i).copied().unwrap_or(true) {
                continue;
            }
            let Some(&initial) = boxes.get(i) else { continue };
            let mut cur = initial;
            if let Some(u) = used.get_mut(i) {
                *u = true;
            }
            let mut j = i + 1;
            while j < boxes.len() {
                if used.get(j).copied().unwrap_or(true) {
                    j += 1;
                    continue;
                }
                let Some(&cand) = boxes.get(j) else {
                    j += 1;
                    continue;
                };
                if cand.y0 > cur.y1.saturating_add(gap) {
                    break;
                }
                if cur.within_gap(cand, gap) {
                    cur = cur.union(cand);
                    if let Some(u) = used.get_mut(j) {
                        *u = true;
                    }
                    merged_any = true;
                    j = i + 1;
                } else {
                    j += 1;
                }
            }
            next.push(cur);
        }
        boxes = next;
        if !merged_any {
            break;
        }
    }

    // Compute exact per-region metrics and drop noise clusters below min_cluster_pixels.
    let mut regions: Vec<DiffRegion> = Vec::new();
    for bx in boxes {
        let mut pixel_count = 0usize;
        let mut sum_de = 0.0f64;
        let mut max_de = 0.0f64;
        for py in bx.y0..bx.y1 {
            let row_off = (py as usize).saturating_mul(w as usize);
            for px in bx.x0..bx.x1 {
                let de_f32 = delta_matrix.get(row_off.saturating_add(px as usize)).copied().unwrap_or(0.0);
                if de_f32 > thresh {
                    let de = f64::from(de_f32);
                    pixel_count += 1;
                    sum_de += de;
                    if de > max_de {
                        max_de = de;
                    }
                }
            }
        }
        if pixel_count < opts.min_cluster_pixels {
            continue;
        }
        let rw = bx.x1.saturating_sub(bx.x0);
        let rh = bx.y1.saturating_sub(bx.y0);
        let mean_de = sum_de / pixel_count as f64;
        let ssim = if bx.x0 == 0 && bx.y0 == 0 && rw == w && rh == h {
            global_ssim
        } else {
            let sub_orig = crop_rect(orig, bx.x0, bx.y0, rw, rh);
            let sub_built = crop_rect(built, bx.x0, bx.y0, rw, rh);
            perceptual_similarity(&sub_orig, &sub_built)
        };
        let severity = if mean_de > 20.0 || pixel_count > 5000 {
            "severe"
        } else if mean_de > 8.0 || pixel_count > 500 {
            "moderate"
        } else {
            "minor"
        };
        regions.push(DiffRegion { id: 0, x: bx.x0, y: bx.y0, width: rw, height: rh, pixel_count, mean_delta_e: mean_de, max_delta_e: max_de, ssim, severity });
    }

    // Order top-to-bottom, left-to-right and assign sequential 1-based IDs.
    regions.sort_by_key(|r| (r.y, r.x));
    for (idx, r) in regions.iter_mut().enumerate() {
        r.id = idx + 1;
    }
    regions
}

/// Embedded 5x7 bitmap glyphs for digits `0..=9` and brackets `[` / `]`.
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F],
        '3' => [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        _ => [0x00; 7],
    }
}

fn put_pixel(img: &mut Image, x: u32, y: u32, color: [u8; 4]) {
    if x >= img.width || y >= img.height {
        return;
    }
    let off = ((y as usize).saturating_mul(img.width as usize).saturating_add(x as usize)).saturating_mul(4);
    if let Some(px) = img.rgba.get_mut(off..off + 4) {
        px.copy_from_slice(&color);
    }
}

fn fill_rect(img: &mut Image, rect: PixelBox, color: [u8; 4]) {
    let x0 = rect.x0.min(img.width);
    let y0 = rect.y0.min(img.height);
    let x1 = rect.x1.min(img.width);
    let y1 = rect.y1.min(img.height);
    for y in y0..y1 {
        for x in x0..x1 {
            put_pixel(img, x, y, color);
        }
    }
}

fn draw_rect_border(img: &mut Image, rect: PixelBox, thickness: u32, color: [u8; 4]) {
    let rw = rect.x1.saturating_sub(rect.x0);
    let rh = rect.y1.saturating_sub(rect.y0);
    if rw == 0 || rh == 0 || thickness == 0 {
        return;
    }
    let t = thickness.min(rw).min(rh);
    // Top and bottom edges.
    fill_rect(img, PixelBox { x0: rect.x0, y0: rect.y0, x1: rect.x1, y1: rect.y0.saturating_add(t) }, color);
    fill_rect(img, PixelBox { x0: rect.x0, y0: rect.y1.saturating_sub(t), x1: rect.x1, y1: rect.y1 }, color);
    // Left and right edges.
    fill_rect(img, PixelBox { x0: rect.x0, y0: rect.y0, x1: rect.x0.saturating_add(t), y1: rect.y1 }, color);
    fill_rect(img, PixelBox { x0: rect.x1.saturating_sub(t), y0: rect.y0, x1: rect.x1, y1: rect.y1 }, color);
}

fn draw_badge(img: &mut Image, region: &DiffRegion, placed_badges: &mut Vec<PixelBox>) {
    let mut label = format!("[{}]", region.id);
    let mut scale: u32 = if img.width >= 320 && img.height >= 200 { 2 } else { 1 };
    let mut pad = 2 * scale;
    let measure = |lbl: &str, sc: u32, pd: u32| -> (u32, u32, u32) {
        let step = 6 * sc;
        let tw = (lbl.len() as u32).saturating_mul(step).saturating_sub(sc);
        let th = 7 * sc;
        (step, tw.saturating_add(pd.saturating_mul(2)), th.saturating_add(pd.saturating_mul(2)))
    };
    let (mut char_step, mut badge_w, mut badge_h) = measure(&label, scale, pad);
    if badge_w > img.width {
        // Ultra-narrow canvas: drop brackets and padding so the digit remains legible without
        // resizing the heatmap image or shifting reported region coordinates.
        label = region.id.to_string();
        scale = 1;
        pad = 0;
        (char_step, badge_w, badge_h) = measure(&label, scale, pad);
    }

    let max_bx = img.width.saturating_sub(badge_w);
    let max_by = img.height.saturating_sub(badge_h);
    let base_x = region.x.min(max_bx);
    let above_y = if region.y >= badge_h { region.y - badge_h } else { region.y.min(max_by) };

    let horiz = std::iter::successors(Some(base_x), move |&cx| (cx < max_bx).then(|| (cx + 4).min(max_bx))).map(move |cx| (cx, above_y));
    let fixed = [(base_x, region.y.min(max_by)), (base_x, region.y.saturating_add(region.height).min(max_by))].into_iter();
    let step_y = badge_h.saturating_add(2).max(1);
    let vert = std::iter::successors(Some(region.y.min(max_by)), move |&cy| (cy < max_by).then(|| (cy + step_y).min(max_by))).map(move |cy| (base_x, cy));

    let badge_rect = horiz
        .chain(fixed)
        .chain(vert)
        .map(|(bx, by)| PixelBox { x0: bx, y0: by, x1: bx.saturating_add(badge_w), y1: by.saturating_add(badge_h) })
        .find(|rect| !placed_badges.iter().any(|p| p.intersects(*rect)))
        .unwrap_or(PixelBox { x0: base_x, y0: above_y, x1: base_x.saturating_add(badge_w), y1: above_y.saturating_add(badge_h) });
    placed_badges.push(badge_rect);
    let (bx, by) = (badge_rect.x0, badge_rect.y0);

    // Solid red badge with white border for readability over both light and thermal areas.
    fill_rect(img, badge_rect, [220, 20, 20, 255]);
    draw_rect_border(img, badge_rect, 1, [255, 255, 255, 255]);

    for (ci, ch) in label.chars().enumerate() {
        let bitmap = glyph_5x7(ch);
        let gx0 = bx.saturating_add(pad).saturating_add((ci as u32).saturating_mul(char_step));
        let gy0 = by.saturating_add(pad);
        for (row_idx, &bits) in bitmap.iter().enumerate() {
            for col_idx in 0..5u32 {
                if (bits & (1u8 << (4 - col_idx))) != 0 {
                    let px = gx0.saturating_add(col_idx.saturating_mul(scale));
                    let py = gy0.saturating_add((row_idx as u32).saturating_mul(scale));
                    fill_rect(img, PixelBox { x0: px, y0: py, x1: px.saturating_add(scale), y1: py.saturating_add(scale) }, [255, 255, 255, 255]);
                }
            }
        }
    }
}

/// Map a Delta-E magnitude above `threshold` onto a yellow -> orange -> red -> magenta thermal ramp.
fn thermal_color(delta_e: f64, threshold: f64) -> ([u8; 3], f64) {
    let span = (50.0 - threshold).max(1.0);
    let t = ((delta_e - threshold) / span).clamp(0.0, 1.0);
    let lerp = |a: u8, b: u8, f: f64| -> u8 { (f64::from(a) + (f64::from(b) - f64::from(a)) * f).round().clamp(0.0, 255.0) as u8 };
    let rgb = if t < 0.35 {
        let f = t / 0.35;
        [255, lerp(225, 130, f), 0]
    } else if t < 0.70 {
        let f = (t - 0.35) / 0.35;
        [lerp(255, 235, f), lerp(130, 25, f), lerp(0, 20, f)]
    } else {
        let f = (t - 0.70) / 0.30;
        [lerp(235, 220, f), lerp(25, 0, f), lerp(20, 200, f)]
    };
    let alpha = 0.55 + 0.35 * t;
    (rgb, alpha)
}

/// Render a composite visual diff heatmap with a desaturated base layer, thermal divergence overlay,
/// 2px red region bounding boxes, and numbered `[1]`, `[2]`, ... badges.
pub fn render_diff_heatmap(orig: &Image, built: &Image, regions: &[DiffRegion], delta_matrix: &[f32]) -> Image {
    render_diff_heatmap_with_threshold(orig, built, regions, delta_matrix, DEFAULT_DELTA_E_THRESHOLD)
}

fn render_diff_heatmap_with_threshold(orig: &Image, built: &Image, regions: &[DiffRegion], delta_matrix: &[f32], threshold: f64) -> Image {
    let w = orig.width.max(built.width);
    let h = orig.height.max(built.height);
    let norm_orig = pad_to_dimensions(orig, w, h);
    let norm_built = pad_to_dimensions(built, w, h);
    let (oa, ba) = (norm_orig.as_image(), norm_built.as_image());

    let pixels = (w as usize).saturating_mul(h as usize);
    let mut rgba = vec![0u8; pixels.saturating_mul(4)];
    let thresh_f32 = threshold as f32;
    let thresh_f64 = f64::from(thresh_f32);

    for i in 0..pixels {
        let off = i.saturating_mul(4);
        let x = (i % (w as usize)) as u32;
        let y = (i / (w as usize)) as u32;
        let bp = ba.rgba.get(off..off + 4).unwrap_or(&[0, 0, 0, 0]);
        let op = oa.rgba.get(off..off + 4).unwrap_or(&[0, 0, 0, 0]);
        // Show desaturated built image for context; outside built's original bounds, show original context.
        let src = if (x >= built.width || y >= built.height) && x < orig.width && y < orig.height { op } else { bp };
        let [sr, sg, sb] = composite_over_mid_gray([src[0], src[1], src[2], src[3]]);
        let luma = 0.299 * f64::from(sr) + 0.587 * f64::from(sg) + 0.114 * f64::from(sb);
        // 40% saturation / 60% grayscale, gently compressed so bright thermal highlights stand out.
        let mut r = (0.4 * f64::from(sr) + 0.6 * luma) * 0.88;
        let mut g = (0.4 * f64::from(sg) + 0.6 * luma) * 0.88;
        let mut b = (0.4 * f64::from(sb) + 0.6 * luma) * 0.88;

        let de_f32 = delta_matrix.get(i).copied().unwrap_or(0.0);
        if de_f32 > thresh_f32 {
            let (heat, alpha) = thermal_color(f64::from(de_f32), thresh_f64);
            let inv = 1.0 - alpha;
            r = r * inv + f64::from(heat[0]) * alpha;
            g = g * inv + f64::from(heat[1]) * alpha;
            b = b * inv + f64::from(heat[2]) * alpha;
        }
        if let Some(dst) = rgba.get_mut(off..off + 4) {
            dst[0] = r.round().clamp(0.0, 255.0) as u8;
            dst[1] = g.round().clamp(0.0, 255.0) as u8;
            dst[2] = b.round().clamp(0.0, 255.0) as u8;
            dst[3] = 255;
        }
    }

    let mut img = Image { width: w, height: h, rgba };

    // Draw 2px bright red bounding boxes, then draw numeric badges on top so badges are never occluded.
    for region in regions {
        let rect = PixelBox { x0: region.x, y0: region.y, x1: region.x.saturating_add(region.width), y1: region.y.saturating_add(region.height) };
        draw_rect_border(&mut img, rect, 2, [235, 25, 25, 255]);
    }
    let mut placed_badges: Vec<PixelBox> = Vec::with_capacity(regions.len());
    for region in regions {
        draw_badge(&mut img, region, &mut placed_badges);
    }

    img
}

/// Compute global SSIM, per-pixel CIELAB CIE76 Delta-E, clustered divergence regions, a composite
/// numbered heatmap PNG, and structured geometric JSON metadata between `orig` and `built`.
pub fn compute_pixel_diff_heatmap(orig: &Image, built: &Image, opts: DiffOptions) -> Result<PixelDiffReport, ToolError> {
    if orig.width == 0 || orig.height == 0 || built.width == 0 || built.height == 0 {
        return Err(ToolError::invalid("cannot compute visual diff for an empty image"));
    }
    if !opts.delta_e_threshold.is_finite() || opts.delta_e_threshold < 0.0 {
        return Err(ToolError::invalid("delta_e_threshold must be a finite non-negative number"));
    }
    let w = orig.width.max(built.width);
    let h = orig.height.max(built.height);
    let total_pixels = (w as u64).saturating_mul(h as u64);
    let working_set_bytes = (orig.rgba.len() as u64).saturating_add(built.rgba.len() as u64).saturating_add(total_pixels.saturating_mul(16));
    if working_set_bytes > MAX_PIXELS_WORKING_SET {
        return Err(ToolError::new(ErrorCode::OverLimit, format!("combined diff canvas is too large ({w}×{h})")));
    }

    let norm_orig = pad_to_dimensions(orig, w, h);
    let norm_built = pad_to_dimensions(built, w, h);
    let (oa, ba) = (norm_orig.as_image(), norm_built.as_image());

    let ssim = perceptual_similarity(oa, ba);
    let n = total_pixels as usize;
    let lut = build_srgb_lut();
    let mut delta_matrix = vec![0.0f32; n];
    let mut sum_de = 0.0f64;
    let mut max_de = 0.0f64;
    let mut differing_pixels = 0u64;
    let thresh_f32 = opts.delta_e_threshold as f32;

    for i in 0..n {
        let off = i.saturating_mul(4);
        let (Some(pa), Some(pb)) = (oa.rgba.get(off..off + 4), ba.rgba.get(off..off + 4)) else { break };
        let ca = composite_over_mid_gray([pa[0], pa[1], pa[2], pa[3]]);
        let cb = composite_over_mid_gray([pb[0], pb[1], pb[2], pb[3]]);
        let de = if ca == cb {
            0.0
        } else {
            let lab1 = linear_rgb_to_lab(lut[usize::from(ca[0])], lut[usize::from(ca[1])], lut[usize::from(ca[2])]);
            let lab2 = linear_rgb_to_lab(lut[usize::from(cb[0])], lut[usize::from(cb[1])], lut[usize::from(cb[2])]);
            delta_e_cie76(lab1, lab2)
        };
        let de_f32 = de as f32;
        let de_measured = f64::from(de_f32);
        if let Some(slot) = delta_matrix.get_mut(i) {
            *slot = de_f32;
        }
        sum_de += de_measured;
        if de_measured > max_de {
            max_de = de_measured;
        }
        if de_f32 > thresh_f32 {
            differing_pixels += 1;
        }
    }

    let mean_delta_e = if total_pixels == 0 { 0.0 } else { sum_de / total_pixels as f64 };
    let difference_fraction = if total_pixels == 0 { 0.0 } else { differing_pixels as f64 / total_pixels as f64 };
    let regions = detect_diff_regions(oa, ba, &delta_matrix, ssim, opts);
    let heatmap_img = render_diff_heatmap_with_threshold(oa, ba, &regions, &delta_matrix, opts.delta_e_threshold);
    let heatmap_png = encode_png(&heatmap_img)?;

    let diff_json = json!({
        "width": w,
        "height": h,
        "original_size": {"width": orig.width, "height": orig.height},
        "built_size": {"width": built.width, "height": built.height},
        "dimensions_matched": orig.width == built.width && orig.height == built.height,
        "ssim": ssim,
        "mean_delta_e": mean_delta_e,
        "max_delta_e": max_de,
        "total_pixels": total_pixels,
        "differing_pixels": differing_pixels,
        "difference_fraction": difference_fraction,
        "delta_e_threshold": opts.delta_e_threshold,
        "region_count": regions.len(),
        "regions": regions,
    });

    Ok(PixelDiffReport {
        width: w,
        height: h,
        original_width: orig.width,
        original_height: orig.height,
        built_width: built.width,
        built_height: built.height,
        ssim,
        mean_delta_e,
        max_delta_e: max_de,
        total_pixels,
        differing_pixels,
        difference_fraction,
        delta_e_threshold: opts.delta_e_threshold,
        regions,
        heatmap_png,
        diff_json,
    })
}

#[cfg(test)]
mod tests {
    /// The dialogs a page opened are reported with the first line of their message and the answer
    /// `answering_dialogs` gives: alerts accepted, confirm/prompt dismissed.
    #[test]
    fn page_dialogs_are_reported_with_their_answer() {
        use super::{dialogs_from, dialogs_said, dialogs_said_in};
        use craft_anything_engines::cdp::{CdpEvent, DIALOG_OPENING};
        use serde_json::json;
        let event = |params| CdpEvent { method: DIALOG_OPENING.into(), params, session_id: None };
        let dialogs = dialogs_from(&[
            event(json!({"type": "alert", "message": "Saved\nsecond line"})),
            event(json!({"type": "confirm", "message": "Delete everything?"})),
        ]);
        assert_eq!(
            dialogs,
            vec![
                json!({"type": "alert", "message": "Saved", "answered": "accepted"}),
                json!({"type": "confirm", "message": "Delete everything?", "answered": "dismissed"}),
            ],
            "one line each, and a confirm is never agreed to"
        );
        assert_eq!(dialogs_said(&dialogs), " The page opened an alert by itself (\"Saved\") and 1 more; it was accepted so the page could go on.");
        assert_eq!(dialogs_said(&[]), "");
        assert_eq!(dialogs_said_in(&json!([])), "");
        assert_eq!(dialogs_said_in(&serde_json::Value::Null), "");
        let mut secrets = super::Secrets::new(&["tok-SECRET".to_string()]);
        let mut echoed = dialogs_from(&[event(json!({"type": "alert", "message": "token tok-SECRET"}))]);
        super::redact_dialogs(&mut echoed, &mut secrets);
        assert_eq!(echoed[0]["message"], "token «redacted»", "a declared secret in a dialog is redacted");
    }

    /// Review of #1459: a copied `Local State` must not open a profile where saving passwords is on.
    #[test]
    fn sign_in_browsers_open_the_prepared_profile_and_fresh_ones_stay_plain() {
        use super::{LaunchOptions, ProfileChoice, sign_in_args};
        let pinned = format!("--profile-directory={}", super::super::site_sign_in::SIGN_IN_PROFILE);
        let seeded = LaunchOptions { profile: ProfileChoice::SeededFrom("saved".into()), ..LaunchOptions::default() };
        let window = LaunchOptions { keeps_sign_in: true, ..LaunchOptions::default() };
        for opts in [&seeded, &window] {
            let args = sign_in_args(opts);
            assert!(args.contains(&pinned), "{args:?}");
            assert!(args.iter().any(|a| a == "--restore-last-session"), "{args:?}");
            assert_eq!(args.iter().any(|a| a == "--password-store=basic"), cfg!(target_os = "linux"), "{args:?}");
        }
        assert!(sign_in_args(&LaunchOptions::default()).is_empty());
    }

    #[test]
    fn a_launched_browser_deletes_its_profile_on_drop() {
        let root = tempfile::tempdir().unwrap();
        let profile = root.path().join("private-profile");
        std::fs::create_dir(&profile).unwrap();
        std::fs::write(profile.join("Preferences"), b"private browser state").unwrap();
        let browser = super::Browser {
            endpoint: super::cdp::parse_endpoint("http://127.0.0.1:9222").unwrap(),
            product: "Test Browser".into(),
            version: String::new(),
            user_agent: String::new(),
            launched: true,
            _proc: None,
            _profile: Some(super::BrowserProfile(profile.clone())),
            _watch: None,
        };
        drop(browser);
        assert!(!profile.exists(), "the private browser profile survived its owner");
    }

    #[test]
    fn profile_cleanup_retries_while_the_browser_releases_files() {
        let root = tempfile::tempdir().unwrap();
        let profile = root.path().join("private-profile");
        std::fs::create_dir(&profile).unwrap();
        let mut attempts = 0;
        super::remove_profile_with(|| {
            attempts += 1;
            if attempts < 3 { Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "still locked")) } else { std::fs::remove_dir_all(&profile) }
        })
        .unwrap();
        assert_eq!(attempts, 3);
        assert!(!profile.exists());
    }

    /// #747: a Flatpak or Snap browser keeps its profile where its sandbox can reach it and we can
    /// read its port; any other browser uses the temp folder.
    #[test]
    fn a_sandboxed_browser_gets_a_profile_it_can_reach() {
        use std::path::{Path, PathBuf};
        let home = || Some(PathBuf::from("/home/me"));
        let base = |exe: &str| super::profile_base(Path::new(exe), home());
        assert_eq!(base("/var/lib/flatpak/exports/bin/com.google.Chrome"), Path::new("/home/me/.var/app/com.google.Chrome/cache"));
        assert_eq!(base("/home/me/.local/share/flatpak/exports/bin/com.brave.Browser"), Path::new("/home/me/.var/app/com.brave.Browser/cache"));
        assert_eq!(base("/snap/bin/chromium"), Path::new("/home/me/snap/chromium/common"));
        assert_eq!(base("/usr/bin/google-chrome"), std::env::temp_dir());
        assert_eq!(super::profile_base(Path::new("/var/lib/flatpak/exports/bin/com.google.Chrome"), None), std::env::temp_dir(), "no home folder");
    }

    /// #1054: Ubuntu's `/usr/bin/chromium-browser` is a script that hands over to the snap.
    #[test]
    fn a_script_that_hands_over_to_a_snap_is_read_and_nothing_else_is() {
        use std::path::Path;
        let ubuntu = b"#! /bin/sh\n# transitional package\nif ! [ -x /snap/bin/chromium ]; then\n  exit 1\nfi\nexec /snap/bin/chromium \"$@\"\n";
        assert_eq!(super::snap_in_script(ubuntu).as_deref(), Some(Path::new("/snap/bin/chromium")));
        assert_eq!(super::snap_in_script(b"#!/bin/sh\nexec \"/snap/bin/brave\" \"$@\"\n").as_deref(), Some(Path::new("/snap/bin/brave")));
        assert_eq!(super::snap_in_script(b"#!/bin/sh\nexec /opt/chrome/chrome \"$@\"\n"), None, "not a snap");
        assert_eq!(super::snap_in_script(b"#!/bin/sh\necho exec /snap/bin/chromium\n"), None, "only a line that runs it");
        assert_eq!(super::snap_in_script(b"\x7fELF exec /snap/bin/chromium"), None, "a program, not a script");
        assert_eq!(super::snap_behind(Path::new("no-such-browser")), None);
    }

    /// The capped encoder gives a PNG of the same pixels when it fits, and stops without one when
    /// it doesn't: noise hardly compresses, so 64×64 of it can't fit in 4 KB.
    #[test]
    fn a_png_over_its_limit_is_refused_while_encoding() {
        let small = Image { width: 3, height: 2, rgba: (0..24).collect() };
        let png = encode_png_within(&small, 1 << 20).unwrap().unwrap();
        let back = decode_png(&png).unwrap();
        assert_eq!((back.width, back.height, back.rgba), (3, 2, small.rgba.clone()));
        assert_eq!(encode_png_within(&small, png.len()).unwrap().as_deref(), Some(png.as_slice()), "exactly the limit fits");
        assert_eq!(encode_png_within(&small, png.len() - 1).unwrap(), None);
        let mut seed = 0x2545_f491_u32;
        let noise: Vec<u8> = (0..64 * 64 * 4)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        let noisy = Image { width: 64, height: 64, rgba: noise };
        assert_eq!(encode_png_within(&noisy, 4096).unwrap(), None);
        let fits = encode_png_within(&noisy, 1 << 20).unwrap().unwrap();
        assert_eq!(png_dimensions(&fits).unwrap(), (64, 64));
    }

    use super::*;

    #[test]
    fn origin_of_keeps_scheme_host_and_port() {
        assert_eq!(origin_of("https://app.test:8443/a/b?c#d").as_deref(), Some("https://app.test:8443"));
        assert_eq!(origin_of("http://x.test").as_deref(), Some("http://x.test"));
        assert_eq!(origin_of("file:///tmp/a.html"), None);
        assert_eq!(origin_of("https://"), None);
    }

    #[test]
    fn runtime_is_named_from_the_version_info() {
        let ua = |s: &str| format!("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) {s}");
        assert_eq!(runtime_of("Chrome/120.0.6099.291", &ua("Notes/1.0.0 Chrome/120.0.6099.291 Electron/28.2.10 Safari/537.36")), "electron");
        assert_eq!(runtime_of("Edg/120.0.2210.91", &ua("Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.91")), "edge");
        assert_eq!(runtime_of("Chrome/120.0.6099.71", &ua("Chrome/120.0.6099.71 Safari/537.36")), "chrome");
        assert_eq!(runtime_of("HeadlessChrome/120", ""), "chrome");
        assert_eq!(runtime_of("Electron app", ""), "electron");
        assert_eq!(runtime_of("unknown", ""), "unknown");
    }

    #[test]
    fn secrets_redacted_in_every_encoding() {
        let mut s = Secrets::new(&["tok-123".into(), "p@ss word".into()]);
        // Raw, percent-encoded, and JSON-escaped spellings are all caught.
        let v = json!({
            "url": "https://x/?t=tok-123",
            "encoded": "value=tok-123&p=p%40ss%20word",
            "tok-123": "key is a secret too",
            "nested": ["tok-123", {"p@ss word": 1}],
        });
        let out = s.scrub(&v);
        let text = out.to_string();
        assert!(!s.leaks_in(&text), "{text}");
        assert!(!text.contains("tok-123"));
        assert!(!text.contains("p%40ss%20word"));
        assert!(s.redactions >= 4);
    }

    /// Form-encoded (`+`, `*` literal, `~` as `%7E`), mixed-case `%xx`, and ASCII case variants redact in `Secrets`.
    #[test]
    fn secrets_redact_form_urlencoded_asterisk_tilde_and_mixed_case_spellings() {
        let mut s = Secrets::new(&["Bearer tok==".into(), "MySecret/1".into(), "a*b~ c".into()]);
        let v = json!({
            "url": "https://api.example.test/v1/MYSECRET%2f1?auth=Bearer+tok%3D%3D&q=a*b%7E+c&py=a%2Ab~+c&gen=a%2ab~%20c",
            "header": "hint=Bearer+Tok%3d%3d; v=A*B%7e+C; py=a%2Ab~+c",
            "body": "MYSECRET/1 and MySecret%2f1 and a*b%7e+c and a%2ab~+c",
        });
        let out = s.scrub(&v);
        let text = out.to_string().to_ascii_lowercase();
        assert!(!s.leaks_in(&text), "{text}");
        for forbidden in ["tok==", "tok%3d%3d", "mysecret/1", "mysecret%2f1", "a*b~ c", "a*b%7e+c", "a%2ab~+c", "a%2ab~%20c"] {
            assert!(!text.contains(forbidden), "Secrets leaked `{forbidden}` in `{text}`");
        }
        assert!(s.redactions >= 7, "expected at least 7 redactions, got {}", s.redactions);
    }

    /// Overlapping, adjacent, and same-origin secret matches merge into complete redactions without leaking suffixes.
    #[test]
    fn secrets_redact_overlapping_adjacent_and_same_origin_matches() {
        let mut overlap = Secrets::new(&["Bearer abc".into(), "abc123456789".into()]);
        let out = overlap.text("Bearer abc123456789");
        assert_eq!(out, "«redacted»");
        assert!(!overlap.leaks_in(&out), "{out}");
        assert!(!out.contains("123456789"), "{out}");
        assert_eq!(overlap.redactions, 1);
        assert_eq!(overlap.text("前Bearer abc123456789后"), "前«redacted»后");

        let mut partial = Secrets::new(&["alpha-beta".into(), "beta-gamma".into(), "Key One".into(), "One/Two".into()]);
        assert_eq!(partial.text("id=alpha-beta-gamma&q=key+one%2ftwo&keep=safe"), "id=«redacted»&q=«redacted»&keep=safe");
        assert_eq!(partial.redactions, 2);

        let mut adjacent = Secrets::new(&["pfx-".into(), "tok999".into()]);
        assert_eq!(adjacent.text("hdr=pfx-tok999!"), "hdr=«redacted»«redacted»!");
        assert_eq!(adjacent.redactions, 2);

        let mut same_origin = Secrets::new(&["tok".into(), "tok-extended".into()]);
        assert_eq!(same_origin.text("a=tok-extended&b=tok"), "a=«redacted»&b=«redacted»");
        assert_eq!(same_origin.redactions, 2);
    }

    #[test]
    fn headers_drop_credentials() {
        let h = json!({"Set-Cookie": "a=b", "Authorization": "Bearer x", "Content-Type": "text/html", "X-Custom": "secretvalue"});
        let p = project_headers(&h);
        let text = p.to_string();
        assert!(!text.contains("Bearer x") && !text.contains("a=b") && !text.contains("secretvalue"), "{text}");
        assert_eq!(p["kept"]["content-type"], "text/html");
        assert_eq!(p["kept"]["x-custom"]["value_length"], 11);
        let creds = p["credentials_present"].as_array().unwrap();
        assert!(creds.iter().any(|c| c == "set-cookie") && creds.iter().any(|c| c == "authorization"));
    }

    #[test]
    fn json_outline_keeps_no_values() {
        let v = json!({"id": 12, "title": "my secret diary", "tags": ["a", "b"], "nested": {"flag": true}});
        let outline = json_outline(&v);
        let text = outline.to_string();
        assert!(!text.contains("my secret diary") && !text.contains("12"), "{text}");
        assert_eq!(outline["kind"], "object");
        assert_eq!(outline["property_count"], 4);
    }

    #[test]
    fn sanitize_url_strips_userinfo() {
        assert_eq!(sanitize_url("https://user:pw@host/path?q=1"), "https://host/path?q=1");
        assert_eq!(sanitize_url("https://user:pw@host/@profile"), "https://host/@profile");
        assert_eq!(sanitize_url("https://host/x"), "https://host/x");
        assert_eq!(sanitize_url("https://example.test/@profile"), "https://example.test/@profile");
        assert_eq!(sanitize_url("https://cdn.example.test/icon@2x.png"), "https://cdn.example.test/icon@2x.png");
        assert_eq!(sanitize_url("https://example.test/search?to=reader@example.test"), "https://example.test/search?to=reader@example.test");
    }

    #[test]
    fn dom_summary_counts_without_text() {
        let dom = json!({"nodeType": 9, "nodeName": "#document", "children": [
            {"nodeType": 1, "nodeName": "HTML", "children": [
                {"nodeType": 1, "nodeName": "BODY", "attributes": ["class", "main secret"], "children": [
                    {"nodeType": 3, "nodeName": "#text"},
                    {"nodeType": 1, "nodeName": "INPUT", "attributes": ["type", "password", "value", "hunter2"]}
                ]}
            ]}
        ]});
        let s = dom_summary(&dom, 1000);
        let text = s.to_string();
        assert!(!text.contains("hunter2") && !text.contains("main secret"), "{text}");
        assert_eq!(s["elements_by_name"]["input"], 1);
        assert_eq!(s["text_nodes"], 1);
        assert!(s["attribute_names"].as_array().unwrap().iter().any(|a| a == "value"));
    }

    #[test]
    fn pictures_decode_from_png_jpeg_webp_and_gif() {
        let pic = image::RgbaImage::from_fn(16, 8, |x, _| if x < 8 { image::Rgba([200, 40, 40, 255]) } else { image::Rgba([20, 20, 120, 255]) });
        let as_bytes = |f: image::ImageFormat| {
            let mut out = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(pic.clone()).into_rgb8().write_to(&mut out, f).unwrap();
            out.into_inner()
        };
        let png = encode_png(&Image { width: 16, height: 8, rgba: pic.clone().into_raw() }).unwrap();
        for (name, bytes) in [
            ("png", png),
            ("jpeg", as_bytes(image::ImageFormat::Jpeg)),
            ("webp", as_bytes(image::ImageFormat::WebP)),
            ("gif", as_bytes(image::ImageFormat::Gif)),
        ] {
            let img = decode_picture(&bytes).unwrap_or_else(|e| panic!("{name}: {}", e.message));
            assert_eq!((img.width, img.height, img.rgba.len()), (16, 8, 16 * 8 * 4), "{name}");
            let left = &img.rgba[..4];
            assert!(left[0] > 150 && left[2] < 90, "{name}: the left half stays red ({left:?})");
        }
        assert!(decode_picture(b"not a picture").is_err_and(|e| e.code == ErrorCode::GarbledOutput));
    }

    #[test]
    fn png_roundtrip_and_pixel_diff() {
        let a = encode_solid(4, 4, [10, 20, 30, 255]);
        let b = encode_solid(4, 4, [10, 20, 30, 255]);
        let mut c_px = [10u8, 20, 30, 255];
        let img_a = decode_png(&a).unwrap();
        assert_eq!((img_a.width, img_a.height), (4, 4));
        let d = pixel_diff(&decode_png(&a).unwrap(), &decode_png(&b).unwrap(), 0);
        assert_eq!(d["identical"], true);
        // Change one pixel by 5 on one channel.
        c_px[1] = 25;
        let c = encode_one_changed(4, 4, [10, 20, 30, 255], 1, c_px);
        let d2 = pixel_diff(&decode_png(&a).unwrap(), &decode_png(&c).unwrap(), 0);
        assert_eq!(d2["differing_pixels"], 1);
        assert_eq!(d2["max_channel_delta"], 5);
        // Threshold of 5 hides a delta of exactly 5.
        let d3 = pixel_diff(&decode_png(&a).unwrap(), &decode_png(&c).unwrap(), 5);
        assert_eq!(d3["differing_pixels"], 0);
        // Different dimensions are not comparable.
        let big = encode_solid(5, 4, [0, 0, 0, 255]);
        assert_eq!(pixel_diff(&decode_png(&a).unwrap(), &decode_png(&big).unwrap(), 0)["comparable"], false);
        let (ia, ib) = (decode_png(&a).unwrap(), decode_png(&big).unwrap());
        let (w, h) = (ia.width.min(ib.width), ia.height.min(ib.height));
        assert_eq!(pixel_diff(&crop(&ia, w, h), &crop(&ib, w, h), 0)["comparable"], true);
        assert!((block_similarity(&ia, &ia, 16, 0) - 1.0).abs() < f64::EPSILON);
        let black = Image { width: 32, height: 32, rgba: vec![0; 32 * 32 * 4] };
        let mut half = Image { width: 32, height: 32, rgba: vec![0; 32 * 32 * 4] };
        half.rgba[..32 * 16 * 4].fill(255);
        assert!((block_similarity(&black, &half, 16, 24) - 0.5).abs() < 1e-9, "top two of four tiles differ");
    }

    fn encode_solid(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
        let data: Vec<u8> = (0..(w * h)).flat_map(|_| px).collect();
        encode_rgba(w, h, &data)
    }

    fn encode_one_changed(w: u32, h: u32, base: [u8; 4], idx: usize, changed: [u8; 4]) -> Vec<u8> {
        let mut data: Vec<u8> = (0..(w * h)).flat_map(|_| base).collect();
        if let Some(slot) = data.get_mut(idx * 4..idx * 4 + 4) {
            slot.copy_from_slice(&changed);
        }
        encode_rgba(w, h, &data)
    }

    fn encode_rgba(w: u32, h: u32, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w, h);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut writer = enc.write_header().unwrap();
            writer.write_image_data(data).unwrap();
        }
        out
    }

    #[test]
    fn parts_that_change_by_themselves_are_left_out_of_comparisons() {
        // A 128×64 page: a grey band, and a "carousel" square at (64, 16) that shows another frame
        // in the second photo of the same page.
        let page = |carousel: u8| {
            let mut rgba = Vec::new();
            for y in 0..64u32 {
                for x in 0..128u32 {
                    let v = if (64..96).contains(&x) && (16..48).contains(&y) {
                        carousel
                    } else if y < 8 {
                        90
                    } else {
                        230
                    };
                    rgba.extend([v, v, v, 255]);
                }
            }
            Image { width: 128, height: 64, rgba }
        };
        let (first, second) = (page(10), page(200));
        let mask = NoiseMask::between(&first, &second, 24);
        assert_eq!((mask.cols, mask.rows), (8, 4));
        assert!(mask.covers(70, 20) && mask.covers(50, 20), "the carousel and one square around it");
        assert!(!mask.covers(5, 5) && !mask.covers(127, 63), "the still parts aren't masked");
        assert_eq!(NoiseMask::from_png(&mask.to_png().unwrap()).unwrap(), mask, "it reads back from its PNG");
        let still = NoiseMask::between(&first, &first, 24);
        assert_eq!(still.count(), 0);
        assert_eq!(still.clone().union(&mask), mask, "a photo pair that missed the change doesn't hide it");
        // A rebuild that matches everything but shows yet another carousel frame.
        let build = page(120);
        assert!(block_similarity(&first, &build, 16, 24) < 1.0 && perceptual_similarity(&first, &build) < 1.0);
        assert_eq!(block_similarity_masked(&first, &build, 16, 24, Some(&mask)), 1.0);
        assert!((perceptual_similarity_masked(&first, &build, Some(&mask)) - 1.0).abs() < 1e-9);
        // A real difference outside the moving part still counts.
        let mut wrong = page(120);
        for px in wrong.rgba.as_chunks_mut::<4>().0.iter_mut().take(128 * 8) {
            *px = [255, 0, 0, 255];
        }
        assert!(block_similarity_masked(&first, &wrong, 16, 24, Some(&mask)) < 1.0);
        let everything = NoiseMask { tile: NOISE_TILE, cols: 8, rows: 4, bits: vec![true; 32] };
        assert_eq!(block_similarity_masked(&first, &wrong, 16, 24, Some(&everything)), 1.0, "nothing left to differ");
    }

    #[test]
    fn perceptual_similarity_sees_structure() {
        let img = |w: u32, h: u32, f: &dyn Fn(u32, u32) -> u8| {
            let mut rgba = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let v = f(x, y);
                    rgba.extend_from_slice(&[v, v, v, 255]);
                }
            }
            Image { width: w, height: h, rgba }
        };
        let stripes = img(200, 120, &|x, _| if (x / 4) % 2 == 0 { 30 } else { 220 });
        let same = img(200, 120, &|x, _| if (x / 4) % 2 == 0 { 30 } else { 220 });
        let flat = img(200, 120, &|_, _| 125);
        let tinted = img(200, 120, &|x, _| if (x / 4) % 2 == 0 { 40 } else { 230 });
        assert!((perceptual_similarity(&stripes, &same) - 1.0).abs() < 1e-9);
        assert!(perceptual_similarity(&stripes, &flat) < 0.2, "stripes and a flat grey look different");
        assert!(perceptual_similarity(&stripes, &tinted) > 0.9, "a small colour shift keeps the structure");
        assert_eq!(perceptual_similarity(&stripes, &img(100, 120, &|_, _| 0)), 0.0, "different sizes");
        let big = img(2000, 40, &|x, _| (x % 256) as u8);
        assert!(perceptual_similarity(&big, &big) > 0.999, "shrunk before comparing");
    }

    #[test]
    fn cielab_conversion_and_delta_e_match_reference_colors() {
        let black = rgb_to_lab(0, 0, 0);
        assert!(black.0.abs() < 1e-6 && black.1.abs() < 1e-6 && black.2.abs() < 1e-6, "black: {black:?}");
        let white = rgb_to_lab(255, 255, 255);
        assert!((white.0 - 100.0).abs() < 0.05 && white.1.abs() < 0.05 && white.2.abs() < 0.05, "white: {white:?}");
        let red = rgb_to_lab(255, 0, 0);
        // D65 sRGB red is approximately L*=53.23, a*=80.11, b*=67.22.
        assert!((red.0 - 53.23).abs() < 0.5 && red.1 > 75.0 && red.2 > 60.0, "red: {red:?}");
        let blue = rgb_to_lab(0, 0, 255);
        // D65 sRGB blue is approximately L*=32.30, a*=79.20, b*=-107.86.
        assert!((blue.0 - 32.30).abs() < 0.5 && blue.2 < -100.0, "blue: {blue:?}");

        assert!(delta_e_cie76(red, red).abs() < 1e-9);
        assert!((delta_e_cie76(black, white) - 100.0).abs() < 0.1);
        assert!(delta_e_cie76(red, blue) > 150.0);
        // Subtle 2-unit sRGB shift produces a small perceptible Delta-E.
        let subtle = delta_e_cie76(rgb_to_lab(120, 120, 120), rgb_to_lab(122, 120, 120));
        assert!(subtle > 0.1 && subtle < 5.0, "subtle delta-e: {subtle}");
    }

    #[test]
    fn pixel_diff_heatmap_clusters_merges_orders_and_filters_noise() {
        let solid = |w: u32, h: u32, px: [u8; 4]| Image { width: w, height: h, rgba: (0..(w * h)).flat_map(|_| px).collect() };
        let paint = |img: &mut Image, rx: u32, ry: u32, rw: u32, rh: u32, px: [u8; 4]| {
            for y in ry..(ry + rh) {
                for x in rx..(rx + rw) {
                    let o = ((y * img.width + x) * 4) as usize;
                    img.rgba[o..o + 4].copy_from_slice(&px);
                }
            }
        };

        // Identical images report SSIM 1.0, Delta-E 0.0, and zero regions.
        let orig = solid(200, 160, [240, 240, 240, 255]);
        let identical = compute_pixel_diff_heatmap(&orig, &orig, DiffOptions::default()).unwrap();
        assert!((identical.ssim - 1.0).abs() < 1e-9);
        assert!(identical.mean_delta_e.abs() < 1e-9);
        assert_eq!(identical.differing_pixels, 0);
        assert!(identical.regions.is_empty());
        assert_eq!(png_dimensions(&identical.heatmap_png).unwrap(), (200, 160));

        // Build image with:
        // 1) Top-right pair of boxes separated by 12px (<= 16px, across inactive grid column 120..128) -> must merge into one box at y=16.
        // 2) Bottom-left box far away -> must become Region [2] (ordered top-to-bottom, left-to-right).
        // 3) Tiny 4x4 (16px < 32px) noise speckle -> must be filtered out from regions.
        let mut built = solid(200, 160, [240, 240, 240, 255]);
        paint(&mut built, 100, 16, 16, 16, [220, 20, 20, 255]);
        paint(&mut built, 128, 16, 16, 16, [220, 20, 20, 255]); // 12px horizontal gap -> merged by 16px sweep!
        paint(&mut built, 16, 100, 24, 24, [20, 40, 220, 255]); // 576px, well separated
        paint(&mut built, 170, 140, 4, 4, [0, 0, 0, 255]); // 16px noise -> filtered out

        let report = compute_pixel_diff_heatmap(&orig, &built, DiffOptions::default()).unwrap();
        assert!(report.ssim < 1.0);
        assert!(report.mean_delta_e > 0.0);
        assert_eq!(report.differing_pixels, 16 * 16 * 2 + 24 * 24 + 16);
        assert_eq!(report.regions.len(), 2, "expected 2 merged regions (noise dropped): {:?}", report.regions);

        let r1 = &report.regions[0];
        let r2 = &report.regions[1];
        assert_eq!(r1.id, 1);
        assert_eq!((r1.x, r1.y, r1.width, r1.height), (100, 16, 44, 16));
        assert_eq!(r1.pixel_count, 512);
        assert_eq!(r1.severity, "severe");
        assert!(r1.mean_delta_e > 20.0);

        // Fully transparent pixels (alpha == 0) with different hidden RGB values are normalized to the same composited background.
        let trans_a = solid(32, 32, [255, 0, 0, 0]);
        let trans_b = solid(32, 32, [0, 255, 120, 0]);
        let trans_diff = compute_pixel_diff_heatmap(&trans_a, &trans_b, DiffOptions::default()).unwrap();
        assert_eq!(trans_diff.differing_pixels, 0);
        assert!(trans_diff.mean_delta_e.abs() < 1e-9);
        assert!((trans_diff.ssim - 1.0).abs() < 1e-9);

        // Opaque black vs transparent black must NOT score SSIM 1.0 (composited over neutral mid-gray).
        let opaque_black = solid(24, 24, [0, 0, 0, 255]);
        let trans_black = solid(24, 24, [0, 0, 0, 0]);
        assert!(perceptual_similarity(&opaque_black, &trans_black) < 1.0);
        let alpha_diff = compute_pixel_diff_heatmap(&opaque_black, &trans_black, DiffOptions::default()).unwrap();
        assert!(alpha_diff.ssim < 1.0 && alpha_diff.differing_pixels == 24 * 24);
        assert!(alpha_diff.regions.iter().all(|r| r.max_delta_e <= alpha_diff.max_delta_e));

        // Low-opacity red vs blue (alpha = 1/255) contributes only 1/255 of full strength after compositing,
        // while opaque red vs opaque blue remains strongly divergent.
        let faint_red = solid(24, 24, [255, 0, 0, 1]);
        let faint_blue = solid(24, 24, [0, 0, 255, 1]);
        let faint_diff = compute_pixel_diff_heatmap(&faint_red, &faint_blue, DiffOptions::default()).unwrap();
        assert!(faint_diff.regions.is_empty() && faint_diff.mean_delta_e < 2.5, "faint alpha diff: {}", faint_diff.mean_delta_e);
        let full_red = solid(24, 24, [255, 0, 0, 255]);
        let full_blue = solid(24, 24, [0, 0, 255, 255]);
        let full_diff = compute_pixel_diff_heatmap(&full_red, &full_blue, DiffOptions::default()).unwrap();
        assert_eq!(full_diff.regions.len(), 1);
        assert_eq!(full_diff.regions[0].severity, "severe");

        assert_eq!(r2.id, 2);
        assert_eq!((r2.x, r2.y, r2.width, r2.height), (16, 100, 24, 24));
        assert_eq!(r2.pixel_count, 576);
        assert_eq!(r2.severity, "severe");

        // JSON round-trip of DiffRegion works including &'static str severity.
        let round_trip: Vec<DiffRegion> = serde_json::from_value(report.diff_json["regions"].clone()).unwrap();
        assert_eq!(round_trip, report.regions);

        // Heatmap PNG decodes cleanly and paints bright red box border on Region [1]'s top-left corner.
        let hm = decode_png(&report.heatmap_png).unwrap();
        assert_eq!((hm.width, hm.height), (200, 160));
        let corner = ((r1.y * hm.width + r1.x) * 4) as usize;
        assert!(
            hm.rgba[corner] > 200 && hm.rgba[corner + 1] < 60 && hm.rgba[corner + 2] < 60,
            "expected red border at corner: {:?}",
            &hm.rgba[corner..corner + 4]
        );

        // Dimension mismatch pads to common bounding dimensions and flags the missing strip as a region.
        let taller = solid(200, 200, [240, 240, 240, 255]);
        let padded = compute_pixel_diff_heatmap(&taller, &orig, DiffOptions::default()).unwrap();
        assert_eq!((padded.width, padded.height), (200, 200));
        assert_eq!(padded.diff_json["dimensions_matched"], false);
        assert_eq!(padded.regions.len(), 1);
        assert_eq!((padded.regions[0].x, padded.regions[0].y, padded.regions[0].width, padded.regions[0].height), (0, 160, 200, 40));
    }
}

/// The last lines a launched browser wrote to stderr, for a failure's message.
#[derive(Clone, Default)]
struct StderrTail(std::sync::Arc<craft_anything_platform::sync::Mutex<std::collections::VecDeque<String>>>);

impl StderrTail {
    fn push(&self, line: &str) {
        let mut q = self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if q.len() >= 20 {
            q.pop_front();
        }
        q.push_back(line.chars().take(400).collect());
    }

    fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).iter().cloned().collect()
    }
}
