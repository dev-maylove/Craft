//! Independent blind visual critic: reviews screenshots (and sequential motion burst frames)
//! from a clean, tool-less, stateless turn with zero access to source code or builder
//! justifications. Enabled whenever the user configures `Role::Reviewer` to an API vision backend
//! or an installed agent CLI (`Codex`, `Antigravity`, `OpenCode`, `Claude Code`).

use super::kinds::{Check, CheckCx};
use crate::AgentEvent;
use crate::agent::ModelSlot;
use crate::config::Price;
use crate::model::{Block, ChatModel, Effort, MAX_IMAGE_BYTES, Msg, Speaker, Stop, TurnRequest, Usage};
use crate::prompt;
use craft_anything_engine::tools::web_common::{
    DiffOptions, DiffRegion, PixelDiffReport, compute_pixel_diff_heatmap, decode_picture, encode_png_within, resize,
};
use craft_anything_engine::workspace::{CheckResult, Claim, TargetRef, Workspace};
use craft_anything_evidence::ToolError;
use craft_anything_platform::sync::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, PoisonError};
use std::time::{Duration, Instant};

/// Stable check id for the blind visual critic.
pub const CHECK_ID: &str = "visual-critic";
/// Report written inside `checks/round-N/`.
pub const REPORT_FILE: &str = "visual-critic.json";
/// Maximum pixel edge sent to vision models so tall full-page captures stay within API limits.
const MAX_IMAGE_EDGE: u32 = 4096;
/// Maximum number of before/after pairs reviewed per round to keep token cost bounded.
const MAX_PAIRS_PER_ROUND: usize = 5;

/// Strict blind-critic prompt modeled on adversarial visual verification: zero source code,
/// zero tolerance for excuses, strict 3-tier severity calibration.
pub const BLIND_CRITIC_SYSTEM: &str = "\
You are an independent, ruthless visual quality critic. You judge a rebuilt UI, website, app or game \
STRICTLY by comparing rendered screenshots (alongside deterministic pixel-diff heatmaps, quantitative \
pixel-analysis metrics, and sequential motion burst frames when provided) against the original reference images.

Rules of blind evaluation:
- You have NO access to source code, implementation constraints, or builder excuses. Judge ONLY what a human user sees on screen.
- Classify every real mismatch into exactly one severity:
  - \"severe\": Noticeable within 1 second — missing or broken sections, wrong palette/background/contrast, overlapping or clipped text/elements, broken typography hierarchy, missing hero/media/cards, dead/static motion where the original clearly animates across burst frames, broken hover/scroll transitions, or glaring rendering artifacts.
  - \"moderate\": Noticeable on careful inspection — misaligned padding/spacing/columns, visibly off corner radii/borders/shadows, mismatched icon or card proportions, uneven easing/flicker/jumps between sequential motion frames, or visible layout shifts.
  - \"minor\": Micro-detail only visible side-by-side at close zoom — 1-2 px spacing differences or subtle font-hinting/anti-aliasing variance.
- Focus on layout, styling, visual hierarchy, alignment, polish, and motion continuity. Neutral placeholder copy or neutral placeholder imagery replacing third-party brand logos/photos is intentional and NOT a defect unless the proportions, styling, or layout break.
- Use the deterministic diff heatmap (which highlights pixel divergence with a thermal overlay, red bounding boxes, and numbered badges [1], [2], ...) and its exact geometric metrics (SSIM, Delta-E, coordinates) to catch subtle alignment, spacing, contrast, and micro-dimension flaws. When a defect maps to a numbered heatmap region, set \"pair\" to the exact pair label (e.g. \"viewport 1440 px\"), set \"region_id\" to that integer, and include the pair, region badge, and pixel coordinates in \"area\".
- Be concrete and surgical: cite exact visible areas, pair labels, region numbers/coordinates, and the exact visual change required.

Reply ONLY with valid JSON matching this schema (no prose outside JSON; include \"pair\" whenever \"region_id\" is set, and omit \"region_id\" or set it to null when a defect is not tied to a single numbered heatmap region):
{
  \"summary\": \"1-2 sentence overall visual verdict\",
  \"defects\": [
    {
      \"severity\": \"severe\" | \"moderate\" | \"minor\",
      \"pair\": \"viewport 1440 px\",
      \"region_id\": 1,
      \"area\": \"navbar (Region [1] at x=140, y=82, 320x48px)\",
      \"summary\": \"padding collapsed by 16px, misaligned with logo\",
      \"fix\": \"increase top padding to 24px and align items center\"
    }
  ]
}";

/// Severity scale for defects found by the blind visual critic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Severe,
    Moderate,
    Minor,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Severe => "severe",
            Severity::Moderate => "moderate",
            Severity::Minor => "minor",
        }
    }

    fn parse(s: &str) -> Option<Severity> {
        match s.trim().to_ascii_lowercase().as_str() {
            "severe" | "critical" | "high" | "blocker" => Some(Severity::Severe),
            "moderate" | "medium" | "major" => Some(Severity::Moderate),
            "minor" | "low" | "nit" | "polish" => Some(Severity::Minor),
            _ => None,
        }
    }
}

/// One visual or motion defect observed strictly from screenshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualDefect {
    pub severity: Severity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<usize>,
    pub area: String,
    pub summary: String,
    pub fix: String,
}

/// Token and cost accounting returned by [`VisualCritic::review`], preserving CLI qualifiers.
#[derive(Debug, Clone, Default)]
pub struct ReviewUsage {
    pub usage: Usage,
    pub cost_usd: Option<f64>,
    pub tokens_unknown: bool,
    pub cost_is_estimate: bool,
}

/// Structured verdict saved to `checks/round-N/visual-critic.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriticReport {
    pub reviewer: String,
    pub model: String,
    pub passed: bool,
    pub severe_count: usize,
    pub moderate_count: usize,
    pub minor_count: usize,
    pub pairs_reviewed: usize,
    pub defects: Vec<VisualDefect>,
    pub summary: String,
}

impl CriticReport {
    /// Build a sorted report where passing requires zero Severe and zero Moderate defects.
    pub fn from_defects(reviewer: String, model: String, pairs_reviewed: usize, summary: String, mut defects: Vec<VisualDefect>) -> CriticReport {
        defects.sort_by_key(|d| d.severity);
        let severe_count = defects.iter().filter(|d| d.severity == Severity::Severe).count();
        let moderate_count = defects.iter().filter(|d| d.severity == Severity::Moderate).count();
        let minor_count = defects.iter().filter(|d| d.severity == Severity::Minor).count();
        let passed = severe_count == 0 && moderate_count == 0;
        CriticReport { reviewer, model, passed, severe_count, moderate_count, minor_count, pairs_reviewed, defects, summary }
    }
}

/// A pair of original and rebuilt images reviewed side by side, optionally accompanied by
/// pre-generated deterministic diff heatmap and geometric metadata paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImagePair {
    pub label: String,
    pub original: PathBuf,
    pub built: PathBuf,
    pub heatmap: Option<PathBuf>,
    pub diff_json: Option<PathBuf>,
}

/// Resolved deterministic visual-diff artifacts and summary metrics for one [`ImagePair`].
#[derive(Debug, Clone)]
pub struct PairDiffSummary {
    pub ssim: f64,
    pub mean_delta_e: f64,
    pub max_delta_e: f64,
    pub total_pixels: u64,
    pub differing_pixels: u64,
    pub difference_fraction: f64,
    pub regions: Vec<DiffRegion>,
    pub heatmap_png: Vec<u8>,
    pub diff_json: Value,
}

impl From<PixelDiffReport> for PairDiffSummary {
    fn from(r: PixelDiffReport) -> Self {
        PairDiffSummary {
            ssim: r.ssim,
            mean_delta_e: r.mean_delta_e,
            max_delta_e: r.max_delta_e,
            total_pixels: r.total_pixels,
            differing_pixels: r.differing_pixels,
            difference_fraction: r.difference_fraction,
            regions: r.regions,
            heatmap_png: r.heatmap_png,
            diff_json: r.diff_json,
        }
    }
}

/// Format quantitative SSIM, Delta-E, and numbered region coordinates for vision/CLI prompts.
pub fn format_quantitative_analysis(label: &str, diff: &PairDiffSummary) -> String {
    format_quantitative_analysis_with_scale(label, diff, None)
}

/// Format quantitative SSIM, Delta-E, coordinate space dimensions, and numbered region coordinates.
pub fn format_quantitative_analysis_with_scale(label: &str, diff: &PairDiffSummary, scale_dims: Option<((u32, u32), (u32, u32))>) -> String {
    let mut out = format!("=== {label} — Quantitative Pixel Analysis ===\n");
    if let Some(((orig_w, orig_h), (sent_w, sent_h))) = scale_dims {
        if (orig_w, orig_h) != (sent_w, sent_h) {
            out.push_str(&format!(
                "- Coordinate Space: full resolution {orig_w}×{orig_h} px (attached image downscaled to {sent_w}×{sent_h} px for API size limits; all Region [N] x, y, width, height coordinates below are in the full-resolution {orig_w}×{orig_h} px space)\n"
            ));
        } else {
            out.push_str(&format!("- Coordinate Space: full resolution {orig_w}×{orig_h} px\n"));
        }
    }
    out.push_str(&format!("- SSIM: {:.3} (1.0 = identical)\n", diff.ssim));
    out.push_str(&format!(
        "- Mean Delta-E: {:.2} ({:.1}% differing pixels, {} of {} px)\n",
        diff.mean_delta_e,
        diff.difference_fraction * 100.0,
        diff.differing_pixels,
        diff.total_pixels,
    ));
    if diff.regions.is_empty() {
        out.push_str("- Divergence Regions Detected: none (below clustering threshold)\n");
    } else {
        out.push_str("- Divergence Regions Detected:\n");
        for r in diff.regions.iter().take(20) {
            out.push_str(&format!(
                "  * Region [{}]: x={}, y={}, {}x{}px ({}, Delta-E {:.1}, max {:.1}, SSIM {:.3}, {} px)\n",
                r.id, r.x, r.y, r.width, r.height, r.severity, r.mean_delta_e, r.max_delta_e, r.ssim, r.pixel_count
            ));
        }
        if diff.regions.len() > 20 {
            out.push_str(&format!("  * ... and {} more region(s) in diff JSON\n", diff.regions.len() - 20));
        }
    }
    out
}

/// Load precomputed diff artifacts when already saved beside `pair`, else compute them on the fly.
pub fn resolve_pair_diff(pair: &ImagePair) -> Result<PairDiffSummary, ToolError> {
    if let (Some(hm_path), Some(json_path)) = (&pair.heatmap, &pair.diff_json)
        && let (Ok(hm_bytes), Ok(json_bytes)) = (std::fs::read(hm_path), std::fs::read(json_path))
        && let Ok(diff_json) = serde_json::from_slice::<Value>(&json_bytes)
    {
        let regions: Vec<DiffRegion> = diff_json.get("regions").cloned().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
        return Ok(PairDiffSummary {
            ssim: diff_json.get("ssim").and_then(Value::as_f64).unwrap_or(0.0),
            mean_delta_e: diff_json.get("mean_delta_e").and_then(Value::as_f64).unwrap_or(0.0),
            max_delta_e: diff_json.get("max_delta_e").and_then(Value::as_f64).unwrap_or(0.0),
            total_pixels: diff_json.get("total_pixels").and_then(Value::as_u64).unwrap_or(0),
            differing_pixels: diff_json.get("differing_pixels").and_then(Value::as_u64).unwrap_or(0),
            difference_fraction: diff_json.get("difference_fraction").and_then(Value::as_f64).unwrap_or(0.0),
            regions,
            heatmap_png: hm_bytes,
            diff_json,
        });
    }
    let orig_bytes = std::fs::read(&pair.original).map_err(|e| ToolError::io(pair.original.display(), e))?;
    let built_bytes = std::fs::read(&pair.built).map_err(|e| ToolError::io(pair.built.display(), e))?;
    let orig_img = decode_picture(&orig_bytes)?;
    let built_img = decode_picture(&built_bytes)?;
    let report = compute_pixel_diff_heatmap(&orig_img, &built_img, DiffOptions::default())?;
    Ok(PairDiffSummary::from(report))
}

/// Deterministically compute and persist `pair-NN-heatmap.png` and `pair-NN-diff.json` into
/// `output_dir` for every pair in `pairs`, attaching the written paths to each [`ImagePair`].
pub fn generate_pair_diffs(pairs: &mut [ImagePair], output_dir: &Path) {
    for (idx, pair) in pairs.iter_mut().enumerate() {
        generate_single_pair_diff(pair, idx + 1, output_dir);
    }
}

fn generate_single_pair_diff(pair: &mut ImagePair, n: usize, output_dir: &Path) {
    if std::fs::create_dir_all(output_dir).is_err() {
        return;
    }
    let Ok(orig_bytes) = std::fs::read(&pair.original) else { return };
    let Ok(built_bytes) = std::fs::read(&pair.built) else { return };
    let (Ok(orig_img), Ok(built_img)) = (decode_picture(&orig_bytes), decode_picture(&built_bytes)) else { return };
    let Ok(report) = compute_pixel_diff_heatmap(&orig_img, &built_img, DiffOptions::default()) else { return };

    let hm_path = output_dir.join(format!("pair-{n:02}-heatmap.png"));
    let json_path = output_dir.join(format!("pair-{n:02}-diff.json"));
    if craft_anything_engine::project::write_atomic(&hm_path, &report.heatmap_png).is_ok() {
        pair.heatmap = Some(hm_path);
    }
    if let Ok(json_bytes) = serde_json::to_vec_pretty(&report.diff_json)
        && craft_anything_engine::project::write_atomic(&json_path, &json_bytes).is_ok()
    {
        pair.diff_json = Some(json_path);
    }
}

/// Stage `pair-NN-original.png`, `pair-NN-rebuilt.png`, `pair-NN-heatmap.png`, and `pair-NN-diff.json`
/// into `review_dir` and return `(prompt, pair_count)` for an isolated agent CLI blind review.
pub fn stage_cli_review_dir(pairs: &[ImagePair], review_dir: &Path) -> Result<(String, usize), ToolError> {
    let mut prompt = String::from(
        "Compare each ORIGINAL image against its REBUILT counterpart in the current folder.\n\
         For each pair, a deterministic pixel-diff heatmap (`pair-NN-heatmap.png`) with numbered red region badges `[1]`, `[2]`, ... \
         and exact geometric metadata (`pair-NN-diff.json` with `(x, y, width, height)`, Delta-E, and SSIM per region) are also provided.\n\
         Open and inspect the original, rebuilt, and heatmap PNG files and consult the diff JSON for precise coordinates before answering:\n\n",
    );
    let mut loaded = 0usize;
    for (idx, pair) in pairs.iter().enumerate() {
        let n = idx + 1;
        let orig_name = format!("pair-{n:02}-original.png");
        let built_name = format!("pair-{n:02}-rebuilt.png");
        let heatmap_name = format!("pair-{n:02}-heatmap.png");
        let diff_name = format!("pair-{n:02}-diff.json");

        let diff = resolve_pair_diff(pair)?;
        let orig_fit = fit_image_file_for_api(&pair.original, &format!("original-{}", pair.label))?;
        let built_fit = fit_image_file_for_api(&pair.built, &format!("rebuilt-{}", pair.label))?;
        let hm_fit = fit_image_for_api(diff.heatmap_png.clone(), &heatmap_name)?;
        let diff_bytes = serde_json::to_vec_pretty(&diff.diff_json).map_err(|e| ToolError::internal(e.to_string()))?;

        std::fs::write(review_dir.join(&orig_name), orig_fit.bytes).map_err(|e| ToolError::io(&orig_name, e))?;
        std::fs::write(review_dir.join(&built_name), built_fit.bytes).map_err(|e| ToolError::io(&built_name, e))?;
        std::fs::write(review_dir.join(&heatmap_name), hm_fit.bytes).map_err(|e| ToolError::io(&heatmap_name, e))?;
        std::fs::write(review_dir.join(&diff_name), diff_bytes).map_err(|e| ToolError::io(&diff_name, e))?;

        let scale =
            if hm_fit.original_dims != hm_fit.sent_dims { (hm_fit.original_dims, hm_fit.sent_dims) } else { (orig_fit.original_dims, orig_fit.sent_dims) };
        prompt.push_str(&format!(
            "{n}. {} — ORIGINAL: `{orig_name}` | REBUILT: `{built_name}` | HEATMAP: `{heatmap_name}` | DIFF JSON: `{diff_name}`\n{}\n",
            pair.label,
            format_quantitative_analysis_with_scale(&pair.label, &diff, Some(scale)),
        ));
        loaded += 1;
    }
    prompt.push_str(
        "Judge ONLY what is visible in these image files and diff metadata. Inspect every `pair-NN-heatmap.png` and consult `pair-NN-diff.json` for exact coordinates and `region_id` numbers. \
         Do not look outside this folder or run shell commands.\n\
         Reply ONLY with the JSON verdict object (`summary` and `defects` ordered from most severe to least severe).",
    );
    Ok((prompt, loaded))
}

enum CriticBackend {
    Api { local: bool, effort: Option<Effort>, prices: BTreeMap<String, Price>, model: Mutex<Box<dyn ChatModel>> },
    Cli { entry: Box<crate::BackendEntry>, key: Option<String>, logs: Option<PathBuf> },
    InitFailed { error: String },
}

/// Temporary isolated directory containing only image pairs for an agent CLI blind review,
/// outside the project folder so the reviewer has zero access to `app/` source code.
struct TempReviewDir {
    path: PathBuf,
}

#[cfg(unix)]
fn private_dir_builder() -> std::fs::DirBuilder {
    use std::os::unix::fs::DirBuilderExt as _;
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    builder
}

#[cfg(not(unix))]
fn private_dir_builder() -> std::fs::DirBuilder {
    std::fs::DirBuilder::new()
}

impl TempReviewDir {
    fn create() -> Result<TempReviewDir, ToolError> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let base = std::env::temp_dir();
        let pid = std::process::id();
        for _ in 0..64 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
            let path = base.join(format!("craft-anything-visual-critic-{pid}-{nanos}-{id}"));
            match private_dir_builder().create(&path) {
                Ok(()) => return Ok(TempReviewDir { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(ToolError::io(path.display(), e)),
            }
        }
        Err(ToolError::internal("couldn't create an exclusive temporary visual review directory"))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempReviewDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

const CLI_REVIEW_TIMEOUT: Duration = Duration::from_secs(300);
const CLI_QUIET_DONE: Duration = Duration::from_secs(2);

/// Stateless, code-blind multimodal visual critic backed by a configured `Role::Reviewer` model
/// (either an API vision model or an installed agent CLI running in an isolated folder).
pub struct VisualCritic {
    pub backend_name: String,
    pub model_id: String,
    runner: CriticBackend,
}

impl VisualCritic {
    pub fn from_slot(slot: ModelSlot, prices: BTreeMap<String, Price>) -> VisualCritic {
        let model_id = slot.model.model();
        VisualCritic {
            backend_name: slot.name,
            model_id,
            runner: CriticBackend::Api { local: slot.local, effort: slot.effort, prices, model: Mutex::new(slot.model) },
        }
    }

    pub fn from_cli(entry: crate::BackendEntry, key: Option<String>, logs: Option<PathBuf>) -> VisualCritic {
        let model_id = entry.options.cli_model.clone().unwrap_or_else(|| entry.config.model.clone());
        let backend_name = entry.config.name.clone();
        VisualCritic { backend_name, model_id, runner: CriticBackend::Cli { entry: Box::new(entry), key, logs } }
    }

    pub fn from_init_error(backend_name: String, model_id: String, error: String) -> VisualCritic {
        VisualCritic { backend_name, model_id, runner: CriticBackend::InitFailed { error } }
    }

    /// Stable identifier included in check cache fingerprints so switching or removing the
    /// reviewer invalidates kept check results.
    pub fn fingerprint_id(&self) -> String {
        format!("{}:{}", self.backend_name, self.model_id)
    }

    /// Execute one clean, tool-less, single-turn review over `pairs` with zero source context.
    pub fn review(&self, pairs: &[ImagePair], stop: &Stop) -> Result<(CriticReport, ReviewUsage), ToolError> {
        match &self.runner {
            CriticBackend::Api { local, effort, prices, model } => self.review_api(pairs, *local, *effort, prices, model, stop),
            CriticBackend::Cli { entry, key, logs } => self.review_cli(pairs, entry, key.as_deref(), logs.as_deref(), stop),
            CriticBackend::InitFailed { error } => Err(ToolError::invalid(error.clone())),
        }
    }

    fn review_api(
        &self,
        pairs: &[ImagePair],
        local: bool,
        effort: Option<Effort>,
        prices: &BTreeMap<String, Price>,
        model: &Mutex<Box<dyn ChatModel>>,
        stop: &Stop,
    ) -> Result<(CriticReport, ReviewUsage), ToolError> {
        let mut blocks = Vec::new();
        let mut loaded = 0usize;
        let mut header = String::from(
            "Compare each ORIGINAL image against its REBUILT counterpart in the exact order shown below.\n\
             Each pair is preceded by deterministic Quantitative Pixel Analysis (SSIM, Mean Delta-E, and numbered divergence regions) \
             and followed by three images: ORIGINAL, REBUILT, and DIFF HEATMAP (highlighting divergent pixels with numbered red badges [1], [2], ...).\n\
             Inspect visual fidelity, typography, colours, spacing, responsive fit, and—when sequential \
             motion burst frames are included—transition/scroll/hover motion continuity.\n",
        );
        for (idx, pair) in pairs.iter().enumerate() {
            header.push_str(&format!("{}. {}\n", idx + 1, pair.label));
        }
        blocks.push(Block::Text { text: header });
        for (idx, pair) in pairs.iter().enumerate() {
            let n = idx + 1;
            let diff = resolve_pair_diff(pair)?;
            let orig_label = format!("original-{}", pair.label);
            let built_label = format!("rebuilt-{}", pair.label);
            let orig_fit = fit_image_file_for_api(&pair.original, &orig_label)?;
            let built_fit = fit_image_file_for_api(&pair.built, &built_label)?;
            let orig = Block::image(&orig_label, &orig_fit.bytes)?;
            let built = Block::image(&built_label, &built_fit.bytes)?;
            let heatmap_name = format!("pair-{n:02}-heatmap.png");
            let hm_fit = fit_image_for_api(diff.heatmap_png.clone(), &heatmap_name)?;
            let heatmap = Block::image(&heatmap_name, &hm_fit.bytes)?;
            let scale =
                if hm_fit.original_dims != hm_fit.sent_dims { (hm_fit.original_dims, hm_fit.sent_dims) } else { (orig_fit.original_dims, orig_fit.sent_dims) };

            blocks.push(Block::Text { text: format_quantitative_analysis_with_scale(&pair.label, &diff, Some(scale)) });
            blocks.push(Block::Text { text: format!("=== {} — ORIGINAL ===", pair.label) });
            blocks.push(orig);
            blocks.push(Block::Text { text: format!("=== {} — REBUILT ===", pair.label) });
            blocks.push(built);
            blocks.push(Block::Text { text: format!("=== {} — DIFF HEATMAP ({heatmap_name}) ===", pair.label) });
            blocks.push(heatmap);
            loaded += 1;
        }
        blocks.push(Block::Text {
            text: "Return ONLY the JSON verdict object with `summary` and `defects` (citing `region_id` and coordinates whenever a defect matches a numbered heatmap region), ordered from most severe to least severe."
                .into(),
        });

        let msg = Msg { role: Speaker::User, blocks };
        let messages = [msg];
        let req = TurnRequest { system: BLIND_CRITIC_SYSTEM, messages: &messages, tools: &[], effort: prompt::choose_effort(effort), max_tokens: 2048 };
        let result = {
            let mut guard = model.lock().unwrap_or_else(PoisonError::into_inner);
            guard.turn(&req, &mut |_| {}, stop)?
        };
        let used_model = if result.model.is_empty() { self.model_id.clone() } else { result.model.clone() };
        let cost = crate::price::cost(&used_model, &result.usage, prices, local);
        let raw = result.message.text();
        let report = parse_critic_response(&raw, &self.backend_name, &used_model, loaded).ok_or_else(|| {
            ToolError::new(
                craft_anything_evidence::ErrorCode::GarbledOutput,
                format!("{} returned a visual review that couldn't be parsed.", self.backend_name),
            )
        })?;
        Ok((report, ReviewUsage { usage: result.usage, cost_usd: cost, tokens_unknown: false, cost_is_estimate: false }))
    }

    fn review_cli(
        &self,
        pairs: &[ImagePair],
        entry: &crate::BackendEntry,
        key: Option<&str>,
        logs: Option<&Path>,
        stop: &Stop,
    ) -> Result<(CriticReport, ReviewUsage), ToolError> {
        if entry.config.ssh_host.as_deref().is_some_and(|h| !h.trim().is_empty()) {
            return Err(ToolError::invalid(format!(
                "{} runs over SSH, so it can't inspect local blind-review screenshot copies. Choose a local Claude Code backend or an API vision backend.",
                self.backend_name
            )));
        }
        if !crate::supports_reviewer(&entry.config) {
            return Err(ToolError::invalid(format!(
                "{} can't enforce code-blind, read-only review scoped to the staged screenshot folder. Choose an API vision backend or a local Claude Code backend.",
                self.backend_name
            )));
        }
        let review_dir = TempReviewDir::create()?;
        let (prompt, loaded) = stage_cli_review_dir(pairs, review_dir.path())?;

        let spec = blind_cli_spec(entry, review_dir.path(), key, logs);
        let mut conv = crate::cli_agents::conversation(spec);
        conv.send(&prompt);
        let started = Instant::now();
        let mut raw = String::new();
        let mut review_usage = ReviewUsage { tokens_unknown: true, ..ReviewUsage::default() };
        let mut quiet_since: Option<Instant> = None;
        loop {
            if stop.is_stopped() {
                conv.stop();
                return Err(ToolError::cancelled());
            }
            if started.elapsed() >= CLI_REVIEW_TIMEOUT {
                conv.stop();
                return Err(ToolError::new(
                    craft_anything_evidence::ErrorCode::TimedOut,
                    format!("{} took too long to finish the blind visual review.", self.backend_name),
                ));
            }
            let events = conv.poll();
            let had_events = !events.is_empty();
            for event in events {
                match event {
                    AgentEvent::TextDelta { text } => raw.push_str(&text),
                    AgentEvent::Error { message, .. } => {
                        conv.stop();
                        return Err(ToolError::invalid(message));
                    }
                    AgentEvent::TurnDone { input_tokens, output_tokens, tokens_unknown, cost_usd, cost_is_estimate, .. } => {
                        review_usage =
                            ReviewUsage { usage: Usage { input_tokens, output_tokens, ..Usage::default() }, cost_usd, tokens_unknown, cost_is_estimate };
                        let report = parse_critic_response(&raw, &self.backend_name, &self.model_id, loaded).ok_or_else(|| {
                            ToolError::new(
                                craft_anything_evidence::ErrorCode::GarbledOutput,
                                format!("{} returned a visual review that couldn't be parsed.", self.backend_name),
                            )
                        })?;
                        return Ok((report, review_usage));
                    }
                    _ => {}
                }
            }
            if had_events {
                quiet_since = None;
            } else if !conv.is_busy() && !raw.trim().is_empty() {
                let since = *quiet_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= CLI_QUIET_DONE {
                    let report = parse_critic_response(&raw, &self.backend_name, &self.model_id, loaded).ok_or_else(|| {
                        ToolError::new(
                            craft_anything_evidence::ErrorCode::GarbledOutput,
                            format!("{} returned a visual review that couldn't be parsed.", self.backend_name),
                        )
                    })?;
                    return Ok((report, review_usage));
                }
            }
            std::thread::sleep(Duration::from_millis(30));
        }
    }
}

/// Build the isolated `CliSpec` for a blind visual critic turn running inside `review_dir`.
/// Uses `blind_review: true` (with `builder: None`, `question: false`, `deep: false`) so only
/// adapters that enforce staging-folder-bounded reads (`supports_blind_review`) run, with zero
/// Craft Anything MCP server attached.
fn blind_cli_spec(entry: &crate::BackendEntry, review_dir: &Path, key: Option<&str>, logs: Option<&Path>) -> crate::cli_agents::CliSpec {
    crate::cli_agents::CliSpec {
        blind_review: true,
        deep: false,
        preamble: Some(format!("{BLIND_CRITIC_SYSTEM}\n\n")),
        key: key.map(str::to_string),
        logs: logs.map(Path::to_path_buf),
        ..crate::cli_agents::CliSpec::new(entry.config.clone(), entry.options.clone(), false, Some(review_dir.to_path_buf()))
    }
}

/// Whether `ws` has any original/reference images that a check round can pair against rebuilt
/// screenshots. Checked by [`VisualCriticCheck::applies`] before earlier checks create `round-N/`.
pub fn has_reference_images(ws: &Workspace, widths: &[u32]) -> bool {
    let capture = ws.capture_dir();
    match ws.manifest.target {
        TargetRef::Website { .. } => {
            widths.iter().take(3).any(|w| capture.join(format!("screenshot-{w}.png")).is_file())
                || capture.join(super::kinds::website::MOTION_DIR).join("motion.json").is_file()
                || capture.join(super::kinds::website::CANVAS3D_DIR).join("canvas-frame-02.png").is_file()
        }
        _ => capture.join("window.png").is_file() || ws.references().into_iter().any(|p| p.is_file()),
    }
}

/// Check that runs [`VisualCritic`] over the current round's screenshots and motion burst frames.
pub struct VisualCriticCheck {
    critic: Arc<VisualCritic>,
}

impl VisualCriticCheck {
    pub fn new(critic: Arc<VisualCritic>) -> VisualCriticCheck {
        VisualCriticCheck { critic }
    }
}

/// Whether another builder turn can address a `visual-critic` result: yes when the critic found
/// actual defects (a failed claim), no when the check is incomplete solely because the critic API
/// itself couldn't answer or be parsed (`unknowns` non-empty with no failed defect claim).
pub fn can_refine_result(result: &CheckResult) -> bool {
    result.claims.iter().any(|c| !c.passed) || result.unknowns.is_empty()
}

impl Check for VisualCriticCheck {
    fn id(&self) -> &'static str {
        CHECK_ID
    }

    fn applies(&self, cx: &CheckCx) -> bool {
        has_reference_images(cx.ws, &cx.build_settings().widths())
    }

    fn can_refine(&self, result: &CheckResult) -> bool {
        can_refine_result(result)
    }

    fn run(&self, cx: &mut CheckCx) -> Result<CheckResult, ToolError> {
        let (mut pairs, omitted) = collect_pairs_with_omitted(cx.ws, cx.round, &cx.build_settings().widths());
        if pairs.is_empty() {
            return Ok(CheckResult {
                unknowns: vec!["no original/rebuilt screenshot pairs were available for visual review".into()],
                ..CheckResult::new(self.id(), cx.round, true, "No reference/build screenshot pairs were available for blind review.")
            });
        }
        let dir = cx.ws.checks_dir().join(format!("round-{}", cx.round));
        for (idx, pair) in pairs.iter_mut().enumerate() {
            cx.stop().check()?;
            generate_single_pair_diff(pair, idx + 1, &dir);
        }
        cx.stop().check()?;
        let stop = Stop::for_tool(cx.cancel_token());
        let (report, review_usage) = match self.critic.review(&pairs, &stop) {
            Ok(ok) => ok,
            Err(e) if e.code == craft_anything_evidence::ErrorCode::Cancelled => return Err(e),
            Err(e) => {
                // An external reviewer API outage stays unverified rather than marking a working
                // build as broken or sending the builder on a futile loop.
                return Ok(CheckResult {
                    unknowns: vec![format!("blind visual critic ({}) couldn't finish", self.critic.backend_name)],
                    ..CheckResult::new(
                        self.id(),
                        cx.round,
                        true,
                        format!("Blind visual critic ({}) couldn't complete this round: {}", self.critic.backend_name, e.message),
                    )
                });
            }
        };
        cx.record_model_usage(&review_usage.usage, review_usage.tokens_unknown, review_usage.cost_usd, review_usage.cost_is_estimate);
        if std::fs::create_dir_all(&dir).is_ok()
            && let Ok(bytes) = serde_json::to_vec_pretty(&report)
        {
            let _ = craft_anything_engine::project::write_atomic(&dir.join(REPORT_FILE), &bytes);
        }
        let mut unknowns = Vec::new();
        if omitted > 0 {
            unknowns.push(format!("{omitted} additional screenshot/motion pair(s) were omitted due to the {MAX_PAIRS_PER_ROUND}-pair per-round review cap"));
        }
        let claim_detail = if report.passed { String::new() } else { format!("{} severe, {} moderate defect(s)", report.severe_count, report.moderate_count) };
        let claim = Claim::new(format!("Blind visual critic ({}) finds no severe or moderate defects", report.reviewer), report.passed, claim_detail);
        let note = format_report_note(&report, cx.round);
        Ok(CheckResult { claims: vec![claim], unknowns, ..CheckResult::new(self.id(), cx.round, true, note) })
    }
}

/// Collect comparable original/built screenshot and motion-burst pairs for `round`.
pub fn collect_pairs(ws: &Workspace, round: u32, widths: &[u32]) -> Vec<ImagePair> {
    collect_pairs_with_omitted(ws, round, widths).0
}

/// Collect up to [`MAX_PAIRS_PER_ROUND`] comparable original/built screenshot pairs for `round`
/// (prioritizing keeping at least one hover and one 3D canvas pair when present) and return the
/// count of any candidate pairs omitted due to the per-round cap.
pub fn collect_pairs_with_omitted(ws: &Workspace, round: u32, widths: &[u32]) -> (Vec<ImagePair>, usize) {
    let capture = ws.capture_dir();
    let round_dir = ws.checks_dir().join(format!("round-{round}"));
    let mut pairs = Vec::new();
    match ws.manifest.target {
        TargetRef::Website { .. } => {
            for &w in widths.iter().take(3) {
                let name = format!("screenshot-{w}.png");
                let (orig, built) = (capture.join(&name), round_dir.join(&name));
                if orig.is_file() && built.is_file() {
                    pairs.push(ImagePair { label: format!("viewport {w} px"), original: orig, built, heatmap: None, diff_json: None });
                }
            }
            // Also include representative sequential motion burst frames (first & last of scroll
            // and hover) when both original and built motion captures exist.
            let orig_motion = capture.join(super::kinds::website::MOTION_DIR);
            let built_motion = round_dir.join(super::kinds::website::MOTION_DIR);
            if orig_motion.join("motion.json").is_file() && built_motion.join("motion.json").is_file() {
                for (prefix, title) in [("burst-scroll-", "scroll burst"), ("burst-hover-", "hover burst")] {
                    let matching: Vec<(String, PathBuf, PathBuf)> = (1..=8)
                        .filter_map(|i| {
                            let name = format!("{prefix}{i:02}.png");
                            let (o, b) = (orig_motion.join(&name), built_motion.join(&name));
                            (o.is_file() && b.is_file()).then_some((name, o, b))
                        })
                        .collect();
                    let picks: Vec<usize> = match matching.len() {
                        0 => Vec::new(),
                        1 => vec![0],
                        n => vec![0, n - 1],
                    };
                    for idx in picks {
                        if let Some((name, o, b)) = matching.get(idx) {
                            pairs.push(ImagePair { label: format!("{title} ({name})"), original: o.clone(), built: b.clone(), heatmap: None, diff_json: None });
                        }
                    }
                }
            }
            let orig_canvas = capture.join(super::kinds::website::CANVAS3D_DIR).join("canvas-frame-02.png");
            let built_canvas = round_dir.join(super::kinds::website::CANVAS3D_DIR).join("canvas-frame-02.png");
            if orig_canvas.is_file() && built_canvas.is_file() {
                pairs.push(ImagePair {
                    label: "3D canvas frame (canvas-frame-02.png)".into(),
                    original: orig_canvas,
                    built: built_canvas,
                    heatmap: None,
                    diff_json: None,
                });
            }
        }
        _ => {
            let built_window = round_dir.join("window.png");
            let built_shot = round_dir.join("screenshot.png");
            let orig_window = capture.join("window.png");
            if orig_window.is_file() {
                if built_window.is_file() {
                    pairs.push(ImagePair {
                        label: "running window".into(),
                        original: orig_window.clone(),
                        built: built_window.clone(),
                        heatmap: None,
                        diff_json: None,
                    });
                } else if built_shot.is_file() {
                    pairs.push(ImagePair {
                        label: "window vs --screenshot".into(),
                        original: orig_window.clone(),
                        built: built_shot.clone(),
                        heatmap: None,
                        diff_json: None,
                    });
                }
            }
            let target_shot = if built_shot.is_file() {
                Some(built_shot)
            } else if built_window.is_file() {
                Some(built_window)
            } else {
                None
            };
            if let Some(built) = target_shot {
                for (idx, reference) in ws.references().into_iter().take(2).enumerate() {
                    if reference.is_file() {
                        let name = reference.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| format!("reference-{}", idx + 1));
                        pairs.push(ImagePair { label: format!("reference {name}"), original: reference, built: built.clone(), heatmap: None, diff_json: None });
                    }
                }
            }
        }
    }
    let total = pairs.len();
    if total > MAX_PAIRS_PER_ROUND {
        // Ensure at least one 3D canvas pair and one hover burst pair remain within the cap when present.
        let canvas_pair = pairs.get(MAX_PAIRS_PER_ROUND..).and_then(|rest| rest.iter().find(|p| p.label.starts_with("3D canvas")).cloned());
        let hover_pair = pairs.get(MAX_PAIRS_PER_ROUND..).and_then(|rest| rest.iter().find(|p| p.label.starts_with("hover burst")).cloned());
        pairs.truncate(MAX_PAIRS_PER_ROUND);
        if let Some(cp) = canvas_pair
            && !pairs.iter().any(|p| p.label.starts_with("3D canvas"))
            && let Some(slot) = pairs.iter_mut().rev().find(|p| !p.label.starts_with("hover burst"))
        {
            *slot = cp;
        }
        if let Some(hp) = hover_pair
            && !pairs.iter().any(|p| p.label.starts_with("hover burst"))
            && let Some(slot) = pairs.iter_mut().rev().find(|p| !p.label.starts_with("3D canvas"))
        {
            *slot = hp;
        }
    }
    let omitted = total.saturating_sub(pairs.len());
    (pairs, omitted)
}

/// Image bytes normalized for vision model limits together with original and sent dimensions.
#[derive(Debug, Clone)]
pub struct FittedImage {
    pub bytes: Vec<u8>,
    pub media_type: String,
    pub original_dims: (u32, u32),
    pub sent_dims: (u32, u32),
}

/// Fit raw image bytes within [`MAX_IMAGE_BYTES`] and [`MAX_IMAGE_EDGE`], always decoding and
/// re-encoding as clean pixel-only PNG (stripping any EXIF/camera metadata and ensuring consistent
/// `image/png` output for both API vision calls and staged `.png` files).
pub fn fit_image_for_api(bytes: Vec<u8>, label: &str) -> Result<FittedImage, ToolError> {
    let mut img = decode_picture(&bytes)?;
    let orig_dims = (img.width, img.height);
    let max_side = img.width.max(img.height);
    if max_side > MAX_IMAGE_EDGE {
        let scale = f64::from(MAX_IMAGE_EDGE) / f64::from(max_side);
        let nw = ((f64::from(img.width) * scale).round() as u32).max(1);
        let nh = ((f64::from(img.height) * scale).round() as u32).max(1);
        img = resize(&img, nw, nh);
    }
    for factor in [1.0_f64, 0.75, 0.55, 0.4, 0.25] {
        if (factor - 1.0).abs() < f64::EPSILON {
            if let Some(png) = encode_png_within(&img, MAX_IMAGE_BYTES)? {
                return Ok(FittedImage { bytes: png, media_type: "image/png".to_string(), original_dims: orig_dims, sent_dims: (img.width, img.height) });
            }
        } else {
            let nw = ((f64::from(img.width) * factor).round() as u32).max(1);
            let nh = ((f64::from(img.height) * factor).round() as u32).max(1);
            let shrunk = resize(&img, nw, nh);
            if let Some(png) = encode_png_within(&shrunk, MAX_IMAGE_BYTES)? {
                return Ok(FittedImage { bytes: png, media_type: "image/png".to_string(), original_dims: orig_dims, sent_dims: (shrunk.width, shrunk.height) });
            }
        }
    }
    Err(ToolError::invalid(format!("{label} couldn't be resized under the picture size limit.")))
}

fn fit_image_file_for_api(path: &Path, label: &str) -> Result<FittedImage, ToolError> {
    let bytes = std::fs::read(path).map_err(|e| ToolError::io(path.display(), e))?;
    fit_image_for_api(bytes, label)
}

/// Normalize raw image bytes to fit [`MAX_IMAGE_BYTES`] and [`MAX_IMAGE_EDGE`], downscaling
/// proportionally when needed.
pub fn normalize_image_bytes(bytes: Vec<u8>, label: &str) -> Result<Vec<u8>, ToolError> {
    fit_image_for_api(bytes, label).map(|fit| fit.bytes)
}

/// Load `path` as model-safe image bytes, downscaling proportionally when the image
/// exceeds [`MAX_IMAGE_BYTES`] or [`MAX_IMAGE_EDGE`] (common on full-page captures).
pub fn prepare_image_bytes(path: &Path, label: &str) -> Result<Vec<u8>, ToolError> {
    let bytes = std::fs::read(path).map_err(|e| ToolError::io(path.display(), e))?;
    normalize_image_bytes(bytes, label)
}

/// Load `path` as a model-safe [`Block::Image`], downscaling proportionally when the image
/// exceeds [`MAX_IMAGE_BYTES`] or [`MAX_IMAGE_EDGE`] (common on full-page captures).
pub fn prepare_image(path: &Path, label: &str) -> Result<Block, ToolError> {
    let bytes = prepare_image_bytes(path, label)?;
    Block::image(label, &bytes)
}

/// Parse the blind critic's JSON output (or resilient `[SEVERE]`/`[MODERATE]`/`[MINOR]` lines).
pub fn parse_critic_response(raw: &str, reviewer: &str, model: &str, pairs_reviewed: usize) -> Option<CriticReport> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(v) = extract_json_object(trimmed)
        && let Some(report) = parse_json_verdict(&v, reviewer, model, pairs_reviewed)
    {
        return Some(report);
    }
    parse_tagged_lines(trimmed, reviewer, model, pairs_reviewed)
}

fn extract_json_object(raw: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(raw)
        && v.is_object()
    {
        return Some(v);
    }
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = raw.get(start..=end)?;
    let v = serde_json::from_str::<Value>(slice).ok()?;
    v.is_object().then_some(v)
}

fn parse_json_verdict(v: &Value, reviewer: &str, model: &str, pairs_reviewed: usize) -> Option<CriticReport> {
    let arr = v.get("defects").and_then(Value::as_array)?;
    let summary = v.get("summary").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).unwrap_or("Blind visual review complete.").to_string();
    let mut defects = Vec::new();
    for item in arr {
        let sev = item.get("severity").and_then(Value::as_str).and_then(Severity::parse)?;
        let pair = item.get("pair").or_else(|| item.get("pair_label")).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        let region_id = item.get("region_id").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()).filter(|&n| n > 0);
        let area = item.get("area").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).unwrap_or("general").to_string();
        let desc = item
            .get("summary")
            .or_else(|| item.get("description"))
            .or_else(|| item.get("issue"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())?
            .to_string();
        let fix = item.get("fix").and_then(Value::as_str).map(str::trim).unwrap_or_default().to_string();
        defects.push(VisualDefect { severity: sev, pair, region_id, area, summary: desc, fix });
    }
    Some(CriticReport::from_defects(reviewer.to_string(), model.to_string(), pairs_reviewed, summary, defects))
}

fn parse_tagged_lines(raw: &str, reviewer: &str, model: &str, pairs_reviewed: usize) -> Option<CriticReport> {
    let mut defects = Vec::new();
    for line in raw.lines() {
        let clean = line.trim().trim_start_matches(['-', '*', '•', ' ']);
        let upper = clean.to_ascii_uppercase();
        let found = [
            ("[SEVERE]", Severity::Severe),
            ("SEVERE:", Severity::Severe),
            ("[MODERATE]", Severity::Moderate),
            ("MODERATE:", Severity::Moderate),
            ("[MINOR]", Severity::Minor),
            ("MINOR:", Severity::Minor),
        ]
        .into_iter()
        .find_map(|(prefix, sev)| upper.starts_with(prefix).then(|| (sev, clean.get(prefix.len()..).unwrap_or_default().trim())));
        if let Some((severity, rest)) = found {
            if rest.is_empty() {
                return None;
            }
            let (area, body) =
                rest.split_once(':').map(|(a, b)| (a.trim().to_string(), b.trim().to_string())).unwrap_or_else(|| ("general".to_string(), rest.to_string()));
            if body.is_empty() {
                return None;
            }
            defects.push(VisualDefect { severity, pair: None, region_id: None, area, summary: body, fix: String::new() });
        }
    }
    if defects.is_empty() {
        return None;
    }
    Some(CriticReport::from_defects(reviewer.to_string(), model.to_string(), pairs_reviewed, format!("Found {} visual defect(s).", defects.len()), defects))
}

/// Concise, severity-ordered note placed inside `CheckResult::note`.
pub fn format_report_note(report: &CriticReport, round: u32) -> String {
    if report.passed {
        let minor = match report.minor_count {
            0 => String::new(),
            n => format!(" ({n} minor polish note(s) logged in checks/round-{round}/{REPORT_FILE})"),
        };
        return format!("Blind critic ({}) passed {} pair(s): 0 severe, 0 moderate defects{minor}. {}", report.reviewer, report.pairs_reviewed, report.summary);
    }
    let mut lines = vec![format!(
        "Blind critic ({}) found {} severe and {} moderate defect(s) across {} pair(s) (details in checks/round-{round}/{REPORT_FILE}): {}",
        report.reviewer, report.severe_count, report.moderate_count, report.pairs_reviewed, report.summary
    )];
    for d in report.defects.iter().filter(|d| d.severity != Severity::Minor).take(8) {
        let fix = if d.fix.is_empty() { String::new() } else { format!(" — Fix: {}", d.fix) };
        let pair_missing = d.pair.as_deref().filter(|p| !d.area.contains(p));
        let region_tag = match (pair_missing, d.region_id) {
            (Some(p), Some(rid)) if !d.area.contains(&format!("[{rid}]")) => format!(" ({p}, Region [{rid}])"),
            (Some(p), Some(_)) => format!(" ({p})"),
            (Some(p), None) => format!(" ({p})"),
            (None, Some(rid)) if !d.area.contains(&format!("[{rid}]")) => format!(" (Region [{rid}])"),
            _ => String::new(),
        };
        lines.push(format!("[{}] {}{region_tag}: {}{fix}", d.severity.as_str().to_ascii_uppercase(), d.area, d.summary));
    }
    lines.join(" | ")
}

/// Refinement message sent to the builder when the blind visual critic rejected the build.
/// Blind-safe: never asks a builder with `sees_pictures == false` to open images.
pub fn refine_prompt(check: &CheckResult, sees_pictures: bool) -> String {
    let inspect = if sees_pictures {
        format!(
            " Inspect the comparison pairs and numbered heatmaps (`pair-*-heatmap.png`) in checks/round-{}/ against capture/ (or reference/) and read checks/round-{}/pair-*-diff.json and checks/round-{}/{REPORT_FILE} for exact pixel coordinates.",
            check.round, check.round, check.round
        )
    } else {
        format!(
            " Read checks/round-{}/{REPORT_FILE} and checks/round-{}/pair-*-diff.json (exact `(x, y, width, height)` region coordinates, Delta-E and SSIM) and compare checks/round-{}/capture.json with capture/capture.json.",
            check.round, check.round, check.round
        )
    };
    format!(
        "Check {} (blind visual critic): {}\n\n\
         An independent visual reviewer judged your rendered output strictly from screenshots without seeing your code.{inspect} \
         Fix every [SEVERE] defect first, then every [MODERATE] defect, in app/ (do not skip or rationalize visual or motion flaws), \
         build it again, and reply with 1-2 plain sentences on what you changed.",
        check.round, check.note
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StopReason, TurnResult, Usage};
    use craft_anything_engine::tools::web_common::{Image, encode_png, png_dimensions};

    struct MockCriticModel {
        reply: String,
        seen_images: Arc<Mutex<usize>>,
        seen_tools: Arc<Mutex<usize>>,
        seen_texts: Arc<Mutex<Vec<String>>>,
    }

    impl ChatModel for MockCriticModel {
        fn turn(&mut self, req: &TurnRequest<'_>, _on_delta: &mut dyn FnMut(crate::model::Delta), _stop: &Stop) -> Result<TurnResult, ToolError> {
            *self.seen_tools.lock().unwrap() = req.tools.len();
            let imgs = req.messages.iter().flat_map(|m| &m.blocks).filter(|b| matches!(b, Block::Image { .. })).count();
            *self.seen_images.lock().unwrap() = imgs;
            let texts: Vec<String> = req
                .messages
                .iter()
                .flat_map(|m| &m.blocks)
                .filter_map(|b| match b {
                    Block::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect();
            *self.seen_texts.lock().unwrap() = texts;
            Ok(TurnResult {
                message: Msg { role: Speaker::Assistant, blocks: vec![Block::Text { text: self.reply.clone() }] },
                stop: StopReason::EndTurn,
                usage: Usage { input_tokens: 1200, output_tokens: 180, cache_read_tokens: 0, cache_write_tokens: 0 },
                model: "claude-sonnet-5-5".into(),
                follow_up: None,
            })
        }

        fn model(&self) -> String {
            "claude-sonnet-5-5".into()
        }
    }

    fn write_solid_png(path: &Path, w: u32, h: u32, rgba: [u8; 4]) {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        let img = Image { width: w, height: h, rgba: (0..(w * h)).flat_map(|_| rgba).collect() };
        std::fs::write(path, encode_png(&img).unwrap()).unwrap();
    }

    #[test]
    fn parser_orders_severity_and_accepts_json_fences_or_tagged_lines() {
        let fenced = "Here is the verdict:\n```json\n{\"summary\":\"Hero is misaligned.\",\"defects\":[\
            {\"severity\":\"minor\",\"area\":\"footer\",\"summary\":\"1px gap\",\"fix\":\"tighten\"},\
            {\"severity\":\"severe\",\"pair\":\"viewport 1440 px\",\"region_id\":1,\"area\":\"hero (Region [1] at x=10, y=20, 100x40px)\",\"summary\":\"headline clipped\",\"fix\":\"remove overflow hidden\"},\
            {\"severity\":\"moderate\",\"pair\":\"scroll burst (burst-scroll-01.png)\",\"region_id\":2,\"area\":\"cards\",\"summary\":\"cards jump\",\"fix\":\"use transform\"}\
        ]}\n```";
        let report = parse_critic_response(fenced, "Claude", "claude-sonnet-5-5", 2).unwrap();
        assert!(!report.passed);
        assert_eq!((report.severe_count, report.moderate_count, report.minor_count), (1, 1, 1));
        assert_eq!(report.defects[0].severity, Severity::Severe);
        assert_eq!(report.defects[0].pair.as_deref(), Some("viewport 1440 px"));
        assert_eq!(report.defects[0].region_id, Some(1));
        assert_eq!(report.defects[1].severity, Severity::Moderate);
        assert_eq!(report.defects[1].pair.as_deref(), Some("scroll burst (burst-scroll-01.png)"));
        assert_eq!(report.defects[1].region_id, Some(2));
        assert_eq!(report.defects[2].severity, Severity::Minor);
        assert_eq!(report.defects[2].region_id, None);
        let note = format_report_note(&report, 1);
        assert!(note.contains("(viewport 1440 px)") && note.contains("(scroll burst (burst-scroll-01.png), Region [2])"), "{note}");

        let pass = parse_critic_response(
            r#"{"summary":"Matches cleanly.","defects":[{"severity":"minor","area":"nav","summary":"slight AA","fix":"none"}]}"#,
            "Claude",
            "m",
            2,
        )
        .unwrap();
        assert!(pass.passed, "minor-only defects pass the gate");

        let tagged = "- [MODERATE] navbar: contrast too low\n- [SEVERE] hero: missing gradient";
        let r2 = parse_critic_response(tagged, "OpenAI", "gpt", 1).unwrap();
        assert!(!r2.passed && r2.severe_count == 1 && r2.moderate_count == 1);
        assert_eq!(r2.defects[0].severity, Severity::Severe);

        // Malformed defect items (unknown severity, empty summary, or empty tagged line) and unstructured prose are rejected.
        assert!(parse_critic_response(r#"{"summary":"x","defects":[{"severity":"unknown","summary":"bad"}]}"#, "Claude", "m", 1).is_none());
        assert!(parse_critic_response(r#"{"summary":"x","defects":[{"severity":"severe","summary":"   "}]}"#, "Claude", "m", 1).is_none());
        assert!(parse_critic_response("[SEVERE] navbar:\n[MINOR] footer: 1px gap", "Claude", "m", 1).is_none());
        assert!(parse_critic_response("The page passed and looks great with no severe defects.", "Claude", "m", 1).is_none());
    }

    #[test]
    fn blind_critic_sends_zero_tools_and_pairs_viewport_and_motion_bursts() {
        let d = tempfile::tempdir().unwrap();
        let ws = Workspace::create(d.path(), "Site", "Clone it", TargetRef::Website { url: "https://example.com".into() }).unwrap();
        write_solid_png(&ws.capture_dir().join("screenshot-1440.png"), 64, 48, [10, 20, 30, 255]);
        // Paint a distinct shifted block on the rebuilt screenshot so deterministic clustering detects Region [1].
        let mut built_rgba: Vec<u8> = (0..(64 * 48)).flat_map(|_| [10u8, 20, 30, 255]).collect();
        for y in 8..24 {
            for x in 12..28 {
                let o = (y * 64 + x) * 4;
                built_rgba[o..o + 4].copy_from_slice(&[230, 40, 40, 255]);
            }
        }
        let built_path = ws.checks_dir().join("round-1/screenshot-1440.png");
        std::fs::create_dir_all(built_path.parent().unwrap()).unwrap();
        std::fs::write(&built_path, encode_png(&Image { width: 64, height: 48, rgba: built_rgba }).unwrap()).unwrap();
        // Motion burst frames (1 and 4 picked).
        std::fs::create_dir_all(ws.capture_dir().join("motion")).unwrap();
        std::fs::create_dir_all(ws.checks_dir().join("round-1/motion")).unwrap();
        std::fs::write(ws.capture_dir().join("motion/motion.json"), "{}").unwrap();
        std::fs::write(ws.checks_dir().join("round-1/motion/motion.json"), "{}").unwrap();
        for i in 1..=4 {
            write_solid_png(&ws.capture_dir().join(format!("motion/burst-scroll-{i:02}.png")), 32, 32, [20, 30, 40, 255]);
            write_solid_png(&ws.checks_dir().join(format!("round-1/motion/burst-scroll-{i:02}.png")), 32, 32, [20, 30, 40, 255]);
        }

        let mut pairs = collect_pairs(&ws, 1, &[1440, 390]);
        assert_eq!(pairs.len(), 3, "1 viewport pair + first & last scroll burst pairs: {pairs:?}");
        let round_dir = ws.checks_dir().join("round-1");
        generate_pair_diffs(&mut pairs, &round_dir);
        assert!(round_dir.join("pair-01-heatmap.png").is_file());
        assert!(round_dir.join("pair-01-diff.json").is_file());
        let diff_json: Value = serde_json::from_slice(&std::fs::read(round_dir.join("pair-01-diff.json")).unwrap()).unwrap();
        assert_eq!(diff_json["regions"].as_array().unwrap().len(), 1);
        assert_eq!(diff_json["regions"][0]["id"], 1);
        assert_eq!(diff_json["regions"][0]["x"], 12);
        assert_eq!(diff_json["regions"][0]["y"], 8);

        // 3D/WebGL/WebGPU canvas frame pair is included when both original and built frames exist.
        write_solid_png(&ws.capture_dir().join("canvas3d/canvas-frame-02.png"), 32, 32, [40, 80, 120, 255]);
        write_solid_png(&ws.checks_dir().join("round-1/canvas3d/canvas-frame-02.png"), 32, 32, [40, 80, 120, 255]);
        let pairs_with_3d = collect_pairs(&ws, 1, &[1440, 390]);
        assert_eq!(pairs_with_3d.len(), 4, "{pairs_with_3d:?}");
        assert!(pairs_with_3d.iter().any(|p| p.label == "3D canvas frame (canvas-frame-02.png)"), "{pairs_with_3d:?}");

        let seen_images = Arc::new(Mutex::new(0usize));
        let seen_tools = Arc::new(Mutex::new(99usize));
        let seen_texts = Arc::new(Mutex::new(Vec::new()));
        let slot = ModelSlot {
            name: "Reviewer".into(),
            model: Box::new(MockCriticModel {
                reply: r#"{"summary":"Missing hover glow.","defects":[{"severity":"severe","region_id":1,"area":"hero (Region [1] at x=12, y=8, 16x16px)","summary":"cta invisible","fix":"set contrast"}]}"#
                    .into(),
                seen_images: seen_images.clone(),
                seen_tools: seen_tools.clone(),
                seen_texts: seen_texts.clone(),
            }),
            local: false,
            effort: Some(Effort::Medium),
            max_turns: 1,
            context_tokens: 0,
            tools: Default::default(),
        };
        let critic = VisualCritic::from_slot(slot, BTreeMap::new());
        let (report, review_usage) = critic.review(&pairs, &Stop::default()).unwrap();
        assert_eq!(*seen_tools.lock().unwrap(), 0, "blind critic never receives tools");
        assert_eq!(*seen_images.lock().unwrap(), 9, "3 pairs = 9 images (original + rebuilt + heatmap per pair)");
        let joined_texts = seen_texts.lock().unwrap().join("\n");
        assert!(joined_texts.contains("Quantitative Pixel Analysis"), "{joined_texts}");
        assert!(joined_texts.contains("SSIM:"), "{joined_texts}");
        assert!(joined_texts.contains("Region [1]: x=12, y=8, 16x16px"), "{joined_texts}");
        assert!(joined_texts.contains("DIFF HEATMAP (pair-01-heatmap.png)"), "{joined_texts}");
        assert!(!report.passed && report.severe_count == 1);
        assert_eq!(report.defects[0].region_id, Some(1));
        assert_eq!(review_usage.usage.input_tokens, 1200);
        assert!(review_usage.cost_usd.is_some());
        assert!(!review_usage.tokens_unknown && !review_usage.cost_is_estimate);

        // CLI staging copies original, rebuilt, heatmap PNG, and diff JSON into the isolated review directory.
        let cli_dir = tempfile::tempdir().unwrap();
        let (cli_prompt, staged_count) = stage_cli_review_dir(&pairs, cli_dir.path()).unwrap();
        assert_eq!(staged_count, 3);
        for n in 1..=3 {
            assert!(cli_dir.path().join(format!("pair-{n:02}-original.png")).is_file());
            assert!(cli_dir.path().join(format!("pair-{n:02}-rebuilt.png")).is_file());
            assert!(cli_dir.path().join(format!("pair-{n:02}-heatmap.png")).is_file());
            assert!(cli_dir.path().join(format!("pair-{n:02}-diff.json")).is_file());
        }
        assert!(
            cli_prompt.contains("pair-01-heatmap.png") && cli_prompt.contains("pair-01-diff.json") && cli_prompt.contains("Region [1]: x=12, y=8"),
            "{cli_prompt}"
        );
    }

    #[test]
    fn oversized_full_page_screenshots_are_downscaled_for_vision_apis() {
        let d = tempfile::tempdir().unwrap();
        let tall = d.path().join("tall.png");
        write_solid_png(&tall, 32, 5000, [100, 120, 140, 255]);
        let Block::Image { media_type, data, .. } = prepare_image(&tall, "tall").unwrap() else {
            panic!("expected image block");
        };
        assert_eq!(media_type, "image/png");
        let raw = craft_anything_engine::tools::web_common::b64_decode(&data).unwrap();
        let (w, h) = png_dimensions(&raw).unwrap();
        assert!(h <= MAX_IMAGE_EDGE && w >= 1, "downscaled to {w}x{h}");
    }

    #[test]
    fn has_reference_images_detects_capture_before_round_screenshots_exist() {
        let d = tempfile::tempdir().unwrap();
        let ws = Workspace::create(d.path(), "Site", "Clone it", TargetRef::Website { url: "https://example.com".into() }).unwrap();
        assert!(!has_reference_images(&ws, &[1440, 390]));
        write_solid_png(&ws.capture_dir().join("screenshot-1440.png"), 16, 16, [10, 20, 30, 255]);
        assert!(has_reference_images(&ws, &[1440, 390]), "applies even when checks/round-1/ has not been created yet");
        assert!(collect_pairs(&ws, 1, &[1440, 390]).is_empty());

        let d2 = tempfile::tempdir().unwrap();
        let app_ws = Workspace::create(d2.path(), "App", "Clone it", TargetRef::None).unwrap();
        assert!(!has_reference_images(&app_ws, &[1440]));
        let ref_pic = d2.path().join("ref.png");
        write_solid_png(&ref_pic, 16, 16, [10, 20, 30, 255]);
        app_ws.add_references(&[ref_pic]).unwrap();
        assert!(has_reference_images(&app_ws, &[1440]));
    }

    #[test]
    fn blind_cli_spec_enforces_review_dir_read_scope_and_rejects_unbounded_cli_adapters() {
        use crate::cli_agents::{McpCommand, adapter_for, build_blind_review_args};
        use crate::{BackendConfig, BackendEntry, BackendKind, BackendOptions, supports_reviewer};

        let mcp = McpCommand { command: "craft-anything-cli".into(), args: vec!["mcp".into()], notes: None };
        let dir = Path::new("/tmp/blind-review");

        // Local Claude Code supports staging-folder-bounded blind review via `--tools Read` and `--allowedTools Read(./**)`.
        let claude_entry = BackendEntry {
            config: BackendConfig {
                name: "Claude Code".into(),
                kind: BackendKind::AgentCli,
                base_url: None,
                model: "claude-code".into(),
                ssh_host: None,
                enabled: true,
            },
            options: BackendOptions::default(),
        };
        assert!(supports_reviewer(&claude_entry.config));
        let spec = blind_cli_spec(&claude_entry, dir, None, None);
        assert!(spec.blind_review && !spec.question && spec.builder.is_none() && !spec.deep);
        assert!(spec.preamble.as_deref().is_some_and(|p| p.ends_with("\n\n")));
        let args =
            build_blind_review_args(adapter_for("claude-code").as_ref(), &claude_entry.options, "Review", spec.preamble.as_deref().unwrap_or_default(), &mcp);
        assert!(!args.iter().any(|a| a.contains("craft-anything-cli")), "must not attach craft-anything MCP: {args:?}");
        assert!(args.contains(&"--strict-mcp-config".to_string()) && args.contains(&"--no-session-persistence".to_string()), "{args:?}");
        assert!(args.windows(2).any(|w| w == ["--tools", "Read"]), "only Read tool enabled: {args:?}");
        assert!(args.windows(2).any(|w| w == ["--allowedTools", "Read(./**)"]), "reads bounded to staged folder: {args:?}");
        assert!(args.contains(&"Glob".to_string()) && args.contains(&"Grep".to_string()) && args.contains(&"Bash".to_string()), "{args:?}");

        // API vision backends are supported; agent CLIs without review_dir read confinement or running over SSH are refused.
        let api_cfg = BackendConfig {
            name: "Anthropic".into(),
            kind: BackendKind::Anthropic,
            base_url: None,
            model: "claude-sonnet-5-5".into(),
            ssh_host: None,
            enabled: true,
        };
        assert!(supports_reviewer(&api_cfg));

        for unsupported in ["codex", "antigravity", "opencode", "gemini-cli"] {
            let entry = BackendEntry {
                config: BackendConfig {
                    name: unsupported.into(),
                    kind: BackendKind::AgentCli,
                    base_url: None,
                    model: unsupported.into(),
                    ssh_host: None,
                    enabled: true,
                },
                options: BackendOptions::default(),
            };
            assert!(!supports_reviewer(&entry.config), "{unsupported} must not be accepted as a blind CLI reviewer");
            let critic = VisualCritic::from_cli(entry, None, None);
            let err = critic.review(&[], &Stop::default()).unwrap_err();
            assert!(err.message.contains("scoped to the staged screenshot folder"), "{unsupported}: {}", err.message);
        }

        // SSH-backed Claude Code cannot read local TempReviewDir copies and is refused cleanly.
        let ssh_entry = BackendEntry {
            config: BackendConfig {
                name: "Remote Claude".into(),
                kind: BackendKind::AgentCli,
                base_url: None,
                model: "claude-code".into(),
                ssh_host: Some("devbox".into()),
                enabled: true,
            },
            options: BackendOptions::default(),
        };
        assert!(!supports_reviewer(&ssh_entry.config));
        let critic = VisualCritic::from_cli(ssh_entry, None, None);
        let err = critic.review(&[], &Stop::default()).unwrap_err();
        assert!(err.message.contains("SSH"), "{}", err.message);
    }
}
