# Behaviour oracle: a scripted experiment run against the original and against the build

Status: proposal · Scope: `crates/engine/src/oracle.rs`, `crates/ai/src/rebuild/oracle_console.rs` ·
Roadmap: R2.5 (behaviour checks for non-web builds).

## The problem

The website check already compares a build with the original: Study photographs the original,
Check photographs the build, and the two are compared. That comparison has an oracle behind it —
the original's own pictures.

A console program has no such comparison. Its check runs the original with a few inputs and runs
the build with the same inputs, and compares the words they printed. That is a real comparison,
but it says nothing about the *behaviour* the words came from, and a build that prints the right
words while doing the wrong thing passes.

The games go further and are not compared at all. The build's `--self-test` report says the core
loop works, and the build wrote that report.

## The seam

A behaviour oracle runs the same scripted experiment against the original and against the build,
and compares what the two saw. The recording and the replay are separate on purpose: **the script
must come from the original, never from the build.** A build that chooses its own test chooses its
own oracle, which is the mistake the self-test makes today.

```rust
pub trait BehaviorOracle: Send + Sync {
    fn id(&self) -> &'static str;
    fn supports(&self, target: &TargetRef) -> bool;
    fn record(&self, cx: &OracleCx) -> Result<Option<Script>, ToolError>;
    fn replay(&self, cx: &OracleCx, script: &Script) -> Result<Vec<Observation>, ToolError>;
    fn compare(&self, script: &Script, build: &[Observation]) -> Vec<Difference> {
        compare(script, build)
    }
}
```

### What a script is

A script is the steps and what each step read back:

```rust
pub struct Script {
    pub version: u32,
    pub oracle: String,
    pub steps: Vec<Value>,
    pub probes: Vec<Probe>,
    /// What the original was seen to do, one observation per step.
    pub original: Vec<Observation>,
}
```

The observations are one per step, so a sequence is compared step by step: the value of a
sequence is in what each step changed, not only in where it ended up.

### What an observation is

```rust
pub struct Observation {
    pub oracle: String,
    pub version: u32,
    /// One entry per probe, in the script's order.
    ///
    /// A probe that could not be read is `None`, not a default value: a missing reading is not a
    /// reading of nothing, and must never count as alike.
    pub readings: Vec<Option<Reading>>,
    pub gaps: Vec<String>,
}
```

### Three rules, the same for every oracle

| Difference | Meaning | Effect |
|---|---|---|
| `Value` | The two sides read a different value | A failed claim |
| `Missing` | One side could read it and the other could not | A failed claim |
| `Unreadable` | Neither side could read it | An unknown, never a pass |

The last rule is the important one. A probe that could not be read produces an unknown, and an
unknown keeps the check from being good enough. That is the rule the mechanics sheet already
follows, generalised.

### What the build is, is a capability

The build can be a program to start, a page to drive, or a report to read. Those are separate
capabilities, each with its own trait, so an oracle declares only the ones it has:

| Trait | The build is | Held by |
|---|---|---|
| `RunsAProgram` | a program to start | `TheProgram` |
| `DrivesAPage` | a page to drive | `ThePage` |
| `ReadsAReport` | a report to read | `TheReport` |
| `DrivesAnApp` | a program to start and watch | `TheApp` |

The set is open: a new kind of capability is a new `Capability`, and nothing else has to change
to make room for it.

The context holds them as trait objects, so the caller says what the build is and the oracle
asks for what it needs. An oracle that needs something the build does not have is refused
plainly, before it starts:

> "The web oracle needs something this build does not have."

### An oracle drives the engine, it does not open its own channel

`OracleCx` reaches the engine through one function:

```rust
pub struct OracleCx<'a> {
    pub call: &'a dyn Fn(&str, Value) -> Result<ToolOutput, ToolError>,
    pub target: &'a TargetRef,
    pub deep: bool,
    pub program: Option<&'a dyn RunsAProgram>,
    pub page: Option<&'a dyn DrivesAPage>,
    pub report: Option<&'a dyn ReadsAReport>,
}
```

So an oracle drives the same tools the builder does and inherits everything the engine
guarantees: external programs run through the platform layer, the session lock is held correctly,
and a run is cancelled when the person pauses. **An oracle never starts a process of its own.**

## The console oracle

The first one. Study runs the original with each of its inputs and keeps what it printed and how
it exited — in the same pass that captures the runs for the study's notes, so the original is not
run twice. Those observations are the script.

The check runs the build with the same inputs and compares:

- the exit code, exactly;
- what was printed, as words, so a program that prints the same words in another order has
  behaved the same way.

The names of both programs are replaced by one word before the comparison, so a usage line naming
the rebuild still matches the original's.

### What it does not do

It does not prove equivalence. It compares a finite, chosen set of behaviours; the original's
behaviour outside them is unknown, and stays unknown.

## Where a script is stored

Not in `craft.json`. The manifest is the project's index — its status, its stack, its approvals —
and a script with its steps and observations is bulk. It belongs beside the capture, in the
project's own folder:

```
capture/
  script.json          the script and the original's observations
```

`craft.json` gains one optional field, `script`, naming that file. Because the field is optional
and the manifest is versioned, adding it is one migration step: a project made before this has no
script and keeps the checks it had.

## Verification

Each of these needs a test that fails before it and passes after:

- Two runs of the same program compare equal.
- A program that differs in one observable way is reported as differing in that way and no other.
- A step neither side could read is an unknown, never a pass.
- A build that stopped early is missing the steps it did not reach.
