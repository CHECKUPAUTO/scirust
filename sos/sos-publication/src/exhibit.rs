//! [`FigureSpec`] and [`TableSpec`] — the regenerable exhibits of a publication.
//!
//! An exhibit is not an image file; it is a **recipe**: the source objects it is
//! computed from, the named deterministic renderer/transform that produces it,
//! its deterministic parameters, and — optionally — the content address the
//! rendered artifact is *expected* to hash to. This is what makes a figure
//! checkable: [`verify_exhibits`](crate::verify::verify_exhibits) compares a
//! freshly re-rendered artifact's address against `expected` and reports drift.
//! Producing the artifact is the Workflow Engine's job; this crate records the
//! recipe and checks the result — it never renders pixels or executes anything.

use serde::{Deserialize, Serialize};
use sos_core::ObjectId;
use sos_core::canonical::{Canonical, CanonicalEncoder};

use crate::key::{FigureKey, TableKey};

/// The name (and version) of a deterministic renderer or table transform, e.g.
/// `"scatter@1"`. It is an identifier, not code: this crate never invokes it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RendererId(pub String);

impl RendererId {
    /// Construct a renderer id.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl Canonical for RendererId {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.0);
    }
}

/// A deterministic parameter value. Deliberately float-free — the canonical
/// encoder is float-free, and a figure parameter that decided output must be
/// exactly representable (fixed-point integers, text, flags), never a
/// non-portable `f64` bit pattern.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ParamValue {
    /// A signed integer (use fixed-point millis/micros for fractional values).
    Int(i64),
    /// A text value.
    Text(String),
    /// A boolean flag.
    Flag(bool),
}

impl ParamValue {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::Int(_) => 0,
            Self::Text(_) => 1,
            Self::Flag(_) => 2,
        }
    }
}

impl Canonical for ParamValue {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        match self
        {
            Self::Int(v) => enc.i64(*v),
            Self::Text(s) => enc.str(s),
            Self::Flag(b) => enc.bool(*b),
        }
    }
}

/// A named deterministic parameter to a renderer/transform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    /// The parameter name.
    pub key: String,
    /// The parameter value.
    pub value: ParamValue,
}

impl Param {
    /// Construct a parameter.
    #[must_use]
    pub fn new(key: impl Into<String>, value: ParamValue) -> Self {
        Self {
            key: key.into(),
            value,
        }
    }
}

impl Canonical for Param {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.key);
        enc.value(&self.value);
    }
}

/// The media type of a rendered exhibit artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MediaType {
    /// Scalable Vector Graphics.
    Svg,
    /// PNG raster image.
    Png,
    /// Comma-separated values (a table export).
    Csv,
    /// JSON (a structured table/figure export).
    Json,
}

impl MediaType {
    const fn discriminant(self) -> u64 {
        match self
        {
            Self::Svg => 0,
            Self::Png => 1,
            Self::Csv => 2,
            Self::Json => 3,
        }
    }

    /// The conventional MIME type.
    #[must_use]
    pub const fn mime(self) -> &'static str {
        match self
        {
            Self::Svg => "image/svg+xml",
            Self::Png => "image/png",
            Self::Csv => "text/csv",
            Self::Json => "application/json",
        }
    }
}

impl Canonical for MediaType {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
    }
}

/// How strictly an exhibit must be regenerable from the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RegenPolicy {
    /// The exhibit **must** re-render to its expected address; drift is a failure.
    MustRegenerate,
    /// A cached artifact is acceptable; re-rendering is not required to match.
    CachedAcceptable,
    /// A static, hand-authored asset (a schematic); not derived from the graph
    /// and therefore not reproducibility-checked.
    StaticAsset,
}

impl RegenPolicy {
    const fn discriminant(self) -> u64 {
        match self
        {
            Self::MustRegenerate => 0,
            Self::CachedAcceptable => 1,
            Self::StaticAsset => 2,
        }
    }

    /// Whether an exhibit under this policy is subject to the drift check.
    #[must_use]
    pub const fn is_checked(self) -> bool {
        matches!(self, Self::MustRegenerate)
    }
}

impl Canonical for RegenPolicy {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
    }
}

/// A figure recipe: what it is computed from, how, and what it should hash to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FigureSpec {
    /// The figure's handle within its publication.
    pub key: FigureKey,
    /// A human-readable caption.
    pub caption: String,
    /// Accessible alternative text (never empty for a real figure).
    pub alt_text: String,
    /// The source objects the figure is computed from.
    pub sources: Vec<ObjectId>,
    /// The named deterministic renderer.
    pub renderer: RendererId,
    /// The renderer's deterministic parameters (held sorted by `key`).
    pub params: Vec<Param>,
    /// The artifact's media type.
    pub media_type: MediaType,
    /// The content address the rendered artifact is expected to hash to, if the
    /// figure has been rendered and pinned. `None` = not yet pinned.
    pub expected: Option<ObjectId>,
    /// How strictly the figure must regenerate.
    pub regeneration: RegenPolicy,
}

impl FigureSpec {
    /// A figure `key` rendered by `renderer` from `sources`. Parameters default
    /// empty (add with [`with_params`](Self::with_params)); `expected` is unpinned;
    /// policy defaults to [`RegenPolicy::MustRegenerate`].
    #[must_use]
    pub fn new(
        key: impl Into<FigureKey>,
        caption: impl Into<String>,
        alt_text: impl Into<String>,
        sources: Vec<ObjectId>,
        renderer: RendererId,
        media_type: MediaType,
    ) -> Self {
        Self {
            key: key.into(),
            caption: caption.into(),
            alt_text: alt_text.into(),
            sources,
            renderer,
            params: Vec::new(),
            media_type,
            expected: None,
            regeneration: RegenPolicy::MustRegenerate,
        }
    }

    /// Set the deterministic parameters (sorted by key for a stable address).
    #[must_use]
    pub fn with_params(mut self, mut params: Vec<Param>) -> Self {
        params.sort_by(|a, b| a.key.cmp(&b.key));
        self.params = params;
        self
    }

    /// Pin the expected rendered-artifact address.
    #[must_use]
    pub fn expecting(mut self, artifact: ObjectId) -> Self {
        self.expected = Some(artifact);
        self
    }

    /// Set the regeneration policy.
    #[must_use]
    pub fn regeneration(mut self, policy: RegenPolicy) -> Self {
        self.regeneration = policy;
        self
    }
}

impl Canonical for FigureSpec {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.key);
        enc.str(&self.caption);
        enc.str(&self.alt_text);
        enc.seq(&self.sources);
        enc.value(&self.renderer);
        enc.seq(&self.params);
        enc.value(&self.media_type);
        enc.option(&self.expected);
        enc.value(&self.regeneration);
    }
}

/// A column of a [`TableSpec`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnDef {
    /// The column header.
    pub name: String,
    /// The column's unit, if any (e.g. `"s"`, `"AU"`).
    pub unit: Option<String>,
}

impl ColumnDef {
    /// A column with no unit.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: None,
        }
    }

    /// A column with a unit.
    #[must_use]
    pub fn with_unit(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            unit: Some(unit.into()),
        }
    }
}

impl Canonical for ColumnDef {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.str(&self.name);
        enc.option(&self.unit);
    }
}

/// The deterministic row ordering of a table — required for a stable data
/// address (unordered rows are not content-addressable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Ordering {
    /// Rows appear in the order the producing computation emitted them (the
    /// producer is itself deterministic).
    AsProduced,
    /// Rows are sorted ascending by the listed column indices, in priority order.
    ByColumns(Vec<u64>),
}

impl Ordering {
    const fn discriminant(&self) -> u64 {
        match self
        {
            Self::AsProduced => 0,
            Self::ByColumns(_) => 1,
        }
    }
}

impl Canonical for Ordering {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.u64(self.discriminant());
        if let Self::ByColumns(cols) = self
        {
            enc.seq(cols);
        }
    }
}

/// A table recipe: its columns, deterministic ordering, source objects, and the
/// canonical data address it should reduce to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableSpec {
    /// The table's handle within its publication.
    pub key: TableKey,
    /// A human-readable caption.
    pub caption: String,
    /// The source objects the table is computed from.
    pub sources: Vec<ObjectId>,
    /// The column definitions, in display order.
    pub columns: Vec<ColumnDef>,
    /// The deterministic row ordering.
    pub ordering: Ordering,
    /// The content address the table's canonical data is expected to hash to, if
    /// pinned. `None` = not yet pinned.
    pub expected: Option<ObjectId>,
    /// How strictly the table must regenerate.
    pub regeneration: RegenPolicy,
}

impl TableSpec {
    /// A table `key` with `columns`, computed from `sources`, ordered by
    /// `ordering`. `expected` is unpinned; policy defaults to
    /// [`RegenPolicy::MustRegenerate`].
    #[must_use]
    pub fn new(
        key: impl Into<TableKey>,
        caption: impl Into<String>,
        sources: Vec<ObjectId>,
        columns: Vec<ColumnDef>,
        ordering: Ordering,
    ) -> Self {
        Self {
            key: key.into(),
            caption: caption.into(),
            sources,
            columns,
            ordering,
            expected: None,
            regeneration: RegenPolicy::MustRegenerate,
        }
    }

    /// Pin the expected canonical-data address.
    #[must_use]
    pub fn expecting(mut self, data: ObjectId) -> Self {
        self.expected = Some(data);
        self
    }

    /// Set the regeneration policy.
    #[must_use]
    pub fn regeneration(mut self, policy: RegenPolicy) -> Self {
        self.regeneration = policy;
        self
    }
}

impl Canonical for TableSpec {
    fn encode(&self, enc: &mut CanonicalEncoder) {
        enc.value(&self.key);
        enc.str(&self.caption);
        enc.seq(&self.sources);
        enc.seq(&self.columns);
        enc.value(&self.ordering);
        enc.option(&self.expected);
        enc.value(&self.regeneration);
    }
}
