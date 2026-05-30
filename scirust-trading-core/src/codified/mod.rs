use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodifiedEvent {
    pub id: Uuid,
    pub source: SourceId,
    pub raw_text: String,
    pub detected_at: DateTime<Utc>,
    pub enrichment: EnrichmentLevel,
    pub category: Option<Category>,
    pub tags: Vec<String>,
    pub polarity: Option<f32>,
    pub magnitude: Option<f32>,
    pub semantic_confidence: Option<f32>,
    pub semantic_summary: Option<String>,
}

impl CodifiedEvent {
    pub fn builder(source: SourceId, raw_text: impl Into<String>) -> CodifiedEventBuilder {
        CodifiedEventBuilder {
            source,
            raw_text: raw_text.into(),
            detected_at: Utc::now(),
            enrichment: EnrichmentLevel::Raw,
            category: None,
            tags: Vec::new(),
            polarity: None,
            magnitude: None,
            semantic_confidence: None,
            semantic_summary: None,
        }
    }
}

pub struct CodifiedEventBuilder {
    source: SourceId,
    raw_text: String,
    detected_at: DateTime<Utc>,
    enrichment: EnrichmentLevel,
    category: Option<Category>,
    tags: Vec<String>,
    polarity: Option<f32>,
    magnitude: Option<f32>,
    semantic_confidence: Option<f32>,
    semantic_summary: Option<String>,
}

impl CodifiedEventBuilder {
    pub fn build(self) -> CodifiedEvent {
        CodifiedEvent {
            id: Uuid::new_v4(),
            source: self.source,
            raw_text: self.raw_text,
            detected_at: self.detected_at,
            enrichment: self.enrichment,
            category: self.category,
            tags: self.tags,
            polarity: self.polarity,
            magnitude: self.magnitude,
            semantic_confidence: self.semantic_confidence,
            semantic_summary: self.semantic_summary,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnrichmentLevel {
    Raw,
    Structural,
    Semantic,
    Contextual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceId(String);
impl SourceId {
    pub fn new(s: impl Into<String>) -> Self { Self(s.into()) }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Category {
    Macro, Regulatory, ExchangeEvent, OnChain, Narrative, Technical, Liquidation, Funding
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EventTiming { Instant, Imminent, Ongoing, Historical }
