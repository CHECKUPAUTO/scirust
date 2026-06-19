//! Named Entity Recognition (NER).
//!
//! Provides:
//! - Rule-based NER with pattern matching and gazetteer lookup.
//! - Simple statistical NER using hand-crafted features + majority classifier.
//! - BIO tagging scheme for entity segmentation.
//! - Entity extraction and classification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Entity types
// ---------------------------------------------------------------------------

/// The type of a named entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percentage,
    Misc,
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Person => write!(f, "PER"),
            Self::Organization => write!(f, "ORG"),
            Self::Location => write!(f, "LOC"),
            Self::Date => write!(f, "DATE"),
            Self::Time => write!(f, "TIME"),
            Self::Money => write!(f, "MONEY"),
            Self::Percentage => write!(f, "PCT"),
            Self::Misc => write!(f, "MISC"),
        }
    }
}

impl EntityType {
    /// Parse a short BIO tag like "B-PER" into `(IsBegin, EntityType)`.
    pub fn from_bio_tag(tag: &str) -> Option<(bool, Self)> {
        let (prefix, label) = match tag.split_once('-')
        {
            Some(("B", rest)) => (true, rest),
            Some(("I", rest)) => (false, rest),
            _ => return None,
        };
        let ty = match label
        {
            "PER" | "PERSON" => Self::Person,
            "ORG" | "ORGANIZATION" => Self::Organization,
            "LOC" | "LOCATION" | "GPE" => Self::Location,
            "DATE" => Self::Date,
            "TIME" => Self::Time,
            "MONEY" => Self::Money,
            "PCT" | "PERCENT" => Self::Percentage,
            "MISC" => Self::Misc,
            _ => return None,
        };
        Some((prefix, ty))
    }
}

// ---------------------------------------------------------------------------
// BIO tag
// ---------------------------------------------------------------------------

/// A BIO label for a single token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BioTag {
    /// Beginning of an entity of the given type.
    Begin(EntityType),
    /// Inside (continuation) of an entity of the given type.
    Inside(EntityType),
    /// Outside any entity.
    Outside,
}

impl BioTag {
    /// Convert to a string tag like `"B-PER"`.
    pub fn as_str(&self) -> String {
        match self
        {
            Self::Begin(t) => format!("B-{}", t),
            Self::Inside(t) => format!("I-{}", t),
            Self::Outside => "O".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Gazetteer-based NER
// ---------------------------------------------------------------------------

/// A gazetteer mapping surface forms to entity types.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Gazetteer {
    entries: HashMap<String, EntityType>,
}

impl Gazetteer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a single entry.
    pub fn add(&mut self, name: &str, entity_type: EntityType) {
        self.entries.insert(name.to_lowercase(), entity_type);
    }

    /// Bulk-add entries.
    pub fn add_many(&mut self, entries: &[(&str, EntityType)]) {
        for &(name, ty) in entries
        {
            self.add(name, ty);
        }
    }

    /// Look up a token (case-insensitive).
    pub fn lookup(&self, token: &str) -> Option<EntityType> {
        self.entries.get(&token.to_lowercase()).copied()
    }
}

// ---------------------------------------------------------------------------
// Rule-based NER
// ---------------------------------------------------------------------------

/// A single extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
    pub score: f64,
}

/// Rule-based NER engine combining gazetteer lookup with regular-expression
/// patterns for numeric entities.
pub struct RuleBasedNer {
    gazetteer: Gazetteer,
    /// Additional patterns: (regex-like substring match, entity type).
    patterns: Vec<(String, EntityType, f64)>,
}

impl RuleBasedNer {
    pub fn new(gazetteer: Gazetteer) -> Self {
        Self {
            gazetteer,
            patterns: Vec::new(),
        }
    }

    /// Add a simple substring pattern with a confidence score.
    pub fn add_pattern(&mut self, substr: &str, entity_type: EntityType, score: f64) {
        self.patterns.push((substr.to_string(), entity_type, score));
    }

    /// Extract entities from `text`.
    pub fn extract(&self, text: &str) -> Vec<Entity> {
        let mut entities = Vec::new();
        let lower = text.to_lowercase();

        // 1. Gazetteer lookup — match longest spans first
        let mut matches: Vec<(usize, usize, EntityType)> = Vec::new();
        for (key, &ty) in &self.gazetteer.entries
        {
            let mut search_from = 0;
            while let Some(pos) = lower[search_from..].find(key.as_str())
            {
                let abs = search_from + pos;
                matches.push((abs, abs + key.len(), ty));
                search_from = abs + 1;
            }
        }
        // Sort by start, then by length descending (prefer longer match)
        matches.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));

        // Greedy non-overlapping selection
        let mut last_end = 0;
        for (start, end, ty) in &matches
        {
            if *start >= last_end
            {
                entities.push(Entity {
                    text: text[*start..*end].to_string(),
                    entity_type: *ty,
                    start: *start,
                    end: *end,
                    score: 0.9,
                });
                last_end = *end;
            }
        }

        // 2. Pattern-based matches for dates, money, percentages
        for (pattern, ty, score) in &self.patterns
        {
            let mut search_from = 0;
            let pat_lower = pattern.to_lowercase();
            while let Some(pos) = lower[search_from..].find(pat_lower.as_str())
            {
                let abs = search_from + pos;
                let end = abs + pattern.len();
                // Only add if not already covered
                if !entities.iter().any(|e| e.start <= abs && e.end > abs)
                {
                    entities.push(Entity {
                        text: text[abs..end].to_string(),
                        entity_type: *ty,
                        start: abs,
                        end,
                        score: *score,
                    });
                }
                search_from = abs + 1;
            }
        }

        // 3. Capitalized-word heuristic: sequences of TitleCase words
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut char_offset = 0;
        let mut cap_start: Option<usize> = None;
        let mut cap_words: Vec<String> = Vec::new();
        for word in &words
        {
            let word_start = text[char_offset..]
                .find(word)
                .map(|p| char_offset + p)
                .unwrap_or(char_offset);
            let word_end = word_start + word.len();
            let is_cap = word
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
                && word.len() > 1;
            if is_cap && !word.chars().all(|c| c.is_ascii_uppercase())
            {
                // TitleCase word — likely part of a named entity
                if cap_start.is_none()
                {
                    cap_start = Some(word_start);
                }
                cap_words.push(word.to_string());
            }
            else
            {
                if cap_words.len() >= 2
                {
                    // Multi-word capitalized span → likely Person or Organization
                    let span_start = cap_start.unwrap();
                    let span_text = cap_words.join(" ");
                    if !entities
                        .iter()
                        .any(|e| e.start <= span_start && e.end >= word_start)
                    {
                        entities.push(Entity {
                            text: span_text,
                            entity_type: EntityType::Person,
                            start: span_start,
                            end: word_start,
                            score: 0.6,
                        });
                    }
                }
                cap_start = None;
                cap_words.clear();
            }
            char_offset = word_end + 1; // +1 for space
        }
        // flush trailing capitalized run
        if cap_words.len() >= 2
        {
            let span_start = cap_start.unwrap();
            let span_text = cap_words.join(" ");
            if !entities.iter().any(|e| e.start <= span_start)
            {
                entities.push(Entity {
                    text: span_text,
                    entity_type: EntityType::Person,
                    start: span_start,
                    end: text.len(),
                    score: 0.6,
                });
            }
        }

        entities.sort_by_key(|e| e.start);
        entities
    }
}

// ---------------------------------------------------------------------------
// Statistical NER: feature extraction
// ---------------------------------------------------------------------------

/// Features extracted from a token in context for statistical NER.
#[derive(Debug, Clone)]
pub struct TokenFeatures {
    pub is_capitalized: bool,
    pub is_all_upper: bool,
    pub is_title_case: bool,
    pub contains_digit: bool,
    pub is_hyphenated: bool,
    pub is_initial: bool,
    pub word_shape: String,
    pub prev_shape: String,
    pub next_shape: String,
    pub position_in_sentence: usize,
    pub sentence_length: usize,
    pub gazetteer_match: Option<EntityType>,
}

/// Compute a simplified word shape: uppercase → 'X', lowercase → 'x',
/// digit → 'd', other → '_'.
pub fn word_shape(word: &str) -> String {
    let mut shape = String::with_capacity(word.len());
    for c in word.chars()
    {
        if c.is_ascii_uppercase()
        {
            shape.push('X');
        }
        else if c.is_ascii_lowercase()
        {
            shape.push('x');
        }
        else if c.is_ascii_digit()
        {
            shape.push('d');
        }
        else
        {
            shape.push('_');
        }
    }
    // Collapse consecutive identical characters
    let mut collapsed = String::with_capacity(shape.len());
    let mut last = '\0';
    for c in shape.chars()
    {
        if c != last
        {
            collapsed.push(c);
            last = c;
        }
    }
    collapsed
}

/// Extract features for every token in `tokens`.
pub fn extract_features(tokens: &[String], gazetteer: &Gazetteer) -> Vec<TokenFeatures> {
    let n = tokens.len();
    let mut features = Vec::with_capacity(n);
    for (i, tok) in tokens.iter().enumerate()
    {
        let is_cap = tok
            .chars()
            .next()
            .map(|c| c.is_ascii_uppercase())
            .unwrap_or(false);
        let is_all_upper = tok.chars().all(|c| c.is_ascii_uppercase()) && tok.len() > 1;
        let is_title = is_cap && tok.chars().skip(1).all(|c| c.is_ascii_lowercase());
        let contains_digit = tok.chars().any(|c| c.is_ascii_digit());
        let is_hyphenated = tok.contains('-');
        let is_initial = tok.len() == 2 && tok.ends_with('.') && is_cap;

        let prev_shape = if i > 0
        {
            word_shape(&tokens[i - 1])
        }
        else
        {
            String::new()
        };
        let next_shape = if i + 1 < n
        {
            word_shape(&tokens[i + 1])
        }
        else
        {
            String::new()
        };

        features.push(TokenFeatures {
            is_capitalized: is_cap,
            is_all_upper,
            is_title_case: is_title,
            contains_digit,
            is_hyphenated,
            is_initial,
            word_shape: word_shape(tok),
            prev_shape,
            next_shape,
            position_in_sentence: i,
            sentence_length: n,
            gazetteer_match: gazetteer.lookup(tok),
        });
    }
    features
}

// ---------------------------------------------------------------------------
// Statistical NER: simple majority classifier
// ---------------------------------------------------------------------------

/// A very simple statistical NER classifier.  For each token, it produces
/// a BIO tag based on hand-crafted feature rules with tunable weights.
pub struct StatisticalNer {
    gazetteer: Gazetteer,
}

impl StatisticalNer {
    pub fn new(gazetteer: Gazetteer) -> Self {
        Self { gazetteer }
    }

    /// Classify a sentence into BIO tags.
    pub fn classify(&self, tokens: &[String]) -> Vec<BioTag> {
        let features = extract_features(tokens, &self.gazetteer);
        let mut tags = Vec::with_capacity(tokens.len());

        for feat in &features
        {
            let tag = self.classify_token(feat);
            tags.push(tag);
        }
        // Enforce BIO consistency: an I- tag must follow a B- or I- of the same type
        for i in 1..tags.len()
        {
            if let BioTag::Inside(ty) = tags[i]
            {
                if !matches!(tags[i - 1], BioTag::Begin(t) | BioTag::Inside(t) if t == ty)
                {
                    tags[i] = BioTag::Begin(ty);
                }
            }
        }
        tags
    }

    fn classify_token(&self, feat: &TokenFeatures) -> BioTag {
        // Gazetteer match → highest priority
        if let Some(ty) = feat.gazetteer_match
        {
            if feat.is_capitalized
            {
                return BioTag::Begin(ty);
            }
            else
            {
                return BioTag::Inside(ty);
            }
        }
        // All caps + not first word → likely org abbreviation
        if feat.is_all_upper && feat.position_in_sentence > 0
        {
            return BioTag::Begin(EntityType::Organization);
        }
        // Contains digit patterns
        if feat.contains_digit
        {
            // Date-like: 12/31/2024, 2024-01-01, Jan 1
            if feat.word_shape.contains('d')
                && (feat.word_shape.contains('_') || feat.prev_shape == "Xx")
            {
                return BioTag::Begin(EntityType::Date);
            }
            // Money-like: $100, €50
            if feat.prev_shape == "_" || feat.word_shape.starts_with('_')
            {
                return BioTag::Inside(EntityType::Money);
            }
        }
        // Title case at start of sentence → possible person/location
        if feat.is_title_case && feat.position_in_sentence == 0
        {
            return BioTag::Begin(EntityType::Person);
        }
        // Title case mid-sentence following a title case → inside entity
        if feat.is_title_case && feat.prev_shape == "Xx"
        {
            return BioTag::Inside(EntityType::Person);
        }
        // Single title-case word mid-sentence
        if feat.is_title_case
        {
            return BioTag::Begin(EntityType::Misc);
        }
        BioTag::Outside
    }
}

// ---------------------------------------------------------------------------
// Entity extraction from BIO tags
// ---------------------------------------------------------------------------

/// Given tokens and their BIO tags, extract entity spans.
pub fn extract_entities(tokens: &[String], tags: &[BioTag]) -> Vec<Entity> {
    let mut entities = Vec::new();
    let mut current: Option<(String, EntityType, usize)> = None;

    for (i, (tok, tag)) in tokens.iter().zip(tags.iter()).enumerate()
    {
        match tag
        {
            BioTag::Begin(ty) =>
            {
                // Flush previous entity
                if let Some((text, ety, start)) = current.take()
                {
                    entities.push(Entity {
                        text,
                        entity_type: ety,
                        start,
                        end: i,
                        score: 1.0,
                    });
                }
                current = Some((tok.clone(), *ty, i));
            },
            BioTag::Inside(ty) =>
            {
                if let Some((ref mut text, ety, start)) = current
                {
                    if ety == *ty
                    {
                        text.push(' ');
                        text.push_str(tok);
                    }
                    else
                    {
                        entities.push(Entity {
                            text: text.clone(),
                            entity_type: ety,
                            start,
                            end: i,
                            score: 1.0,
                        });
                        current = Some((tok.clone(), *ty, i));
                    }
                }
                else
                {
                    current = Some((tok.clone(), *ty, i));
                }
            },
            BioTag::Outside =>
            {
                if let Some((text, ety, start)) = current.take()
                {
                    entities.push(Entity {
                        text,
                        entity_type: ety,
                        start,
                        end: i,
                        score: 1.0,
                    });
                }
            },
        }
    }
    // Flush trailing entity
    if let Some((text, ety, start)) = current
    {
        entities.push(Entity {
            text,
            entity_type: ety,
            start,
            end: tokens.len(),
            score: 1.0,
        });
    }
    entities
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_gazetteer() -> Gazetteer {
        let mut g = Gazetteer::new();
        g.add_many(&[
            ("Albert Einstein", EntityType::Person),
            ("New York", EntityType::Location),
            ("Google", EntityType::Organization),
            ("Microsoft", EntityType::Organization),
        ]);
        g
    }

    #[test]
    fn test_gazetteer_lookup() {
        let g = make_gazetteer();
        assert_eq!(g.lookup("Albert Einstein"), Some(EntityType::Person));
        assert_eq!(g.lookup("google"), Some(EntityType::Organization));
        assert!(g.lookup("unknown").is_none());
    }

    #[test]
    fn test_rule_based_ner_person() {
        let ner = RuleBasedNer::new(make_gazetteer());
        let entities = ner.extract("Albert Einstein was a physicist");
        assert!(
            entities
                .iter()
                .any(|e| e.text == "Albert Einstein" && e.entity_type == EntityType::Person)
        );
    }

    #[test]
    fn test_rule_based_ner_location() {
        let ner = RuleBasedNer::new(make_gazetteer());
        let entities = ner.extract("New York is a city");
        assert!(
            entities
                .iter()
                .any(|e| e.text == "New York" && e.entity_type == EntityType::Location)
        );
    }

    #[test]
    fn test_rule_based_ner_pattern() {
        let mut ner = RuleBasedNer::new(Gazetteer::new());
        ner.add_pattern("$100", EntityType::Money, 0.95);
        let entities = ner.extract("The cost is $100.");
        assert!(entities.iter().any(|e| e.entity_type == EntityType::Money));
    }

    #[test]
    fn test_word_shape() {
        assert_eq!(word_shape("Hello"), "Xx"); // X + xxxx → Xx
        assert_eq!(word_shape("NYC"), "X"); // XXX → X
        assert_eq!(word_shape("2024"), "d"); // dddd → d
        assert_eq!(word_shape("Mr."), "Xx_");
    }

    #[test]
    fn test_bio_tag_display() {
        assert_eq!(BioTag::Begin(EntityType::Person).as_str(), "B-PER");
        assert_eq!(BioTag::Inside(EntityType::Organization).as_str(), "I-ORG");
        assert_eq!(BioTag::Outside.as_str(), "O");
    }

    #[test]
    fn test_statistical_ner() {
        let ner = StatisticalNer::new(make_gazetteer());
        let tokens: Vec<String> = "Albert Einstein was born in Germany"
            .split_whitespace()
            .map(String::from)
            .collect();
        let tags = ner.classify(&tokens);
        assert_eq!(tags.len(), tokens.len());
        // "Albert" should be B-PER or similar
        assert!(matches!(tags[0], BioTag::Begin(EntityType::Person)));
    }

    #[test]
    fn test_extract_entities() {
        let tokens: Vec<String> = "Albert Einstein was born in Germany"
            .split_whitespace()
            .map(String::from)
            .collect();
        let tags = vec![
            BioTag::Begin(EntityType::Person),
            BioTag::Inside(EntityType::Person),
            BioTag::Outside,
            BioTag::Outside,
            BioTag::Outside,
            BioTag::Begin(EntityType::Location),
        ];
        let entities = extract_entities(&tokens, &tags);
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "Albert Einstein");
        assert_eq!(entities[0].entity_type, EntityType::Person);
        assert_eq!(entities[1].text, "Germany");
        assert_eq!(entities[1].entity_type, EntityType::Location);
    }

    #[test]
    fn test_bio_consistency() {
        let ner = StatisticalNer::new(Gazetteer::new());
        let tokens: Vec<String> = "Google is in Mountain View"
            .split_whitespace()
            .map(String::from)
            .collect();
        let tags = ner.classify(&tokens);
        // Verify no I- follows O
        for i in 1..tags.len()
        {
            if let BioTag::Inside(_) = tags[i]
            {
                assert!(matches!(tags[i - 1], BioTag::Begin(_) | BioTag::Inside(_)));
            }
        }
    }

    #[test]
    fn test_entity_type_display() {
        assert_eq!(EntityType::Person.to_string(), "PER");
        assert_eq!(EntityType::Organization.to_string(), "ORG");
        assert_eq!(EntityType::Location.to_string(), "LOC");
    }

    #[test]
    fn test_entity_type_from_bio_tag() {
        assert_eq!(
            EntityType::from_bio_tag("B-PER"),
            Some((true, EntityType::Person))
        );
        assert_eq!(
            EntityType::from_bio_tag("I-ORG"),
            Some((false, EntityType::Organization))
        );
        assert!(EntityType::from_bio_tag("O").is_none());
        assert!(EntityType::from_bio_tag("X-FOO").is_none());
    }
}
