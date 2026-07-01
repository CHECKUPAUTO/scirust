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

    /// Look up a surface form (case-insensitive).
    ///
    /// The `token` may be a single word or a whitespace-separated multi-word
    /// phrase (e.g. `"Albert Einstein"`); it is matched against the stored
    /// entries verbatim after lowercasing.
    pub fn lookup(&self, token: &str) -> Option<EntityType> {
        self.entries.get(&token.to_lowercase()).copied()
    }

    /// The largest number of whitespace-separated words in any stored entry
    /// (0 if the gazetteer is empty).  Used to bound multi-word (n-gram)
    /// matching so callers never have to scan more tokens than necessary.
    pub fn max_entry_words(&self) -> usize {
        self.entries
            .keys()
            .map(|k| k.split_whitespace().count())
            .max()
            .unwrap_or(0)
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

/// Lowercase `text` while recording, for every byte of the lowercased string,
/// the byte offset of the character it originated from in `text`.
///
/// Returns `(lower, map)` where `map` has `lower.len() + 1` entries: `map[i]`
/// is the original byte offset for lowercase byte `i`, and `map[lower.len()]`
/// equals `text.len()`.  Every entry is guaranteed to fall on a `char`
/// boundary of `text`, so slicing `text[map[a]..map[b]]` for any lowercase
/// char boundaries `a <= b` never panics.
///
/// This is needed because `str::to_lowercase` may change the byte length of a
/// string (e.g. `"İ"` (2 bytes) → `"i̇"` (3 bytes)); byte offsets found in the
/// lowercased text therefore cannot be applied to the original text directly.
fn lowercase_with_offsets(text: &str) -> (String, Vec<usize>) {
    let mut lower = String::with_capacity(text.len());
    let mut map = Vec::with_capacity(text.len() + 1);
    let mut buf = [0u8; 4];
    for (byte_idx, ch) in text.char_indices()
    {
        for lc in ch.to_lowercase()
        {
            let encoded = lc.encode_utf8(&mut buf);
            for _ in 0..encoded.len()
            {
                map.push(byte_idx);
            }
            lower.push_str(encoded);
        }
    }
    map.push(text.len());
    (lower, map)
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
        // Case-insensitive search is done on the lowercased text, but the
        // returned spans must refer to the *original* text.  Lowercasing can
        // change the byte length of a string (e.g. Unicode "İ" → "i̇"), so
        // offsets computed on `lower` cannot be applied to `text` directly.
        // `lower_map` translates any byte offset in `lower` back to the byte
        // offset of the originating character in `text`.
        let (lower, lower_map) = lowercase_with_offsets(text);

        // 1. Gazetteer lookup — match longest spans first.
        // Offsets in `matches` are byte offsets into `lower`.
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
                // Translate lowercase offsets back to original-text offsets.
                let orig_start = lower_map[*start];
                let orig_end = lower_map[*end];
                entities.push(Entity {
                    text: text[orig_start..orig_end].to_string(),
                    entity_type: *ty,
                    start: orig_start,
                    end: orig_end,
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
                let end = abs + pat_lower.len();
                // Translate lowercase offsets back to original-text offsets.
                let orig_start = lower_map[abs];
                let orig_end = lower_map[end];
                // Only add if not already covered
                if !entities
                    .iter()
                    .any(|e| e.start <= orig_start && e.end > orig_start)
                {
                    entities.push(Entity {
                        text: text[orig_start..orig_end].to_string(),
                        entity_type: *ty,
                        start: orig_start,
                        end: orig_end,
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
    /// Entity type of the gazetteer entry this token belongs to, if any.  For a
    /// multi-word entry (e.g. `"Albert Einstein"`) every token of the matched
    /// span carries the type.
    pub gazetteer_match: Option<EntityType>,
    /// True when this token is the *first* token of a gazetteer match; false
    /// for continuation tokens of a multi-word match (and when there is no
    /// match).  Lets the classifier emit `B-` for the head and `I-` for the
    /// tail of a multi-word entity.
    pub gazetteer_begin: bool,
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

    // Gazetteer matching. Entries may be multi-word (e.g. "Albert Einstein"),
    // so a per-token lookup can never match them. Instead, tile the token
    // stream with greedy longest-first matches: at each unclaimed position, try
    // the longest window (bounded by the longest stored entry) and shrink until
    // a stored entry matches. Every token of a match records the entity type;
    // only the first token of a match is flagged as the beginning of the span.
    let max_words = gazetteer.max_entry_words().min(n);
    let mut gaz_match: Vec<Option<EntityType>> = vec![None; n];
    let mut gaz_begin: Vec<bool> = vec![false; n];
    if max_words >= 1
    {
        let mut i = 0;
        while i < n
        {
            let mut matched = false;
            let max_len = max_words.min(n - i);
            for len in (1..=max_len).rev()
            {
                let phrase = tokens[i..i + len].join(" ");
                if let Some(ty) = gazetteer.lookup(&phrase)
                {
                    gaz_begin[i] = true;
                    for slot in gaz_match.iter_mut().skip(i).take(len)
                    {
                        *slot = Some(ty);
                    }
                    i += len;
                    matched = true;
                    break;
                }
            }
            if !matched
            {
                i += 1;
            }
        }
    }

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
            gazetteer_match: gaz_match[i],
            gazetteer_begin: gaz_begin[i],
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
        // Gazetteer match → highest priority. The first token of the matched
        // (possibly multi-word) span begins the entity; the remaining tokens
        // continue it, so a multi-word entry yields one contiguous span.
        if let Some(ty) = feat.gazetteer_match
        {
            if feat.gazetteer_begin
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
    fn test_extract_features_multiword_gazetteer_match() {
        // Regression: the gazetteer stores multi-word entries (e.g.
        // "New York"), but features were looked up one token at a time, so a
        // single token could never match a multi-word key. Before the fix both
        // tokens of "New York" had `gazetteer_match == None`; now the whole
        // span is matched, the head token begins it and the tail continues it.
        let g = make_gazetteer();
        let tokens: Vec<String> = "I live in New York now"
            .split_whitespace()
            .map(String::from)
            .collect();
        let feats = extract_features(&tokens, &g);

        // "New" (index 3) begins a Location span; "York" (index 4) continues it.
        assert_eq!(feats[3].gazetteer_match, Some(EntityType::Location));
        assert!(feats[3].gazetteer_begin);
        assert_eq!(feats[4].gazetteer_match, Some(EntityType::Location));
        assert!(!feats[4].gazetteer_begin);

        // Non-entity tokens stay unmatched.
        assert_eq!(feats[0].gazetteer_match, None);
        assert_eq!(feats[5].gazetteer_match, None);
    }

    #[test]
    fn test_statistical_ner_multiword_gazetteer() {
        // End-to-end: a multi-word gazetteer entry must drive the classifier
        // and be recovered as a single contiguous entity via BIO tags. Before
        // the fix "New York" produced no gazetteer-driven tags at all.
        let ner = StatisticalNer::new(make_gazetteer());
        let tokens: Vec<String> = "flights to New York depart"
            .split_whitespace()
            .map(String::from)
            .collect();
        let tags = ner.classify(&tokens);
        assert_eq!(tags[2], BioTag::Begin(EntityType::Location));
        assert_eq!(tags[3], BioTag::Inside(EntityType::Location));

        let entities = extract_entities(&tokens, &tags);
        let loc = entities
            .iter()
            .find(|e| e.entity_type == EntityType::Location)
            .expect("New York should be recovered as one Location entity");
        assert_eq!(loc.text, "New York");
    }

    #[test]
    fn test_max_entry_words() {
        let g = make_gazetteer();
        // "Albert Einstein" / "New York" are 2 words; the rest are 1.
        assert_eq!(g.max_entry_words(), 2);
        assert_eq!(Gazetteer::new().max_entry_words(), 0);
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
    fn test_rule_based_ner_non_ascii_offsets() {
        // Regression: lowercasing can change byte length (Unicode "İ" → "i̇"),
        // so offsets found on the lowercased text must be mapped back to the
        // original text. Before the fix this panicked with
        // "byte index N is not a char boundary" or produced a corrupted slice.
        let mut g = Gazetteer::new();
        g.add("İé", EntityType::Location);
        g.add("Café", EntityType::Organization);
        let ner = RuleBasedNer::new(g);

        // "İéé" is the minimal case that panicked on the original code:
        // lowercasing "İé" grows by one byte, pushing the end offset into the
        // trailing multibyte 'é' of the source text.
        let text = "İéé and Café here";
        let entities = ner.extract(text);

        // Gazetteer matches must be exact slices of the ORIGINAL text and must
        // preserve original casing.
        let loc = entities
            .iter()
            .find(|e| e.entity_type == EntityType::Location)
            .expect("İé should be found");
        assert_eq!(loc.text, "İé");
        assert_eq!(&text[loc.start..loc.end], "İé");

        let org = entities
            .iter()
            .find(|e| e.entity_type == EntityType::Organization)
            .expect("Café should be found");
        assert_eq!(org.text, "Café");
        assert_eq!(&text[org.start..org.end], "Café");

        // Every entity span must be a valid, in-bounds slice of the original.
        for e in &entities
        {
            assert!(e.end <= text.len());
            assert!(text.is_char_boundary(e.start));
            assert!(text.is_char_boundary(e.end));
            assert_eq!(e.text, text[e.start..e.end]);
        }
    }

    #[test]
    fn test_rule_based_ner_non_ascii_pattern() {
        // The pattern branch had the same offset bug (and additionally mixed a
        // lowercased search offset with the original-case pattern length).
        let mut ner = RuleBasedNer::new(Gazetteer::new());
        ner.add_pattern("İé", EntityType::Misc, 0.8);
        let text = "prefix İéé suffix";
        let entities = ner.extract(text);
        let m = entities
            .iter()
            .find(|e| e.entity_type == EntityType::Misc)
            .expect("İé pattern should match");
        assert_eq!(m.text, "İé");
        assert_eq!(&text[m.start..m.end], "İé");
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
