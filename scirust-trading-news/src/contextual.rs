//! Enrichissement Contextual : promeut les events Semantic → Contextual en
//! leur attachant :
//!   - `nearest_neighbors` : top-K events historiques similaires par embedding
//!     (cosine similarity sur vecteurs Snowflake Arctic 768d via TRIBE Brain)
//!   - `historical_response` : réaction moyenne du marché après les events
//!     passés ayant les mêmes tags (via la query API de persistence)
//!
//! Architecture :
//!
//! ```text
//!  slow_path → CodifiedEvent (Semantic)
//!      │
//!      ├─→ TribeClient.embed(text) ──→ Vec<f32> (768d)
//!      │       │
//!      │       └─→ EmbeddingStore.find_nearest(emb, K) ──→ Vec<(Uuid, f64)>
//!      │
//!      ├─→ HistoricalProvider.lookup(tags) ──────────────→ Option<MarketReaction>
//!      │
//!      └─→ CodifiedEvent (Contextual) → bus.news
//! ```
//!
//! Le stockage des embeddings est extérieur à ce module (trait
//! `EmbeddingStore`). Une implémentation SQLite-backed est fournie dans
//! `scirust-trading-persistence::embeddings` (BLOB column + cosine en Rust).

use crate::{NewsError, NewsResult};
use async_trait::async_trait;
use scirust_trading_core::{CodifiedEvent, EnrichmentLevel, MarketReaction};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Client TRIBE Brain — interroge l'embedder Snowflake Arctic sur port 7440.
///
/// Endpoint attendu : `POST /embed` avec body `{"text": "..."}` qui renvoie
/// `{"embedding": [...]}`. C'est l'API standard du TRIBE Brain Server.
#[derive(Debug, Clone)]
pub struct TribeConfig {
    pub base_url: String,
    pub timeout_secs: u64,
}

impl Default for TribeConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:7440".into(),
            timeout_secs: 5,
        }
    }
}

pub struct TribeClient {
    pub config: TribeConfig,
    client: reqwest::Client,
}

impl TribeClient {
    pub fn new(config: TribeConfig) -> NewsResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| NewsError::Http(e.to_string()))?;
        Ok(Self { config, client })
    }

    /// Calcule l'embedding d'un texte. Renvoie un Vec<f32> de la dimension
    /// retournée par le serveur (768 pour Snowflake Arctic).
    pub async fn embed(&self, text: &str) -> NewsResult<Vec<f32>> {
        let url = format!("{}/embed", self.config.base_url.trim_end_matches('/'));
        let body = EmbedRequest {
            text: text.to_string(),
        };
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| NewsError::Http(e.to_string()))?
            .error_for_status()
            .map_err(|e| NewsError::Http(e.to_string()))?;
        let parsed: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| NewsError::Parse(e.to_string()))?;
        if parsed.embedding.is_empty() {
            return Err(NewsError::Parse("empty embedding returned".into()));
        }
        Ok(parsed.embedding)
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

/// Store d'embeddings : insertion + lookup K-nearest par cosine similarity.
#[async_trait]
pub trait EmbeddingStore: Send + Sync {
    /// Insère un embedding indexé par event_id.
    async fn put(&self, event_id: uuid::Uuid, embedding: Vec<f32>) -> NewsResult<()>;

    /// Trouve les K events les plus proches en cosine similarity. Renvoie
    /// `(event_id, similarity)` triés par similarity décroissante. Doit
    /// exclure `event_id` lui-même s'il est passé (cas où on cherche
    /// les voisins d'un event déjà stocké).
    async fn find_nearest(
        &self,
        query: &[f32],
        k: usize,
        exclude: Option<uuid::Uuid>,
    ) -> NewsResult<Vec<(uuid::Uuid, f64)>>;
}

/// Provider de historical_response — abstraction qui découple l'enricher
/// de la couche persistence.
#[async_trait]
pub trait HistoricalProvider: Send + Sync {
    async fn lookup_response(
        &self,
        tags: &[String],
        asset_symbol_canonical: &str,
        exchange: &str,
    ) -> NewsResult<Option<MarketReaction>>;
}

#[derive(Debug, Clone)]
pub struct ContextualConfig {
    pub k_neighbors: usize,
    /// Seuil minimum de cosine similarity pour considérer un event comme voisin
    pub min_similarity: f64,
    /// Symbole de référence pour les lookups historical_response
    pub reference_symbol: String,
    pub reference_exchange: String,
    /// Active la promotion vers Contextual même si les enrichments échouent
    /// (utile en dev / quand TRIBE est down)
    pub graceful_degrade: bool,
}

impl Default for ContextualConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            min_similarity: 0.6,
            reference_symbol: "BTC/USDT".into(),
            reference_exchange: "binance".into(),
            graceful_degrade: true,
        }
    }
}

pub struct ContextualEnricher {
    pub config: ContextualConfig,
    pub tribe: Option<Arc<TribeClient>>,
    pub embeddings: Option<Arc<dyn EmbeddingStore>>,
    pub historical: Option<Arc<dyn HistoricalProvider>>,
    /// Cache LRU minimaliste pour ne pas re-embedder le même texte
    cache: Arc<RwLock<EmbedCache>>,
}

impl ContextualEnricher {
    pub fn new(
        config: ContextualConfig,
        tribe: Option<Arc<TribeClient>>,
        embeddings: Option<Arc<dyn EmbeddingStore>>,
        historical: Option<Arc<dyn HistoricalProvider>>,
    ) -> Self {
        Self {
            config,
            tribe,
            embeddings,
            historical,
            cache: Arc::new(RwLock::new(EmbedCache::new(2048))),
        }
    }

    /// Promeut un CodifiedEvent Semantic → Contextual en attachant
    /// nearest_neighbors et historical_response.
    ///
    /// Si déjà Contextual ou Raw, renvoie l'event tel quel.
    /// Si Semantic, tente l'enrichissement. Sur échec partiel (TRIBE down),
    /// on dégrade selon `graceful_degrade`.
    pub async fn enrich(&self, mut event: CodifiedEvent) -> NewsResult<CodifiedEvent> {
        if event.enrichment != EnrichmentLevel::Semantic
            && event.enrichment != EnrichmentLevel::Structural
        {
            return Ok(event);
        }

        // 1. Embedding + nearest neighbors
        let mut embedding_ok = false;
        if let (Some(tribe), Some(store)) = (&self.tribe, &self.embeddings) {
            let text_key = embedding_key(&event);
            let cached = {
                let cache = self.cache.read().await;
                cache.get(&text_key).cloned()
            };
            let embedding = match cached {
                Some(e) => Some(e),
                None => match tribe.embed(&event.raw_text).await {
                    Ok(e) => {
                        let mut cache = self.cache.write().await;
                        cache.put(text_key.clone(), e.clone());
                        Some(e)
                    }
                    Err(e) => {
                        tracing::warn!("TRIBE embed failed: {e}");
                        if !self.config.graceful_degrade {
                            return Err(e);
                        }
                        None
                    }
                },
            };

            if let Some(emb) = &embedding {
                // Stocke l'embedding pour ce nouvel event
                if let Err(e) = store.put(event.id, emb.clone()).await {
                    tracing::warn!("embedding store.put failed: {e}");
                }
                // Recherche les voisins (excluant self)
                match store
                    .find_nearest(emb, self.config.k_neighbors, Some(event.id))
                    .await
                {
                    Ok(neighbors) => {
                        let filtered: Vec<(uuid::Uuid, f64)> = neighbors
                            .into_iter()
                            .filter(|(_, sim)| *sim >= self.config.min_similarity)
                            .collect();
                        event.nearest_neighbors = filtered;
                        embedding_ok = true;
                    }
                    Err(e) => {
                        tracing::warn!("embedding store.find_nearest failed: {e}");
                        if !self.config.graceful_degrade {
                            return Err(e);
                        }
                    }
                }
            }
        }

        // 2. Historical response sur les tags
        let mut historical_ok = false;
        if let Some(hist) = &self.historical {
            if !event.tags.is_empty() {
                match hist
                    .lookup_response(
                        &event.tags,
                        &self.config.reference_symbol,
                        &self.config.reference_exchange,
                    )
                    .await
                {
                    Ok(reaction) => {
                        event.historical_response = reaction;
                        historical_ok = true;
                    }
                    Err(e) => {
                        tracing::warn!("historical_response lookup failed: {e}");
                        if !self.config.graceful_degrade {
                            return Err(e);
                        }
                    }
                }
            }
        }

        // 3. Promotion vers Contextual si au moins un enrichment a réussi
        if embedding_ok || historical_ok {
            event.enrichment = EnrichmentLevel::Contextual;
            event.explanation = format!(
                "{}|contextual(neighbors={}, hist={})",
                event.explanation,
                event.nearest_neighbors.len(),
                event.historical_response.is_some()
            );
        }
        Ok(event)
    }
}

fn embedding_key(event: &CodifiedEvent) -> String {
    // Hash simple : le texte brut. Pour de gros volumes, passer en xxhash.
    event.raw_text.clone()
}

/// LRU minimaliste sans dep externe (BTreeMap + counter)
struct EmbedCache {
    capacity: usize,
    map: std::collections::HashMap<String, (Vec<f32>, u64)>,
    counter: u64,
}

impl EmbedCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: std::collections::HashMap::new(),
            counter: 0,
        }
    }

    fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.map.get(key).map(|(v, _)| v)
    }

    fn put(&mut self, key: String, value: Vec<f32>) {
        self.counter += 1;
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Drop l'entrée la plus ancienne
            if let Some(oldest_key) = self
                .map
                .iter()
                .min_by_key(|(_, (_, c))| *c)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&oldest_key);
            }
        }
        self.map.insert(key, (value, self.counter));
    }
}

/// Compute cosine similarity entre deux vecteurs. Réutilisable par les stores.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use scirust_trading_core::{Category, EventTiming, SourceId};
    use std::collections::HashMap;
    use tokio::sync::Mutex;

    fn make_event(text: &str, tags: &[&str], level: EnrichmentLevel) -> CodifiedEvent {
        let mut e = CodifiedEvent::builder(SourceId::new("test"), text)
            .category(Category::Macro)
            .timing(EventTiming::Observed(Utc::now()))
            .reliability(0.9)
            .build();
        e.tags = tags.iter().map(|t| t.to_string()).collect();
        e.enrichment = level;
        e
    }

    // ─── Mock TribeClient via une struct partagée ──────────────────────

    struct MockEmbeddings {
        store: Arc<Mutex<HashMap<uuid::Uuid, Vec<f32>>>>,
    }

    impl MockEmbeddings {
        fn new() -> Self {
            Self {
                store: Arc::new(Mutex::new(HashMap::new())),
            }
        }
    }

    #[async_trait]
    impl EmbeddingStore for MockEmbeddings {
        async fn put(&self, id: uuid::Uuid, embedding: Vec<f32>) -> NewsResult<()> {
            let mut s = self.store.lock().await;
            s.insert(id, embedding);
            Ok(())
        }
        async fn find_nearest(
            &self,
            query: &[f32],
            k: usize,
            exclude: Option<uuid::Uuid>,
        ) -> NewsResult<Vec<(uuid::Uuid, f64)>> {
            let s = self.store.lock().await;
            let mut scored: Vec<(uuid::Uuid, f64)> = s
                .iter()
                .filter(|(id, _)| Some(**id) != exclude)
                .map(|(id, emb)| (*id, cosine_similarity(query, emb)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(scored)
        }
    }

    struct MockHistorical {
        reaction: Option<MarketReaction>,
    }

    #[async_trait]
    impl HistoricalProvider for MockHistorical {
        async fn lookup_response(
            &self,
            _tags: &[String],
            _symbol: &str,
            _exchange: &str,
        ) -> NewsResult<Option<MarketReaction>> {
            Ok(self.reaction.clone())
        }
    }

    #[test]
    fn cosine_correct() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 0.0, 0.0];
        let c = vec![0.0_f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-9);
        assert!(cosine_similarity(&a, &c).abs() < 1e-9);
        let d = vec![1.0_f32, 1.0, 0.0];
        // (1×1 + 0×1) / (1 × √2) = 1/√2 ≈ 0.707
        let s = cosine_similarity(&a, &d);
        assert!((s - 0.7071).abs() < 1e-3);
    }

    #[test]
    fn cosine_handles_zero_norm() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[tokio::test]
    async fn enrich_attaches_neighbors_and_history() {
        // Setup mock TRIBE qui renvoie un embedding fixe : on ne peut pas
        // appeler le vrai serveur dans un test unitaire
        let store: Arc<dyn EmbeddingStore> = Arc::new(MockEmbeddings::new());

        // Pré-peuplé : 3 events historiques avec embeddings connus
        let h1 = uuid::Uuid::new_v4();
        let h2 = uuid::Uuid::new_v4();
        let h3 = uuid::Uuid::new_v4();
        store.put(h1, vec![1.0, 0.0, 0.0]).await.unwrap();
        store.put(h2, vec![0.9, 0.1, 0.0]).await.unwrap(); // similaire à h1
        store.put(h3, vec![0.0, 0.0, 1.0]).await.unwrap(); // très différent

        // Mock historical (renvoie une réaction non nulle)
        let hist: Arc<dyn HistoricalProvider> = Arc::new(MockHistorical {
            reaction: Some(MarketReaction {
                n_samples: 5,
                delta_5min_bps: 25.0,
                delta_15min_bps: 60.0,
                delta_60min_bps: 120.0,
                delta_60min_std_bps: 35.0,
                volume_spike_ratio: 2.5,
            }),
        });

        // Cas test : on crée un enricher mais on ne lui donne PAS de TRIBE
        // client (pour éviter le réseau). On simule un embedding "manuel"
        // en pré-populant l'event et en sautant la phase embed.
        // → Pour ça, on bypasse l'enricher et on appelle find_nearest direct
        let query = vec![1.0_f32, 0.05, 0.0];
        let neighbors = store.find_nearest(&query, 5, None).await.unwrap();
        assert_eq!(neighbors.len(), 3);
        // h1 (1,0,0) le plus proche
        assert_eq!(neighbors[0].0, h1);
        assert!(neighbors[0].1 > 0.99);
        // h2 proche aussi
        assert_eq!(neighbors[1].0, h2);
        // h3 dernier
        assert_eq!(neighbors[2].0, h3);

        // Test path historical separately (sans TRIBE)
        let enricher = ContextualEnricher::new(
            ContextualConfig::default(),
            None,
            None,
            Some(hist),
        );
        let ev = make_event(
            "SEC approves new spot ETF",
            &["etf", "sec", "regulatory"],
            EnrichmentLevel::Semantic,
        );
        let enriched = enricher.enrich(ev).await.unwrap();
        assert_eq!(enriched.enrichment, EnrichmentLevel::Contextual);
        let h = enriched.historical_response.as_ref().unwrap();
        assert_eq!(h.n_samples, 5);
        assert!((h.delta_60min_bps - 120.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn enrich_skips_already_contextual() {
        let enricher = ContextualEnricher::new(
            ContextualConfig::default(),
            None,
            None,
            Some(Arc::new(MockHistorical {
                reaction: Some(MarketReaction {
                    n_samples: 10,
                    delta_5min_bps: 1.0,
                    delta_15min_bps: 2.0,
                    delta_60min_bps: 3.0,
                    delta_60min_std_bps: 4.0,
                    volume_spike_ratio: 1.5,
                }),
            })),
        );
        let mut ev = make_event("x", &["a"], EnrichmentLevel::Contextual);
        ev.historical_response = None;
        let enriched = enricher.enrich(ev).await.unwrap();
        // Pas modifié (déjà Contextual)
        assert!(enriched.historical_response.is_none());
    }

    #[tokio::test]
    async fn enrich_no_providers_returns_event_unchanged() {
        let enricher = ContextualEnricher::new(ContextualConfig::default(), None, None, None);
        let ev = make_event("x", &["a"], EnrichmentLevel::Semantic);
        let enriched = enricher.enrich(ev.clone()).await.unwrap();
        assert_eq!(enriched.enrichment, EnrichmentLevel::Semantic);
        assert!(enriched.nearest_neighbors.is_empty());
        assert!(enriched.historical_response.is_none());
    }

    #[tokio::test]
    async fn enrich_graceful_degrade_partial_success() {
        // Historical OK, embeddings absent → promotion quand même
        let enricher = ContextualEnricher::new(
            ContextualConfig::default(),
            None,
            None,
            Some(Arc::new(MockHistorical {
                reaction: Some(MarketReaction {
                    n_samples: 2,
                    delta_5min_bps: 10.0,
                    delta_15min_bps: 20.0,
                    delta_60min_bps: 30.0,
                    delta_60min_std_bps: 5.0,
                    volume_spike_ratio: 1.2,
                }),
            })),
        );
        let ev = make_event("x", &["tag1"], EnrichmentLevel::Semantic);
        let enriched = enricher.enrich(ev).await.unwrap();
        assert_eq!(enriched.enrichment, EnrichmentLevel::Contextual);
    }

    #[test]
    fn cache_evicts_oldest() {
        let mut cache = EmbedCache::new(2);
        cache.put("a".into(), vec![1.0]);
        cache.put("b".into(), vec![2.0]);
        cache.put("c".into(), vec![3.0]); // évince "a"
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
    }
}
