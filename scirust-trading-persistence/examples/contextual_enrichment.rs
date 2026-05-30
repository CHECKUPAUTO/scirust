//! Démo de l'enrichissement Contextual : on simule un flux d'events
//! Semantic, on les passe par le ContextualEnricher, on observe la
//! promotion vers Contextual avec nearest_neighbors + historical_response.
//!
//! cargo run --example contextual_enrichment -p scirust-trading-persistence

use async_trait::async_trait;
use chrono::{Duration, Utc};
use scirust_trading_core::{
    Category, CodifiedEvent, EnrichmentLevel, EventTiming, Exchange,
    MarketState, Polarity, SourceId, Symbol, Target,
};
use scirust_trading_news::{
    ContextualConfig, ContextualEnricher, EmbeddingStore, HistoricalProvider,
    NewsResult,
};
use scirust_trading_persistence::{
    decisions::flush_decisions, writer::flush_events, writer::flush_market, QueryApi,
    SqliteEmbeddingStore,
};
use std::sync::Arc;
use uuid::Uuid;

/// Mock TRIBE qui renvoie un embedding déterministe basé sur les hash des mots.
/// Permet de démontrer le pipeline sans serveur réel sur port 7440.
struct DeterministicEmbedder;

impl DeterministicEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; 16];
        for word in text.split_whitespace() {
            let h = word.to_lowercase().bytes().fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
            let idx = (h % 16) as usize;
            v[idx] += 1.0;
        }
        // L2 normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in &mut v {
            *x /= norm;
        }
        v
    }
}

/// EmbeddingStore qui utilise notre DeterministicEmbedder + SqliteEmbeddingStore
struct DemoStore {
    embedder: DeterministicEmbedder,
    backing: SqliteEmbeddingStore,
}

#[async_trait]
impl EmbeddingStore for DemoStore {
    async fn put(&self, id: Uuid, _embedding: Vec<f32>) -> NewsResult<()> {
        // Pas utilisé directement — on stocke via seed_event
        self.backing.put(id, _embedding).await
    }
    async fn find_nearest(
        &self,
        query: &[f32],
        k: usize,
        exclude: Option<Uuid>,
    ) -> NewsResult<Vec<(Uuid, f64)>> {
        self.backing.find_nearest(query, k, exclude).await
    }
}

impl DemoStore {
    async fn seed_with_text(&self, id: Uuid, text: &str) -> NewsResult<()> {
        let emb = self.embedder.embed(text);
        self.backing.put(id, emb).await
    }
}

#[tokio::main]
async fn main() {
    println!("scirust contextual enrichment demo");
    println!("══════════════════════════════════════════════════════════════");

    // ─── Setup ────────────────────────────────────────────────────────
    // 1. Persistance avec embeddings store
    let store = Arc::new(DemoStore {
        embedder: DeterministicEmbedder,
        backing: SqliteEmbeddingStore::open_in_memory("snowflake-arctic-v2").unwrap(),
    });
    let api = QueryApi::open_in_memory().unwrap();

    // 2. Pré-peuple 5 events historiques avec leurs embeddings
    println!("\n[1] Pré-peuplement de 5 events historiques avec embeddings");
    let historical_events = vec![
        ("SEC approves spot Bitcoin ETF from BlackRock", vec!["etf", "sec", "approval", "regulatory"]),
        ("SEC delays ruling on Ethereum ETF application", vec!["etf", "sec", "delay"]),
        ("FOMC raises interest rates by 25 basis points", vec!["fomc", "rate_decision", "hawkish"]),
        ("FOMC holds rates steady, signals patience", vec!["fomc", "rate_decision", "dovish"]),
        ("Whale moves 5000 BTC to Binance hot wallet", vec!["whale", "on_chain"]),
    ];
    let now = Utc::now();
    let mut historical_ids = Vec::new();

    {
        let c = api.conn.lock().await;
        for (i, (text, tags)) in historical_events.iter().enumerate() {
            let mut ev = CodifiedEvent::builder(SourceId::new("seed"), *text)
                .category(Category::Macro)
                .timing(EventTiming::Observed(now - Duration::days((i as i64 + 1) * 7)))
                .reliability(0.9)
                .build();
            ev.tags = tags.iter().map(|t| t.to_string()).collect();
            ev.detected_at = now - Duration::days((i as i64 + 1) * 7);
            ev.semantic_summary = Some(format!("Historical: {}", text));
            ev.polarity = Some(Polarity::new(if i == 0 || i == 3 { 0.6 } else { -0.3 }));
            ev.magnitude = Some(0.7);
            ev.semantic_confidence = Some(0.85);
            ev.targets = vec![Target::All];
            ev.enrichment = EnrichmentLevel::Semantic;
            historical_ids.push(ev.id);
            store.seed_with_text(ev.id, text).await.unwrap();
            flush_events(&c, &[ev]).unwrap();
            println!("    + {}", text);
        }
    }
    println!("    → {} embeddings stockés", store.backing.count().await.unwrap());

    // 3. Pré-peuple aussi des market_states autour de certains events pour
    //    permettre à historical_response_for_tags de retourner quelque chose
    println!("\n[2] Pré-peuplement de market_states pour historical_response");
    {
        let c = api.conn.lock().await;
        // 3 events FOMC à T-21j, T-14j (déjà créés), + on en simule 2 autres anciens
        let fomc_dates = vec![
            now - Duration::days(60),
            now - Duration::days(45),
            now - Duration::days(30),
        ];
        for fomc_date in &fomc_dates {
            // Insert un event historique FOMC sans embedding (juste pour le tag matching)
            let mut ev = CodifiedEvent::builder(SourceId::new("seed"), "FOMC historical")
                .category(Category::Macro)
                .timing(EventTiming::Observed(*fomc_date))
                .reliability(0.95)
                .build();
            ev.tags = vec!["fomc".into(), "rate_decision".into(), "hawkish".into()];
            ev.detected_at = *fomc_date;
            ev.targets = vec![Target::All];
            flush_events(&c, &[ev]).unwrap();
            // Market states : 50000 avant, 50500 à +5min, 50800 à +15min, 51000 à +60min
            flush_market(
                &c,
                &[
                    MarketState {
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTC", "USDT"),
                        timestamp: *fomc_date,
                        mid: 50_000.0,
                        microprice: 50_000.0,
                        spread_bps: 1.0,
                        imbalance_5: 0.0,
                        imbalance_20: 0.0,
                        realized_vol_pct: 30.0,
                        volume_1min: 10.0,
                        flow_imbalance_1min: 0.0,
                        trade_count_1min: 50,
                    },
                    MarketState {
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTC", "USDT"),
                        timestamp: *fomc_date + Duration::minutes(5),
                        mid: 50_500.0,
                        microprice: 50_500.0,
                        spread_bps: 1.0,
                        imbalance_5: 0.0,
                        imbalance_20: 0.0,
                        realized_vol_pct: 45.0,
                        volume_1min: 18.0,
                        flow_imbalance_1min: 0.2,
                        trade_count_1min: 90,
                    },
                    MarketState {
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTC", "USDT"),
                        timestamp: *fomc_date + Duration::minutes(15),
                        mid: 50_800.0,
                        microprice: 50_800.0,
                        spread_bps: 1.0,
                        imbalance_5: 0.0,
                        imbalance_20: 0.0,
                        realized_vol_pct: 40.0,
                        volume_1min: 15.0,
                        flow_imbalance_1min: 0.15,
                        trade_count_1min: 80,
                    },
                    MarketState {
                        exchange: Exchange::Binance,
                        symbol: Symbol::new("BTC", "USDT"),
                        timestamp: *fomc_date + Duration::minutes(60),
                        mid: 51_000.0,
                        microprice: 51_000.0,
                        spread_bps: 1.0,
                        imbalance_5: 0.0,
                        imbalance_20: 0.0,
                        realized_vol_pct: 32.0,
                        volume_1min: 8.0,
                        flow_imbalance_1min: 0.0,
                        trade_count_1min: 40,
                    },
                ],
            )
            .unwrap();
        }
    }

    // ─── Enrichissement d'un nouvel event ────────────────────────────
    let _ = flush_decisions; // silence
    println!("\n[3] Arrivée d'un nouvel event FOMC (Semantic → Contextual)");

    let enricher = ContextualEnricher::new(
        ContextualConfig {
            k_neighbors: 3,
            min_similarity: 0.3,
            reference_symbol: "BTC/USDT".into(),
            reference_exchange: "binance".into(),
            graceful_degrade: true,
        },
        None, // pas de TRIBE — on inject manuellement
        Some(store.clone() as Arc<dyn EmbeddingStore>),
        Some(Arc::new(api) as Arc<dyn HistoricalProvider>),
    );

    let new_text = "FOMC announces 25 basis point rate hike, signals hawkish bias";
    let mut new_event = CodifiedEvent::builder(SourceId::new("fomc.calendar"), new_text)
        .category(Category::Macro)
        .timing(EventTiming::Observed(Utc::now()))
        .reliability(0.98)
        .build();
    new_event.tags = vec!["fomc".into(), "rate_decision".into(), "hawkish".into()];
    new_event.semantic_summary = Some("Fed raised rates, hawkish guidance".into());
    new_event.polarity = Some(Polarity::new(-0.4));
    new_event.magnitude = Some(0.8);
    new_event.semantic_confidence = Some(0.9);
    new_event.targets = vec![Target::All];
    new_event.enrichment = EnrichmentLevel::Semantic;

    println!("    Input  : '{}'", new_text);
    println!("    Tags   : {:?}", new_event.tags);
    println!("    Level  : {:?}", new_event.enrichment);

    // Pré-embed manuellement pour bypasser le TRIBE network call dans la démo
    let new_embedding = store.embedder.embed(new_text);
    store.put(new_event.id, new_embedding.clone()).await.unwrap();

    // Patch : l'enricher a besoin du tribe pour faire l'embed lui-même.
    // En production, le tribe HTTP call ferait ça. Ici on a déjà l'embedding
    // dans le store, alors on bypass en mettant nearest_neighbors directement
    let neighbors = store
        .find_nearest(&new_embedding, 3, Some(new_event.id))
        .await
        .unwrap();
    new_event.nearest_neighbors = neighbors;

    // Pour historical_response, on appelle l'enricher (sans tribe)
    let enriched = enricher.enrich(new_event).await.unwrap();

    println!("\n[4] Résultat de l'enrichissement Contextual");
    println!("    Level  : {:?}", enriched.enrichment);
    println!("    Neighbors (top {}):", enriched.nearest_neighbors.len());
    for (id, sim) in &enriched.nearest_neighbors {
        let label = historical_events
            .iter()
            .zip(&historical_ids)
            .find(|(_, hid)| **hid == *id)
            .map(|((text, _), _)| *text)
            .unwrap_or("?");
        println!("      sim={:.3} : {}", sim, label);
    }
    if let Some(h) = &enriched.historical_response {
        println!("    Historical response (sur {} échantillons):", h.n_samples);
        println!("      Δ +5 min  : {:+.2} bps", h.delta_5min_bps);
        println!("      Δ +15 min : {:+.2} bps", h.delta_15min_bps);
        println!("      Δ +60 min : {:+.2} bps ± {:.2}", h.delta_60min_bps, h.delta_60min_std_bps);
        println!("      vol spike : {:.2}×", h.volume_spike_ratio);
        println!("      significant? : {}", h.is_significant());
    } else {
        println!("    Historical response : aucune (pas assez de données)");
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("Le score composite de l'event est maintenant calculable :");
    println!("  weighted_score = polarity × magnitude × confidence × reliability × decay");
    println!("                 = {:.2} × {:.2} × {:.2} × {:.2} × decay(now)",
        enriched.polarity.unwrap().value(),
        enriched.magnitude.unwrap(),
        enriched.semantic_confidence.unwrap(),
        enriched.source_reliability.value(),
    );
    println!("Score signé : {:+.3}", enriched.weighted_score(Utc::now()));
    println!();
    println!("Combiné à la `historical_response`, le decision engine sait que");
    println!("ce type d'event a historiquement produit +200 bps à +60 min,");
    println!("et que l'event courant est très similaire à 'FOMC raises rates'");
    println!("→ signal de qualité maximale pour le sizing.");
}
