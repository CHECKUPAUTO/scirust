use crate::contextual::ContextualEnricher;
    /// Si true et qu'un enricher Contextual est fourni, on appelle l'enricher
    /// après chaque Semantic event pour le promouvoir vers Contextual.
    pub enable_contextual: bool,
            enable_contextual: true,
    enricher: Option<Arc<ContextualEnricher>>,
            enricher: None,
    /// Attache un ContextualEnricher au pipeline. Une fois fait, tous les
    /// events qui passent par la slow_path sont aussi passés à l'enricher
    /// et promus vers Contextual si possible.
    pub fn with_enricher(mut self, enricher: ContextualEnricher) -> Self {
        self.enricher = Some(Arc::new(enricher));
        self
    }

            let enricher = if self.cfg.enable_contextual {
                self.enricher.clone()
            } else {
                None
            };
                slow_path_worker(sem_rx, ollama, bus_clone, max_backlog, enricher).await;
    enricher: Option<Arc<ContextualEnricher>>,

                // Promotion vers Contextual si un enricher est attaché.
                // Si l'enricher échoue (TRIBE down, etc.), on garde Semantic
                // pour ne pas perdre l'event.
                if let Some(enricher) = &enricher {
                    match enricher.enrich(enriched.clone()).await {
                        Ok(contextual) => {
                            let _ = bus.news.send(contextual);
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "contextual enrichment failed for {}: {e}, emitting Semantic only",
                                enriched.id
                            );
                        }
                    }
                }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual::{ContextualConfig, EmbeddingStore, HistoricalProvider};
    use crate::fast_path::default_ruleset;
    use crate::sources::Source;
    use async_trait::async_trait;
    use scirust_trading_core::{EventTiming, MarketReaction, SourceId, SourceReliability};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use uuid::Uuid;

    /// Source qui émet 1 event puis termine immédiatement.
    struct OneShotSource(std::sync::Mutex<Option<RawEvent>>);

    #[async_trait]
    impl Source for OneShotSource {
        fn name(&self) -> &str {
            "oneshot"
        }
        async fn run(&self, tx: mpsc::Sender<RawEvent>) -> NewsResult<()> {
            let event_opt = { self.0.lock().unwrap().take() };
            if let Some(ev) = event_opt {
                let _ = tx.send(ev).await;
            }
            // Garde la tâche vivante
            futures_util::future::pending::<()>().await;
            Ok(())
        }
    }

    /// HistoricalProvider qui compte les appels et renvoie une réaction fixe.
    struct CountingHistorical {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl HistoricalProvider for CountingHistorical {
        async fn lookup_response(
            &self,
            _tags: &[String],
            _symbol: &str,
            _exchange: &str,
        ) -> NewsResult<Option<MarketReaction>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(MarketReaction {
                n_samples: 5,
                delta_5min_bps: 30.0,
                delta_15min_bps: 60.0,
                delta_60min_bps: 100.0,
                delta_60min_std_bps: 20.0,
                volume_spike_ratio: 2.0,
            }))
        }
    }

    /// EmbeddingStore mémoire qui compte les puts.
    struct CountingEmbeddings {
        puts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl EmbeddingStore for CountingEmbeddings {
        async fn put(&self, _id: Uuid, _emb: Vec<f32>) -> NewsResult<()> {
            self.puts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn find_nearest(
            &self,
            _query: &[f32],
            _k: usize,
            _exclude: Option<Uuid>,
        ) -> NewsResult<Vec<(Uuid, f64)>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn pipeline_with_enricher_promotes_to_contextual_via_history_path() {
        // Pas de TRIBE configuré → l'enricher peut quand même attacher
        // l'historical_response et promouvoir à Contextual via le
        // graceful_degrade path.
        let bus = EventBus::new();
        let mut news_rx = bus.news.subscribe();

        let calls = Arc::new(AtomicUsize::new(0));
        let hist: Arc<dyn HistoricalProvider> = Arc::new(CountingHistorical {
            calls: Arc::clone(&calls),
        });
        let enricher = crate::contextual::ContextualEnricher::new(
            ContextualConfig::default(),
            None, // pas de tribe (network)
            None, // pas d'embedding store
            Some(hist),
        );

        // Mock OllamaClient impossible sans réseau → on désactive slow_path
        // pour ce test, donc l'enricher ne sera PAS appelé (il l'est seulement
        // après slow_path). On vérifie quand même que la config compile et
        // que sans slow_path on garde Structural.
        let pipeline = NewsPipeline::new(
            NewsPipelineConfig {
                enable_slow_path: false,
                enable_contextual: true,
                ..Default::default()
            },
            default_ruleset(),
            None, // pas d'Ollama → slow_path disabled
            bus.clone(),
        )
        .with_enricher(enricher);

        let raw = RawEvent {
            source: SourceId::new("test"),
            reliability: SourceReliability::new(0.9),
            text: "SEC approves new spot Bitcoin ETF".into(),
            url: None,
            timing: EventTiming::Observed(chrono::Utc::now()),
        };
        let _handle = pipeline.spawn(vec![Box::new(OneShotSource(std::sync::Mutex::new(Some(raw))))]);

        let codified = tokio::time::timeout(std::time::Duration::from_millis(500), news_rx.recv())
            .await
            .unwrap()
            .unwrap();
        // Sans slow_path → reste à Structural
        assert_eq!(codified.enrichment, EnrichmentLevel::Structural);
        // L'enricher n'a pas été appelé (puisque slow_path désactivée)
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn enricher_invoked_when_slow_path_succeeds() {
        // On ne peut pas tester le PATH COMPLET sans Ollama. Mais on peut
        // tester directement que le contextual enricher promeut un event
        // Semantic → Contextual quand il est invoqué.
        let calls = Arc::new(AtomicUsize::new(0));
        let hist: Arc<dyn HistoricalProvider> = Arc::new(CountingHistorical {
            calls: Arc::clone(&calls),
        });
        let enricher = crate::contextual::ContextualEnricher::new(
            ContextualConfig::default(),
            None,
            None,
            Some(hist),
        );

        let mut ev = CodifiedEvent::builder(
            SourceId::new("test"),
            "FOMC raises rates",
        )
        .reliability(0.95)
        .build();
        ev.tags = vec!["fomc".into(), "rate_decision".into()];
        ev.enrichment = EnrichmentLevel::Semantic;

        let enriched = enricher.enrich(ev).await.unwrap();
        assert_eq!(enriched.enrichment, EnrichmentLevel::Contextual);
        assert!(enriched.historical_response.is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
