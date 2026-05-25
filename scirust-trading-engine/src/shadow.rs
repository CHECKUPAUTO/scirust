//! Le shadow maintient en mémoire une fenêtre glissante d'events actifs
//! (filtre les expirés à chaque tick).

use crate::atr::AtrTracker;
use crate::decision::Decision;
use crate::evaluator::{Evaluator, EvaluatorContext};
use crate::portfolio::PortfolioState;
    /// Active la mise à jour des positions virtuelles à partir des décisions
    /// Open/Close (utile pour simuler une session paper)
    pub track_virtual_positions: bool,
    /// Si Some, maintient un AtrTracker alimenté par `bus.bars` avec cette
    /// période (typique : 14). L'ATR courant est injecté dans
    /// `EvaluatorContext::atr`, ce qui active `SizingMethod::AtrTargetRisk`.
    pub atr_period: Option<usize>,
}

impl Default for ShadowConfig {
            decisions_channel_cap: 512,
            virtual_equity_quote: 10_000.0,
            track_virtual_positions: true,
            atr_period: Some(14),
        }
    }
}
    /// (un event peut être ré-émis sur le bus à un niveau d'enrichissement
    /// plus élevé)
    events: Arc<Mutex<HashMap<uuid::Uuid, CodifiedEvent>>>,
    /// ATR tracker, alimenté par `bus.bars`. None si désactivé.
    atr_tracker: Option<Arc<Mutex<AtrTracker>>>,
}

impl ShadowEvaluator {
    pub fn new(evaluator: Evaluator, config: ShadowConfig) -> Self {
        let (decisions_tx, _) = broadcast::channel(config.decisions_channel_cap);
        let atr_tracker = config
            .atr_period
            .map(|p| Arc::new(Mutex::new(AtrTracker::new(p))));
        Self {
            evaluator: Arc::new(evaluator),
            portfolio: Arc::new(Mutex::new(PortfolioState::virtual_paper(
            config,
            decisions_tx,
            events: Arc::new(Mutex::new(HashMap::new())),
            atr_tracker,
        }
    }

    pub fn spawn(self: Arc<Self>, bus: &EventBus) -> tokio::task::JoinHandle<()> {
        let mut news_rx = bus.news.subscribe();
        let mut market_rx = bus.market.subscribe();
        let mut bars_rx = bus.bars.subscribe();
        let me = Arc::clone(&self);

        tokio::spawn(async move {
                            tracing::warn!("shadow market_rx lagged by {n}");
                        }
                    },
                    b = bars_rx.recv() => match b {
                        Ok(bar) => {
                            if let Some(tracker) = &me.atr_tracker {
                                tracker.lock().await.ingest(&bar);
                            }
                        }
                        Err(broadcast::error::RecvError::Closed) => return,
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!("shadow bars_rx lagged by {n}");
                        }
                    },
                }
            }
        })
            events.values().filter(|e| e.is_active(now)).cloned().collect()
        };

        // 3. Lookup de l'ATR courant pour ce symbole (None si pas de tracker
        // ou pas encore assez de bars)
        let atr = if let Some(tracker) = &self.atr_tracker {
            tracker.lock().await.current(&market.symbol.canonical())
        } else {
            None
        };

        // 4. Évalue
        let decision = {
            let portfolio = self.portfolio.lock().await;
            let ctx = EvaluatorContext {
                market,
                events: &active_events,
                portfolio: &portfolio,
                atr,
            };
            self.evaluator.evaluate(&ctx)?
        };
    use crate::decision::DecisionAction;
    use chrono::{Duration, Utc};
    use scirust_trading_core::{
        BarKind, Bar, Category, CodifiedEvent, DecisionSchema, EventBus, EventTiming,
        Exchange, Gate, GateCondition, MarketState, Polarity, Sizing, SizingMethod,
        SourceId, Symbol, Target,
    };

    fn schema_with_no_gates() -> DecisionSchema {
        let pos = p1.position(Exchange::Binance, &Symbol::new("BTC", "USDT")).unwrap();
        assert!(pos.is_long());
    }

    fn make_bar(mid_price: f64, range: f64) -> Bar {
        let now = Utc::now();
        Bar {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTC", "USDT"),
            kind: BarKind::Tick { ticks_per_bar: 500 },
            start: now,
            end: now,
            open: mid_price - range / 2.0,
            high: mid_price + range / 2.0,
            low: mid_price - range / 2.0,
            close: mid_price + range / 2.0,
            volume: 1.0,
            trade_count: 500,
            buy_volume: 0.5,
            sell_volume: 0.5,
            vwap: mid_price,
        }
    }

    #[tokio::test]
    async fn shadow_consumes_bars_to_build_atr() {
        let bus = EventBus::new();
        let schema = schema_with_no_gates();
        let evaluator = Evaluator::new(schema);
        let shadow = Arc::new(ShadowEvaluator::new(
            evaluator,
            ShadowConfig {
                atr_period: Some(3),
                ..Default::default()
            },
        ));
        let _handle = Arc::clone(&shadow).spawn(&bus);

        // Pousse 3 bars avec range constant 250
        for _ in 0..3 {
            bus.bars.send(make_bar(50_000.0, 250.0)).unwrap();
        }
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;

        // Vérifie que l'AtrTracker a accumulé : l'ATR pour BTC/USDT existe
        let tracker = shadow.atr_tracker.as_ref().expect("tracker should exist");
        let atr_value = tracker.lock().await.current("BTC/USDT");
        assert!(atr_value.is_some());
        let v = atr_value.unwrap();
        // 3 bars de range 250 → ATR converge vers 250
        assert!((v - 250.0).abs() < 1.0, "ATR={v}, expected ~250");
    }
}
