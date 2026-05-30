//!
//! Si on détecte un gap (séquence), on invalide et on resync.

use crate::bars::BarAggregator;
use crate::features::FeatureWindow;
use crate::orderbook::{LocalOrderBook, OrderBookEvent};
use crate::{ObserverError, ObserverResult};
    /// chaque diff, mais on n'émet qu'à cette cadence pour ne pas saturer
    /// les consommateurs.
    pub emit_interval_ms: u64,
    /// Si Some, on agrège les trades en tick bars de N ticks et on les
    /// publie sur `bus.bars`. Utile pour alimenter l'AtrTracker.
    pub tick_bar_size: Option<u32>,
}

impl BinanceConfig {
            depth_speed_ms: 100,
            reconnect_delay_sec: 2,
            emit_interval_ms: 500,
            tick_bar_size: Some(500),
        }
    }

            depth_speed_ms: 100,
            reconnect_delay_sec: 2,
            emit_interval_ms: 500,
            tick_bar_size: Some(500),
        }
    }

    pub bus: EventBus,
    book: Arc<Mutex<LocalOrderBook>>,
    features: Arc<Mutex<FeatureWindow>>,
    bar_agg: Option<Arc<Mutex<crate::bars::TickBarAggregator>>>,
}

impl BinanceObserver {
    pub fn new(config: BinanceConfig, bus: EventBus) -> Self {
        let book = LocalOrderBook::new(config.exchange(), config.symbol.clone());
        let bar_agg = config.tick_bar_size.map(|n| {
            Arc::new(Mutex::new(crate::bars::TickBarAggregator::new(
                config.exchange(),
                config.symbol.clone(),
                n,
            )))
        });
        Self {
            config,
            bus,
            book: Arc::new(Mutex::new(book)),
            features: Arc::new(Mutex::new(FeatureWindow::new(60))),
            bar_agg,
        }
    }

                                    CombinedEvent::Depth(ev) => buffer.push(ev),
                                    CombinedEvent::Trade(t) => {
                                        let _ = self.bus.trades.send(t.clone());
                                        self.features.lock().await.push_trade(t.clone());
                                        if let Some(agg) = &self.bar_agg {
                                            let mut a = agg.lock().await;
                                            if let Some(bar) = a.ingest(&t) {
                                                let _ = self.bus.bars.send(bar);
                                            }
                                        }
                                    }
                                }
                            }
                                }
                                Ok(CombinedEvent::Trade(t)) => {
                                    let _ = self.bus.trades.send(t.clone());
                                    self.features.lock().await.push_trade(t.clone());
                                    if let Some(agg) = &self.bar_agg {
                                        let mut a = agg.lock().await;
                                        if let Some(bar) = a.ingest(&t) {
                                            let _ = self.bus.bars.send(bar);
                                        }
                                    }
                                }
                                Err(e) => tracing::warn!("parse error: {e}"),
                            }
