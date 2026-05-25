use async_trait::async_trait;
use std::sync::Arc;
use scirust_trading_core::EventBus;

pub struct BinanceObserver;

impl BinanceObserver {
    pub fn new(_config: BinanceConfig, _bus: EventBus) -> Self { Self }
    pub async fn run(self: Arc<Self>) {
        loop { tokio::time::sleep(tokio::time::Duration::from_secs(1)).await; }
    }
}

pub struct BinanceConfig {
    pub symbol: String,
    pub exchange: String,
}

impl BinanceConfig {
    pub fn futures(symbol: String) -> Self {
        Self { symbol, exchange: "binance-futures".into() }
    }
}
