use async_trait::async_trait;
use scirust_trading_core::EventBus;

pub struct ShadowEvaluator;

impl ShadowEvaluator {
    pub fn new() -> Self { Self }
    pub fn spawn(&self, _bus: &EventBus) -> tokio::sync::broadcast::Receiver<Vec<u8>> {
        let (tx, rx) = tokio::sync::broadcast::channel(100);
        rx
    }
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Vec<u8>> {
        let (tx, rx) = tokio::sync::broadcast::channel(100);
        rx
    }
}
