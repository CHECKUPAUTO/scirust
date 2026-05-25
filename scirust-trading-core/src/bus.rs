use tokio::sync::broadcast;
use std::sync::Arc;

#[derive(Clone)]
pub struct EventBus {
    pub market: broadcast::Sender<Vec<u8>>,
    pub news: broadcast::Sender<Vec<u8>>,
    pub decisions: broadcast::Sender<Vec<u8>>,
    pub bars: broadcast::Sender<Vec<u8>>,
    pub control: broadcast::Sender<Vec<u8>>,
}

impl EventBus {
    pub fn new() -> Self {
        let (market, _) = broadcast::channel(1024);
        let (news, _) = broadcast::channel(1024);
        let (decisions, _) = broadcast::channel(1024);
        let (bars, _) = broadcast::channel(128);
        let (control, _) = broadcast::channel(16);
        Self { market, news, decisions, bars, control }
    }

    pub fn subscribe(&self) -> (broadcast::Receiver<Vec<u8>>, broadcast::Receiver<Vec<u8>>) {
        (self.news.subscribe(), self.market.subscribe())
    }
}
