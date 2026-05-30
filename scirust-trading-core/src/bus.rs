use crate::codified::CodifiedEvent;
use crate::market::MarketState;
use crate::types::{Bar, Order, Trade};
use tokio::sync::broadcast;

#[derive(Debug, Clone)]
pub enum Event {
    Market(MarketState),
    News(CodifiedEvent),
    Trade(Trade),
    OrderUpdate(Order),
    Bar(Bar),
}

pub struct EventBus {
    pub market: broadcast::Sender<MarketState>,
    pub news: broadcast::Sender<CodifiedEvent>,
    pub trades: broadcast::Sender<Trade>,
    pub orders: broadcast::Sender<Order>,
    pub bars: broadcast::Sender<Bar>,
}

impl EventBus {
    pub fn new() -> Self {
        let (market, _) = broadcast::channel(1024);
        let (news, _) = broadcast::channel(256);
        let (trades, _) = broadcast::channel(4096);
        let (orders, _) = broadcast::channel(1024);
        let (bars, _) = broadcast::channel(1024);
        Self {
            market,
            news,
            trades,
            orders,
            bars,
        }
    }

    pub fn subscribe(&self) -> EventBusHandle {
        EventBusHandle {
            market_rx: self.market.subscribe(),
            news_rx: self.news.subscribe(),
            trades_rx: self.trades.subscribe(),
            orders_rx: self.orders.subscribe(),
            bars_rx: self.bars.subscribe(),
        }
    }
}

pub struct EventBusHandle {
    pub market_rx: broadcast::Receiver<MarketState>,
    pub news_rx: broadcast::Receiver<CodifiedEvent>,
    pub trades_rx: broadcast::Receiver<Trade>,
    pub orders_rx: broadcast::Receiver<Order>,
    pub bars_rx: broadcast::Receiver<Bar>,
}
