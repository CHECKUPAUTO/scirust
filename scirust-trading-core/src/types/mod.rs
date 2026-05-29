use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub kind: BarKind,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BarKind { Min1, Min5, Hour1, Day1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub price: f64,
    pub quantity: f64,
    pub side: crate::market::Side,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: String,
    pub status: String,
}
