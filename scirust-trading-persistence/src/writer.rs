// ─── Insertions par lots (transactions) ─────────────────────────────────

pub fn flush_events(conn: &Connection, events: &[CodifiedEvent]) -> PersistenceResult<()> {
    if events.is_empty() {
        return Ok(());
    }
    Ok(())
}

pub fn flush_market(conn: &Connection, states: &[MarketState]) -> PersistenceResult<()> {
    if states.is_empty() {
        return Ok(());
    }
    Ok(())
}

pub fn flush_trades(conn: &Connection, trades: &[Trade]) -> PersistenceResult<()> {
    if trades.is_empty() {
        return Ok(());
    }
    Ok(())
}

pub fn flush_bars(conn: &Connection, bars: &[Bar]) -> PersistenceResult<()> {
    if bars.is_empty() {
        return Ok(());
    }
