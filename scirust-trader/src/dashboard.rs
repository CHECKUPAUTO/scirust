//! Self-contained HTML dashboards — the visual layer on top of the toolbox.
//!
//! Where [`crate::chart`] emits a single SVG, this module assembles a whole
//! **report page**: the ranked opportunities from a [`ScanReport`], metric
//! cards + an embedded equity-curve SVG + a trade log for each
//! [`BacktestReport`]. The output is one self-contained HTML string (inline CSS,
//! embedded SVG, no external requests, light/dark aware) an agent can hand
//! straight to the user to open — turning "show me what you found" into a
//! visual, shareable page rather than a wall of JSON.
//!
//! Pure Rust, deterministic, no dependencies.

use crate::backtest::BacktestReport;
use crate::chart::{ChartOptions, equity_curve_svg};
use crate::scanner::ScanReport;

/// Page framing.
#[derive(Debug, Clone)]
pub struct DashboardOptions {
    pub title: String,
    pub subtitle: String,
}

impl Default for DashboardOptions {
    fn default() -> Self {
        Self {
            title: "SciRust Trader".to_string(),
            subtitle: "Deterministic, auditable, simulation-first".to_string(),
        }
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn cls(x: f32) -> &'static str {
    if x > 0.0
    {
        "pos"
    }
    else if x < 0.0
    {
        "neg"
    }
    else
    {
        "flat"
    }
}

fn stat_card(label: &str, value: &str, tone: &str) -> String {
    format!(
        "<div class=\"card\"><div class=\"card-label\">{}</div><div class=\"card-value {}\">{}</div></div>",
        esc(label),
        tone,
        esc(value)
    )
}

/// Render the full dashboard page. Either or both sections may be empty.
pub fn render_dashboard(
    scan: Option<&ScanReport>,
    backtests: &[(String, &BacktestReport)],
    opts: &DashboardOptions,
) -> String {
    let mut body = String::new();

    // --- header ---
    body.push_str(&format!(
        "<header><h1>{}</h1><p class=\"sub\">{}</p></header>",
        esc(&opts.title),
        esc(&opts.subtitle)
    ));

    // --- scan section ---
    if let Some(report) = scan
    {
        body.push_str("<section><h2>Opportunities</h2>");
        body.push_str("<div class=\"cards\">");
        body.push_str(&stat_card(
            "Candidates",
            &report.num_candidates.to_string(),
            "",
        ));
        body.push_str(&stat_card("Matched", &report.num_matched.to_string(), ""));
        body.push_str(&stat_card("Symbols", &report.num_symbols.to_string(), ""));
        body.push_str(&stat_card(
            "Shown",
            &report.opportunities.len().to_string(),
            "",
        ));
        body.push_str("</div>");
        body.push_str(&format!(
            "<p class=\"proof\">Report proof: <code>{}</code> — {}</p>",
            &report.manifest_hash[..report.manifest_hash.len().min(16)],
            if report.verify()
            {
                "<span class=\"pos\">✓ verified</span>"
            }
            else
            {
                "<span class=\"neg\">✗ invalid</span>"
            }
        ));

        if report.opportunities.is_empty()
        {
            body.push_str("<p class=\"muted\">No opportunities matched the constraints.</p>");
        }
        else
        {
            let mut rows = String::new();
            for (i, o) in report.opportunities.iter().enumerate()
            {
                let side_tone = match o.action
                {
                    crate::agent::Action::Long => "pos",
                    crate::agent::Action::Short => "neg",
                    crate::agent::Action::Flat => "flat",
                };
                rows.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td class=\"mono\">{}</td>\
                     <td class=\"{}\">{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.2}</td>\
                     <td class=\"{}\">{:+.2}%</td><td>{:.2}</td>\
                     <td><div class=\"bar\"><span style=\"width:{:.0}%\"></span></div>{:.3}</td>\
                     <td class=\"mono muted\">{}</td></tr>",
                    i + 1,
                    esc(&o.symbol),
                    esc(&o.strategy),
                    side_tone,
                    o.action.label(),
                    o.entry,
                    o.stop_loss,
                    o.take_profit,
                    cls(o.backtest_total_return),
                    o.backtest_total_return * 100.0,
                    o.backtest_sharpe,
                    (o.score * 100.0).clamp(0.0, 100.0),
                    o.score,
                    &o.proof_hash[..o.proof_hash.len().min(10)],
                ));
            }
            body.push_str(&format!(
                "<div class=\"tw\"><table><thead><tr>\
                 <th>#</th><th>Symbol</th><th>Strategy</th><th>Side</th><th>Entry</th>\
                 <th>Stop</th><th>Target</th><th>Backtest</th><th>Sharpe</th><th>Score</th>\
                 <th>Proof</th></tr></thead><tbody>{}</tbody></table></div>",
                rows
            ));
        }
        body.push_str("</section>");
    }

    // --- backtest sections ---
    for (label, report) in backtests
    {
        body.push_str(&format!("<section><h2>Backtest — {}</h2>", esc(label)));
        body.push_str(&format!(
            "<p class=\"sub\">{} · {} · {}</p>",
            esc(&report.symbol),
            esc(&report.strategy),
            esc(&report.interval)
        ));
        let p = &report.performance;
        body.push_str("<div class=\"cards\">");
        body.push_str(&stat_card(
            "Total return",
            &format!("{:+.2}%", report.total_return * 100.0),
            cls(report.total_return),
        ));
        body.push_str(&stat_card(
            "Buy & hold",
            &format!("{:+.2}%", report.buy_hold_return * 100.0),
            cls(report.buy_hold_return),
        ));
        body.push_str(&stat_card(
            "Sharpe",
            &format!("{:.2}", p.sharpe),
            cls(p.sharpe),
        ));
        body.push_str(&stat_card(
            "Sortino",
            &format!("{:.2}", p.sortino),
            cls(p.sortino),
        ));
        body.push_str(&stat_card(
            "Max drawdown",
            &format!("{:.2}%", p.max_drawdown * 100.0),
            "neg",
        ));
        body.push_str(&stat_card(
            "Win rate",
            &format!("{:.0}%", p.trades.win_rate * 100.0),
            "",
        ));
        let pf = if p.trades.profit_factor.is_finite()
        {
            format!("{:.2}", p.trades.profit_factor)
        }
        else
        {
            "∞".to_string()
        };
        body.push_str(&stat_card("Profit factor", &pf, ""));
        body.push_str(&stat_card("Trades", &report.num_trades.to_string(), ""));
        body.push_str("</div>");

        // Equity curve SVG.
        let svg = equity_curve_svg(
            &report.equity_curve,
            &ChartOptions {
                width: 860,
                height: 300,
                title: format!("Equity — {}", report.symbol),
            },
        );
        body.push_str(&format!("<div class=\"chart\">{}</div>", svg));

        // Recent trades.
        if !report.trades.is_empty()
        {
            let mut rows = String::new();
            for t in report.trades.iter().rev().take(12)
            {
                rows.push_str(&format!(
                    "<tr><td class=\"{}\">{}</td><td>{:.2}</td><td>{:.2}</td><td>{:.4}</td>\
                     <td class=\"{}\">{:+.2}</td><td class=\"{}\">{:+.2}%</td><td>{}</td></tr>",
                    match t.action
                    {
                        crate::agent::Action::Long => "pos",
                        crate::agent::Action::Short => "neg",
                        crate::agent::Action::Flat => "flat",
                    },
                    t.action.label(),
                    t.entry_price,
                    t.exit_price,
                    t.qty,
                    cls(t.net_pnl),
                    t.net_pnl,
                    cls(t.return_pct),
                    t.return_pct * 100.0,
                    t.bars_held,
                ));
            }
            body.push_str(&format!(
                "<div class=\"tw\"><table><thead><tr><th>Side</th><th>Entry</th><th>Exit</th>\
                 <th>Qty</th><th>Net PnL</th><th>Return</th><th>Bars</th></tr></thead>\
                 <tbody>{}</tbody></table></div>",
                rows
            ));
        }
        body.push_str("</section>");
    }

    body.push_str(
        "<footer>Generated by SciRust Trader — deterministic &amp; simulation-first. \
         No real orders were placed.</footer>",
    );

    wrap_page(&opts.title, &body)
}

fn wrap_page(title: &str, body: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>{}</title><style>{}</style></head><body>{}</body></html>",
        esc(title),
        CSS,
        body
    )
}

const CSS: &str = "\
:root{--bg:#f7f8fa;--panel:#ffffff;--fg:#1c2230;--muted:#6b7280;--line:#e5e7eb;\
--pos:#0f9d6b;--neg:#e5484d;--flat:#8892a0;--accent:#3b82f6;}\
@media(prefers-color-scheme:dark){:root{--bg:#0e1117;--panel:#161b22;--fg:#e6edf3;\
--muted:#8b949e;--line:#232a33;--pos:#26a67a;--neg:#f0616d;--flat:#6b7684;}}\
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);\
font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;\
line-height:1.5;padding:24px;max-width:960px;margin:0 auto}\
header h1{margin:0 0 2px;font-size:22px}.sub{color:var(--muted);margin:0 0 4px;font-size:13px}\
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;\
padding:18px;margin:18px 0}h2{margin:0 0 12px;font-size:16px}\
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin:8px 0 14px}\
.card{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px}\
.card-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}\
.card-value{font-size:19px;font-weight:600;margin-top:3px}\
.pos{color:var(--pos)}.neg{color:var(--neg)}.flat{color:var(--flat)}.muted{color:var(--muted)}\
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}\
.proof{font-size:12px;color:var(--muted)}.proof code{color:var(--fg)}\
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;min-width:520px}\
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}\
th:nth-child(-n+3),td:nth-child(-n+3){text-align:left}\
thead th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}\
tbody tr:hover{background:var(--bg)}\
.bar{display:inline-block;width:44px;height:6px;background:var(--line);border-radius:3px;\
overflow:hidden;vertical-align:middle;margin-right:6px}\
.bar span{display:block;height:100%;background:var(--accent)}\
.chart{margin:10px 0;overflow-x:auto}.chart svg{max-width:100%;height:auto}\
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:18px}";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::{BacktestConfig, run_backtest};
    use crate::market::{Candle, MarketSnapshot};
    use crate::scanner::{OpportunityConstraints, ScanRiskConfig, scan};
    use crate::strategy::MaCross;

    fn uptrend(symbol: &str, n: usize) -> MarketSnapshot {
        let candles = (0..n)
            .map(|i| {
                let c = 100.0 + i as f32;
                Candle {
                    ts_ms: i as i64 * 60_000,
                    open: c,
                    high: c + 1.0,
                    low: c - 1.0,
                    close: c,
                    volume: 100.0,
                }
            })
            .collect();
        MarketSnapshot {
            exchange: "test".to_string(),
            symbol: symbol.to_string(),
            interval: "1h".to_string(),
            candles,
        }
    }

    #[test]
    fn dashboard_is_wellformed_html() {
        let series = vec![uptrend("BTC/USDT", 200)];
        let report = scan(
            &series,
            &OpportunityConstraints::default(),
            &ScanRiskConfig::default(),
        );
        let bt = run_backtest(
            &MaCross::sma(10, 30),
            &series[0].candles,
            &BacktestConfig::default(),
        );
        let html = render_dashboard(
            Some(&report),
            &[("SMA cross".to_string(), &bt)],
            &DashboardOptions::default(),
        );
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.trim_end().ends_with("</html>"));
        assert!(html.contains("<style>"));
        assert!(html.contains("Opportunities"));
        assert!(html.contains("Backtest"));
        assert!(html.contains("<svg")); // embedded equity curve
        assert!(html.contains("Total return"));
        // Balanced-ish: the section wrapper appears for scan + backtest.
        assert!(html.matches("<section>").count() >= 1);
    }

    #[test]
    fn dashboard_handles_empty_scan() {
        let html = render_dashboard(None, &[], &DashboardOptions::default());
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("SciRust Trader"));
    }

    #[test]
    fn dashboard_escapes_untrusted_text() {
        let opts = DashboardOptions {
            title: "A<script>&\"x".to_string(),
            subtitle: "s".to_string(),
        };
        let html = render_dashboard(None, &[], &opts);
        assert!(html.contains("A&lt;script&gt;&amp;&quot;x"));
        assert!(!html.contains("<script>"));
    }
}
