//! Pure-Rust SVG charting — so the agent can *show* what it found, not just
//! describe it.
//!
//! No dependencies: every chart is a self-contained SVG string the LLM can drop
//! straight into a message, an HTML artifact, or a file. Supports candlestick
//! price charts with indicator overlays and entry/exit markers, plus an equity
//! curve renderer for backtest results. Colours are chosen to read on both light
//! and dark backgrounds; the background itself is transparent by default.

use crate::market::Candle;

/// A named line drawn over the price panel (e.g. an SMA or a band). `values`
/// must align 1:1 with the candles; `NaN` entries are gaps (not drawn).
#[derive(Debug, Clone)]
pub struct Overlay {
    pub name: String,
    pub color: String,
    pub values: Vec<f32>,
}

impl Overlay {
    pub fn new(name: &str, color: &str, values: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            color: color.to_string(),
            values,
        }
    }
}

/// An entry/exit annotation at a candle index and price.
#[derive(Debug, Clone)]
pub struct Marker {
    pub index: usize,
    pub price: f32,
    pub bullish: bool,
    pub label: String,
}

/// Chart geometry & labelling.
#[derive(Debug, Clone)]
pub struct ChartOptions {
    pub width: u32,
    pub height: u32,
    pub title: String,
}

impl Default for ChartOptions {
    fn default() -> Self {
        Self {
            width: 900,
            height: 480,
            title: String::new(),
        }
    }
}

const MARGIN_L: f32 = 60.0;
const MARGIN_R: f32 = 16.0;
const MARGIN_T: f32 = 34.0;
const MARGIN_B: f32 = 28.0;
const UP_COLOR: &str = "#26a69a";
const DOWN_COLOR: &str = "#ef5350";
const AXIS_COLOR: &str = "#8892a0";
const GRID_COLOR: &str = "#8892a033";

fn fmt(x: f32) -> String {
    // Compact fixed formatting; avoids locale/exponent surprises in SVG.
    format!("{:.2}", x)
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Map a value in `[vmin, vmax]` to a y pixel (inverted: high value = low y).
fn y_of(v: f32, vmin: f32, vmax: f32, top: f32, bottom: f32) -> f32 {
    if (vmax - vmin).abs() < 1e-12
    {
        return (top + bottom) / 2.0;
    }
    bottom - (v - vmin) / (vmax - vmin) * (bottom - top)
}

/// Render a candlestick chart with optional overlays and markers. Returns a
/// complete `<svg>…</svg>` document string.
pub fn candlestick_svg(
    candles: &[Candle],
    overlays: &[Overlay],
    markers: &[Marker],
    opts: &ChartOptions,
) -> String {
    let w = opts.width as f32;
    let h = opts.height as f32;
    if candles.is_empty()
    {
        return empty_svg(opts, "no data");
    }
    let plot_l = MARGIN_L;
    let plot_r = w - MARGIN_R;
    let plot_t = MARGIN_T;
    let plot_b = h - MARGIN_B;
    let plot_w = (plot_r - plot_l).max(1.0);

    // Price range across highs/lows and finite overlay values.
    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for c in candles
    {
        vmin = vmin.min(c.low);
        vmax = vmax.max(c.high);
    }
    for ov in overlays
    {
        for &v in &ov.values
        {
            if v.is_finite()
            {
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
    }
    if !vmin.is_finite() || !vmax.is_finite()
    {
        return empty_svg(opts, "no finite prices");
    }
    let pad = (vmax - vmin) * 0.05 + 1e-6;
    vmin -= pad;
    vmax += pad;

    let n = candles.len();
    let slot = plot_w / n as f32;
    let body_w = (slot * 0.6).clamp(1.0, 14.0);
    let x_center = |i: usize| plot_l + (i as f32 + 0.5) * slot;

    let mut s = String::with_capacity(4096 + n * 120);
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" font-family=\"ui-sans-serif,system-ui,sans-serif\">",
        opts.width, opts.height, opts.width, opts.height
    ));

    // Title.
    if !opts.title.is_empty()
    {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"20\" font-size=\"15\" font-weight=\"600\" fill=\"{}\">{}</text>",
            plot_l,
            AXIS_COLOR,
            escape(&opts.title)
        ));
    }

    // Horizontal gridlines + price labels (5 levels).
    for k in 0..=4
    {
        let frac = k as f32 / 4.0;
        let price = vmax - frac * (vmax - vmin);
        let y = plot_t + frac * (plot_b - plot_t);
        s.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\"/>",
            plot_l, y, plot_r, y, GRID_COLOR
        ));
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{}\" text-anchor=\"end\">{}</text>",
            plot_l - 6.0,
            y + 3.0,
            AXIS_COLOR,
            fmt(price)
        ));
    }

    // Candles.
    for (i, c) in candles.iter().enumerate()
    {
        let xc = x_center(i);
        let up = c.close >= c.open;
        let color = if up { UP_COLOR } else { DOWN_COLOR };
        let y_high = y_of(c.high, vmin, vmax, plot_t, plot_b);
        let y_low = y_of(c.low, vmin, vmax, plot_t, plot_b);
        let y_open = y_of(c.open, vmin, vmax, plot_t, plot_b);
        let y_close = y_of(c.close, vmin, vmax, plot_t, plot_b);
        let y_body_top = y_open.min(y_close);
        let y_body_h = (y_open - y_close).abs().max(1.0);
        // Wick.
        s.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\"/>",
            xc, y_high, xc, y_low, color
        ));
        // Body.
        s.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"/>",
            xc - body_w / 2.0,
            y_body_top,
            body_w,
            y_body_h,
            color
        ));
    }

    // Overlays as polylines.
    for ov in overlays
    {
        let mut pts = String::new();
        let mut started = false;
        for (i, &v) in ov.values.iter().enumerate().take(n)
        {
            if !v.is_finite()
            {
                started = false;
                continue;
            }
            let x = x_center(i);
            let y = y_of(v, vmin, vmax, plot_t, plot_b);
            if started
            {
                pts.push_str(&format!(" L{:.1},{:.1}", x, y));
            }
            else
            {
                pts.push_str(&format!("M{:.1},{:.1}", x, y));
                started = true;
            }
        }
        if !pts.is_empty()
        {
            s.push_str(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.6\"/>",
                pts, ov.color
            ));
        }
    }

    // Markers (triangles: up=bullish/green below the bar, down=bearish/red above).
    for m in markers
    {
        if m.index >= n
        {
            continue;
        }
        let x = x_center(m.index);
        let y = y_of(m.price, vmin, vmax, plot_t, plot_b);
        let (color, dy) = if m.bullish { (UP_COLOR, 10.0) } else { (DOWN_COLOR, -10.0) };
        let (a, b, c) = if m.bullish
        {
            ((x, y + dy - 8.0), (x - 5.0, y + dy), (x + 5.0, y + dy))
        }
        else
        {
            ((x, y + dy + 8.0), (x - 5.0, y + dy), (x + 5.0, y + dy))
        };
        s.push_str(&format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{}\"/>",
            a.0, a.1, b.0, b.1, c.0, c.1, color
        ));
    }

    // Overlay legend.
    if !overlays.is_empty()
    {
        let mut lx = plot_l;
        let ly = plot_t - 6.0;
        for ov in overlays
        {
            s.push_str(&format!(
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"10\" height=\"10\" fill=\"{}\"/>",
                lx, ly - 9.0, ov.color
            ));
            s.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{}\">{}</text>",
                lx + 14.0,
                ly,
                AXIS_COLOR,
                escape(&ov.name)
            ));
            lx += 16.0 + ov.name.len() as f32 * 6.5 + 12.0;
        }
    }

    // Axis frame.
    s.push_str(&format!(
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1\"/>",
        plot_l,
        plot_t,
        plot_r - plot_l,
        plot_b - plot_t,
        AXIS_COLOR
    ));

    s.push_str("</svg>");
    s
}

/// Render an equity curve (line chart) — the standard backtest visual.
pub fn equity_curve_svg(equity: &[f32], opts: &ChartOptions) -> String {
    let w = opts.width as f32;
    let h = opts.height as f32;
    if equity.len() < 2
    {
        return empty_svg(opts, "no equity data");
    }
    let plot_l = MARGIN_L;
    let plot_r = w - MARGIN_R;
    let plot_t = MARGIN_T;
    let plot_b = h - MARGIN_B;
    let plot_w = (plot_r - plot_l).max(1.0);

    let mut vmin = f32::INFINITY;
    let mut vmax = f32::NEG_INFINITY;
    for &v in equity
    {
        if v.is_finite()
        {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if !vmin.is_finite()
    {
        return empty_svg(opts, "no finite equity");
    }
    let pad = (vmax - vmin) * 0.05 + 1e-6;
    vmin -= pad;
    vmax += pad;

    let n = equity.len();
    let x_of = |i: usize| plot_l + i as f32 / (n as f32 - 1.0) * plot_w;

    let mut s = String::with_capacity(2048 + n * 16);
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" font-family=\"ui-sans-serif,system-ui,sans-serif\">",
        opts.width, opts.height, opts.width, opts.height
    ));
    if !opts.title.is_empty()
    {
        s.push_str(&format!(
            "<text x=\"{}\" y=\"20\" font-size=\"15\" font-weight=\"600\" fill=\"{}\">{}</text>",
            plot_l,
            AXIS_COLOR,
            escape(&opts.title)
        ));
    }
    for k in 0..=4
    {
        let frac = k as f32 / 4.0;
        let val = vmax - frac * (vmax - vmin);
        let y = plot_t + frac * (plot_b - plot_t);
        s.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\"/>",
            plot_l, y, plot_r, y, GRID_COLOR
        ));
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{}\" text-anchor=\"end\">{}</text>",
            plot_l - 6.0,
            y + 3.0,
            AXIS_COLOR,
            fmt(val)
        ));
    }
    // The curve.
    let start = equity[0];
    let color = if equity[n - 1] >= start { UP_COLOR } else { DOWN_COLOR };
    let mut pts = String::new();
    for (i, &v) in equity.iter().enumerate()
    {
        let x = x_of(i);
        let y = y_of(v, vmin, vmax, plot_t, plot_b);
        if i == 0
        {
            pts.push_str(&format!("M{:.1},{:.1}", x, y));
        }
        else
        {
            pts.push_str(&format!(" L{:.1},{:.1}", x, y));
        }
    }
    s.push_str(&format!(
        "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.8\"/>",
        pts, color
    ));
    // Starting-capital baseline.
    let y0 = y_of(start, vmin, vmax, plot_t, plot_b);
    s.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\" stroke-dasharray=\"4 3\"/>",
        plot_l, y0, plot_r, y0, AXIS_COLOR
    ));
    s.push_str(&format!(
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1\"/>",
        plot_l,
        plot_t,
        plot_r - plot_l,
        plot_b - plot_t,
        AXIS_COLOR
    ));
    s.push_str("</svg>");
    s
}

fn empty_svg(opts: &ChartOptions, msg: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><text x=\"{}\" y=\"{}\" font-size=\"13\" fill=\"{}\" text-anchor=\"middle\">{}</text></svg>",
        opts.width,
        opts.height,
        opts.width,
        opts.height,
        opts.width / 2,
        opts.height / 2,
        AXIS_COLOR,
        escape(msg)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candles() -> Vec<Candle> {
        (0..30)
            .map(|i| {
                let base = 100.0 + (i as f32 * 0.4).sin() * 5.0;
                Candle {
                    ts_ms: i as i64 * 60_000,
                    open: base,
                    high: base + 2.0,
                    low: base - 2.0,
                    close: base + (if i % 2 == 0 { 1.0 } else { -1.0 }),
                    volume: 100.0,
                }
            })
            .collect()
    }

    #[test]
    fn candlestick_svg_is_wellformed() {
        let c = candles();
        let ov = Overlay::new("SMA10", "#f6c343", crate::indicators::sma(&c.iter().map(|k| k.close).collect::<Vec<_>>(), 10));
        let markers = vec![Marker { index: 5, price: c[5].close, bullish: true, label: "entry".into() }];
        let svg = candlestick_svg(&c, &[ov], &markers, &ChartOptions { title: "BTC/USDT".into(), ..Default::default() });
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("<rect")); // candle bodies
        assert!(svg.contains("<path")); // overlay
        assert!(svg.contains("<polygon")); // marker
        assert!(svg.contains("BTC/USDT"));
    }

    #[test]
    fn equity_curve_svg_is_wellformed() {
        let eq: Vec<f32> = (0..50).map(|i| 10_000.0 + i as f32 * 20.0).collect();
        let svg = equity_curve_svg(&eq, &ChartOptions { title: "Equity".into(), ..Default::default() });
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<path"));
        assert!(svg.contains("Equity"));
    }

    #[test]
    fn empty_inputs_are_safe() {
        let svg = candlestick_svg(&[], &[], &[], &ChartOptions::default());
        assert!(svg.contains("no data"));
        let svg2 = equity_curve_svg(&[1.0], &ChartOptions::default());
        assert!(svg2.contains("no equity data"));
    }

    #[test]
    fn text_is_escaped() {
        let c = candles();
        let svg = candlestick_svg(&c, &[], &[], &ChartOptions { title: "A<b>&c".into(), ..Default::default() });
        assert!(svg.contains("A&lt;b&gt;&amp;c"));
        assert!(!svg.contains("A<b>&c"));
    }
}
