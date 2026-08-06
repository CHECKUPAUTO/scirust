use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub struct SignalHistory {
    current: BTreeMap<String, f64>,
    past: Vec<BTreeMap<String, f64>>,
}

impl SignalHistory {
    pub fn new(current: BTreeMap<String, f64>, past: Vec<BTreeMap<String, f64>>) -> Self {
        Self { current, past }
    }

    pub fn current(&self, name: &str) -> Option<f64> {
        self.current.get(name).copied()
    }

    pub fn lag(&self, name: &str, lag: usize) -> Option<f64> {
        if lag == 0 {
            return self.current(name);
        }
        self.past
            .get(lag - 1)
            .and_then(|signals| signals.get(name))
            .copied()
    }

    pub fn ema(&self, name: &str, window: usize) -> Option<f64> {
        if window == 0 {
            return None;
        }
        let alpha = 2.0 / (window as f64 + 1.0);
        let mut samples = Vec::with_capacity(window);
        for lag in (0..window).rev() {
            if let Some(value) = self.lag(name, lag) {
                samples.push(value);
            }
        }
        let mut iter = samples.into_iter();
        let mut value = iter.next()?;
        for sample in iter {
            value = alpha.mul_add(sample, (1.0 - alpha) * value);
        }
        Some(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Signal(String, usize),
    Plus,
    Minus,
    Star,
    Slash,
    LeftParen,
    RightParen,
    Comma,
    Function(String),
}

pub fn evaluate_expression(expression: &str, history: &SignalHistory) -> Result<f64, String> {
    let tokens = tokenize(expression)?;
    let mut parser = Parser {
        tokens: &tokens,
        position: 0,
        history,
    };
    let value = parser.parse_expression()?;
    if parser.position != tokens.len() {
        return Err(format!(
            "unexpected token at position {} in `{expression}`",
            parser.position
        ));
    }
    if !value.is_finite() {
        return Err(format!("expression `{expression}` produced a non-finite value"));
    }
    Ok(value)
}

fn tokenize(expression: &str) -> Result<Vec<Token>, String> {
    let bytes = expression.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            index += 1;
            continue;
        }
        match byte {
            b'+' => {
                tokens.push(Token::Plus);
                index += 1;
            }
            b'-' => {
                tokens.push(Token::Minus);
                index += 1;
            }
            b'*' => {
                tokens.push(Token::Star);
                index += 1;
            }
            b'/' => {
                tokens.push(Token::Slash);
                index += 1;
            }
            b'(' => {
                tokens.push(Token::LeftParen);
                index += 1;
            }
            b')' => {
                tokens.push(Token::RightParen);
                index += 1;
            }
            b',' => {
                tokens.push(Token::Comma);
                index += 1;
            }
            b'0'..=b'9' | b'.' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && matches!(bytes[index], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    if matches!(bytes[index], b'+' | b'-')
                        && !matches!(bytes[index - 1], b'e' | b'E')
                    {
                        break;
                    }
                    index += 1;
                }
                let raw = &expression[start..index];
                let value = raw
                    .parse::<f64>()
                    .map_err(|error| format!("invalid number `{raw}`: {error}"))?;
                tokens.push(Token::Number(value));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && matches!(bytes[index], b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')
                {
                    index += 1;
                }
                let name = expression[start..index].to_string();
                if index < bytes.len() && bytes[index] == b'[' {
                    index += 1;
                    if index >= bytes.len() || bytes[index] != b't' {
                        return Err(format!("invalid temporal index for signal `{name}`"));
                    }
                    index += 1;
                    let lag = if index < bytes.len() && bytes[index] == b']' {
                        index += 1;
                        0
                    } else {
                        if index >= bytes.len() || bytes[index] != b'-' {
                            return Err(format!("invalid temporal index for signal `{name}`"));
                        }
                        index += 1;
                        let lag_start = index;
                        while index < bytes.len() && bytes[index].is_ascii_digit() {
                            index += 1;
                        }
                        if lag_start == index || index >= bytes.len() || bytes[index] != b']' {
                            return Err(format!("invalid temporal index for signal `{name}`"));
                        }
                        let lag = expression[lag_start..index]
                            .parse::<usize>()
                            .map_err(|error| format!("invalid lag for `{name}`: {error}"))?;
                        index += 1;
                        lag
                    };
                    tokens.push(Token::Signal(name, lag));
                } else if index < bytes.len() && bytes[index] == b'(' {
                    tokens.push(Token::Function(name));
                } else {
                    tokens.push(Token::Signal(name, 0));
                }
            }
            other => {
                return Err(format!(
                    "unsupported character `{}` in expression",
                    char::from(other)
                ));
            }
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    position: usize,
    history: &'a SignalHistory,
}

impl Parser<'_> {
    fn parse_expression(&mut self) -> Result<f64, String> {
        let mut value = self.parse_term()?;
        loop {
            match self.tokens.get(self.position) {
                Some(Token::Plus) => {
                    self.position += 1;
                    value += self.parse_term()?;
                }
                Some(Token::Minus) => {
                    self.position += 1;
                    value -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_term(&mut self) -> Result<f64, String> {
        let mut value = self.parse_factor()?;
        loop {
            match self.tokens.get(self.position) {
                Some(Token::Star) => {
                    self.position += 1;
                    value *= self.parse_factor()?;
                }
                Some(Token::Slash) => {
                    self.position += 1;
                    let denominator = self.parse_factor()?;
                    if denominator.abs() <= f64::EPSILON {
                        return Err("division by zero".to_string());
                    }
                    value /= denominator;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn parse_factor(&mut self) -> Result<f64, String> {
        match self.tokens.get(self.position).cloned() {
            Some(Token::Minus) => {
                self.position += 1;
                Ok(-self.parse_factor()?)
            }
            Some(Token::Number(value)) => {
                self.position += 1;
                Ok(value)
            }
            Some(Token::Signal(name, lag)) => {
                self.position += 1;
                self.history
                    .lag(&name, lag)
                    .ok_or_else(|| format!("missing signal `{name}` at lag {lag}"))
            }
            Some(Token::LeftParen) => {
                self.position += 1;
                let value = self.parse_expression()?;
                self.expect_right_paren()?;
                Ok(value)
            }
            Some(Token::Function(name)) => self.parse_function(&name),
            other => Err(format!("expected expression factor, found {other:?}")),
        }
    }

    fn parse_function(&mut self, name: &str) -> Result<f64, String> {
        self.position += 1;
        self.expect_left_paren()?;
        match name {
            "abs" => {
                let value = self.parse_expression()?;
                self.expect_right_paren()?;
                Ok(value.abs())
            }
            name if name.starts_with("ema_") => {
                let window = name[4..]
                    .parse::<usize>()
                    .map_err(|error| format!("invalid EMA window in `{name}`: {error}"))?;
                let signal = match self.tokens.get(self.position).cloned() {
                    Some(Token::Signal(signal, 0)) => signal,
                    _ => return Err(format!("`{name}` requires a current signal argument")),
                };
                self.position += 1;
                self.expect_right_paren()?;
                self.history
                    .ema(&signal, window)
                    .ok_or_else(|| format!("insufficient history for `{name}({signal})`"))
            }
            _ => Err(format!("unsupported function `{name}`")),
        }
    }

    fn expect_left_paren(&mut self) -> Result<(), String> {
        match self.tokens.get(self.position) {
            Some(Token::LeftParen) => {
                self.position += 1;
                Ok(())
            }
            _ => Err("expected `(`".to_string()),
        }
    }

    fn expect_right_paren(&mut self) -> Result<(), String> {
        match self.tokens.get(self.position) {
            Some(Token::RightParen) => {
                self.position += 1;
                Ok(())
            }
            _ => Err("expected `)`".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history() -> SignalHistory {
        SignalHistory::new(
            BTreeMap::from([
                ("drift".to_string(), 0.4),
                ("skip_margin".to_string(), 0.8),
                ("cache_age".to_string(), 3.0),
            ]),
            vec![
                BTreeMap::from([("drift".to_string(), 0.3)]),
                BTreeMap::from([("drift".to_string(), 0.1)]),
                BTreeMap::from([("drift".to_string(), 0.0)]),
            ],
        )
    }

    #[test]
    fn evaluates_current_interaction() {
        let value = evaluate_expression("skip_margin / (1 + cache_age)", &history()).unwrap();
        assert!((value - 0.2).abs() < 1e-12);
    }

    #[test]
    fn evaluates_temporal_curvature() {
        let value = evaluate_expression(
            "drift[t] - 2 * drift[t-1] + drift[t-2]",
            &history(),
        )
        .unwrap();
        assert!((value + 0.1).abs() < 1e-12);
    }

    #[test]
    fn evaluates_ema_residual() {
        let value = evaluate_expression("drift[t] - ema_4(drift)", &history()).unwrap();
        assert!(value.is_finite());
    }

    #[test]
    fn rejects_missing_history() {
        let error = evaluate_expression("drift[t-5]", &history()).unwrap_err();
        assert!(error.contains("missing signal"));
    }

    #[test]
    fn rejects_division_by_zero() {
        let error = evaluate_expression("drift / (cache_age - 3)", &history()).unwrap_err();
        assert_eq!(error, "division by zero");
    }
}
