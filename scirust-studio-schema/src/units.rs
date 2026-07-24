//! A small, explicit unit-symbol table.
//!
//! This is deliberately not a general unit-parsing engine:
//! `scirust_units::Quantity`/`Dimension` are purely programmatic (see
//! `docs/studio/REPOSITORY_AUDIT.md` §7) and have no string parser, so
//! resolving a scenario's `unit = "m/s"` string into a checked
//! [`scirust_units::Dimension`] needs *some* symbol table — this is that
//! table, kept intentionally small: only the symbols an actual Studio
//! scenario currently needs are recognised. Adding a symbol is a one-line,
//! reviewable change; adding a general parser (compound exponents, prefixes)
//! is future work and should not be assumed done just because this module
//! exists.

use scirust_units::Dimension;

/// A recognised unit symbol's dimension and its SI conversion factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitEntry {
    /// The dimension the symbol denotes.
    pub dimension: Dimension,
    /// Multiply a value expressed in this unit by this factor to get the
    /// SI-coherent value `scirust_units::Quantity` expects.
    pub to_si_factor: f64,
}

/// Look up a unit symbol, e.g. `"m/s"`. Returns `None` for anything not in
/// the table, including the empty string — use `"1"` for a dimensionless
/// quantity.
pub fn lookup(symbol: &str) -> Option<UnitEntry> {
    let dimension = match symbol
    {
        "1" => Dimension::DIMENSIONLESS,
        "m" => Dimension::LENGTH,
        "kg" => Dimension::MASS,
        "s" => Dimension::TIME,
        "A" => Dimension::CURRENT,
        "K" => Dimension::TEMPERATURE,
        "mol" => Dimension::AMOUNT,
        "cd" => Dimension::LUMINOUS,
        "m/s" => Dimension::VELOCITY,
        "m/s^2" => Dimension::ACCELERATION,
        "N" => Dimension::FORCE,
        "J" => Dimension::ENERGY,
        "W" => Dimension::POWER,
        "Pa" => Dimension::PRESSURE,
        "Hz" => Dimension::FREQUENCY,
        "C" => Dimension::CHARGE,
        "V" => Dimension::VOLTAGE,
        "Ohm" => Dimension::RESISTANCE,
        "kg/s" => Dimension::MASS.div(Dimension::TIME),
        "kg/s^2" => Dimension::MASS.div(Dimension::TIME.powi(2)),
        "kg/m^3" => Dimension::MASS.div(Dimension::LENGTH.powi(3)),
        // Rate constants (SIR beta/gamma, Robertson k1/k2/k3 in their
        // dimensionless-fraction formulation): dimensionally identical to
        // Hz, but "1/s" reads naturally for a rate constant that is not an
        // oscillation frequency.
        "1/s" => Dimension::FREQUENCY,
        // Gravitational parameter mu = G*M (orbital two-body problem).
        "m^3/s^2" => Dimension::LENGTH.powi(3).div(Dimension::TIME.powi(2)),
        // Henry (inductance): H = J/A^2 (from E = 1/2 L I^2).
        "H" => Dimension::ENERGY.div(Dimension::CURRENT.powi(2)),
        // Farad (capacitance): F = C/V.
        "F" => Dimension::CHARGE.div(Dimension::VOLTAGE),
        _ => return None,
    };
    Some(UnitEntry {
        dimension,
        to_si_factor: 1.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_the_symbols_the_spring_mass_damper_example_uses() {
        assert_eq!(lookup("kg").unwrap().dimension, Dimension::MASS);
        assert_eq!(lookup("m").unwrap().dimension, Dimension::LENGTH);
        assert_eq!(lookup("s").unwrap().dimension, Dimension::TIME);
        assert_eq!(lookup("m/s").unwrap().dimension, Dimension::VELOCITY);
        assert_eq!(
            lookup("kg/s").unwrap().dimension,
            Dimension::MASS.div(Dimension::TIME)
        );
        assert_eq!(
            lookup("kg/s^2").unwrap().dimension,
            Dimension::MASS.div(Dimension::TIME.powi(2))
        );
    }

    #[test]
    fn every_supported_unit_has_a_unit_conversion_factor() {
        for symbol in ["1", "m", "kg", "s", "A", "K", "mol", "cd", "m/s", "N"]
        {
            assert_eq!(lookup(symbol).unwrap().to_si_factor, 1.0, "symbol {symbol}");
        }
    }

    #[test]
    fn unknown_symbol_is_rejected_not_guessed() {
        assert_eq!(lookup(""), None);
        assert_eq!(lookup("m/s^3"), None);
        assert_eq!(lookup("furlong"), None);
    }

    #[test]
    fn recognises_the_phase_2a_adapter_units() {
        assert_eq!(lookup("1/s").unwrap().dimension, Dimension::FREQUENCY);
        assert_eq!(
            lookup("m^3/s^2").unwrap().dimension,
            Dimension::LENGTH.powi(3).div(Dimension::TIME.powi(2))
        );
        assert_eq!(
            lookup("H").unwrap().dimension,
            Dimension::ENERGY.div(Dimension::CURRENT.powi(2))
        );
        assert_eq!(
            lookup("F").unwrap().dimension,
            Dimension::CHARGE.div(Dimension::VOLTAGE)
        );
    }
}
