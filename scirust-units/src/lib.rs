//! `scirust-units` — runtime dimensional analysis over the 7 SI base dimensions.
//!
//! This crate provides two core types:
//!
//! * [`Dimension`] — a `Copy` bag of seven integer exponents, one per SI base
//!   dimension (length, mass, time, electric current, thermodynamic temperature,
//!   amount of substance, luminous intensity). Dimensions multiply by *adding*
//!   exponents and divide by *subtracting* them.
//! * [`Quantity`] — a `f64` magnitude (expressed in coherent SI units) tagged
//!   with a [`Dimension`]. Arithmetic that would mix incompatible dimensions is
//!   rejected through a checked, `Result`-returning API instead of panicking.
//!
//! Dimensional mistakes (adding a length to a time, taking the square root of an
//! odd-exponent dimension, dividing by a zero magnitude) surface as
//! [`UnitsError`] values rather than silent numerical nonsense.
//!
//! # Example
//!
//! ```
//! use scirust_units::{Dimension, Quantity};
//!
//! // Newton's second law: F = m * a.
//! let mass = Quantity::kilograms(2.0);
//! let accel = Quantity::new(9.81, Dimension::ACCELERATION);
//! let force = mass.mul(&accel);
//! assert_eq!(force.dim, Dimension::FORCE);
//! assert!((force.value - 19.62).abs() < 1e-9);
//! assert_eq!(force.dim.to_string(), "m·kg·s^-2");
//!
//! // Energy is force times distance, and has the dimension of the joule.
//! let work = force.mul(&Quantity::meters(3.0));
//! assert_eq!(work.dim, Dimension::ENERGY);
//!
//! // Mixing incompatible dimensions is a checked error, never a panic.
//! let length = Quantity::meters(3.0);
//! let time = Quantity::seconds(1.0);
//! assert!(length.try_add(&time).is_err());
//! assert!(length.try_add(&Quantity::meters(2.0)).is_ok());
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Number of SI base dimensions tracked by a [`Dimension`].
const N: usize = 7;

/// Short unit symbols for each base dimension, in canonical index order:
/// length, mass, time, current, temperature, amount, luminous intensity.
const SYMBOLS: [&str; N] = ["m", "kg", "s", "A", "K", "mol", "cd"];

/// A physical dimension expressed as the seven SI base-dimension exponents.
///
/// The exponents are, in order: length, mass, time, electric current,
/// thermodynamic temperature, amount of substance and luminous intensity.
/// For example `m·kg·s^-2` (a force) is `[1, 1, -2, 0, 0, 0, 0]`.
///
/// Multiplication adds exponents and division subtracts them, mirroring the
/// algebra of physical units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimension {
    /// Exponents for each base dimension, indexed as in [`SYMBOLS`].
    exponents: [i8; N],
}

impl Dimension {
    /// Builds a dimension directly from its seven base-dimension exponents,
    /// ordered length, mass, time, current, temperature, amount, luminous.
    #[must_use]
    pub const fn from_exponents(exponents: [i8; N]) -> Self {
        Self { exponents }
    }

    /// The dimensionless dimension (all exponents zero); prints as `1`.
    pub const DIMENSIONLESS: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 0]);
    /// Length (metre, `m`).
    pub const LENGTH: Dimension = Dimension::from_exponents([1, 0, 0, 0, 0, 0, 0]);
    /// Mass (kilogram, `kg`).
    pub const MASS: Dimension = Dimension::from_exponents([0, 1, 0, 0, 0, 0, 0]);
    /// Time (second, `s`).
    pub const TIME: Dimension = Dimension::from_exponents([0, 0, 1, 0, 0, 0, 0]);
    /// Electric current (ampere, `A`).
    pub const CURRENT: Dimension = Dimension::from_exponents([0, 0, 0, 1, 0, 0, 0]);
    /// Thermodynamic temperature (kelvin, `K`).
    pub const TEMPERATURE: Dimension = Dimension::from_exponents([0, 0, 0, 0, 1, 0, 0]);
    /// Amount of substance (mole, `mol`).
    pub const AMOUNT: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 1, 0]);
    /// Luminous intensity (candela, `cd`).
    pub const LUMINOUS: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 1]);

    /// Area, `length^2` (`m^2`).
    pub const AREA: Dimension = Dimension::LENGTH.powi(2);
    /// Volume, `length^3` (`m^3`).
    pub const VOLUME: Dimension = Dimension::LENGTH.powi(3);
    /// Velocity, `length / time` (`m·s^-1`).
    pub const VELOCITY: Dimension = Dimension::from_exponents([1, 0, -1, 0, 0, 0, 0]);
    /// Acceleration, `length / time^2` (`m·s^-2`).
    pub const ACCELERATION: Dimension = Dimension::from_exponents([1, 0, -2, 0, 0, 0, 0]);
    /// Force (newton), `mass·length / time^2` (`m·kg·s^-2`).
    pub const FORCE: Dimension = Dimension::from_exponents([1, 1, -2, 0, 0, 0, 0]);
    /// Energy (joule), `force·length` (`m^2·kg·s^-2`).
    pub const ENERGY: Dimension = Dimension::from_exponents([2, 1, -2, 0, 0, 0, 0]);
    /// Power (watt), `energy / time` (`m^2·kg·s^-3`).
    pub const POWER: Dimension = Dimension::from_exponents([2, 1, -3, 0, 0, 0, 0]);
    /// Pressure (pascal), `force / area` (`m^-1·kg·s^-2`).
    pub const PRESSURE: Dimension = Dimension::from_exponents([-1, 1, -2, 0, 0, 0, 0]);
    /// Frequency (hertz), `1 / time` (`s^-1`).
    pub const FREQUENCY: Dimension = Dimension::from_exponents([0, 0, -1, 0, 0, 0, 0]);
    /// Electric charge (coulomb), `current·time` (`s·A`).
    pub const CHARGE: Dimension = Dimension::from_exponents([0, 0, 1, 1, 0, 0, 0]);
    /// Electric potential (volt), `power / current` (`m^2·kg·s^-3·A^-1`).
    pub const VOLTAGE: Dimension = Dimension::from_exponents([2, 1, -3, -1, 0, 0, 0]);
    /// Electric resistance (ohm), `voltage / current` (`m^2·kg·s^-3·A^-2`).
    pub const RESISTANCE: Dimension = Dimension::from_exponents([2, 1, -3, -2, 0, 0, 0]);

    /// Returns the dimensionless dimension (all exponents zero).
    #[must_use]
    pub const fn dimensionless() -> Dimension {
        Dimension::DIMENSIONLESS
    }

    /// Returns the length base dimension.
    #[must_use]
    pub const fn length() -> Dimension {
        Dimension::LENGTH
    }

    /// Returns the mass base dimension.
    #[must_use]
    pub const fn mass() -> Dimension {
        Dimension::MASS
    }

    /// Returns the time base dimension.
    #[must_use]
    pub const fn time() -> Dimension {
        Dimension::TIME
    }

    /// Returns the electric-current base dimension.
    #[must_use]
    pub const fn current() -> Dimension {
        Dimension::CURRENT
    }

    /// Returns the thermodynamic-temperature base dimension.
    #[must_use]
    pub const fn temperature() -> Dimension {
        Dimension::TEMPERATURE
    }

    /// Returns the amount-of-substance base dimension.
    #[must_use]
    pub const fn amount() -> Dimension {
        Dimension::AMOUNT
    }

    /// Returns the luminous-intensity base dimension.
    #[must_use]
    pub const fn luminous() -> Dimension {
        Dimension::LUMINOUS
    }

    /// Returns the raw exponent array in canonical order (length, mass, time,
    /// current, temperature, amount, luminous).
    #[must_use]
    pub const fn exponents(&self) -> [i8; N] {
        self.exponents
    }

    /// Multiplies two dimensions by adding their exponents component-wise.
    #[must_use]
    pub const fn mul(self, other: Dimension) -> Dimension {
        let mut e = self.exponents;
        let mut i = 0;
        while i < N
        {
            e[i] += other.exponents[i];
            i += 1;
        }
        Dimension::from_exponents(e)
    }

    /// Divides two dimensions by subtracting the divisor's exponents.
    #[must_use]
    pub const fn div(self, other: Dimension) -> Dimension {
        let mut e = self.exponents;
        let mut i = 0;
        while i < N
        {
            e[i] -= other.exponents[i];
            i += 1;
        }
        Dimension::from_exponents(e)
    }

    /// Raises the dimension to an integer power by scaling every exponent by `n`.
    #[must_use]
    pub const fn powi(self, n: i32) -> Dimension {
        let mut e = self.exponents;
        let mut i = 0;
        while i < N
        {
            e[i] = (e[i] as i32 * n) as i8;
            i += 1;
        }
        Dimension::from_exponents(e)
    }

    /// Attempts the dimensional square root, halving every exponent.
    ///
    /// # Errors
    ///
    /// Returns [`UnitsError::NonSquareDimension`] if any exponent is odd, since
    /// its half would not be an integer.
    pub fn try_sqrt(self) -> Result<Dimension, UnitsError> {
        let mut e = self.exponents;
        for x in &mut e
        {
            if *x % 2 != 0
            {
                return Err(UnitsError::NonSquareDimension { dim: self });
            }
            *x /= 2;
        }
        Ok(Dimension::from_exponents(e))
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut wrote = false;
        for (i, &e) in self.exponents.iter().enumerate()
        {
            if e == 0
            {
                continue;
            }
            if wrote
            {
                write!(f, "·")?;
            }
            wrote = true;
            if e == 1
            {
                write!(f, "{}", SYMBOLS[i])?;
            }
            else
            {
                write!(f, "{}^{}", SYMBOLS[i], e)?;
            }
        }
        if !wrote
        {
            write!(f, "1")?;
        }
        Ok(())
    }
}

/// Errors produced by the dimension-checked API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitsError {
    /// Two quantities with different dimensions were combined by an operation
    /// (such as addition) that requires them to match.
    IncompatibleDimensions {
        /// Dimension of the left-hand operand.
        left: Dimension,
        /// Dimension of the right-hand operand.
        right: Dimension,
    },
    /// A division was attempted with a divisor whose numeric value is zero.
    DivisionByZero,
    /// A square root was requested for a dimension with an odd exponent, which
    /// has no integer-exponent root.
    NonSquareDimension {
        /// The offending dimension.
        dim: Dimension,
    },
}

impl fmt::Display for UnitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            UnitsError::IncompatibleDimensions { left, right } =>
            {
                write!(f, "incompatible dimensions: {left} vs {right}")
            },
            UnitsError::DivisionByZero => write!(f, "division by zero magnitude"),
            UnitsError::NonSquareDimension { dim } =>
            {
                write!(
                    f,
                    "cannot take the square root of non-square dimension {dim}"
                )
            },
        }
    }
}

impl std::error::Error for UnitsError {}

/// A physical quantity: a numeric magnitude in coherent SI units tagged with a
/// [`Dimension`].
///
/// The magnitude is always stored in the coherent SI unit for its dimension
/// (metres, kilograms, seconds, newtons, joules, …), so quantities can be
/// combined without unit-prefix bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// Numeric magnitude, expressed in the coherent SI unit for [`dim`](Self::dim).
    pub value: f64,
    /// Physical dimension of this quantity.
    pub dim: Dimension,
}

impl Quantity {
    /// Creates a quantity from a magnitude and a dimension.
    #[must_use]
    pub const fn new(value: f64, dim: Dimension) -> Quantity {
        Quantity { value, dim }
    }

    /// Returns `true` if the two quantities share the same dimension.
    #[must_use]
    pub fn is_compatible(&self, other: &Quantity) -> bool {
        self.dim == other.dim
    }

    /// Adds two quantities of identical dimension.
    ///
    /// # Errors
    ///
    /// Returns [`UnitsError::IncompatibleDimensions`] if the dimensions differ.
    pub fn try_add(&self, other: &Quantity) -> Result<Quantity, UnitsError> {
        if self.dim != other.dim
        {
            return Err(UnitsError::IncompatibleDimensions {
                left: self.dim,
                right: other.dim,
            });
        }
        Ok(Quantity::new(self.value + other.value, self.dim))
    }

    /// Subtracts a quantity of identical dimension from this one.
    ///
    /// # Errors
    ///
    /// Returns [`UnitsError::IncompatibleDimensions`] if the dimensions differ.
    pub fn try_sub(&self, other: &Quantity) -> Result<Quantity, UnitsError> {
        if self.dim != other.dim
        {
            return Err(UnitsError::IncompatibleDimensions {
                left: self.dim,
                right: other.dim,
            });
        }
        Ok(Quantity::new(self.value - other.value, self.dim))
    }

    /// Multiplies two quantities: magnitudes multiply and dimensions combine.
    #[must_use]
    pub fn mul(&self, other: &Quantity) -> Quantity {
        Quantity::new(self.value * other.value, self.dim.mul(other.dim))
    }

    /// Divides this quantity by another: magnitudes divide and dimensions
    /// subtract.
    ///
    /// # Errors
    ///
    /// Returns [`UnitsError::DivisionByZero`] if the divisor's magnitude is
    /// zero.
    pub fn div(&self, other: &Quantity) -> Result<Quantity, UnitsError> {
        if other.value == 0.0
        {
            return Err(UnitsError::DivisionByZero);
        }
        Ok(Quantity::new(
            self.value / other.value,
            self.dim.div(other.dim),
        ))
    }

    /// Scales the magnitude by a dimensionless factor, leaving the dimension
    /// unchanged.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Quantity {
        Quantity::new(self.value * factor, self.dim)
    }

    /// Raises the quantity to an integer power: the magnitude uses
    /// [`f64::powi`] and the dimension's exponents scale by `n`.
    #[must_use]
    pub fn powi(&self, n: i32) -> Quantity {
        Quantity::new(self.value.powi(n), self.dim.powi(n))
    }

    /// Attempts the square root of the quantity.
    ///
    /// # Errors
    ///
    /// Returns [`UnitsError::NonSquareDimension`] if the dimension has any odd
    /// exponent (see [`Dimension::try_sqrt`]).
    pub fn try_sqrt(&self) -> Result<Quantity, UnitsError> {
        let dim = self.dim.try_sqrt()?;
        Ok(Quantity::new(self.value.sqrt(), dim))
    }

    /// Creates a length in metres.
    #[must_use]
    pub const fn meters(value: f64) -> Quantity {
        Quantity::new(value, Dimension::LENGTH)
    }

    /// Creates a mass in kilograms.
    #[must_use]
    pub const fn kilograms(value: f64) -> Quantity {
        Quantity::new(value, Dimension::MASS)
    }

    /// Creates a duration in seconds.
    #[must_use]
    pub const fn seconds(value: f64) -> Quantity {
        Quantity::new(value, Dimension::TIME)
    }

    /// Creates an electric current in amperes.
    #[must_use]
    pub const fn amperes(value: f64) -> Quantity {
        Quantity::new(value, Dimension::CURRENT)
    }

    /// Creates a temperature in kelvin.
    #[must_use]
    pub const fn kelvins(value: f64) -> Quantity {
        Quantity::new(value, Dimension::TEMPERATURE)
    }

    /// Creates an amount of substance in moles.
    #[must_use]
    pub const fn moles(value: f64) -> Quantity {
        Quantity::new(value, Dimension::AMOUNT)
    }

    /// Creates a luminous intensity in candela.
    #[must_use]
    pub const fn candelas(value: f64) -> Quantity {
        Quantity::new(value, Dimension::LUMINOUS)
    }

    /// Creates a force in newtons (`kg·m·s^-2`).
    #[must_use]
    pub const fn newtons(value: f64) -> Quantity {
        Quantity::new(value, Dimension::FORCE)
    }

    /// Creates an energy in joules (`kg·m^2·s^-2`).
    #[must_use]
    pub const fn joules(value: f64) -> Quantity {
        Quantity::new(value, Dimension::ENERGY)
    }

    /// Creates a power in watts (`kg·m^2·s^-3`).
    #[must_use]
    pub const fn watts(value: f64) -> Quantity {
        Quantity::new(value, Dimension::POWER)
    }

    /// Creates a pressure in pascals (`kg·m^-1·s^-2`).
    #[must_use]
    pub const fn pascals(value: f64) -> Quantity {
        Quantity::new(value, Dimension::PRESSURE)
    }

    /// Creates a frequency in hertz (`s^-1`).
    #[must_use]
    pub const fn hertz(value: f64) -> Quantity {
        Quantity::new(value, Dimension::FREQUENCY)
    }

    /// Creates an electric charge in coulombs (`s·A`).
    #[must_use]
    pub const fn coulombs(value: f64) -> Quantity {
        Quantity::new(value, Dimension::CHARGE)
    }

    /// Creates an electric potential in volts (`kg·m^2·s^-3·A^-1`).
    #[must_use]
    pub const fn volts(value: f64) -> Quantity {
        Quantity::new(value, Dimension::VOLTAGE)
    }

    /// Creates an electric resistance in ohms (`kg·m^2·s^-3·A^-2`).
    #[must_use]
    pub const fn ohms(value: f64) -> Quantity {
        Quantity::new(value, Dimension::RESISTANCE)
    }
}

/// Ergonomic-but-unchecked addition.
///
/// This operator **panics** if the operands have different dimensions. It
/// exists purely for concise expressions where the dimensions are known to
/// match; prefer [`Quantity::try_add`] for the checked, non-panicking path.
/// Library code in this crate never relies on this operator.
impl std::ops::Add for Quantity {
    type Output = Quantity;

    fn add(self, rhs: Quantity) -> Quantity {
        match self.try_add(&rhs)
        {
            Ok(q) => q,
            Err(e) => panic!("Quantity::add: {e}"),
        }
    }
}

/// Ergonomic-but-unchecked subtraction.
///
/// This operator **panics** if the operands have different dimensions. Prefer
/// [`Quantity::try_sub`] for the checked, non-panicking path. Library code in
/// this crate never relies on this operator.
impl std::ops::Sub for Quantity {
    type Output = Quantity;

    fn sub(self, rhs: Quantity) -> Quantity {
        match self.try_sub(&rhs)
        {
            Ok(q) => q,
            Err(e) => panic!("Quantity::sub: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn dimensionless_prints_one() {
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "1");
    }

    #[test]
    fn base_dimension_display() {
        assert_eq!(Dimension::LENGTH.to_string(), "m");
        assert_eq!(Dimension::MASS.to_string(), "kg");
        assert_eq!(Dimension::TIME.to_string(), "s");
    }

    #[test]
    fn derived_dimension_display() {
        assert_eq!(Dimension::FORCE.to_string(), "m·kg·s^-2");
        assert_eq!(Dimension::ENERGY.to_string(), "m^2·kg·s^-2");
        assert_eq!(Dimension::POWER.to_string(), "m^2·kg·s^-3");
        assert_eq!(Dimension::FREQUENCY.to_string(), "s^-1");
    }

    #[test]
    fn newtons_second_law_dimension_and_value() {
        // F = m * a, with a expressed as length / time^2.
        let mass = Quantity::kilograms(3.0);
        let accel = Quantity::meters(4.0)
            .div(&Quantity::seconds(2.0).powi(2))
            .expect("non-zero divisor");
        let force = mass.mul(&accel);

        assert_eq!(force.dim, Dimension::FORCE);
        assert_eq!(force.dim, Quantity::newtons(1.0).dim);
        // 3 kg * (4 / 4) m/s^2 = 3 N
        assert!((force.value - 3.0).abs() < EPS);
    }

    #[test]
    fn energy_and_power_dimensions() {
        let force = Quantity::newtons(10.0);
        let distance = Quantity::meters(2.0);
        let energy = force.mul(&distance);
        assert_eq!(energy.dim, Dimension::ENERGY);
        assert_eq!(energy.dim, Quantity::joules(1.0).dim);
        assert!((energy.value - 20.0).abs() < EPS);

        let time = Quantity::seconds(4.0);
        let power = energy.div(&time).expect("non-zero divisor");
        assert_eq!(power.dim, Dimension::POWER);
        assert_eq!(power.dim, Quantity::watts(1.0).dim);
        assert!((power.value - 5.0).abs() < EPS);
    }

    #[test]
    fn pressure_is_force_over_area() {
        let force = Quantity::newtons(50.0);
        let area = Quantity::meters(5.0).powi(2);
        let pressure = force.div(&area).expect("non-zero divisor");
        assert_eq!(pressure.dim, Dimension::PRESSURE);
        assert_eq!(pressure.dim, Quantity::pascals(1.0).dim);
        assert!((pressure.value - 2.0).abs() < EPS);
    }

    #[test]
    fn electrical_dimensions() {
        let charge = Quantity::amperes(2.0).mul(&Quantity::seconds(3.0));
        assert_eq!(charge.dim, Dimension::CHARGE);
        assert_eq!(charge.dim, Quantity::coulombs(1.0).dim);

        let voltage = Quantity::watts(12.0)
            .div(&Quantity::amperes(3.0))
            .expect("non-zero divisor");
        assert_eq!(voltage.dim, Dimension::VOLTAGE);
        assert_eq!(voltage.dim, Quantity::volts(1.0).dim);

        let resistance = voltage
            .div(&Quantity::amperes(2.0))
            .expect("non-zero divisor");
        assert_eq!(resistance.dim, Dimension::RESISTANCE);
        assert_eq!(resistance.dim, Quantity::ohms(1.0).dim);
    }

    #[test]
    fn try_add_requires_matching_dimensions() {
        let length = Quantity::meters(3.0);
        let time = Quantity::seconds(2.0);
        match length.try_add(&time)
        {
            Err(UnitsError::IncompatibleDimensions { left, right }) =>
            {
                assert_eq!(left, Dimension::LENGTH);
                assert_eq!(right, Dimension::TIME);
            },
            other => panic!("expected IncompatibleDimensions, got {other:?}"),
        }

        let sum = length
            .try_add(&Quantity::meters(4.0))
            .expect("same dimension adds");
        assert_eq!(sum.dim, Dimension::LENGTH);
        assert!((sum.value - 7.0).abs() < EPS);
    }

    #[test]
    fn try_sub_requires_matching_dimensions() {
        let a = Quantity::joules(10.0);
        let b = Quantity::joules(4.0);
        let diff = a.try_sub(&b).expect("same dimension subtracts");
        assert!((diff.value - 6.0).abs() < EPS);
        assert_eq!(diff.dim, Dimension::ENERGY);

        assert!(a.try_sub(&Quantity::watts(1.0)).is_err());
    }

    #[test]
    fn is_compatible_reflects_dimension_equality() {
        assert!(Quantity::meters(1.0).is_compatible(&Quantity::meters(9.0)));
        assert!(!Quantity::meters(1.0).is_compatible(&Quantity::seconds(1.0)));
    }

    #[test]
    fn division_by_zero_value_is_rejected() {
        let numerator = Quantity::meters(5.0);
        let zero = Quantity::seconds(0.0);
        assert_eq!(numerator.div(&zero), Err(UnitsError::DivisionByZero));
    }

    #[test]
    fn powi_and_try_sqrt_roundtrip() {
        let length = Quantity::meters(3.0);
        let area = length.powi(2);
        assert_eq!(area.dim, Dimension::AREA);
        assert!((area.value - 9.0).abs() < EPS);

        let recovered = area
            .try_sqrt()
            .expect("area is a perfect dimensional square");
        assert_eq!(recovered.dim, Dimension::LENGTH);
        assert!((recovered.value - 3.0).abs() < EPS);
    }

    #[test]
    fn try_sqrt_rejects_odd_exponents() {
        // Volume has an odd exponent (m^3) so it has no dimensional square root.
        match Dimension::VOLUME.try_sqrt()
        {
            Err(UnitsError::NonSquareDimension { dim }) =>
            {
                assert_eq!(dim, Dimension::VOLUME);
            },
            other => panic!("expected NonSquareDimension, got {other:?}"),
        }
        assert!(Quantity::meters(8.0).try_sqrt().is_err());
    }

    #[test]
    fn dimensionless_ratio_equals_dimensionless() {
        let ratio = Quantity::meters(6.0)
            .div(&Quantity::meters(2.0))
            .expect("non-zero divisor");
        assert_eq!(ratio.dim, Dimension::DIMENSIONLESS);
        assert!((ratio.value - 3.0).abs() < EPS);
        assert_eq!(ratio.dim.to_string(), "1");
    }

    #[test]
    fn dimension_mul_div_powi_algebra() {
        assert_eq!(
            Dimension::LENGTH.mul(Dimension::TIME.powi(-1)),
            Dimension::VELOCITY
        );
        assert_eq!(
            Dimension::VELOCITY.div(Dimension::TIME),
            Dimension::ACCELERATION
        );
        assert_eq!(Dimension::FORCE.mul(Dimension::LENGTH), Dimension::ENERGY);
        assert_eq!(Dimension::ENERGY.div(Dimension::TIME), Dimension::POWER);
    }

    #[test]
    fn scale_keeps_dimension() {
        let f = Quantity::newtons(2.0).scale(2.5);
        assert_eq!(f.dim, Dimension::FORCE);
        assert!((f.value - 5.0).abs() < EPS);
    }

    #[test]
    fn add_operator_matches_try_add() {
        let total = Quantity::meters(1.0) + Quantity::meters(2.0);
        assert_eq!(total.dim, Dimension::LENGTH);
        assert!((total.value - 3.0).abs() < EPS);

        let diff = Quantity::joules(5.0) - Quantity::joules(2.0);
        assert!((diff.value - 3.0).abs() < EPS);
    }

    #[test]
    #[should_panic(expected = "incompatible dimensions")]
    fn add_operator_panics_on_mismatch() {
        let _ = Quantity::meters(1.0) + Quantity::seconds(1.0);
    }

    #[test]
    fn error_display_is_human_readable() {
        let err = UnitsError::IncompatibleDimensions {
            left: Dimension::LENGTH,
            right: Dimension::TIME,
        };
        assert_eq!(err.to_string(), "incompatible dimensions: m vs s");
        assert_eq!(
            UnitsError::DivisionByZero.to_string(),
            "division by zero magnitude"
        );
    }
}
