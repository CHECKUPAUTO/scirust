use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};

use crate::ReachabilityReport;

/// Rapport rationnel exact, sans conversion flottante.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactRatio {
    numerator: BigUint,
    denominator: BigUint,
}

/// Erreurs de construction d'une signature TDI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureError {
    EmptyReport,
    MissingLayer { depth: usize },
    ZeroPathCount { depth: usize },
}

/// Signature prospective minimale de TDI-1.
///
/// Elle conserve séparément :
/// - le nombre d'états accessibles à chaque profondeur ;
/// - le nombre de chemins admissibles ;
/// - la proportion exacte des chemins revenant à l'état initial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TdiSignature {
    reachable_profile: Vec<usize>,
    path_profile: Vec<u128>,
    return_profile: Vec<ExactRatio>,
}

impl ExactRatio {
    /// Construit et réduit une fraction exacte.
    #[must_use]
    pub fn new(numerator: u128, denominator: u128) -> Option<Self> {
        Self::from_biguint(BigUint::from(numerator), BigUint::from(denominator))
    }

    fn from_biguint(numerator: BigUint, denominator: BigUint) -> Option<Self> {
        if denominator.is_zero()
        {
            return None;
        }

        let divisor = numerator.gcd(&denominator);

        Some(Self {
            numerator: numerator / &divisor,
            denominator: denominator / divisor,
        })
    }

    #[must_use]
    pub const fn numerator(&self) -> &BigUint {
        &self.numerator
    }

    #[must_use]
    pub const fn denominator(&self) -> &BigUint {
        &self.denominator
    }

    /// Extrait les composantes sous forme `u128` lorsqu'elles tiennent
    /// toutes les deux dans cette représentation.
    #[must_use]
    pub fn components_u128(&self) -> Option<(u128, u128)> {
        Some((self.numerator.to_u128()?, self.denominator.to_u128()?))
    }

    /// Conversion destinée uniquement à l'affichage et aux modèles statistiques.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        if self.numerator.is_zero()
        {
            return 0.0;
        }

        if let (Some(numerator), Some(denominator)) =
            (self.numerator.to_f64(), self.denominator.to_f64())
        {
            if numerator.is_finite() && denominator.is_finite()
            {
                return numerator / denominator;
            }
        }

        let numerator_bits = self.numerator.bits();
        let denominator_bits = self.denominator.bits();

        let numerator_shift = numerator_bits.saturating_sub(53) as usize;

        let denominator_shift = denominator_bits.saturating_sub(53) as usize;

        let numerator_top = (&self.numerator >> numerator_shift)
            .to_u64()
            .expect("top 53 numerator bits fit in u64");

        let denominator_top = (&self.denominator >> denominator_shift)
            .to_u64()
            .expect("top 53 denominator bits fit in u64");

        let exponent_difference = numerator_shift as i64 - denominator_shift as i64;

        if exponent_difference < i64::from(i32::MIN)
        {
            return 0.0;
        }

        if exponent_difference > i64::from(i32::MAX)
        {
            return f64::INFINITY;
        }

        (numerator_top as f64 / denominator_top as f64) * 2.0_f64.powi(exponent_difference as i32)
    }

    /// Addition rationnelle exacte sans limite de taille fixe.
    #[must_use]
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        let divisor = self.denominator.gcd(&other.denominator);

        let left_factor = &other.denominator / &divisor;

        let right_factor = &self.denominator / &divisor;

        let numerator = &self.numerator * &left_factor + &other.numerator * &right_factor;

        let denominator = &self.denominator * left_factor;

        Self::from_biguint(numerator, denominator)
    }

    /// Division exacte par un entier strictement positif.
    #[must_use]
    pub fn checked_div_u128(&self, divisor: u128) -> Option<Self> {
        if divisor == 0
        {
            return None;
        }

        let divisor = BigUint::from(divisor);

        let cancellation = self.numerator.gcd(&divisor);

        let numerator = &self.numerator / &cancellation;

        let remaining_divisor = divisor / cancellation;

        let denominator = &self.denominator * remaining_divisor;

        Self::from_biguint(numerator, denominator)
    }

    /// Comparaison rationnelle exacte par produits arbitrairement grands.
    #[must_use]
    pub fn checked_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let left = &self.numerator * &other.denominator;

        let right = &other.numerator * &self.denominator;

        Some(left.cmp(&right))
    }
}

impl TdiSignature {
    /// Extrait une signature exacte depuis un rapport d'exploration.
    pub fn from_report(report: &ReachabilityReport) -> Result<Self, SignatureError> {
        if report.horizon() == 0
        {
            return Err(SignatureError::EmptyReport);
        }

        let mut reachable_profile = Vec::with_capacity(report.horizon());
        let mut path_profile = Vec::with_capacity(report.horizon());
        let mut return_profile = Vec::with_capacity(report.horizon());

        for depth in 1..=report.horizon()
        {
            let reachable = report
                .reachable_count(depth)
                .ok_or(SignatureError::MissingLayer { depth })?;

            let total_paths = report
                .path_count(depth)
                .ok_or(SignatureError::MissingLayer { depth })?;

            if total_paths == 0
            {
                return Err(SignatureError::ZeroPathCount { depth });
            }

            let returned_paths = report
                .return_path_count(depth)
                .ok_or(SignatureError::MissingLayer { depth })?;

            let return_ratio = ExactRatio::new(returned_paths, total_paths)
                .ok_or(SignatureError::ZeroPathCount { depth })?;

            reachable_profile.push(reachable);
            path_profile.push(total_paths);
            return_profile.push(return_ratio);
        }

        Ok(Self {
            reachable_profile,
            path_profile,
            return_profile,
        })
    }

    #[must_use]
    pub fn horizon(&self) -> usize {
        self.reachable_profile.len()
    }

    #[must_use]
    pub fn reachable_profile(&self) -> &[usize] {
        &self.reachable_profile
    }

    #[must_use]
    pub fn path_profile(&self) -> &[u128] {
        &self.path_profile
    }

    #[must_use]
    pub fn return_profile(&self) -> &[ExactRatio] {
        &self.return_profile
    }
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;

    use crate::{Action, ExactRatio, SignatureError, State, TableSystem, TdiSignature, explore};

    #[test]
    fn reduces_exact_ratios() {
        let ratio = ExactRatio::new(6, 8).expect("non-zero denominator");

        assert_eq!(ratio.numerator(), &BigUint::from(3_u8));
        assert_eq!(ratio.denominator(), &BigUint::from(4_u8));
        assert_eq!(ratio.as_f64(), 0.75);
    }

    #[test]
    fn rejects_zero_denominator() {
        assert_eq!(ExactRatio::new(1, 0), None);
    }

    #[test]
    fn supports_denominators_larger_than_u128() {
        let mut ratio = ExactRatio::new(1, 1).expect("valid ratio");

        for _ in 0..100
        {
            ratio = ratio
                .checked_div_u128(3)
                .expect("arbitrary-precision division succeeds");
        }

        let expected = BigUint::from(3_u8).pow(100);

        assert_eq!(ratio.numerator(), &BigUint::from(1_u8));
        assert_eq!(ratio.denominator(), &expected);
        assert!(ratio.denominator().bits() > 128);
        assert!(ratio.as_f64().is_finite());
        assert!(ratio.as_f64() > 0.0);
    }

    #[test]
    fn compares_large_ratios_without_cross_product_overflow() {
        let maximum = u128::MAX;

        let left = ExactRatio::new(maximum - 1, maximum).expect("valid ratio");

        let right = ExactRatio::new(maximum - 2, maximum - 1).expect("valid ratio");

        assert_eq!(left.checked_cmp(&right), Some(std::cmp::Ordering::Greater));

        assert_eq!(right.checked_cmp(&left), Some(std::cmp::Ordering::Less));

        assert_eq!(left.checked_cmp(&left), Some(std::cmp::Ordering::Equal));
    }

    #[test]
    fn extracts_branching_and_return_profiles() {
        let zero = State::new(0b00, 2).expect("valid state");
        let one = State::new(0b01, 2).expect("valid state");
        let two = State::new(0b10, 2).expect("valid state");

        let mut system = TableSystem::new(2).expect("valid system");
        system
            .insert(zero, Action::Noop, vec![one, two])
            .expect("valid transition");
        system
            .insert(one, Action::Noop, vec![zero])
            .expect("valid transition");
        system
            .insert(two, Action::Noop, vec![zero])
            .expect("valid transition");

        let report =
            explore(&system, zero, &[Action::Noop, Action::Noop]).expect("exploration succeeds");

        let signature = TdiSignature::from_report(&report).expect("valid signature");

        assert_eq!(signature.horizon(), 2);
        assert_eq!(signature.reachable_profile(), &[2, 1]);
        assert_eq!(signature.path_profile(), &[2, 2]);
        assert_eq!(
            signature.return_profile(),
            &[
                ExactRatio::new(0, 2).expect("valid ratio"),
                ExactRatio::new(2, 2).expect("valid ratio")
            ]
        );
    }

    #[test]
    fn rejects_an_empty_report() {
        let state = State::new(0, 1).expect("valid state");
        let system = TableSystem::new(1).expect("valid system");

        let report = explore(&system, state, &[]).expect("zero-horizon exploration succeeds");

        assert_eq!(
            TdiSignature::from_report(&report),
            Err(SignatureError::EmptyReport)
        );
    }
}
