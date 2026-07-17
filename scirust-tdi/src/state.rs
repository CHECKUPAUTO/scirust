use core::fmt;

/// Configuration booléenne finie contenant au maximum 64 constituants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct State {
    bits: u64,
    width: u8,
}

/// Erreurs de construction ou d'accès à un état.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    WidthOutOfRange { width: u8 },
    BitsOutsideWidth { bits: u64, width: u8 },
    NodeOutOfRange { node: u8, width: u8 },
}

impl State {
    /// Construit un état booléen de largeur comprise entre 1 et 64.
    pub fn new(bits: u64, width: u8) -> Result<Self, StateError> {
        if width == 0 || width > 64
        {
            return Err(StateError::WidthOutOfRange { width });
        }

        if width < 64
        {
            let allowed_mask = (1_u64 << width) - 1;
            if bits & !allowed_mask != 0
            {
                return Err(StateError::BitsOutsideWidth { bits, width });
            }
        }

        Ok(Self { bits, width })
    }

    /// Construit l'état nul.
    pub fn zero(width: u8) -> Result<Self, StateError> {
        Self::new(0, width)
    }

    /// Retourne la représentation binaire compacte.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// Retourne le nombre de constituants.
    #[must_use]
    pub const fn width(self) -> u8 {
        self.width
    }

    /// Lit la valeur d'un constituant.
    pub fn get(self, node: u8) -> Result<bool, StateError> {
        self.check_node(node)?;
        Ok((self.bits & (1_u64 << node)) != 0)
    }

    /// Retourne un nouvel état dont un constituant prend la valeur demandée.
    pub fn with_bit(self, node: u8, value: bool) -> Result<Self, StateError> {
        self.check_node(node)?;

        let mask = 1_u64 << node;
        let bits = if value
        {
            self.bits | mask
        }
        else
        {
            self.bits & !mask
        };

        Ok(Self {
            bits,
            width: self.width,
        })
    }

    /// Retourne un nouvel état après inversion d'un constituant.
    pub fn flip(self, node: u8) -> Result<Self, StateError> {
        self.check_node(node)?;

        Ok(Self {
            bits: self.bits ^ (1_u64 << node),
            width: self.width,
        })
    }

    /// Calcule la distance de Hamming entre deux états compatibles.
    pub fn hamming_distance(self, other: Self) -> Result<u32, StateError> {
        if self.width != other.width
        {
            return Err(StateError::WidthOutOfRange { width: other.width });
        }

        Ok((self.bits ^ other.bits).count_ones())
    }

    fn check_node(self, node: u8) -> Result<(), StateError> {
        if node >= self.width
        {
            return Err(StateError::NodeOutOfRange {
                node,
                width: self.width,
            });
        }

        Ok(())
    }
}

impl fmt::Display for State {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:0width$b}",
            self.bits,
            width = usize::from(self.width)
        )
    }
}

impl fmt::Display for StateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::WidthOutOfRange { width } =>
            {
                write!(formatter, "state width must be in 1..=64, got {width}")
            },
            Self::BitsOutsideWidth { bits, width } =>
            {
                write!(
                    formatter,
                    "bits {bits:#x} exceed the declared state width {width}"
                )
            },
            Self::NodeOutOfRange { node, width } =>
            {
                write!(
                    formatter,
                    "node index {node} is outside the state width {width}"
                )
            },
        }
    }
}

impl std::error::Error for StateError {}

#[cfg(test)]
mod tests {
    use super::{State, StateError};

    #[test]
    fn rejects_invalid_widths() {
        assert_eq!(
            State::new(0, 0),
            Err(StateError::WidthOutOfRange { width: 0 })
        );
        assert_eq!(
            State::new(0, 65),
            Err(StateError::WidthOutOfRange { width: 65 })
        );
    }

    #[test]
    fn rejects_bits_outside_declared_width() {
        assert_eq!(
            State::new(0b1_0000, 4),
            Err(StateError::BitsOutsideWidth {
                bits: 0b1_0000,
                width: 4
            })
        );
    }

    #[test]
    fn reads_sets_and_flips_bits() {
        let state = State::new(0b0101, 4).expect("valid state");

        assert_eq!(state.get(0), Ok(true));
        assert_eq!(state.get(1), Ok(false));
        assert_eq!(state.with_bit(1, true).map(State::bits), Ok(0b0111));
        assert_eq!(state.flip(2).map(State::bits), Ok(0b0001));
    }

    #[test]
    fn computes_hamming_distance() {
        let left = State::new(0b0101, 4).expect("valid state");
        let right = State::new(0b1110, 4).expect("valid state");

        assert_eq!(left.hamming_distance(right), Ok(3));
    }

    #[test]
    fn formats_with_leading_zeroes() {
        let state = State::new(0b0011, 4).expect("valid state");
        assert_eq!(state.to_string(), "0011");
    }
}
