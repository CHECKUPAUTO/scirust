// scirust-core/src/error.rs
//
// Type d'erreur centralisé pour SciRust.
//
// Philosophie de migration :
//   - Les nouvelles API publiques renvoient `Result<T, SciRustError>`
//   - Les API internes (qui restent dans le crate) peuvent continuer à
//     utiliser des assertions/panics tant que l'erreur ne traverse pas
//     une frontière publique
//   - Les helpers `expect_shape!`, `expect_device!` permettent d'écrire
//     des checks lisibles qui produisent une erreur structurée
//
// Pour cette PR v9, on convertit les API publiques principales :
//   - Tape::try_input → Result<Var, SciRustError>
//   - Module::try_forward (méthode optionnelle, default délègue à forward)
//   - DataLoader::try_iter / batch
//   - load_idx_images / load_idx_labels (déjà des io::Result, on enrichit)
//
// Les API non-result restent disponibles (panic en cas de problème) pour
// ne pas casser l'usage existant. Migration progressive.

use std::fmt;
// use crate::tensor::device::Device;

/// Computational device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device { Cpu, Gpu }


// ================================================================== //
//  Type d'erreur principal                                            //
// ================================================================== //

#[derive(Debug)]
pub enum SciRustError {
    /// Shapes incompatibles entre opérandes
    ShapeMismatch {
        op:        &'static str,
        expected:  (usize, usize),
        got:       (usize, usize),
    },
    /// Tenseurs sur des devices différents
    DeviceMismatch {
        op:        &'static str,
        expected:  Device,
        got:       Device,
    },
    /// Tenseur sur le mauvais device pour cette op
    WrongDevice {
        op:        &'static str,
        expected:  Device,
        got:       Device,
        hint:      &'static str,
    },
    /// Configuration invalide (hyperparamètres, dimensions)
    InvalidConfig(String),
    /// GPU demandé mais indisponible
    GpuNotAvailable,
    /// Erreur GPU (compilation kernel, allocation, etc.)
    GpuError(String),
    /// I/O sur fichiers (datasets, checkpoints)
    IoError(std::io::Error),
    /// Format de fichier corrompu ou invalide
    InvalidFormat {
        what:     &'static str,
        details:  String,
    },
    /// Index hors bornes (tape, dataset, batch)
    IndexOutOfBounds {
        what:     &'static str,
        index:    usize,
        bound:    usize,
    },
}

impl fmt::Display for SciRustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SciRustError::ShapeMismatch { op, expected, got } => {
                write!(f, "shape mismatch in '{op}': expected {:?}, got {:?}",
                       expected, got)
            }
            SciRustError::DeviceMismatch { op, expected, got } => {
                write!(f, "device mismatch in '{op}': expected {:?}, got {:?} \
                          (use to_cpu/to_gpu to align)", expected, got)
            }
            SciRustError::WrongDevice { op, expected, got, hint } => {
                write!(f, "wrong device for '{op}': expected {:?}, got {:?} — {hint}",
                       expected, got)
            }
            SciRustError::InvalidConfig(msg) => {
                write!(f, "invalid configuration: {msg}")
            }
            SciRustError::GpuNotAvailable => {
                write!(f, "GPU requested but no GPU adapter available — \
                          rebuild with --features wgpu and ensure a compatible adapter is present")
            }
            SciRustError::GpuError(msg) => {
                write!(f, "GPU error: {msg}")
            }
            SciRustError::IoError(e) => {
                write!(f, "I/O error: {e}")
            }
            SciRustError::InvalidFormat { what, details } => {
                write!(f, "invalid {what} format: {details}")
            }
            SciRustError::IndexOutOfBounds { what, index, bound } => {
                write!(f, "{what} index out of bounds: {index} >= {bound}")
            }
        }
    }
}

impl std::error::Error for SciRustError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SciRustError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SciRustError {
    fn from(e: std::io::Error) -> Self { SciRustError::IoError(e) }
}

// ================================================================== //
//  Type Result alias                                                  //
// ================================================================== //

pub type Result<T> = std::result::Result<T, SciRustError>;

// ================================================================== //
//  Helpers — vérification de shapes / devices                         //
// ================================================================== //

/// Vérifie que deux shapes sont identiques. Renvoie ShapeMismatch sinon.
pub fn check_shape(op: &'static str, expected: (usize, usize),
                   got: (usize, usize)) -> Result<()>
{
    if expected != got {
        Err(SciRustError::ShapeMismatch { op, expected, got })
    } else {
        Ok(())
    }
}

/// Vérifie que le device courant correspond à celui attendu.
pub fn check_device(op: &'static str, expected: Device, got: Device) -> Result<()> {
    if expected != got {
        Err(SciRustError::DeviceMismatch { op, expected, got })
    } else {
        Ok(())
    }
}

/// Macro pratique pour bail-out d'une fonction renvoyant Result :
///   bail!("invalid kernel size: {}", k);
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::error::SciRustError::InvalidConfig(format!($($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_mismatch_displays_clearly() {
        let e = SciRustError::ShapeMismatch {
            op: "matmul", expected: (2, 3), got: (2, 4)
        };
        let s = format!("{}", e);
        assert!(s.contains("matmul"));
        assert!(s.contains("(2, 3)"));
        assert!(s.contains("(2, 4)"));
    }

    #[test]
    fn check_shape_passes_when_equal() {
        let r = check_shape("test", (1, 2), (1, 2));
        assert!(r.is_ok());
    }

    #[test]
    fn check_shape_fails_when_different() {
        let r = check_shape("test", (1, 2), (3, 4));
        assert!(matches!(r, Err(SciRustError::ShapeMismatch { .. })));
    }

    #[test]
    fn from_io_error_works() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let sci_err: SciRustError = io_err.into();
        assert!(matches!(sci_err, SciRustError::IoError(_)));
    }

    #[test]
    fn error_implements_std_error() {
        // Validation que SciRustError est compatible avec ? sur des fonctions
        // qui renvoient Box<dyn std::error::Error>
        fn returns_boxed() -> std::result::Result<(), Box<dyn std::error::Error>> {
            let _: SciRustError = SciRustError::GpuNotAvailable;
            Ok(())
        }
        assert!(returns_boxed().is_ok());
    }
}
