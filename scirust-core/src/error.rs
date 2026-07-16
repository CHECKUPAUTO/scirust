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
pub enum Device {
    Cpu,
    Gpu,
}

// ================================================================== //
//  Type d'erreur principal                                            //
// ================================================================== //

#[derive(Debug)]
#[non_exhaustive]
pub enum SciRustError {
    /// Shapes incompatibles entre opérandes
    ShapeMismatch {
        op: &'static str,
        expected: (usize, usize),
        got: (usize, usize),
    },
    /// Dimension intérieure d'un produit incompatible (a.cols != b.rows).
    DimMismatch {
        op: &'static str,
        a_cols: usize,
        b_rows: usize,
    },
    /// Tenseurs sur des devices différents
    DeviceMismatch {
        op: &'static str,
        expected: Device,
        got: Device,
    },
    /// Tenseur sur le mauvais device pour cette op
    WrongDevice {
        op: &'static str,
        expected: Device,
        got: Device,
        hint: &'static str,
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
    InvalidFormat { what: &'static str, details: String },
    /// Index hors bornes (tape, dataset, batch)
    IndexOutOfBounds {
        what: &'static str,
        index: usize,
        bound: usize,
    },
    /// Rang (nombre de dimensions) inattendu pour cette op N-D
    RankMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    /// Axe hors bornes pour le rang du tenseur
    AxisOutOfBounds {
        op: &'static str,
        axis: usize,
        rank: usize,
    },
    /// Nombre total d'éléments incompatible entre deux shapes N-D (reshape…)
    NumelMismatch {
        op: &'static str,
        from_shape: Vec<usize>,
        to_shape: Vec<usize>,
    },
    /// Broadcast impossible d'une shape source vers une shape cible
    BroadcastIncompatible {
        op: &'static str,
        from: Vec<usize>,
        to: Vec<usize>,
    },
    /// Overflow des gradients détecté (mixed precision / loss scaling)
    GradientOverflow { loss_scale: f32 },
}

impl fmt::Display for SciRustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            SciRustError::ShapeMismatch { op, expected, got } =>
            {
                write!(
                    f,
                    "shape mismatch in '{op}': expected {:?}, got {:?}",
                    expected, got
                )
            },
            SciRustError::DimMismatch { op, a_cols, b_rows } =>
            {
                write!(
                    f,
                    "inner dimension mismatch in '{op}': left has {a_cols} column(s) \
                     but right has {b_rows} row(s)"
                )
            },
            SciRustError::DeviceMismatch { op, expected, got } =>
            {
                write!(
                    f,
                    "device mismatch in '{op}': expected {:?}, got {:?} \
                          (use to_cpu/to_gpu to align)",
                    expected, got
                )
            },
            SciRustError::WrongDevice {
                op,
                expected,
                got,
                hint,
            } =>
            {
                write!(
                    f,
                    "wrong device for '{op}': expected {:?}, got {:?} — {hint}",
                    expected, got
                )
            },
            SciRustError::InvalidConfig(msg) =>
            {
                write!(f, "invalid configuration: {msg}")
            },
            SciRustError::GpuNotAvailable =>
            {
                write!(
                    f,
                    "GPU requested but no GPU adapter available — \
                          rebuild with --features wgpu and ensure a compatible adapter is present"
                )
            },
            SciRustError::GpuError(msg) =>
            {
                write!(f, "GPU error: {msg}")
            },
            SciRustError::IoError(e) =>
            {
                write!(f, "I/O error: {e}")
            },
            SciRustError::InvalidFormat { what, details } =>
            {
                write!(f, "invalid {what} format: {details}")
            },
            SciRustError::IndexOutOfBounds { what, index, bound } =>
            {
                write!(f, "{what} index out of bounds: {index} >= {bound}")
            },
            SciRustError::RankMismatch { op, expected, got } =>
            {
                write!(
                    f,
                    "rank mismatch in '{op}': expected {expected} dimension(s), got {got}"
                )
            },
            SciRustError::AxisOutOfBounds { op, axis, rank } =>
            {
                write!(
                    f,
                    "axis {axis} out of bounds in '{op}': tensor has rank {rank}"
                )
            },
            SciRustError::NumelMismatch {
                op,
                from_shape,
                to_shape,
            } =>
            {
                let from_numel: usize = from_shape.iter().product();
                let to_numel: usize = to_shape.iter().product();
                write!(
                    f,
                    "element count mismatch in '{op}': cannot view {from_shape:?} \
                     ({from_numel} element(s)) as {to_shape:?} ({to_numel} element(s))"
                )
            },
            SciRustError::BroadcastIncompatible { op, from, to } =>
            {
                write!(f, "cannot broadcast {from:?} to {to:?} in '{op}'")
            },
            SciRustError::GradientOverflow { loss_scale } =>
            {
                write!(
                    f,
                    "gradient overflow detected, loss scale reduced to {loss_scale}"
                )
            },
        }
    }
}

impl SciRustError {
    /// A short, machine-stable code for the error category — useful for logging,
    /// dashboards, and scripting (`E_SHAPE`, `E_GPU`, …). Stable across releases
    /// even as the human message text is refined.
    pub fn code(&self) -> &'static str {
        match self
        {
            SciRustError::ShapeMismatch { .. } => "E_SHAPE",
            SciRustError::DimMismatch { .. } => "E_DIM",
            SciRustError::DeviceMismatch { .. } => "E_DEVICE",
            SciRustError::WrongDevice { .. } => "E_DEVICE",
            SciRustError::InvalidConfig(_) => "E_CONFIG",
            SciRustError::GpuNotAvailable => "E_GPU_ABSENT",
            SciRustError::GpuError(_) => "E_GPU",
            SciRustError::IoError(_) => "E_IO",
            SciRustError::InvalidFormat { .. } => "E_FORMAT",
            SciRustError::IndexOutOfBounds { .. } => "E_BOUNDS",
            SciRustError::RankMismatch { .. } => "E_RANK",
            SciRustError::AxisOutOfBounds { .. } => "E_AXIS",
            SciRustError::NumelMismatch { .. } => "E_NUMEL",
            SciRustError::BroadcastIncompatible { .. } => "E_BROADCAST",
            SciRustError::GradientOverflow { .. } => "E_OVERFLOW",
        }
    }

    /// A one-line, actionable hint on how to resolve the error — the affordance
    /// the Rust compiler and `miette` give (`help: …`). `None` when no generic
    /// remedy applies. Callers can print it as a second line under the message.
    pub fn hint(&self) -> Option<&'static str> {
        match self
        {
            SciRustError::ShapeMismatch { .. } =>
            {
                Some("check the operand shapes; transpose or reshape one side so they match")
            },
            SciRustError::DimMismatch { .. } => Some(
                "for `a · b`, `a.cols` must equal `b.rows`; transpose one operand or fix the layer width",
            ),
            SciRustError::DeviceMismatch { .. } | SciRustError::WrongDevice { .. } => Some(
                "move both tensors to the same device with `.to_cpu()` / `.to_gpu()` before the op",
            ),
            SciRustError::GpuNotAvailable => Some(
                "rebuild with `--features wgpu` and ensure a compatible GPU adapter is visible, or run on CPU",
            ),
            SciRustError::InvalidFormat { .. } => Some(
                "the file is corrupt or from an unsupported version; re-export it or check the producer",
            ),
            SciRustError::IndexOutOfBounds { .. } =>
            {
                Some("the index is past the end; check the length before indexing")
            },
            SciRustError::RankMismatch { .. } => Some(
                "reshape or add/remove axes so the operand has the expected number of dimensions",
            ),
            SciRustError::AxisOutOfBounds { .. } =>
            {
                Some("valid axes are 0..rank; check the axis argument against `ndim()`")
            },
            SciRustError::NumelMismatch { .. } => Some(
                "the total element count must be preserved; pick a target shape whose product matches",
            ),
            SciRustError::BroadcastIncompatible { .. } => Some(
                "shapes broadcast right-aligned: each axis must match or be 1 on the source side",
            ),
            SciRustError::GradientOverflow { .. } => Some(
                "skip this optimizer step and retry; dynamic loss scaling already lowered the scale",
            ),
            _ => None,
        }
    }
}

impl std::error::Error for SciRustError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            SciRustError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SciRustError {
    fn from(e: std::io::Error) -> Self {
        SciRustError::IoError(e)
    }
}

impl From<String> for SciRustError {
    fn from(s: String) -> Self {
        SciRustError::InvalidConfig(s)
    }
}

impl From<&str> for SciRustError {
    fn from(s: &str) -> Self {
        SciRustError::InvalidConfig(s.to_string())
    }
}

impl From<serde_json::Error> for SciRustError {
    fn from(e: serde_json::Error) -> Self {
        SciRustError::InvalidFormat {
            what: "json",
            details: e.to_string(),
        }
    }
}

// ================================================================== //
//  Type Result alias                                                  //
// ================================================================== //

pub type Result<T> = std::result::Result<T, SciRustError>;

// ================================================================== //
//  Helpers — vérification de shapes / devices                         //
// ================================================================== //

/// Vérifie que deux shapes sont identiques. Renvoie ShapeMismatch sinon.
pub fn check_shape(op: &'static str, expected: (usize, usize), got: (usize, usize)) -> Result<()> {
    if expected != got
    {
        Err(SciRustError::ShapeMismatch { op, expected, got })
    }
    else
    {
        Ok(())
    }
}

/// Vérifie que les dimensions internes d'un matmul concordent (a.cols == b.rows).
pub fn check_inner_dim(op: &'static str, a_cols: usize, b_rows: usize) -> Result<()> {
    if a_cols != b_rows
    {
        Err(SciRustError::DimMismatch { op, a_cols, b_rows })
    }
    else
    {
        Ok(())
    }
}

/// Vérifie que le device courant correspond à celui attendu.
pub fn check_device(op: &'static str, expected: Device, got: Device) -> Result<()> {
    if expected != got
    {
        Err(SciRustError::DeviceMismatch { op, expected, got })
    }
    else
    {
        Ok(())
    }
}

/// Vérifie qu'un index est dans les bornes `[0, bound)`. Renvoie
/// `IndexOutOfBounds` sinon — à préférer à un `panic!`/indexation directe sur
/// les frontières publiques.
pub fn check_index(what: &'static str, index: usize, bound: usize) -> Result<()> {
    if index >= bound
    {
        Err(SciRustError::IndexOutOfBounds { what, index, bound })
    }
    else
    {
        Ok(())
    }
}

/// Vérifie qu'un tenseur N-D a le rang attendu. Renvoie RankMismatch sinon.
pub fn check_rank(op: &'static str, expected: usize, got: usize) -> Result<()> {
    if expected != got
    {
        Err(SciRustError::RankMismatch { op, expected, got })
    }
    else
    {
        Ok(())
    }
}

/// Vérifie qu'un axe est valide pour un tenseur de rang `rank` (axe < rang).
/// Renvoie `AxisOutOfBounds` sinon.
pub fn check_axis(op: &'static str, axis: usize, rank: usize) -> Result<()> {
    if axis >= rank
    {
        Err(SciRustError::AxisOutOfBounds { op, axis, rank })
    }
    else
    {
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
            op: "matmul",
            expected: (2, 3),
            got: (2, 4),
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
    fn dim_mismatch_has_code_and_actionable_hint() {
        let e = check_inner_dim("matmul", 3, 4).unwrap_err();
        assert_eq!(e.code(), "E_DIM");
        let msg = format!("{e}");
        assert!(msg.contains("3") && msg.contains("4"));
        let hint = e.hint().expect("dim mismatch should carry a hint");
        assert!(hint.contains("a.cols") || hint.contains("transpose"));
    }

    #[test]
    fn every_variant_has_a_stable_code() {
        // A representative set; `code()` must be total (compiles only if so).
        assert_eq!(SciRustError::GpuNotAvailable.code(), "E_GPU_ABSENT");
        assert_eq!(SciRustError::InvalidConfig("x".into()).code(), "E_CONFIG");
    }

    #[test]
    fn rank_mismatch_has_code_and_actionable_hint() {
        let e = check_rank("to_tensor_2d", 2, 4).unwrap_err();
        assert_eq!(e.code(), "E_RANK");
        let msg = format!("{e}");
        assert!(msg.contains("to_tensor_2d"));
        assert!(msg.contains('2') && msg.contains('4'));
        let hint = e.hint().expect("rank mismatch should carry a hint");
        assert!(hint.contains("dimensions") || hint.contains("axes"));
    }

    #[test]
    fn check_rank_passes_when_equal() {
        assert!(check_rank("test", 3, 3).is_ok());
    }

    #[test]
    fn axis_out_of_bounds_has_code_and_actionable_hint() {
        let e = check_axis("slice_axis", 5, 3).unwrap_err();
        assert_eq!(e.code(), "E_AXIS");
        let msg = format!("{e}");
        assert!(msg.contains("slice_axis"));
        assert!(msg.contains('5') && msg.contains('3'));
        let hint = e.hint().expect("axis out of bounds should carry a hint");
        assert!(hint.contains("rank") || hint.contains("ndim"));
    }

    #[test]
    fn check_axis_passes_when_in_bounds() {
        assert!(check_axis("test", 2, 3).is_ok());
        assert!(matches!(
            check_axis("test", 3, 3),
            Err(SciRustError::AxisOutOfBounds { .. })
        ));
    }

    #[test]
    fn numel_mismatch_displays_both_shapes_and_counts() {
        let e = SciRustError::NumelMismatch {
            op: "reshape",
            from_shape: vec![2, 3, 4],
            to_shape: vec![7, 4],
        };
        assert_eq!(e.code(), "E_NUMEL");
        let msg = format!("{e}");
        assert!(msg.contains("reshape"));
        assert!(msg.contains("[2, 3, 4]") && msg.contains("[7, 4]"));
        assert!(msg.contains("24") && msg.contains("28"));
        let hint = e.hint().expect("numel mismatch should carry a hint");
        assert!(hint.contains("element count"));
    }

    #[test]
    fn broadcast_incompatible_displays_both_shapes() {
        let e = SciRustError::BroadcastIncompatible {
            op: "broadcast_to",
            from: vec![3, 1],
            to: vec![2, 3],
        };
        assert_eq!(e.code(), "E_BROADCAST");
        let msg = format!("{e}");
        assert!(msg.contains("[3, 1]") && msg.contains("[2, 3]"));
        assert!(msg.contains("broadcast_to"));
        let hint = e.hint().expect("broadcast error should carry a hint");
        assert!(hint.contains("right-aligned") || hint.contains("broadcast"));
    }

    #[test]
    fn gradient_overflow_has_code_and_reports_scale() {
        let e = SciRustError::GradientOverflow { loss_scale: 0.5 };
        assert_eq!(e.code(), "E_OVERFLOW");
        let msg = format!("{e}");
        assert!(msg.contains("overflow") && msg.contains("0.5"));
        assert!(e.hint().is_some());
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
