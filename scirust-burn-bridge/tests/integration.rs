//! Tests d'intégration du bridge avec un backend Burn réel (`burn-ndarray`).
//!
//! Ces tests sont des **oracles** : on construit un réseau aux poids *connus*,
//! on dérive la sortie attendue à la main, et on vérifie forme + données au
//! bit près. On vérifie aussi que la donnée traverse le bridge sans
//! transposition ni réordonnancement (layout row-major préservé).

use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    module::Param,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, TensorData},
};
use scirust_burn_bridge::{InferenceOnly, Policy};

type B = NdArray<f32>;

/// Construit une couche linéaire `d_input -> d_output` aux poids/biais fixés.
///
/// Convention Burn (vérifiée contre burn-nn 0.20) : `weight` a la forme
/// `[d_input, d_output]` et `forward` calcule `output = input @ weight + bias`.
/// Le vecteur `weight` est donc rangé en row-major sur `[d_input, d_output]`.
fn linear_with_weights(
    device: &NdArrayDevice,
    d_input: usize,
    d_output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
) -> Linear<B> {
    assert_eq!(
        weight.len(),
        d_input * d_output,
        "weight length must be d_input*d_output"
    );
    assert_eq!(bias.len(), d_output, "bias length must be d_output");

    let mut lin = LinearConfig::new(d_input, d_output).init(device);
    let w = TensorData::new(weight, [d_input, d_output]);
    let b = TensorData::new(bias, [d_output]);
    lin.weight = Param::from_data(w, device);
    lin.bias = Some(Param::from_data(b, device));
    lin
}

/// Politique = une seule couche linéaire aux poids connus.
struct LinearPolicy {
    inner: Linear<B>,
}

impl Policy<B> for LinearPolicy {
    type Input = Tensor<B, 2>;
    type Output = Tensor<B, 2>;

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.inner.forward(input)
    }
}

/// Oracle de forme + données pour un forward `[1,3] -> [1,2]`.
///
/// weight = [[1,2],[3,4],[5,6]] (row-major sur [3,2]), bias = [10,20].
/// Pour input = [1,1,1] :
///   out[0] = 1·1 + 1·3 + 1·5 + 10 = 19
///   out[1] = 1·2 + 1·4 + 1·6 + 20 = 32
#[test]
fn linear_forward_exact_shape_and_data() {
    let device = NdArrayDevice::Cpu;
    let policy = LinearPolicy {
        inner: linear_with_weights(
            &device,
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0],
        ),
    };
    let bridge = InferenceOnly::new(policy, device);

    let input = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 1.0, 1.0], [1, 3]),
        bridge.device(),
    );
    let out = bridge.eval(input);

    // Shape mapping : [batch=1, d_input=3] -> [batch=1, d_output=2].
    assert_eq!(
        out.dims(),
        [1, 2],
        "Linear must map d_input=3 -> d_output=2"
    );

    let got: Vec<f32> = out.into_data().to_vec().expect("to_vec");
    assert_eq!(got, vec![19.0f32, 32.0], "hand-derived output mismatch");
}

/// Deuxième vecteur d'entrée pour éviter tout faux positif lié à la symétrie.
///
/// input = [2,0,1] :
///   out[0] = 2·1 + 0·3 + 1·5 + 10 = 17
///   out[1] = 2·2 + 0·4 + 1·6 + 20 = 30
#[test]
fn linear_forward_asymmetric_input() {
    let device = NdArrayDevice::Cpu;
    let policy = LinearPolicy {
        inner: linear_with_weights(
            &device,
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0],
        ),
    };
    let bridge = InferenceOnly::new(policy, device);

    let input = Tensor::<B, 2>::from_data(
        TensorData::new(vec![2.0f32, 0.0, 1.0], [1, 3]),
        bridge.device(),
    );
    let got: Vec<f32> = bridge.eval(input).into_data().to_vec().expect("to_vec");

    assert_eq!(got, vec![17.0f32, 30.0]);
}

/// Forward batché : deux lignes traitées indépendamment, dans l'ordre.
///
/// input = [[1,1,1],[2,0,1]] -> [[19,32],[17,30]]
/// Le `Vec` aplati row-major attendu est donc [19, 32, 17, 30].
#[test]
fn linear_forward_batched_rows_independent() {
    let device = NdArrayDevice::Cpu;
    let policy = LinearPolicy {
        inner: linear_with_weights(
            &device,
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![10.0, 20.0],
        ),
    };
    let bridge = InferenceOnly::new(policy, device);

    let input = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.0f32, 1.0, 1.0, 2.0, 0.0, 1.0], [2, 3]),
        bridge.device(),
    );
    let out = bridge.eval(input);

    assert_eq!(out.dims(), [2, 2], "[batch=2, d_output=2]");
    let got: Vec<f32> = out.into_data().to_vec().expect("to_vec");
    assert_eq!(got, vec![19.0f32, 32.0, 17.0, 30.0]);
}

/// Round-trip exact : un `Vec<f32>` SciRust -> `Tensor` Burn -> `Vec<f32>`.
///
/// Doit préserver les valeurs ET l'ordre row-major, sans transposition.
/// On passe par une `IdentityPolicy` pour exercer le chemin complet du bridge.
#[test]
fn roundtrip_preserves_data_and_row_major_order() {
    struct IdentityPolicy;
    impl Policy<B> for IdentityPolicy {
        type Input = Tensor<B, 2>;
        type Output = Tensor<B, 2>;
        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            input
        }
    }

    let device = NdArrayDevice::Cpu;
    let bridge = InferenceOnly::new(IdentityPolicy, device);

    // Matrice 2x3 distincte et asymétrique pour détecter toute transposition.
    let original: Vec<f32> = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
    let input =
        Tensor::<B, 2>::from_data(TensorData::new(original.clone(), [2, 3]), bridge.device());

    let out = bridge.eval(input);
    assert_eq!(out.dims(), [2, 3], "round-trip must preserve shape");

    let back: Vec<f32> = out.into_data().to_vec().expect("to_vec");
    assert_eq!(
        back, original,
        "round-trip must preserve data + row-major order"
    );

    // Sentinelle anti-transposition : la transposée [3,2] aurait l'ordre
    // [10,20,11,21,12,22] ; on s'assure de NE PAS l'obtenir.
    let transposed_order: Vec<f32> = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
    assert_ne!(
        back, transposed_order,
        "data must not be transposed by the bridge"
    );
}

/// Vérifie que `device()` expose bien le device passé au constructeur et que
/// `policy()` rend la politique sous-jacente (accès lecture seule réellement
/// branché, pas un placeholder).
#[test]
fn accessors_expose_real_state() {
    struct Tagged {
        tag: u32,
    }
    impl Policy<B> for Tagged {
        type Input = Tensor<B, 2>;
        type Output = Tensor<B, 2>;
        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            input
        }
    }

    let device = NdArrayDevice::Cpu;
    // `NdArrayDevice` is `Copy`, so the bridge gets its own copy.
    let bridge = InferenceOnly::new(Tagged { tag: 7 }, device);

    assert_eq!(bridge.device(), &device);
    assert_eq!(bridge.policy().tag, 7);
}
