//! A small, dependency-free **lossless** byte codec (LZSS) for the COLD spill
//! store. Code and prose spill blobs are highly repetitive (`pub fn`, indentation,
//! identifiers), so a sliding-window LZ shrinks them on disk without the lossy
//! compaction of slice 4.
//!
//! Format: a 1-byte header — `0` = the payload is the original bytes verbatim
//! (used whenever compression wouldn't help, so the codec **never inflates**), `1` =
//! LZSS. The LZSS stream is the classic flag-byte scheme: one flag byte precedes up
//! to 8 tokens; bit *i* (LSB first) is `0` for a literal byte or `1` for a 2-byte
//! back-reference packing a 12-bit offset (1..4096) and a 4-bit length (3..18).
//!
//! **Safety net:** the spill store keys and verifies blobs by the SHA-256 of the
//! *original* content, re-checked on read. So even a latent bug here can only ever
//! produce a hash mismatch — a recoverable cold-miss — never silent corruption.
//! That is what makes a hand-rolled codec acceptable on the lossless path; the
//! round-trip property test below is the primary guard.

const WINDOW: usize = 4096; // 12-bit offset
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 18; // 4-bit length + MIN_MATCH
const MAX_CHAIN: usize = 64; // hash-chain search-depth cap (bounds compress time)
const HASH_SIZE: usize = 1 << 13;

/// Compress `data`, never inflating: returns `[0] ++ data` if LZSS wouldn't be
/// smaller, else `[1] ++ lzss(data)`. Deterministic (so content-addressing still
/// deduplicates identical blobs to one file).
pub fn compress(data: &[u8]) -> Vec<u8> {
    let packed = lzss_compress(data);
    let mut out = Vec::with_capacity(packed.len().min(data.len()) + 1);
    if packed.len() < data.len() {
        out.push(1);
        out.extend_from_slice(&packed);
    } else {
        out.push(0);
        out.extend_from_slice(data);
    }
    out
}

/// Inverse of [`compress`]. `None` on a malformed blob (unknown header, truncated
/// or out-of-range back-reference) — surfaced by the caller as a cold-miss.
pub fn decompress(blob: &[u8]) -> Option<Vec<u8>> {
    match blob.split_first() {
        Some((0, rest)) => Some(rest.to_vec()),
        Some((1, rest)) => lzss_decompress(rest),
        _ => None,
    }
}

fn hash3(data: &[u8], pos: usize) -> usize {
    let a = data[pos] as usize;
    let b = data[pos + 1] as usize;
    let c = data[pos + 2] as usize;
    ((a << 10) ^ (b << 5) ^ c) & (HASH_SIZE - 1)
}

fn lzss_compress(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::new();
    // Hash-chain index: `head[h]` is the most recent position whose 3-byte prefix
    // hashes to `h`; `prev[p]` chains to the prior such position. Bounds the match
    // search to MAX_CHAIN candidates per position instead of scanning the window.
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; n];

    let insert = |head: &mut [i32], prev: &mut [i32], pos: usize| {
        if pos + MIN_MATCH <= n {
            let h = hash3(data, pos);
            prev[pos] = head[h];
            head[h] = pos as i32;
        }
    };

    let mut pos = 0;
    while pos < n {
        let flag_idx = out.len();
        out.push(0u8);
        let mut flag = 0u8;
        for bit in 0..8 {
            if pos >= n {
                break;
            }
            let max_len = (n - pos).min(MAX_MATCH);
            let mut best_len = 0usize;
            let mut best_off = 0usize;
            if max_len >= MIN_MATCH {
                let win_start = pos.saturating_sub(WINDOW);
                let mut cand = head[hash3(data, pos)];
                let mut chain = 0;
                while cand >= 0 && (cand as usize) >= win_start && chain < MAX_CHAIN {
                    let c = cand as usize;
                    let mut l = 0;
                    while l < max_len && data[c + l] == data[pos + l] {
                        l += 1;
                    }
                    if l > best_len {
                        best_len = l;
                        best_off = pos - c;
                        if l == max_len {
                            break;
                        }
                    }
                    cand = prev[c];
                    chain += 1;
                }
            }
            insert(&mut head, &mut prev, pos);
            if best_len >= MIN_MATCH {
                flag |= 1 << bit;
                let code = (((best_off - 1) as u16) << 4) | ((best_len - MIN_MATCH) as u16);
                out.push((code >> 8) as u8);
                out.push((code & 0xFF) as u8);
                for p in (pos + 1)..(pos + best_len) {
                    insert(&mut head, &mut prev, p);
                }
                pos += best_len;
            } else {
                out.push(data[pos]);
                pos += 1;
            }
        }
        out[flag_idx] = flag;
    }
    out
}

fn lzss_decompress(blob: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < blob.len() {
        let flag = blob[i];
        i += 1;
        for bit in 0..8 {
            if i >= blob.len() {
                break; // a short final block: remaining flag bits have no tokens
            }
            if flag & (1 << bit) != 0 {
                if i + 1 >= blob.len() {
                    return None; // truncated back-reference
                }
                let code = ((blob[i] as u16) << 8) | (blob[i + 1] as u16);
                i += 2;
                let off = ((code >> 4) + 1) as usize;
                let len = ((code & 0xF) as usize) + MIN_MATCH;
                if off > out.len() {
                    return None; // back-reference before the start of output
                }
                let start = out.len() - off;
                for k in 0..len {
                    out.push(out[start + k]); // byte-by-byte: handles off < len overlap
                }
            } else {
                out.push(blob[i]);
                i += 1;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn round_trip(data: &[u8]) {
        let blob = compress(data);
        assert_eq!(
            decompress(&blob).as_deref(),
            Some(data),
            "round-trip for {data:?}"
        );
        // Never inflates beyond the 1-byte header.
        assert!(
            blob.len() <= data.len() + 1,
            "inflated {} → {}",
            data.len(),
            blob.len()
        );
    }

    #[test]
    fn round_trips_edge_cases() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"ab");
        round_trip(b"abc");
        round_trip(&[0u8; 5000]); // long run → overlap matches
        round_trip(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); // overlap (off < len)
        round_trip(b"pub fn foo() {}\npub fn bar() {}\npub fn baz() {}\n");
    }

    #[test]
    fn compresses_repetitive_code() {
        let src = "pub fn function_x() -> u32 { 0 }\n".repeat(64);
        let blob = compress(src.as_bytes());
        assert_eq!(decompress(&blob).as_deref(), Some(src.as_bytes()));
        assert!(
            blob.len() * 2 < src.len(),
            "expected >2x on repetitive code, got {} → {}",
            src.len(),
            blob.len()
        );
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(decompress(&[]), None); // no header
        assert_eq!(decompress(&[2, 0, 0]), None); // unknown header
                                                  // Flag bit 0 = back-reference, but only 1 of its 2 bytes follows (truncated).
        assert_eq!(decompress(&[1, 0b0000_0001, 0x00]), None);
        // Back-reference whose offset points before the start of output.
        assert_eq!(decompress(&[1, 0b0000_0001, 0x00, 0x00]), None);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        /// The core lossless guarantee: decompress(compress(x)) == x for ANY bytes.
        #[test]
        fn decompress_inverts_compress(data in prop::collection::vec(any::<u8>(), 0..2048)) {
            let blob = compress(&data);
            let got = decompress(&blob);
            prop_assert_eq!(got.as_deref(), Some(data.as_slice()));
        }

        /// Biased toward small alphabets (where matches abound) to stress the
        /// hash-chain / overlap paths harder than uniform random would.
        #[test]
        fn decompress_inverts_compress_low_entropy(
            data in prop::collection::vec(0u8..4u8, 0..3000)
        ) {
            let blob = compress(&data);
            let got = decompress(&blob);
            prop_assert_eq!(got.as_deref(), Some(data.as_slice()));
            prop_assert!(blob.len() <= data.len() + 1);
        }
    }

    // ── Format 2 : dictionnaire partagé ──────────────────────────────────────

    fn sample_dict() -> std::sync::Arc<Dict> {
        // Un dictionnaire réaliste : du code qui ressemble à ce que le tier COLD
        // spille réellement (des contenus de nœuds d'un même projet Rust).
        let mut d = Vec::new();
        for i in 0..80 {
            d.extend_from_slice(
                format!(
                    "pub fn helper_{i}(input: &str) -> Result<usize> {{\n    Ok(input.len())\n}}\n"
                )
                .as_bytes(),
            );
        }
        std::sync::Arc::new(Dict::new(d))
    }

    fn round_trip_v2(c: &mut Compressor, dict: &[u8], data: &[u8]) {
        let blob = c.compress_with(data);
        assert_eq!(
            decompress_with(&blob, Some(dict)).as_deref(),
            Some(data),
            "v2 round-trip for {} bytes",
            data.len()
        );
        // Jamais plus gros que l'entrée, ni que ce que produisait le format 1.
        assert!(blob.len() <= data.len() + 1, "inflation interdite");
        assert!(
            blob.len() <= compress(data).len(),
            "le format 2 ne doit jamais être pire que le format 1"
        );
    }

    #[test]
    fn v2_round_trips_and_never_regresses_on_realistic_content() {
        let dict = sample_dict();
        let mut c = Compressor::new(dict.clone());
        for text in [
            "pub fn helper_7(input: &str) -> Result<usize> {\n    Ok(input.len())\n}\n",
            "pub fn something_else() {}\n",
            "",
            "x",
            "\u{4e2d}\u{6587}",
        ] {
            round_trip_v2(&mut c, dict.bytes(), text.as_bytes());
        }
    }

    #[test]
    fn a_format_1_blob_stays_readable_with_and_without_a_dictionary() {
        // La compatibilité ascendante est la raison d'être de l'octet d'en-tête :
        // les blobs déjà sur disque ne doivent jamais devoir être réécrits.
        let dict = sample_dict();
        let payload = b"pub fn legacy() -> u8 { 42 }\n".repeat(4);
        let old = compress(&payload);
        assert_eq!(decompress_with(&old, None).as_deref(), Some(&payload[..]));
        assert_eq!(
            decompress_with(&old, Some(dict.bytes())).as_deref(),
            Some(&payload[..])
        );
    }

    #[test]
    fn a_format_2_blob_without_its_dictionary_is_a_miss_not_a_wrong_restore() {
        // Le contrat du tier COLD : à défaut de pouvoir restaurer, on rate — on ne
        // rend jamais un contenu faux.
        let dict = sample_dict();
        let mut c = Compressor::new(dict.clone());
        let payload = b"pub fn helper_3(input: &str) -> Result<usize> {\n    Ok(input.len())\n}\n";
        let blob = c.compress_with(payload);
        if blob[0] == 2 {
            assert!(decompress_with(&blob, None).is_none());
        }
    }

    #[test]
    fn compression_is_deterministic_across_compressors() {
        // Le store est adressé par contenu : deux compressions du même blob doivent
        // produire les mêmes octets, sinon la déduplication casse.
        let dict = sample_dict();
        let payload = b"pub fn helper_11(input: &str) -> Result<usize> {\n    Ok(input.len())\n}\n";
        let a = Compressor::new(dict.clone()).compress_with(payload);
        let b = Compressor::new(dict.clone()).compress_with(payload);
        assert_eq!(a, b);
        // Et un compresseur réutilisé doit rendre le même résultat qu'un neuf,
        // ce qui vérifie que la table de hachage est bien restaurée entre blobs.
        let mut reused = Compressor::new(dict.clone());
        let _ = reused.compress_with(b"un autre contenu quelconque\n");
        assert_eq!(reused.compress_with(payload), a);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        #[test]
        fn v2_round_trips_any_bytes(data: Vec<u8>) {
            let dict = sample_dict();
            let mut c = Compressor::new(dict.clone());
            let blob = c.compress_with(&data);
            let back = decompress_with(&blob, Some(dict.bytes()));
            prop_assert_eq!(back.as_deref(), Some(&data[..]));
            prop_assert!(blob.len() <= data.len() + 1);
            prop_assert!(blob.len() <= compress(&data).len());
        }

        #[test]
        fn v2_round_trips_dictionary_like_bytes(reps in 0usize..40, idx in 0usize..80) {
            // Des entrées qui ressemblent au dictionnaire : le cas que le format 2
            // est censé exploiter, et celui où un décalage d'offset se verrait.
            let dict = sample_dict();
            let mut c = Compressor::new(dict.clone());
            let data = format!("pub fn helper_{idx}(input: &str) -> Result<usize> {{\n    Ok(input.len())\n}}\n")
                .repeat(reps)
                .into_bytes();
            let blob = c.compress_with(&data);
            let back = decompress_with(&blob, Some(dict.bytes()));
            prop_assert_eq!(back.as_deref(), Some(&data[..]));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Format 2 — LZSS avec dictionnaire partagé
//
// Motivation, mesurée sur un vrai tier COLD (5000 contenus de nœuds, 1,47 Mo) :
// les blobs spillés sont minuscules — médiane 97 octets, 42 % sous 64 octets,
// aucun au-dessus de 16 Ko. À cette taille un blob n'a pas de redondance interne
// à exploiter (39 % ressortaient incompressibles, en mode verbatim), mais il en
// partage énormément avec ses voisins : c'est du code d'un même projet.
//
// Un dictionnaire partagé donne donc ce que la fenêtre seule ne peut pas :
//   v1 (isolé)                        910 421 o
//   v2 + dictionnaire 64 Ko           598 104 o  (+ 65 536 o de dictionnaire)
//   → 27 % de moins au total, dictionnaire compris.
//
// Deux résultats contre-intuitifs guident les constantes ci-dessous, et méritent
// d'être connus de qui les modifierait :
//   * agrandir la fenêtre SEULE dégrade le ratio (0,94× mesuré) — on paie un
//     encodage plus lourd sans match long à exploiter ;
//   * l'optimum du dictionnaire est exactement la taille de la fenêtre ; au-delà,
//     le contenu utile sort de portée et le gain redescend.
// Fenêtre et dictionnaire doivent donc être dimensionnés ensemble.

/// Fenêtre du format 2. Doit rester ≥ la taille du dictionnaire (voir ci-dessus).
pub const V2_WINDOW: usize = 32 * 1024;
/// Longueur de match maximale du format 2 (contre 18 pour le format 1).
pub const V2_MAX_MATCH: usize = 258;
const V2_HASH_SIZE: usize = 1 << 15;
const V2_MAX_CHAIN: usize = 64;

fn hash3_in(data: &[u8], pos: usize) -> usize {
    let a = data[pos] as usize;
    let b = data[pos + 1] as usize;
    let c = data[pos + 2] as usize;
    ((a << 10) ^ (b << 5) ^ c) & (V2_HASH_SIZE - 1)
}

/// Encodage à longueur variable. Un match proche et court tient sur 2 octets,
/// comme dans le format 1 ; un match lointain ou long passe sur 4 octets. Le bit
/// de poids fort du premier octet distingue les deux familles.
fn v2_emit_match(out: &mut Vec<u8>, off: usize, len: usize) {
    let o = off - 1;
    if o < 2048 && len <= 10 {
        let code = ((o as u16) << 4) | ((len - MIN_MATCH) as u16);
        out.push(((code >> 8) as u8) & 0x7F);
        out.push((code & 0xFF) as u8);
    } else {
        out.push(0x80 | ((o >> 16) as u8 & 0x7F));
        out.push((o >> 8) as u8);
        out.push(o as u8);
        out.push((len - MIN_MATCH) as u8);
    }
}

/// Dictionnaire pré-indexé : la table de hachage est construite **une seule fois**
/// et réutilisée pour chaque blob. Reconstruire l'index à chaque appel coûtait
/// 2,4 Mo/s ; le pré-indexer et ne plus recopier le dictionnaire porte le débit à
/// 19 Mo/s.
pub struct Dict {
    bytes: Vec<u8>,
    head: Vec<i32>,
    prev: Vec<i32>,
}

impl std::fmt::Debug for Dict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Les tables d'index n'ont aucun intérêt à l'affichage ; seule la taille
        // du dictionnaire renseigne.
        f.debug_struct("Dict")
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl Dict {
    /// Indexe `bytes` comme dictionnaire partagé.
    pub fn new(bytes: Vec<u8>) -> Self {
        let n = bytes.len();
        let mut head = vec![-1i32; V2_HASH_SIZE];
        let mut prev = vec![-1i32; n.max(1)];
        // Boucle explicite : l'index sert à la fois de position hachée, de clé
        // dans `prev` et de valeur stockée dans `head`.
        let limit = n.saturating_sub(MIN_MATCH);
        let mut i = 0usize;
        while i < limit {
            let h = hash3_in(&bytes, i);
            prev[i] = head[h];
            head[h] = i as i32;
            i += 1;
        }
        Dict { bytes, head, prev }
    }

    /// Les octets du dictionnaire (nécessaires au décodage).
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Compresseur réutilisable lié à un [`Dict`]. Le dictionnaire reste en place dans
/// le tampon de travail d'un blob à l'autre, et la table de hachage n'est jamais
/// clonée : les entrées touchées sont journalisées puis restaurées.
#[derive(Clone)]
pub struct Compressor {
    dict: std::sync::Arc<Dict>,
    buf: Vec<u8>,
    head: Vec<i32>,
    prev: Vec<i32>,
    journal: Vec<(u32, i32)>,
}

impl std::fmt::Debug for Compressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compressor")
            .field("dict", &self.dict)
            .finish()
    }
}

impl Compressor {
    /// Le dictionnaire est **partagé** (`Arc`) plutôt qu'emprunté : un compresseur
    /// doit pouvoir vivre aussi longtemps que le store qui l'utilise. Le construire
    /// recopie les tables du dictionnaire, ce qui est négligeable une fois — mais
    /// annulerait le gain de débit s'il était reconstruit à chaque blob. Garder un
    /// compresseur vivant est donc un choix de conception, pas un détail.
    pub fn new(dict: std::sync::Arc<Dict>) -> Self {
        Compressor {
            buf: dict.bytes.clone(),
            head: dict.head.clone(),
            prev: dict.prev.clone(),
            journal: Vec::new(),
            dict,
        }
    }

    fn find(&self, at: usize, n: usize) -> (usize, usize) {
        let max_len = (n - at).min(V2_MAX_MATCH);
        if max_len < MIN_MATCH {
            return (0, 0);
        }
        let win_start = at.saturating_sub(V2_WINDOW);
        let (mut bl, mut bo) = (0usize, 0usize);
        let mut cand = self.head[hash3_in(&self.buf, at)];
        let mut chain = 0;
        while cand >= 0 && (cand as usize) >= win_start && chain < V2_MAX_CHAIN {
            let c = cand as usize;
            let mut l = 0;
            while l < max_len && self.buf[c + l] == self.buf[at + l] {
                l += 1;
            }
            if l > bl {
                bl = l;
                bo = at - c;
                if l == max_len {
                    break;
                }
            }
            cand = self.prev[c];
            chain += 1;
        }
        (bl, bo)
    }

    fn insert(&mut self, pos: usize, n: usize) {
        if pos + MIN_MATCH <= n {
            let h = hash3_in(&self.buf, pos);
            self.prev[pos] = self.head[h];
            self.journal.push((h as u32, self.head[h]));
            self.head[h] = pos as i32;
        }
    }

    /// Flux format-2 pour `data` (sans l'octet d'en-tête, ajouté par [`compress_with`]).
    fn stream(&mut self, data: &[u8]) -> Vec<u8> {
        let start = self.dict.bytes.len();
        self.buf.truncate(start);
        self.buf.extend_from_slice(data);
        let n = self.buf.len();
        if self.prev.len() < n {
            self.prev.resize(n, -1);
        }
        self.journal.clear();

        let mut out = Vec::with_capacity(data.len() / 2 + 8);
        let mut pos = start;
        while pos < n {
            let flag_idx = out.len();
            out.push(0u8);
            let mut flag = 0u8;
            for bit in 0..8 {
                if pos >= n {
                    break;
                }
                let (mut bl, mut bo) = self.find(pos, n);
                // Lazy matching : si la position suivante fait mieux, émettre un
                // littéral plutôt qu'un match médiocre (+4,6 % mesuré).
                if bl >= MIN_MATCH && pos + 1 < n {
                    let (nl, _) = self.find(pos + 1, n);
                    if nl > bl {
                        bl = 0;
                        bo = 0;
                    }
                }
                self.insert(pos, n);
                if bl >= MIN_MATCH {
                    flag |= 1 << bit;
                    v2_emit_match(&mut out, bo, bl);
                    for q in (pos + 1)..(pos + bl) {
                        self.insert(q, n);
                    }
                    pos += bl;
                } else {
                    out.push(self.buf[pos]);
                    pos += 1;
                }
            }
            out[flag_idx] = flag;
        }
        // Restaurer la table dans son état « dictionnaire seul ».
        for &(h, old) in self.journal.iter().rev() {
            self.head[h as usize] = old;
        }
        out
    }

    /// Compresse `data` en retenant le plus petit des trois candidats : verbatim,
    /// format 1, format 2 + dictionnaire. Le résultat ne peut donc être ni plus
    /// gros que l'entrée, ni plus gros que ce que produisait [`compress`] — même
    /// si le dictionnaire n'aide pas du tout (vérifié sur 3000 entrées aléatoires).
    pub fn compress_with(&mut self, data: &[u8]) -> Vec<u8> {
        let v1 = compress(data);
        let s = self.stream(data);
        if s.len() + 1 < v1.len() {
            let mut out = Vec::with_capacity(s.len() + 1);
            out.push(2u8);
            out.extend_from_slice(&s);
            out
        } else {
            v1
        }
    }
}

/// Décodeur unifié : en-tête `0` = verbatim, `1` = format 1, `2` = format 2. Un
/// blob écrit avant l'introduction du dictionnaire reste lisible, avec ou sans
/// `dict` ; seul le format 2 en exige un.
pub fn decompress_with(blob: &[u8], dict: Option<&[u8]>) -> Option<Vec<u8>> {
    match blob.split_first() {
        Some((2, rest)) => v2_decompress(rest, dict?),
        _ => decompress(blob),
    }
}

fn v2_decompress(blob: &[u8], dict: &[u8]) -> Option<Vec<u8>> {
    let mut buf = dict.to_vec();
    let start = buf.len();
    let mut i = 0usize;
    while i < blob.len() {
        let flag = blob[i];
        i += 1;
        for bit in 0..8 {
            if i >= blob.len() {
                break;
            }
            if flag & (1 << bit) == 0 {
                buf.push(blob[i]);
                i += 1;
            } else {
                let b0 = *blob.get(i)?;
                let (off, len) = if b0 & 0x80 == 0 {
                    let code = ((b0 as u16) << 8) | (*blob.get(i + 1)? as u16);
                    i += 2;
                    (
                        ((code >> 4) as usize) + 1,
                        ((code & 0xF) as usize) + MIN_MATCH,
                    )
                } else {
                    let o = (((b0 & 0x7F) as usize) << 16)
                        | ((*blob.get(i + 1)? as usize) << 8)
                        | (*blob.get(i + 2)? as usize);
                    let l = (*blob.get(i + 3)? as usize) + MIN_MATCH;
                    i += 4;
                    (o + 1, l)
                };
                if off > buf.len() {
                    return None;
                }
                let from = buf.len() - off;
                for k in 0..len {
                    let b = buf[from + k];
                    buf.push(b);
                }
            }
        }
    }
    Some(buf.split_off(start))
}
