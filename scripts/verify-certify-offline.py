#!/usr/bin/env python3
# =============================================================================
# SciRust — vérification hors ligne des entrées NON CERTIFIÉES de la campagne
# d'arrondi correct (proof_portable_f32 --certify).
# -----------------------------------------------------------------------------
# Pour chaque fonction, lit `proof-certify-<nom>.txt` (bit patterns f32 des
# entrées que le certificat d'intervalle n'a pas pu prouver), obtient la sortie
# du code expédié via `proof_portable_f32 --eval`, calcule la référence en
# précision arbitraire (Decimal, 60 chiffres — implémentations indépendantes :
# exp/ln natifs de Decimal, séries pour tanh/sigmoid/erf, réduction par π de
# Chudnovsky pour sin/cos), détermine le f32 CORRECTEMENT ARRONDI par
# comparaison EXACTE aux milieux (fractions rationnelles, pas de double
# arrondi), et classe chaque entrée : correctement arrondie / fidèle (1 ulp).
#
# Usage :  python3 scripts/verify-certify-offline.py
#          (après `cargo run --release -p scirust-core --bin proof_portable_f32
#           -- --certify`, depuis la racine du dépôt)
# =============================================================================
import struct
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 60

# --- π en 80 chiffres (Chudnovsky, comme pour les bits de 2/π du volet 113) --
def pi_chudnovsky():
    from decimal import getcontext as g
    old = g().prec
    g().prec = 100
    C = 426880 * Decimal(10005).sqrt()
    K, M, X, L = 6, 1, 1, 13591409
    S = Decimal(13591409)
    for i in range(1, 10):
        M = M * (K**3 - 16 * K) // (i**3)
        L += 545140134
        X *= -262537412640768000
        S += Decimal(M * L) / X
        K += 12
    pi = C / S
    g().prec = old
    return pi

PI = pi_chudnovsky()

def d_exp(x):
    return x.exp()

def d_ln(x):
    return x.ln()

def d_tanh(x):
    t = (-2 * abs(x)).exp()
    r = (1 - t) / (1 + t)
    return r if x >= 0 else -r

def d_sigmoid(x):
    if x >= 0:
        return 1 / (1 + (-x).exp())
    t = x.exp()
    return t / (1 + t)

def d_erf(x):
    s, t, n = Decimal(0), x, 0
    while True:
        c = t / (2 * n + 1)
        s += c
        if abs(c) < Decimal(10) ** -58 * (abs(s) + Decimal(10) ** -30):
            break
        n += 1
        t = t * (-x * x) / n
    two_over_sqrtpi = 2 / PI.sqrt()
    return s * two_over_sqrtpi

def d_sincos(x, want_sin):
    # réduction : r = x mod 2π (précision conservée : |x| ≤ 3,4e38, π sur
    # 100 chiffres ⇒ r juste à ~10⁻⁶⁰ près)
    two_pi = 2 * PI
    k = (x / two_pi).to_integral_value(rounding="ROUND_FLOOR")
    r = x - k * two_pi
    # Taylor depuis r (|r| < 2π : converge en ~40 termes)
    s, c = Decimal(0), Decimal(0)
    term = Decimal(1)
    n = 0
    while True:
        if n % 4 == 0:
            c += term
        elif n % 4 == 1:
            s += term
        elif n % 4 == 2:
            c -= term
        else:
            s -= term
        n += 1
        term = term * r / n
        if abs(term) < Decimal(10) ** -58:
            break
    return s if want_sin else c

FUNCS = {
    "exp": d_exp,
    "ln": d_ln,
    "tanh": d_tanh,
    "sigmoid": d_sigmoid,
    "sin": lambda x: d_sincos(x, True),
    "cos": lambda x: d_sincos(x, False),
    "erf": d_erf,
}

def bits_to_f32(b):
    return struct.unpack("f", struct.pack("I", b))[0]

def f32_to_bits(x):
    return struct.unpack("I", struct.pack("f", x))[0]

def next_up(x):
    b = f32_to_bits(x)
    if b == 0x80000000:
        return bits_to_f32(1)
    return bits_to_f32(b + 1 if x >= 0 else b - 1)

def next_down(x):
    return -next_up(-x)

def correctly_rounded_f32(v: Decimal) -> float:
    """Le f32 le plus proche de v — milieux comparés EXACTEMENT (Fraction)."""
    vf = Fraction(v)
    c = bits_to_f32(f32_to_bits(float(v)))  # candidat (à ≤ 1 ulp du bon)
    for _ in range(4):
        hi, lo = next_up(c), next_down(c)
        mid_hi = (Fraction(c) + Fraction(hi)) / 2
        mid_lo = (Fraction(lo) + Fraction(c)) / 2
        if vf > mid_hi:
            c = hi
            continue
        if vf < mid_lo:
            c = lo
            continue
        if vf == mid_hi:  # égalité exacte : pair gagne
            return c if f32_to_bits(c) % 2 == 0 else hi
        if vf == mid_lo:
            return c if f32_to_bits(c) % 2 == 0 else lo
        return c
    raise RuntimeError("non convergé")

def main():
    binary = Path("target/release/proof_portable_f32")
    if not binary.exists():
        sys.exit("compiler d'abord : cargo build --release -p scirust-core "
                 "--bin proof_portable_f32")
    grand_total, grand_cr, grand_faithful = 0, 0, 0
    for name in FUNCS:
        path = Path(f"proof-certify-{name}.txt")
        if not path.exists():
            print(f"{name}: pas de fichier (0 entrée non certifiée ?)")
            continue
        out = subprocess.run(
            [str(binary), "--eval", name, str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        cr, faithful, worse = 0, 0, []
        for line in out.strip().splitlines():
            in_hex, out_hex = line.split()
            x = Decimal(repr(bits_to_f32(int(in_hex, 16))))
            got_bits = int(out_hex, 16)
            ref = correctly_rounded_f32(FUNCS[name](x))
            ref_bits = f32_to_bits(ref)
            if got_bits == ref_bits:
                cr += 1
            elif abs(got_bits - ref_bits) == 1:
                faithful += 1
            else:
                worse.append(in_hex)
        total = cr + faithful + len(worse)
        grand_total += total
        grand_cr += cr
        grand_faithful += faithful
        print(f"{name}: {total} vérifiées → {cr} correctement arrondies, "
              f"{faithful} fidèles (1 ulp), {len(worse)} pires{worse[:8]}")
    print(f"\nTOTAL : {grand_total} entrées non certifiées vérifiées → "
          f"{grand_cr} correctement arrondies, {grand_faithful} fidèles (1 ulp)")

if __name__ == "__main__":
    main()
