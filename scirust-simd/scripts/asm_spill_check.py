#!/usr/bin/env python3
"""Analyse un fichier assembleur AArch64 émis par `--emit asm` et compte, pour
chaque boucle chaude hypercomplexe (symboles `scirust_probe_*`), le trafic de
pile.

Métrique principale : chargements/stockages de registres VECTORIELS (q/v/d)
référençant `sp` — les spills/reloads. Zéro ⇒ noyau register-résident.

Métrique secondaire (informative) : trafic de registres généraux (x/w) vers
`sp`, dû au passage d'arguments SIMD par valeur via l'ABI, hors pression
registre interne du calcul.

Usage : asm_spill_check.py <fichier.s>
Sortie : tableau lisible + bloc JSON pour comparaison machine avant/après.
"""

from __future__ import annotations

import json
import re
import sys

# Symboles sondes émis par src/hypercomplex/asm_probe.rs.
PROBES = [
    "scirust_probe_quat_mul",
    "scirust_probe_oct_mul",
    "scirust_probe_sed_mul",
]

# Un stockage/chargement mémoire AArch64 : (ld|st)(r|p) <reg>, ... [sp...]
# On distingue le premier registre : q/v/d = vectoriel, x/w = général.
MEM_RE = re.compile(
    r"""^\s*
        (?P<op>ld[rp]|st[rp])\s+          # ldr/ldp/str/stp
        (?P<reg>[a-z])\d+                 # premier registre (lettre de classe)
    """,
    re.VERBOSE,
)
# La référence [sp...] apparaît dans l'opérande adresse (avec ou sans offset).
SP_RE = re.compile(r"\[sp\b")

# Instructions utiles pour le contexte (débit du noyau).
FMA_RE = re.compile(r"^\s*fml[as]\b")
FMUL_RE = re.compile(r"^\s*fmul\b")
FADD_RE = re.compile(r"^\s*f(add|sub)\b")
SHUFFLE_RE = re.compile(r"^\s*(tbl|tbx|ext|rev\d+|zip[12]|uzp[12]|trn[12]|dup|ins|mov\s+v)\b")


LABEL_RE = re.compile(r"^(\.?L[A-Za-z0-9_$.]+):\s*$")
# Branches AArch64 : b, b.<cc>, cbz/cbnz, tbz/tbnz — dernier opérande = cible.
BRANCH_RE = re.compile(r"^\s*(b(?:\.[a-z]+)?|cbn?z|tbn?z)\b.*?(\.?L[A-Za-z0-9_$.]+)\s*$")


def extract_symbol_body(lines: list[str], symbol: str) -> list[str] | None:
    """Renvoie les lignes entre `symbol:` et le `.cfi_endproc` correspondant."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"{symbol}:":
            start = i + 1
            break
    if start is None:
        return None
    body = []
    for line in lines[start:]:
        if ".cfi_endproc" in line:
            break
        body.append(line)
    return body


def loop_mask(body: list[str]) -> list[bool]:
    """Marque chaque ligne du corps comme appartenant (ou non) à un corps de
    boucle, détecté par arête arrière : une branche vers un label situé PLUS
    HAUT délimite `[label, branche]`. Robuste aux boucles imbriquées (union)."""
    label_line = {}
    for i, line in enumerate(body):
        m = LABEL_RE.match(line.strip())
        if m:
            label_line[m.group(1)] = i
    in_loop = [False] * len(body)
    for i, line in enumerate(body):
        m = BRANCH_RE.match(line)
        if not m:
            continue
        target = m.group(2)
        j = label_line.get(target)
        if j is not None and j <= i:  # arête arrière
            for k in range(j, i + 1):
                in_loop[k] = True
    return in_loop


# Registres vectoriels callee-saved par l'ABI AArch64 : v8–v15 (seuls les
# 64 bits bas, d8–d15, sont préservés). Leur sauvegarde/restauration en
# prologue/épilogue est un coût FIXE par appel, PAS un spill de boucle chaude.
ABI_CALLEE_SAVED = re.compile(r"^(?:d|q|v)(?:8|9|1[0-5])$")

# Tous les registres nommés dans une instruction mémoire (ldp/stp en ont deux).
REG_TOKEN_RE = re.compile(r"\b([dqv])(\d+)\b")


def analyze_body(body: list[str]) -> dict:
    in_loop = loop_mask(body)
    abi = 0
    hot_store = hot_load = 0       # spills vectoriels DANS la boucle chaude
    setup_spills = 0               # spills vectoriels hors boucle (prologue/setup)
    gp_store = gp_load = 0
    fma = fmul = fadd = shuffle = 0
    hot_spill_lines: list[str] = []

    for i, line in enumerate(body):
        m = MEM_RE.match(line)
        if m and SP_RE.search(line):
            op = m.group("op")
            reg_class = m.group("reg")
            is_store = op.startswith("st")
            if reg_class in ("q", "v", "d"):
                regs = [f"{c}{n}" for c, n in REG_TOKEN_RE.findall(line)]
                is_abi = bool(regs) and all(ABI_CALLEE_SAVED.match(r) for r in regs)
                if is_abi:
                    # Sauvegarde ABI de v8–v15 : coût fixe par appel, pas un
                    # spill de boucle. (Sa présence trahit tout de même une
                    # forte pression, mais elle ne se paie pas par itération.)
                    abi += 1
                elif in_loop[i]:
                    if is_store:
                        hot_store += 1
                    else:
                        hot_load += 1
                    hot_spill_lines.append(line.strip())
                else:
                    setup_spills += 1
            elif reg_class in ("x", "w"):
                if is_store:
                    gp_store += 1
                else:
                    gp_load += 1

        if FMA_RE.match(line):
            fma += 1
        elif FMUL_RE.match(line):
            fmul += 1
        elif FADD_RE.match(line):
            fadd += 1
        if SHUFFLE_RE.match(line):
            shuffle += 1

    return {
        # Métrique principale : spills vectoriels DANS la boucle chaude
        # (le coût qui se paie à chaque itération = la vraie pression registre).
        "hot_spills_store": hot_store,
        "hot_spills_load": hot_load,
        "hot_spills_total": hot_store + hot_load,
        # Spills vectoriels de setup (hors boucle, une fois par appel).
        "setup_spills": setup_spills,
        # Sauvegardes ABI callee-saved v8–v15 (prologue/épilogue).
        "abi_saves": abi,
        # Trafic GP sp-relatif (passage d'args SIMD par valeur), informatif.
        "gp_sp_store": gp_store,
        "gp_sp_load": gp_load,
        "fma": fma,
        "fmul": fmul,
        "faddsub": fadd,
        "shuffle": shuffle,
        "hot_spill_lines": hot_spill_lines,
    }


def main() -> int:
    args = sys.argv[1:]
    matrix_row = False
    if args and args[0] == "--matrix-row":
        matrix_row = True
        args = args[1:]
    if len(args) != 1:
        print("usage: asm_spill_check.py [--matrix-row] <fichier.s>", file=sys.stderr)
        return 2
    with open(args[0], encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    # Mode compact : une ligne « quat | oct | sed » pour la matrice multi-CPU.
    if matrix_row:
        cells = []
        for sym in PROBES:
            body = extract_symbol_body(lines, sym)
            if body is None:
                cells.append("ABSENT")
            else:
                cells.append(str(analyze_body(body)["hot_spills_total"]))
        print(f"{cells[0]:<9} | {cells[1]:<9} | {cells[2]:<9}")
        return 0

    results: dict[str, dict] = {}
    header = (
        f"{'kernel':<24} {'HOT_SPILL':>9} {'(st/ld)':>9} "
        f"{'setup':>5} {'abi':>4} {'fma':>5} {'fmul':>5} {'shuf':>5}"
    )
    print(header)
    print("-" * len(header))
    for sym in PROBES:
        body = extract_symbol_body(lines, sym)
        if body is None:
            print(f"{sym:<24} {'ABSENT':>9}")
            continue
        a = analyze_body(body)
        results[sym] = a
        stld = f"{a['hot_spills_store']}/{a['hot_spills_load']}"
        print(
            f"{sym:<24} {a['hot_spills_total']:>9} {stld:>9} "
            f"{a['setup_spills']:>5} {a['abi_saves']:>4} "
            f"{a['fma']:>5} {a['fmul']:>5} {a['shuffle']:>5}"
        )

    # Détail des vrais spills de boucle chaude (les coupables de la pression).
    for sym in PROBES:
        a = results.get(sym)
        if a and a["hot_spill_lines"]:
            print(f"\n--- spills de boucle chaude dans {sym} ---")
            for ln in a["hot_spill_lines"]:
                print(f"    {ln}")

    # Bloc JSON compact pour diff machine avant/après.
    summary = {
        sym: {
            "hot_spills_total": results[sym]["hot_spills_total"],
            "hot_spills_store": results[sym]["hot_spills_store"],
            "hot_spills_load": results[sym]["hot_spills_load"],
            "setup_spills": results[sym]["setup_spills"],
            "abi_saves": results[sym]["abi_saves"],
            "fma": results[sym]["fma"],
            "shuffle": results[sym]["shuffle"],
        }
        for sym in PROBES
        if sym in results
    }
    print("\nJSON " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
