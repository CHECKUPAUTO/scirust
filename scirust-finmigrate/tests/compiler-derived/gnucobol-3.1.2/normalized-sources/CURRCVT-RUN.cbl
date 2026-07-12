*>*****************************************************************
*> PROGRAM-ID.  CURRCVT
*> PURPOSE.     National-to-national euro conversion by triangulation,
*>              per Council Regulation (EC) No 1103/97.
*>
*> Sixth migration unit. It reproduces a LEGALLY DEFINED algorithm whose
*> rounding steps are the whole point:
*>
*>   1. Conversion rates are "1 euro = X national currency" quoted to
*>      SIX SIGNIFICANT FIGURES and MUST NOT be rounded or truncated
*>      further when converting.
*>   2. Converting one national currency to another MUST go THROUGH the
*>      euro (triangulation). A direct/implicit cross-rate is lawful only
*>      if it yields the identical triangulated result — it often does not.
*>   3. The intermediate euro amount is rounded to NOT LESS THAN three
*>      decimal places (this unit uses exactly 3).
*>   4. The final amount is rounded to the TARGET currency's minor unit
*>      — 2 dp for most, 0 dp for ITL / ESP — NEAREST-AWAY-FROM-ZERO.
*>
*> Scope: both source and target are NATIONAL currencies (neither is the
*> euro); that is the case the triangulation rule governs.
*>
*> Gap-R (variable target minor unit) is implemented, not hard-coded: the
*> wrapper ACCEPTs WS-MINOR-UNIT (the target currency's minor unit — 0 for
*> ITL/ESP, 2 otherwise) after the two rates, then rounds into a result
*> field of the matching scale and DISPLAYs it. A 0 dp target DISPLAYs
*> WS-RESULT-0 (no decimals); a 2 dp target DISPLAYs WS-RESULT-2. This is
*> the only change from the pre-2026-07-12 wrapper, which stored every
*> result in a fixed 2-dp field and so diverged from the model by a minor
*> unit on lira/peseta targets (see RESULTS.md — Gap-R reconciliation).
*>*****************************************************************
IDENTIFICATION DIVISION.
PROGRAM-ID. CURRCVT-RUN.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  WS-AMOUNT      PIC S9(11)V99  COMP-3.
01  WS-RATE-FROM   PIC S9(5)V9(6) COMP-3.
01  WS-RATE-TO     PIC S9(5)V9(6) COMP-3.
01  WS-MINOR-UNIT  PIC 9.
01  WS-EURO        PIC S9(13)V999 COMP-3.
01  WS-RESULT-0    PIC S9(13)     COMP-3.
01  WS-RESULT-2    PIC S9(11)V99  COMP-3.
PROCEDURE DIVISION.
0000-CONVERT.
    ACCEPT WS-AMOUNT
    ACCEPT WS-RATE-FROM
    ACCEPT WS-RATE-TO
    ACCEPT WS-MINOR-UNIT
*>    Step 1: source national amount -> euro, ROUNDED to 3 dp
*>    (intermediate; never fewer than 3 decimals).
    COMPUTE WS-EURO ROUNDED = WS-AMOUNT / WS-RATE-FROM.
*>    Step 2: euro -> target national amount, ROUNDED to the target's
*>    minor unit (0 dp for ITL/ESP, else 2 dp) — Gap-R.
    IF WS-MINOR-UNIT = 0
        COMPUTE WS-RESULT-0 ROUNDED = WS-EURO * WS-RATE-TO
        DISPLAY WS-EURO
        DISPLAY WS-RESULT-0
    ELSE
        COMPUTE WS-RESULT-2 ROUNDED = WS-EURO * WS-RATE-TO
        DISPLAY WS-EURO
        DISPLAY WS-RESULT-2
    END-IF
    GOBACK.
