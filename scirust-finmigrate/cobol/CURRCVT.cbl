      ******************************************************************
      * PROGRAM-ID.  CURRCVT
      * PURPOSE.     National-to-national euro conversion by triangulation,
      *              per Council Regulation (EC) No 1103/97.
      *
      * Sixth migration unit. It reproduces a LEGALLY DEFINED algorithm whose
      * rounding steps are the whole point:
      *
      *   1. Conversion rates are "1 euro = X national currency" quoted to
      *      SIX SIGNIFICANT FIGURES and MUST NOT be rounded or truncated
      *      further when converting.
      *   2. Converting one national currency to another MUST go THROUGH the
      *      euro (triangulation). A direct/implicit cross-rate is lawful only
      *      if it yields the identical triangulated result — it often does not.
      *   3. The intermediate euro amount is rounded to NOT LESS THAN three
      *      decimal places (this unit uses exactly 3).
      *   4. The final amount is rounded to the TARGET currency's minor unit
      *      — 2 dp for most, 0 dp for ITL / ESP — NEAREST-AWAY-FROM-ZERO.
      *
      * Scope: both source and target are NATIONAL currencies (neither is the
      * euro); that is the case the triangulation rule governs.
      *
      * Gap-R (variable target minor unit) is implemented here rather than
      * hard-coded: WS-MINOR-UNIT carries the target currency's minor unit
      * (from the currency master — 0 for ITL/ESP, 2 otherwise) and the final
      * ROUNDED COMPUTE targets a result field of the matching scale. The
      * caller reads WS-RESULT-0 when WS-MINOR-UNIT = 0, otherwise WS-RESULT-2.
      * (Audit trail 2026-07-12: this closes the Gap-R divergence the GnuCOBOL
      * compiler evidence exposed, where a fixed 2-dp result field mis-rounded
      * every lira/peseta amount by a minor unit.)
      ******************************************************************
       IDENTIFICATION DIVISION.
       PROGRAM-ID. CURRCVT.
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
      *    Step 1: source national amount -> euro, ROUNDED to 3 dp
      *    (intermediate; never fewer than 3 decimals).
           COMPUTE WS-EURO ROUNDED = WS-AMOUNT / WS-RATE-FROM.
      *    Step 2: euro -> target national amount, ROUNDED to the target's
      *    minor unit (0 dp for ITL/ESP, else 2 dp) — Gap-R.
           IF WS-MINOR-UNIT = 0
               COMPUTE WS-RESULT-0 ROUNDED = WS-EURO * WS-RATE-TO
           ELSE
               COMPUTE WS-RESULT-2 ROUNDED = WS-EURO * WS-RATE-TO
           END-IF.
           GOBACK.
