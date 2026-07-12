      ******************************************************************
      * PROGRAM-ID.  BRKTCALC
      * PURPOSE.     Progressive (marginal) bracketed tax on a base amount.
      *
      * Fifth migration unit. It introduces a COBOL pattern the earlier
      * units did not exercise: a TABLE (OCCURS) of tax brackets walked by
      * PERFORM VARYING, with MARGINAL arithmetic. The failure modes are
      * distinct from plain COMPUTE units:
      *
      *   - MARGINAL, not flat: each bracket's rate applies only to the
      *     portion of the base that falls WITHIN that bracket, never to
      *     the whole base. Applying the top bracket's rate to the entire
      *     base (a very common porting mistake) massively over-taxes.
      *   - BOUNDARY inclusivity: bracket i covers (LOWER(i) .. LOWER(i+1)].
      *     A base exactly on a threshold must fill the lower bracket and
      *     leave the next one empty.
      *   - SINGLE rounding event: the marginal amounts are accumulated at
      *     full precision and the TOTAL is ROUNDED once (NEAREST-AWAY-
      *     FROM-ZERO). Rounding each bracket then summing would drift.
      *
      * The bracket table below is a simplified graduated schedule; the
      * exact numbers are the migration contract (cobol/SEMANTICS_BRKT.md),
      * not tax advice.
      ******************************************************************
       IDENTIFICATION DIVISION.
       PROGRAM-ID. BRKTCALC.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-BASE            PIC S9(9)V99 COMP-3.
       01  WS-TAX             PIC S9(9)V99 COMP-3.
      *    Bracket table: LOWER threshold (inclusive floor) + marginal RATE.
      *    The final bracket has an open top (LOWER = 999999999.99 sentinel
      *    is NOT used; the loop treats the last bracket as unbounded).
       01  WS-BRACKETS.
           05  WS-BRK OCCURS 5 TIMES.
               10  WS-LOWER   PIC S9(9)V99 COMP-3.
               10  WS-RATE    PIC S V9(5)  COMP-3.
       01  WS-IDX             PIC 9(2)     COMP-3.
       01  WS-UPPER           PIC S9(9)V99 COMP-3.
       01  WS-PORTION         PIC S9(9)V99 COMP-3.
       01  WS-ACCUM           PIC S9(13)V9(7) COMP-3.
       PROCEDURE DIVISION.
       0000-MAIN.
           PERFORM 1000-LOAD-TABLE.
           MOVE 0 TO WS-ACCUM.
           PERFORM VARYING WS-IDX FROM 1 BY 1 UNTIL WS-IDX > 5
      *        Upper edge of this bracket = lower of the next, or the base
      *        itself for the last (unbounded) bracket.
               IF WS-IDX < 5
                   MOVE WS-LOWER (WS-IDX + 1) TO WS-UPPER
               ELSE
                   MOVE WS-BASE TO WS-UPPER
               END-IF
      *        Clamp the upper edge to the base (bracket may be partial/empty).
               IF WS-UPPER > WS-BASE
                   MOVE WS-BASE TO WS-UPPER
               END-IF
      *        Portion of the base within this bracket (never negative).
               IF WS-UPPER > WS-LOWER (WS-IDX)
                   COMPUTE WS-PORTION = WS-UPPER - WS-LOWER (WS-IDX)
               ELSE
                   MOVE 0 TO WS-PORTION
               END-IF
      *        Accumulate the marginal tax at FULL precision (no rounding).
               COMPUTE WS-ACCUM =
                   WS-ACCUM + (WS-PORTION * WS-RATE (WS-IDX))
           END-PERFORM.
      *    Single rounding event into the 2 dp result.
           COMPUTE WS-TAX ROUNDED = WS-ACCUM.
           GOBACK.

       1000-LOAD-TABLE.
      *    0 .. 10,000       @  0%
      *    10,000 .. 40,000  @ 10%
      *    40,000 .. 85,000  @ 22%
      *    85,000 .. 165,000 @ 24%
      *    165,000 ..        @ 32%
           MOVE 0.00        TO WS-LOWER (1)  MOVE 0.00000 TO WS-RATE (1).
           MOVE 10000.00    TO WS-LOWER (2)  MOVE 0.10000 TO WS-RATE (2).
           MOVE 40000.00    TO WS-LOWER (3)  MOVE 0.22000 TO WS-RATE (3).
           MOVE 85000.00    TO WS-LOWER (4)  MOVE 0.24000 TO WS-RATE (4).
           MOVE 165000.00   TO WS-LOWER (5)  MOVE 0.32000 TO WS-RATE (5).
