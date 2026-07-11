      ******************************************************************
      * PROGRAM-ID.  INTACCR
      * PURPOSE.     Monthly interest accrual for a savings account.
      *
      * This is the LEGACY SOURCE OF TRUTH for the migration. It is a
      * faithful, self-contained reproduction of the arithmetic core of
      * the legacy batch posting routine. It deliberately exercises the
      * four "hidden dependency" classes called out in audit_report.md:
      *
      *   1. COMP-3 (packed decimal) storage of every monetary field.
      *   2. Fixed implied scale (V99 = 2 dp, SV9(5) = 5 dp) that is
      *      enforced on every STORE, not just on print.
      *   3. ROUNDED on the COMPUTE, using the COBOL default rounding
      *      mode NEAREST-AWAY-FROM-ZERO (round-half-up, sign-aware).
      *   4. A single rounding event at the final store of WS-MONTHLY-INT;
      *      the multiply/divide chain is carried at high intermediate
      *      precision and is NOT rounded per operation.
      *
      * The un-ROUNDED companion field WS-MONTHLY-TRUNC captures the
      * COBOL default (truncation toward zero) so the sandbox can prove
      * that the Rust port reproduces BOTH rounding disciplines exactly.
      ******************************************************************
       IDENTIFICATION DIVISION.
       PROGRAM-ID. INTACCR.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
      *    Principal balance:  9 integer digits + 2 decimal, signed.
       01  WS-PRINCIPAL       PIC S9(9)V99   COMP-3.
      *    Annual nominal rate as a pure fraction, 5 decimal places.
      *    e.g. 0.03500 represents 3.500 % per annum.
       01  WS-ANNUAL-RATE     PIC S V9(5)    COMP-3.
      *    Posted monthly interest (ROUNDED, 2 dp).
       01  WS-MONTHLY-INT     PIC S9(9)V99   COMP-3.
      *    Monthly interest under the COBOL default (truncated, 2 dp).
       01  WS-MONTHLY-TRUNC   PIC S9(9)V99   COMP-3.
      *    Resulting balance after posting the ROUNDED interest.
       01  WS-NEW-BALANCE     PIC S9(9)V99   COMP-3.
       PROCEDURE DIVISION.
       0000-ACCRUE.
      *    Single rounding event: the whole expression is evaluated at
      *    high intermediate precision, then rounded once into a 2 dp
      *    field using NEAREST-AWAY-FROM-ZERO.
           COMPUTE WS-MONTHLY-INT ROUNDED =
               WS-PRINCIPAL * WS-ANNUAL-RATE / 12.
      *    Default discipline (no ROUNDED): truncate toward zero.
           COMPUTE WS-MONTHLY-TRUNC =
               WS-PRINCIPAL * WS-ANNUAL-RATE / 12.
      *    Post the ROUNDED interest onto the principal.
           ADD WS-MONTHLY-INT TO WS-PRINCIPAL GIVING WS-NEW-BALANCE.
           GOBACK.
