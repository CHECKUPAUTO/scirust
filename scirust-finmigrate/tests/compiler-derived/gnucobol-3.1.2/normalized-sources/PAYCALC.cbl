*>*****************************************************************
*> PROGRAM-ID.  PAYCALC
*> PURPOSE.     Level (annuity) payment for a fully-amortizing loan.
*>
*> Third migration unit. It computes the fixed monthly payment that
*> AMORTSCH consumes as an input — closing the loop — and it is the
*> unit that confronts COBOL's exponentiation semantics head-on.
*>
*> The textbook annuity formula uses a NEGATIVE exponent:
*>     payment = P*i / (1 - (1+i)**(-n))
*> In COBOL a fractional OR negative exponent forces the whole
*> expression into LONG floating-point (operands promoted to COMP-2),
*> which would drag IEEE-754 error into the money path and violate the
*> project's no-floating-point mandate.
*>
*> This program deliberately uses the algebraically-equivalent form
*> with a POSITIVE INTEGER exponent:
*>     f = (1+i)**n            <- integer power: a succession of
*>                                fixed-point multiplications (NOT float)
*>     payment = P*i*f / (f-1)
*> so every operation stays fixed-point decimal. See
*> cobol/SEMANTICS_PAY.md for the exact contract and the citations.
*>
*> WS-FACTOR is stored at a FIXED 9-dp scale: a single rounding event
*> at a scale far coarser than any compiler-specific intermediate cap,
*> so the result is insensitive to ARITH(COMPAT|EXTEND).
*>*****************************************************************
IDENTIFICATION DIVISION.
PROGRAM-ID. PAYCALC.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  WS-PRINCIPAL       PIC S9(9)V99    COMP-3.
01  WS-RATE            PIC SV9(7)     COMP-3.
01  WS-NUM-PERIODS     PIC 9(3)        COMP-3.
01  WS-FACTOR          PIC S9(5)V9(9)  COMP-3.
01  WS-PAYMENT         PIC S9(7)V99    COMP-3.
PROCEDURE DIVISION.
0000-COMPUTE-PAYMENT.
    IF WS-RATE = ZERO
*>        Zero-rate loan: straight-line principal / periods. The
*>        annuity form would divide by (f-1) = 0, so special-case it.
        COMPUTE WS-FACTOR ROUNDED = 1
        COMPUTE WS-PAYMENT ROUNDED =
            WS-PRINCIPAL / WS-NUM-PERIODS
    ELSE
*>        Integer power => succession of fixed-point multiplications.
        COMPUTE WS-FACTOR ROUNDED = (1 + WS-RATE) ** WS-NUM-PERIODS
        COMPUTE WS-PAYMENT ROUNDED =
            (WS-PRINCIPAL * WS-RATE * WS-FACTOR) / (WS-FACTOR - 1)
    END-IF.
    GOBACK.
