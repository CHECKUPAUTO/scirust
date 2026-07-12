*>*****************************************************************
*> PROGRAM-ID.  AMORTSCH
*> PURPOSE.     Fixed-payment loan amortization schedule.
*>
*> Second migration unit. Where INTACCR was a single arithmetic
*> store, AMORTSCH carries STATE across periods, so it adds two
*> failure modes that a one-shot routine cannot exhibit:
*>
*>   A. ACCUMULATED ROUNDING DRIFT. Each period's interest is
*>      independently ROUNDED to the cent (NEAREST-AWAY-FROM-ZERO).
*>      Over N periods those half-cent roundings accumulate; a port
*>      that rounds even slightly differently drifts by whole pennies
*>      by the final period.
*>
*>   B. FINAL-PAYMENT RECONCILIATION. Because of (A) the scheduled
*>      payment cannot close the balance to exactly zero. The legacy
*>      rule: on the LAST period, or as soon as the principal portion
*>      would meet-or-exceed the remaining balance, set the principal
*>      portion equal to the whole remaining balance and let the
*>      actual payment absorb the difference. The ending balance MUST
*>      be exactly 0.00. Getting the >= boundary or the last-period
*>      test wrong leaves a residual penny (an audit finding).
*>
*> All arithmetic is fixed-point decimal (COMP-3). No floating point,
*> no exponentiation — the scheduled payment is an INPUT, not derived
*> from the annuity formula (which would drag in COBOL's float-valued
*> ** operator). See cobol/SEMANTICS_AMORT.md for the exact contract.
*>*****************************************************************
IDENTIFICATION DIVISION.
PROGRAM-ID. AMORTSCH-RUN.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  WS-ORIG-PRINCIPAL  PIC S9(9)V99 COMP-3.
01  WS-BALANCE         PIC S9(9)V99 COMP-3.
01  WS-MONTHLY-RATE    PIC SV9(7)  COMP-3.
01  WS-PAYMENT         PIC S9(7)V99 COMP-3.
01  WS-NUM-PERIODS     PIC 9(3)     COMP-3.
01  WS-INTEREST        PIC S9(9)V99 COMP-3.
01  WS-PRINC-PORTION   PIC S9(9)V99 COMP-3.
01  WS-ACTUAL-PAYMENT  PIC S9(9)V99 COMP-3.
01  WS-PERIOD          PIC 9(3)     COMP-3.
PROCEDURE DIVISION.
0000-BUILD-SCHEDULE.
    ACCEPT WS-ORIG-PRINCIPAL
    ACCEPT WS-MONTHLY-RATE
    ACCEPT WS-PAYMENT
    ACCEPT WS-NUM-PERIODS
    MOVE WS-ORIG-PRINCIPAL TO WS-BALANCE.
    PERFORM VARYING WS-PERIOD FROM 1 BY 1
            UNTIL WS-PERIOD > WS-NUM-PERIODS
               OR WS-BALANCE = ZERO
*>        One rounding event: interest on the CURRENT balance,
*>        NEAREST-AWAY-FROM-ZERO into a 2 dp field.
        COMPUTE WS-INTEREST ROUNDED = WS-BALANCE * WS-MONTHLY-RATE
*>        Principal portion of the scheduled payment (exact, 2 dp).
        COMPUTE WS-PRINC-PORTION = WS-PAYMENT - WS-INTEREST
*>        Final-payment reconciliation (rule B). Note the >= boundary
*>        and the last-period test — both are load-bearing.
        IF WS-PERIOD = WS-NUM-PERIODS
           OR WS-PRINC-PORTION >= WS-BALANCE
            MOVE WS-BALANCE TO WS-PRINC-PORTION
            COMPUTE WS-ACTUAL-PAYMENT =
                WS-INTEREST + WS-PRINC-PORTION
        ELSE
            MOVE WS-PAYMENT TO WS-ACTUAL-PAYMENT
        END-IF
        SUBTRACT WS-PRINC-PORTION FROM WS-BALANCE
        DISPLAY WS-PERIOD
        DISPLAY WS-INTEREST
        DISPLAY WS-PRINC-PORTION
        DISPLAY WS-ACTUAL-PAYMENT
        DISPLAY WS-BALANCE
*>        (emit: WS-PERIOD, WS-INTEREST, WS-PRINC-PORTION,
*>                WS-ACTUAL-PAYMENT, WS-BALANCE)
    END-PERFORM.
    GOBACK.
