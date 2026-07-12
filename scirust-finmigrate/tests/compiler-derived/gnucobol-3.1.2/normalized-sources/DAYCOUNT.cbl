*>*****************************************************************
*> PROGRAM-ID.  DAYCOUNT
*> PURPOSE.     Accrued interest between two dates on the
*>              30/360 US (NASD bond-basis) day-count convention.
*>
*> Fourth migration unit. The arithmetic is trivial; the RISK is the
*> day-count itself. "30/360 US" is ambiguous in the wild:
*>   - the SIFMA/NASD BOND BASIS applies February end-of-month rules;
*>   - Excel DAYS360 (US) does NOT.
*> The two disagree by up to several days around Feb/31st month-ends
*> (e.g. 28-Feb -> 31-Aug is 180 days NASD but 183 Excel). Picking the
*> wrong one silently mis-accrues interest. This program implements the
*> NASD bond basis; the divergence from Excel is documented and
*> cross-checked in the sandbox. See cobol/SEMANTICS_DAY.md.
*>
*> The four adjustment rules are applied IN ORDER; the EOM flags are
*> read from the ORIGINAL dates before any adjustment. Rule 3 tests
*> "D1 = 30 OR 31" so it catches a D1 that rule 4 has not yet reduced.
*>*****************************************************************
IDENTIFICATION DIVISION.
PROGRAM-ID. DAYCOUNT.
DATA DIVISION.
WORKING-STORAGE SECTION.
01  WS-PRINCIPAL      PIC S9(9)V99 COMP-3.
01  WS-ANNUAL-RATE    PIC SV9(7)  COMP-3.
01  WS-DATE1.
    05  WS-Y1         PIC 9(4).
    05  WS-M1         PIC 9(2).
    05  WS-D1         PIC 9(2).
01  WS-DATE2.
    05  WS-Y2         PIC 9(4).
    05  WS-M2         PIC 9(2).
    05  WS-D2         PIC 9(2).
01  WS-AD1            PIC 9(2).
01  WS-AD2            PIC 9(2).
01  WS-DAYS           PIC S9(6)    COMP-3.
01  WS-INTEREST       PIC S9(9)V99 COMP-3.
01  WS-FLAGS.
    05  WS-FEB1       PIC X.
    05  WS-FEB2       PIC X.
PROCEDURE DIVISION.
0000-MAIN.
    PERFORM 2000-DAY-COUNT.
*>    One rounding event: interest for the accrual period.
    COMPUTE WS-INTEREST ROUNDED =
        WS-PRINCIPAL * WS-ANNUAL-RATE * WS-DAYS / 360.
    GOBACK.

2000-DAY-COUNT.
    MOVE WS-D1 TO WS-AD1.
    MOVE WS-D2 TO WS-AD2.
*>    EOM flags from the ORIGINAL dates.
    MOVE 'N' TO WS-FEB1 WS-FEB2.
    PERFORM 2100-FEB-FLAG-1.
    PERFORM 2100-FEB-FLAG-2.
*>    Rule 1: both Date1 and Date2 are last day of February.
    IF WS-FEB1 = 'Y' AND WS-FEB2 = 'Y'
        MOVE 30 TO WS-AD2
    END-IF.
*>    Rule 2: Date1 is last day of February.
    IF WS-FEB1 = 'Y'
        MOVE 30 TO WS-AD1
    END-IF.
*>    Rule 3: D2 = 31 and D1 is 30 or 31.
    IF WS-AD2 = 31 AND (WS-AD1 = 30 OR WS-AD1 = 31)
        MOVE 30 TO WS-AD2
    END-IF.
*>    Rule 4: D1 = 31.
    IF WS-AD1 = 31
        MOVE 30 TO WS-AD1
    END-IF.
    COMPUTE WS-DAYS = 360 * (WS-Y2 - WS-Y1)
                    +  30 * (WS-M2 - WS-M1)
                    +       (WS-AD2 - WS-AD1).

2100-FEB-FLAG-1.
*>    Set WS-FEB1 = 'Y' if Date1 is the last day of February.
    IF WS-M1 = 2
        IF (WS-D1 = 29)
            OR (WS-D1 = 28 AND NOT (FUNCTION MOD(WS-Y1,4) = 0
                AND (FUNCTION MOD(WS-Y1,100) NOT = 0
                     OR FUNCTION MOD(WS-Y1,400) = 0)))
            MOVE 'Y' TO WS-FEB1
        END-IF
    END-IF.

2100-FEB-FLAG-2.
    IF WS-M2 = 2
        IF (WS-D2 = 29)
            OR (WS-D2 = 28 AND NOT (FUNCTION MOD(WS-Y2,4) = 0
                AND (FUNCTION MOD(WS-Y2,100) NOT = 0
                     OR FUNCTION MOD(WS-Y2,400) = 0)))
            MOVE 'Y' TO WS-FEB2
        END-IF
    END-IF.
