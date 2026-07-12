# SciRust real COBOL corpus — GnuCOBOL syntax baselines

Source dataset:
- X-COBOL: A Dataset of COBOL Repositories
- Zenodo record: 7968845
- Original archive MD5: 1a05a95e5320bde93fadcecea4c1926a

Selected public COBOL sources:
- etalab/taxe-fonciere
- tylerbro93/COBOL-SALES-INVENTORY-REPORT-DEMONSTRATION

Compiler:
- GnuCOBOL 3.1.2.0

Validated source programs:
- CTXTA3B.cob: syntax accepted, 0 warnings, 0 errors
- CTXTA3N.cob: syntax accepted, 0 warnings, 0 errors
- EFITA3B8.cob: syntax accepted, 7 warnings, 0 errors
- EFITA3N8.cob: syntax accepted, 8 warnings, 0 errors
- project2.cbl: syntax accepted, 0 warnings, 0 errors

Validation command:

    cobc -fsyntax-only -Wall -I copybooks src/<program>

Important:
- These are real public COBOL sources taken from existing repositories.
- The current evidence establishes parser/compiler acceptance baselines.
- It does not yet establish executable runtime-output baselines because
  representative production input files and runtime environments are absent.
- Compiler warnings are preserved verbatim and must not be silently removed.
