# Quotation (draft for estimator review), sheet sheet-10833-scan.png

Prices are from the prototype's illustrative catalogue, not a supplier price list. Every line names the evidence it rests on: S = symbol box, T = text box, clause numbers from the specification, R = rule.

## Board as read
- Busbar rating 1600 A; fault level 80 kA; Form 4 Type 4; IP41; default 4-pole; 2 outgoing ways in 2 sections.

## Bill of materials

| # | Part | Description | Qty | Unit £ | Total £ | Conf | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | METER-CT 1600/5A | BAR A 1600A Cu 80 kA/1s | Panel meter (CT 1600/5A | BAR A 1600A Cu 80 kA/1s) | 1 | 260.00 | 260.00 | 0.27 | S5 S5t1 S5t2 R-METERING |
| 2 | METER-CT1600/5A | 1600 A Cu 80 kA/1s BARB | Panel meter (CT1600/5A | 1600 A Cu 80 kA/1s BARB) | 1 | 260.00 | 260.00 | 0.52 | S6 S6t1 S6t2 R-METERING |
| 3 | CT-SET | Current transformers 1600/5 A, set of three, class 1 | 1 | 280.00 | 280.00 | 0.29 | S7 S7t2 R-METERING |
| 4 | CT-SET | Current transformers 1600/5 A, set of three, class 1 | 1 | 280.00 | 280.00 | 0.41 | S8 S8t2 R-METERING |
| 5 | BSS-1600-4P | Bus-section switch 1600 A 4-pole, 80 kA | 1 | 4,340.00 | 4,340.00 | 0.24 | S9 S9t2 R-BUS-SECTION |
| 6 | FU-KIT-BS | Bus-section functional unit and interlocking | 1 | 900.00 | 900.00 | 0.24 | S9 S9t2 R-FU-KIT |
| 7 | FSW-400-3P | Fuse-switch-disconnector 400 A 3-pole; feeds MCC-7chillers | 1 | 780.00 | 780.00 | 0.39 | S19 S19t4 S19t3 R-FSW-SELECT |
| 8 | FUSE-400 | Fuse links 400 A, one per pole | 3 | 46.00 | 138.00 | 0.39 | S19 S19t4 S19t3 R-FUSE-LINKS |
| 9 | FU-KIT-FUSE_SWITCH | Outgoing functional unit: compartment door, terminals, labels | 1 | 380.00 | 380.00 | 0.39 | S19 S19t4 S19t3 R-FU-KIT |
| 10 | FSW-800-4P | Fuse-switch-disconnector 800 A 4-pole; feeds kitchen | 1 | 1,854.40 | 1,854.40 | 0.31 | S21 S21t4 S21t3 R-FSW-SELECT |
| 11 | FUSE-800 | Fuse links 800 A, one per pole | 4 | 94.00 | 376.00 | 0.31 | S21 S21t4 S21t3 R-FUSE-LINKS |
| 12 | FU-KIT-FUSE_SWITCH | Outgoing functional unit: compartment door, terminals, labels | 1 | 380.00 | 380.00 | 0.31 | S21 S21t4 S21t3 R-FU-KIT |
| 13 | BUSBAR-CU-1600 | Copper busbar set 1600 A (4-bar), 2 sections, about 58 kg | 1 | 662.40 | 662.40 | 0.9 | T12 T12 T60 R-BUSBAR-SIZE |
| 14 | ENC-SECTION | Enclosure section, Form 4 Type 4, IP41, RAL 7035 | 2 | 3,160.04 | 6,320.08 | 0.95 | T54 T54 T56 T58 R-ENCLOSURE |
| 15 | LAB-ASSY | Assembly, wiring and routine verification: 168 hours | 1 | 8,064.00 | 8,064.00 | 0.9 |  R-LABOUR |
| 16 | DOC | Documentation, as-built drawings, test certificates | 1 | 600.00 | 600.00 | 1.0 |  R-DOC |
| 17 | CERT | Design-verification certificates per specification | 1 | 0.00 | 0.00 | 1.0 | 7.7 R-SPEC-CERT |
| 18 | DEL | Delivery to site, crane and offload excluded | 1 | 850.00 | 850.00 | 1.0 |  R-DELIVERY |

Subtotal £26,724.88; margin £4,008.73; **total £30,733.61**

## Clarifications and exclusions (from flagged items)
- [missing] D3 (ACB): rating not read. Action: read from drawing / schedule. Evidence: S3 S3t1 S3t2
- [missing] D4 (ACB): rating not read. Action: read from drawing / schedule. Evidence: S4 S4t1 S4t2
- [missing] D10 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S10 S10t5 S10t3
- [missing] D11 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S11 S11t3 S11t2
- [missing] D12 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S12 S12t5 S12t2
- [missing] D13 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S13 S13t5 S13t2
- [missing] D14 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S14 S14t9 S14t4
- [missing] D15 (CONTACTOR): kW not read. Action: read from drawing / schedule. Evidence: S15 S15t5 S15t3
- [missing] D16 (CONTACTOR): kW not read. Action: read from drawing / schedule. Evidence: S16 S16t5 S16t2
- [missing] D17 (ACB): rating not read. Action: read from drawing / schedule. Evidence: S17 S17t4 S17t3
- [missing] D22 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S22 S22t2 S22t1
- [missing] D24 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S24 S24t2 S24t1
- [missing] D25 (FUSE_SWITCH): rating not read. Action: read from drawing / schedule. Evidence: S25 S25t4 S25t3
- [conflict] fault_kA: drawing says 80, specification clause 7.3 says 50. Action: clarify with the consultant before order; priced to the drawing. Evidence: T54 T54 T56 T58 7.3
- [low_confidence] devices read with low confidence: D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26, D27. Action: check against the drawing. Evidence: D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13 D16 D17 D18 D19 D20 D21 D22 D23 D24 D25 D26 D27

## Specification clauses matched

- standard: clause 7.1 -> BS EN 61439-2
- form: clause 7.2 -> Form 4 Type 4
- fault_kA: clause 7.3 -> 50
- poles: clause 7.4 -> 4
- ip: clause 7.5 -> IP41
- busbar_material: clause 7.6 -> copper
- certificates: clause 7.7 -> True
- other (not priced automatically): clause 7.8: Two years' spares for all trip units shall be included.

Generated by sld2quote 0.1.0 in 28.0 s; reader deskew -0.7 deg.
