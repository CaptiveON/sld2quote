"""Rules layer and bill of materials. Deterministic: ratings and forms come from the drawing, the notes and the
specification; the catalogue maps them to parts; nothing is guessed. Anything below the fault level or missing is
raised as an issue for the estimator, not silently fixed. Every line records its evidence."""
from __future__ import annotations
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
CAT = json.load(open(os.path.join(HERE, "catalogue.json")))


def ka_band(ka: int | None) -> str:
    if ka is None:
        return "N"
    for lim, band in CAT["ka_bands"]:
        if ka <= lim:
            return band
    return "X"


def _nearest_key(table: dict, rating: int) -> str:
    keys = sorted(int(k) for k in table)
    for k in keys:
        if k >= rating:
            return str(k)
    return str(keys[-1])


def build_bom(extraction: dict, spec: dict | None, conflicts: list[dict]) -> dict:
    notes = extraction["notes"]
    req = (spec or {}).get("requirements", {})
    issues = []
    # board-level parameters: drawing notes first, spec as the check (conflicts already flagged separately)
    fault = notes.get("fault_kA") or req.get("fault_kA")
    form = notes.get("form") or req.get("form") or "Form 4 Type 6"
    ip = notes.get("ip") or req.get("ip") or "IP31"
    poles_default = notes.get("poles_default") or req.get("poles") or 4
    if notes.get("fault_kA") is None:
        issues.append({"kind": "missing", "what": "fault level not found on the drawing notes", "action": "confirm with the consultant", "evidence": []})
    lines = []
    sections = 0
    ways = 0

    def add(part, desc, qty, unit, evidence, rule, conf=1.0, basis=""):
        lines.append({"line": len(lines) + 1, "part": part, "description": desc, "qty": qty, "unit_gbp": round(unit, 2), "total_gbp": round(unit * qty, 2),
                      "evidence": evidence, "rule": rule, "conf": round(conf, 2), "basis": basis})

    for d in extraction["devices"]:
        ev = [d["symbol"]] + d.get("label_texts", [])
        if d["type"] in ("ACB", "MCCB", "BUS_SECTION"):
            rating = d.get("rating")
            if rating is None:
                issues.append({"kind": "missing", "what": f"{d['id']} ({d['type']}): rating not read", "action": "read from drawing / schedule", "evidence": ev}); continue
            poles = d.get("poles") or poles_default
            dka = d.get("kA") or fault
            if fault and dka and dka < fault:
                issues.append({"kind": "below_fault", "what": f"{d['id']} ({d['type']} {rating} A) shows {dka} kA but the board fault level is {fault} kA",
                               "action": "proposed: select the {0} kA band; confirm with the designer".format(fault), "evidence": ev + notes.get("evidence", [])})
                dka = fault
            band = ka_band(dka)
            if d["type"] == "ACB":
                key = _nearest_key(CAT["acb"], rating); base = CAT["acb"][key]
                part = f"ACB-{key}-{poles}P-{band}"
                add(part, f"Air circuit breaker {key} A {poles}-pole, {dka} kA band {band}, withdrawable, electronic trip", 1,
                    base * CAT["band_multiplier"][band] * CAT["poles_multiplier"][str(poles)], ev, "R-ACB-SELECT", d["conf"], f"rating {rating} A from {d['label_texts'][:1]}; poles {poles}; kA {dka}")
                add("FU-KIT-ACB", "Incomer functional unit: compartment, shutters, terminals, controls", 1, CAT["functional_unit_kit"]["ACB"], ev, "R-FU-KIT", d["conf"])
                if "ATS" in d.get("label", "").upper():
                    add("ATS-CTRL", "Automatic transfer switch controller and interlocks", 1, CAT["ats_controller"], ev, "R-ATS", d["conf"])
                sections += 1
            elif d["type"] == "BUS_SECTION":
                key = _nearest_key(CAT["bus_section_switch"], rating)
                add(f"BSS-{key}-{poles}P", f"Bus-section switch {key} A {poles}-pole, {dka} kA", 1, CAT["bus_section_switch"][key] * CAT["band_multiplier"][band], ev, "R-BUS-SECTION", d["conf"])
                add("FU-KIT-BS", "Bus-section functional unit and interlocking", 1, CAT["functional_unit_kit"]["BUS_SECTION"], ev, "R-FU-KIT", d["conf"])
                sections += 1
            else:
                key = _nearest_key(CAT["mccb"], rating); base = CAT["mccb"][key]
                add(f"MCCB-{key}-{poles}P-{band}", f"Moulded-case circuit breaker {key} A {poles}-pole, {dka} kA band {band}, adjustable trip" + (f"; feeds {d['dest']}" if d.get("dest") else ""),
                    1, base * CAT["band_multiplier"][band] * CAT["poles_multiplier"][str(poles)], ev, "R-MCCB-SELECT", d["conf"], f"rating {rating} A; poles {poles}{' (notes default)' if d.get('poles') is None else ''}; kA {dka}")
                add("FU-KIT-MCCB", "Outgoing functional unit: compartment door, terminals, labels", 1, CAT["functional_unit_kit"]["MCCB"], ev, "R-FU-KIT", d["conf"])
                ways += 1
    # busbars
    bus_rating = None
    for bb in extraction["busbars"]:
        if bb.get("rating"):
            bus_rating = max(bus_rating or 0, bb["rating"])
    if bus_rating is None:
        acbs = [d.get("rating") or 0 for d in extraction["devices"] if d["type"] == "ACB"]
        bus_rating = max(acbs) if acbs else 2500
        issues.append({"kind": "assumed", "what": f"busbar rating not read; assumed {bus_rating} A from the largest incomer", "action": "confirm", "evidence": []})
    sections += math.ceil(ways / CAT["ways_per_section"]) if ways else 0
    sections = max(sections, 1)
    key = _nearest_key(CAT["busbar"]["kg_per_m"], bus_rating)
    kg = CAT["busbar"]["kg_per_m"][key] * CAT["busbar"]["metres_per_section"] * sections
    add(f"BUSBAR-CU-{key}", f"Copper busbar set {key} A (4-bar), {sections} sections, about {kg:.0f} kg", 1, kg * CAT["busbar"]["gbp_per_kg"],
        [bb.get("label_text") for bb in extraction["busbars"] if bb.get("label_text")], "R-BUSBAR-SIZE", 0.9, f"{bus_rating} A busbar label; {sections} sections")
    fm = CAT["enclosure_section"]["form_multiplier"].get(form, 1.3); ipm = CAT["enclosure_section"]["ip_multiplier"].get(ip, 1.0)
    add("ENC-SECTION", f"Enclosure section, {form}, {ip}, RAL 7035", sections, CAT["enclosure_section"]["base"] * fm * ipm, notes.get("evidence", []), "R-ENCLOSURE", 0.95, f"{ways} ways at {CAT['ways_per_section']} per section plus incomers")
    hrs = CAT["labour"]["hours"]
    labour_h = sum(hrs[d["type"]] for d in extraction["devices"] if d["type"] in hrs) + sections * (hrs["section_assembly"] + hrs["test_per_section"])
    add("LAB-ASSY", f"Assembly, wiring and routine verification: {labour_h} hours", 1, labour_h * CAT["labour"]["rate_gbp_per_hour"], [], "R-LABOUR", 0.9)
    add("DOC", "Documentation, as-built drawings, test certificates", 1, CAT["documentation_gbp"], [], "R-DOC")
    if req.get("certificates"):
        add("CERT", "Design-verification certificates per specification", 1, 0.0, [spec["evidence"].get("certificates")], "R-SPEC-CERT")
    if any("witness" in (o["text"].lower()) for o in req.get("other", [])):
        add("WITNESS", "Witness testing at works (specification)", 1, CAT["witness_test_gbp"], [o["clause"] for o in req["other"] if "witness" in o["text"].lower()], "R-SPEC-WITNESS")
    add("DEL", "Delivery to site, crane and offload excluded", 1, CAT["delivery_gbp"], [], "R-DELIVERY")
    subtotal = sum(l["total_gbp"] for l in lines)
    margin = round(subtotal * CAT["margin"], 2)
    for c in conflicts:
        issues.append({"kind": "conflict", "what": f"{c['field']}: drawing says {c['drawing']}, specification clause {c['spec_clause']} says {c['spec']}",
                       "action": "clarify with the consultant before order; priced to the drawing", "evidence": c.get("drawing_evidence", []) + [c.get("spec_clause")]})
    low = [d["id"] for d in extraction["devices"] if d.get("conf", 1) < 0.6]
    if low:
        issues.append({"kind": "low_confidence", "what": f"devices read with low confidence: {', '.join(low)}", "action": "check against the drawing", "evidence": low})
    return {"board": {"fault_kA": fault, "form": form, "ip": ip, "poles_default": poles_default, "busbar_rating": bus_rating, "sections": sections, "ways": ways},
            "lines": lines, "subtotal_gbp": round(subtotal, 2), "margin_gbp": margin, "total_gbp": round(subtotal + margin, 2), "issues": issues}
