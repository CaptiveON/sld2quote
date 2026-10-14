"""Specification reader: clauses -> requirements with the clause as evidence, and a cross-check against the drawing notes.

Deliberately rule-based: in the KTP this is where an LLM with a fixed schema and clause citations goes, evaluated
against estimator-agreed golden extractions. The interface (requirements + evidence + conflicts) stays the same.
"""
from __future__ import annotations
import re
from .reader import RX_FORM, RX_IP, RX_KA, RX_POLE_DEFAULT, LOOKALIKE

RX_CLAUSE = re.compile(r"^\s*(\d+\.\d+)\s+(.*\S)\s*$")


def read_spec(path: str) -> dict:
    clauses = []
    for line in open(path, encoding="utf-8"):
        m = RX_CLAUSE.match(line)
        if m:
            clauses.append({"id": m.group(1), "text": m.group(2)})
    req = {"standard": None, "form": None, "fault_kA": None, "poles": None, "ip": None, "busbar_material": None, "certificates": False, "other": []}
    ev = {}
    for c in clauses:
        s = c["text"].translate(LOOKALIKE); low = s.lower()
        if "61439" in s and req["standard"] is None:
            req["standard"] = re.search(r"BS EN(?: IEC)? 61439-\d(?::\d{4})?", s).group(0); ev["standard"] = c["id"]
        m = RX_FORM.search(s)
        if m and ("separation" in low or "form" in low) and req["form"] is None:
            req["form"] = f"Form {m.group(1)}" + (f" Type {m.group(2)}" if m.group(2) else (m.group(3) if m.group(3) else "")); ev["form"] = c["id"]
        m = RX_KA.search(s)
        if m and ("short-circuit" in low or "fault" in low) and req["fault_kA"] is None:
            req["fault_kA"] = int(m.group(1)); ev["fault_kA"] = c["id"]
        m = RX_POLE_DEFAULT.search(s)
        if m and req["poles"] is None:
            req["poles"] = int(m.group(1)); ev["poles"] = c["id"]
        m = RX_IP.search(s)
        if m and req["ip"] is None:
            req["ip"] = f"IP{m.group(1)}"; ev["ip"] = c["id"]
        if "copper" in low and req["busbar_material"] is None:
            req["busbar_material"] = "copper"; ev["busbar_material"] = c["id"]
        if "certificate" in low or "verification" in low and "certificate" in low:
            req["certificates"] = True; ev["certificates"] = c["id"]
        if not any(k in low for k in ["61439", "separation", "short-circuit", "fault", "pole", "protection", "copper", "certificate"]):
            req["other"].append({"clause": c["id"], "text": s})
    return {"clauses": clauses, "requirements": req, "evidence": ev}


def cross_check(drawing_notes: dict, spec: dict) -> list[dict]:
    """Drawing-versus-specification conflicts. Flagged, never resolved."""
    r, ev = spec["requirements"], spec["evidence"]
    out = []
    for field, dkey in [("fault_kA", "fault_kA"), ("form", "form"), ("poles", "poles_default"), ("ip", "ip")]:
        dv, sv = drawing_notes.get(dkey), r.get(field)
        if dv is not None and sv is not None and str(dv).lower() != str(sv).lower():
            out.append({"field": field, "drawing": dv, "spec": sv, "drawing_evidence": drawing_notes.get("evidence", []), "spec_clause": ev.get(field)})
    return out
