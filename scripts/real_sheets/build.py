"""Build data/real from spec.json: render sheets, write manifest.json, contact_sheet.png and the tables of SOURCES.md.
usage: python -I build.py <spec.json> <prose.md>   (prose.md holds the hand-written sections, with {SHEETS} and {DOCS} placeholders)"""
import hashlib, json, os, sys
import pypdfium2 as pdfium
import pypdfium2.raw as raw
from PIL import Image, ImageDraw, ImageFont

ROOT = "/Users/captiveon/Documents/techproject/experimental/thmillion/practical/sld2quote/data/real"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def note(source_id):
    d = {}
    for line in open(os.path.join(ROOT, "raw", source_id, "note.txt")):
        k, _, v = line.partition(":")
        d[k.strip()] = v.strip()
    d["sha256"] = d["file"].split("sha256:")[1].strip()
    return d


def render(s, doc):
    pg = pdfium.PdfDocument(os.path.join(ROOT, "raw", s["source_id"], doc["file"]))[s["pdf_page"] - 1]
    if s.get("drop_stamp"):  # the archive's CANCELLED stamp is the page's only form object; the drawing is not in it
        for o in [o for o in pg.get_objects(max_depth=1) if o.type == raw.FPDF_PAGEOBJ_FORM]:
            pg.remove_obj(o)
        pg.gen_content()
    img = pg.render(scale=s["dpi"] / 72, grayscale=True).to_pil().convert("L")
    w, h = img.size
    x0, y0, x1, y1 = s["crop"]
    img = img.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    return img.rotate(-s["rotate_cw"], expand=True) if s.get("rotate_cw") else img


def contact_sheet(items, path, cols=5, cw=420, th=300):
    f, fh = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 22)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("L", (cols * cw, 50 + rows * (th + 40)), 255)
    d = ImageDraw.Draw(sheet)
    d.text((12, 12), f"sld2quote real test set: {len(items)} sheets from {len({s['source_id'] for s, _ in items})} US federal documents (public domain)", fill=0, font=fh)
    for k, (s, img) in enumerate(items):
        x, y = (k % cols) * cw, 50 + (k // cols) * (th + 40)
        t = img.copy(); t.thumbnail((cw - 20, th - 10))
        sheet.paste(t, (x + (cw - t.width) // 2, y + (th - t.height) // 2))
        d.rectangle([x + 4, y, x + cw - 5, y + th + 34], outline=170)
        lw = d.textlength(s["sheet_id"], font=f)
        d.text((x + (cw - lw) / 2, y + th + 8), s["sheet_id"], fill=0, font=f)
    sheet.save(path, optimize=True)


def cell(v):
    return str(v).replace("|", "/").replace("\n", " ")


def main():
    spec, prose = json.load(open(sys.argv[1])), open(sys.argv[2]).read()
    docs = {d["source_id"]: d for d in spec["documents"]}
    os.makedirs(os.path.join(ROOT, "sheets"), exist_ok=True)
    manifest, items = [], []
    for s in spec["sheets"]:
        doc, n = docs[s["source_id"]], note(s["source_id"])
        img = render(s, doc)
        rel = f"sheets/{s['sheet_id']}.png"
        img.save(os.path.join(ROOT, rel), optimize=True)
        items.append((s, img))
        manifest.append({
            "sheet_id": s["sheet_id"], "image": rel, "width": img.width, "height": img.height,
            "image_sha256": hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest(),
            "source_id": s["source_id"], "document_title": doc["title"], "agency": doc["agency"],
            "publication_date": doc["published"], "url": n["url"], "final_url": n["final_url"],
            "retrieved_utc": n["retrieved_utc"], "raw_file": f"raw/{s['source_id']}/{doc['file']}", "raw_sha256": n["sha256"],
            "pdf_page": s["pdf_page"], "printed_page": s["printed_page"], "figure": s["figure"], "caption": s["caption"],
            "credit_on_figure": s.get("credit", "none"), "rights_statement": doc["statement"], "provenance_basis": doc["basis"],
            "description": s["description"], "kind": s["kind"], "native_resolution": s["native"], "quality": s["quality"],
            "voltage": s["voltage"], "approx_devices": s["devices"], "ratings_shown": s["ratings"],
            "symbol_style": "US ANSI/IEEE 315 style (arc or square breakers, coil or zig-zag transformers, circled G)",
            "breaker_symbol_px": s["breaker_px"],
            "render": {"tool": "pypdfium2 5.14.0 (PDFium)", "dpi": s["dpi"], "grey": True,
                       "crop_page_fraction_x0y0x1y1": s["crop"], "rotate_clockwise_deg": s.get("rotate_cw", 0),
                       "archive_stamp_removed": bool(s.get("drop_stamp"))},
            "labelled": False, "use": "test only, never training"})
        print(s["sheet_id"], img.size)
    json.dump({"set": "sld2quote real test set", "built": "2026-10-18", "sheets": manifest},
              open(os.path.join(ROOT, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    contact_sheet(items, os.path.join(ROOT, "contact_sheet.png"))
    rows = ["| Sheet | Document (agency, date) | URL | Page (PDF / printed) | Figure caption | Rights statement found | Description | Kind, quality | Render, size |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m, s in zip(manifest, spec["sheets"]):
        d = docs[m["source_id"]]
        rows.append("| " + " | ".join(cell(v) for v in [
            f"`{m['sheet_id']}`", d['short'],
            m["url"], f"{m['pdf_page']} / {m['printed_page']}", m["caption"] + (f"; credit: {s['credit']}" if s.get("credit") else ""),
            d["statement_short"], f"{m['description']} About {m['approx_devices']} devices.",
            f"{m['kind']} ({m['native_resolution']}), {m['quality']}",
            f"{s['dpi']} dpi{', rotated 90 cw' if s.get('rotate_cw') else ''}{', stamp removed' if s.get('drop_stamp') else ''}; {m['width']} x {m['height']} px; breaker about {m['breaker_symbol_px']} px"]) + " |")
    drows = ["| Source id | Title | Agency | Published | URL | Retrieved (UTC) | Statement found | Provenance basis |", "|---|---|---|---|---|---|---|---|"]
    for sid, d in docs.items():
        n = note(sid)
        drows.append("| " + " | ".join(cell(v) for v in [f"`{sid}`", d["title"], d["agency"], d["published"], n["url"], n["retrieved_utc"], d["statement"], d["basis"]]) + " |")
    open(os.path.join(ROOT, "SOURCES.md"), "w").write(prose.replace("{SHEETS}", "\n".join(rows)).replace("{DOCS}", "\n".join(drows)))


if __name__ == "__main__":
    main()
