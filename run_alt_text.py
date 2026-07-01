"""Generate accessibility alt-text for Yale Center for British Art images.

Reads a CSV of catalog URLs (``https://collections.britishart.yale.edu/catalog/tms:<id>``),
turns each into its IIIF Presentation manifest URL
(``https://manifests.collections.yale.edu/ycba/obj/<id>``), downloads the image(s) for
that object, and asks Gemini (``gemini-3.1-flash-lite`` on Vertex AI) to write alt-text
that describes only what is visible in the picture.

The pipeline is built on the chai-engine:

    IIIFDirFileProvider  ->  Iterator  ->  GeminiDescriber

One chai Workflow is built and run per manifest; the manifests themselves are processed
concurrently with a thread pool (default 20 workers).

Run:

    /opt/anaconda3/bin/python run_alt_text.py

Requires Google Cloud application-default credentials for the Vertex project
(``gcloud auth application-default login``).
"""

import csv
import os
import re
import sys
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ----------------------------------------------------------

CHAI_ENGINE_PATH = "/Users/wjm55/chai-engine"

PROJECT_ID = "cultural-heritage-gemini"   # Vertex AI project
LOCATION = "global"                        # Vertex AI location
MODEL = "gemini-3.1-flash-lite"            # via google-genai on Vertex

WORKERS = 20                               # manifests processed concurrently
URLS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "urls.csv")
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
RUN_LABEL = "test-run-100"

# Very specific alt-text prompt. The image is the only thing the model may rely on:
# no outside knowledge, no historical context, no named people. Written for an 8th
# grade reading level using short, plain sentences that name and describe objects,
# and kept to 4-6 sentences total.
PROMPT = """You are writing alt-text to make an image accessible to people who cannot see it.

Describe ONLY what is actually visible in this image. Do not use any outside knowledge about the image, and do not try to identify it.

Follow these rules exactly:
- Write between 4 and 6 sentences. No more, no fewer.
- Write at an 8th-grade reading level.
- Use short, plain, direct sentences.
- Name the objects you can see in the image, then describe each one in a basic way (its shape, color, size, and where it sits in the image).
- Describe only what is visible. Do not guess.
- Do not guess at historical context, time period, meaning, or symbolism.
- Do not name specific real people, artists, places, or events.
- Do not begin with phrases like "Image of", "Picture of", "This image shows", or "Alt text:". Start with the description itself.

Return only the alt-text as plain text."""

# --- Vertex AI: set project BEFORE importing chai (gemini backend reads env) --

os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

if CHAI_ENGINE_PATH not in sys.path:
    sys.path.insert(0, CHAI_ENGINE_PATH)

from chai.workflow import Workflow  # noqa: E402
import chai.provider as _provider  # noqa: E402


class FirstImageIIIFProvider(_provider.IIIFDirFileProvider):
    """IIIF provider that downloads only the FIRST canvas image of the manifest.

    The base provider fetches one image per canvas; here we keep just the first
    canvas so each object yields a single image (usually the primary recto view).
    """

    def get_images_info(self, manifest):
        canvases = super().get_images_info(manifest)
        return canvases[:1]


# Register on chai.provider so importClass("provider.FirstImageIIIFProvider") resolves it.
_provider.FirstImageIIIFProvider = FirstImageIIIFProvider


CATALOG_ID_RE = re.compile(r"tms:(\d+)")


def catalog_to_manifest(catalog_url):
    """Map a catalog URL to (obj_id, manifest_url), or (None, None) if it doesn't match."""
    m = CATALOG_ID_RE.search(catalog_url)
    if not m:
        return None, None
    obj_id = m.group(1)
    return obj_id, f"https://manifests.collections.yale.edu/ycba/obj/{obj_id}"


def read_catalog_urls(path, limit=None):
    """Read the urls.csv (single `url` column, with header). Optionally cap at `limit` rows."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if not row:
                continue
            url = row[0].strip()
            if i == 0 and url.lower() == "url":
                continue  # header
            if url:
                rows.append(url)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _label_text(label):
    """Flatten a IIIF label ({"en": [..]}) or plain value into a single string."""
    if isinstance(label, dict):
        parts = []
        for v in label.values():
            if isinstance(v, (list, tuple)):
                parts.extend(str(x) for x in v)
            else:
                parts.append(str(v))
        return " / ".join(parts)
    if isinstance(label, (list, tuple)):
        return " / ".join(str(x) for x in label)
    return str(label) if label else ""


DESCRIBER_ID = "alt_text"


def build_workflow(obj_id, images_dir):
    """Build a per-manifest chai Workflow: IIIF provider -> Iterator -> GeminiDescriber."""
    config = {
        "type": "Workflow",
        "id": f"alt_{obj_id}",
        "steps": [
            {
                "type": "provider.FirstImageIIIFProvider",
                "id": "iiif",
                "settings": {"directory": images_dir},
                "steps": [
                    {
                        "type": "iterator.Iterator",
                        "id": "each_image",
                        "settings": {"continue_on_error": True},
                        "steps": [
                            {
                                "type": "describer.GeminiDescriber",
                                "id": DESCRIBER_ID,
                                "settings": {
                                    "model": MODEL,
                                    "location": LOCATION,
                                    "expected_output": "text",
                                    "tools": [],            # no Google Search grounding
                                    "temperature": 0.2,
                                    "max_output_tokens": 1024,
                                    "prompt": PROMPT,
                                    "retries": 2,
                                    "retry_delay": 2.0,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return Workflow(config)


def collect_describer_results(result, out):
    """Walk the finished result tree and record every GeminiDescriber output.

    The provenance chain (``result.input``) is only backfilled after each
    component finishes, so we read it from the returned tree (not from live
    events). Each describer ``ItemResult``'s ``input`` is the source image
    ``FileItemResult``, giving us the file name. We only descend into list
    values so we never trigger a ``FileItemResult``'s lazy on-disk read.
    """
    # Local import so the module can be introspected without importing chai.
    from chai.result import Result

    if not isinstance(result, Result):
        return
    cls = type(result).__name__
    proc = getattr(result, "processor", None)
    if proc is not None and getattr(proc, "id", None) == DESCRIBER_ID and cls == "ItemResult":
        src = getattr(result.input, "file_name", None)
        out.append({
            "image_file": os.path.basename(src) if src else None,
            "alt_text": result.value if isinstance(result.value, str) else str(result.value),
            "metadata": dict(result.metadata or {}),
        })
        return
    if cls in ("ListResult", "DirectoryListResult", "LabelListResult"):
        for child in result.value:
            if isinstance(child, Result):
                collect_describer_results(child, out)


def load_image_info(images_dir):
    """Read the provider's _info.json: {filename: {url, canvas}}."""
    info_path = os.path.join(images_dir, "_info.json")
    if not os.path.exists(info_path):
        return {}
    try:
        with open(info_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def process_manifest(catalog_url, downloads_root):
    """Run the full pipeline for one catalog URL. Returns a list of per-image dicts."""
    obj_id, manifest_url = catalog_to_manifest(catalog_url)
    base = {
        "catalog_url": catalog_url,
        "obj_id": obj_id,
        "manifest_url": manifest_url,
    }
    if not obj_id:
        return [dict(base, image_file=None, image_url=None, canvas_label=None,
                     alt_text=None, status="error", tokens_total=None,
                     duration_s=None, error="could not parse tms id from catalog url")]

    images_dir = os.path.join(downloads_root, obj_id)
    os.makedirs(images_dir, exist_ok=True)

    captured = []
    try:
        wf = build_workflow(obj_id, images_dir)
        result = wf.run(manifest_url)
        collect_describer_results(result, captured)
    except Exception as e:
        return [dict(base, image_file=None, image_url=None, canvas_label=None,
                     alt_text=None, status="error", tokens_total=None,
                     duration_s=None, error=f"{type(e).__name__}: {e}")]

    info = load_image_info(images_dir)
    by_file = {c["image_file"]: c for c in captured if c.get("image_file")}

    rows = []
    filenames = sorted(info.keys()) if info else sorted(by_file.keys())
    if not filenames:
        return [dict(base, image_file=None, image_url=None, canvas_label=None,
                     alt_text=None, status="error", tokens_total=None,
                     duration_s=None, error="no images found in manifest")]

    for fn in filenames:
        meta = info.get(fn, {})
        canvas = meta.get("canvas", {}) if isinstance(meta, dict) else {}
        image_url = meta.get("url") if isinstance(meta, dict) else None
        canvas_label = _label_text(canvas.get("label")) if isinstance(canvas, dict) else ""

        cap = by_file.get(fn)
        if cap:
            usage = (cap.get("metadata") or {}).get("token_usage") or {}
            rows.append(dict(
                base,
                image_file=fn,
                image_url=image_url,
                canvas_label=canvas_label,
                alt_text=cap.get("alt_text"),
                status="ok",
                tokens_total=usage.get("total"),
                duration_s=round(cap.get("metadata", {}).get("duration", 0) or 0, 2),
                error="",
            ))
        else:
            rows.append(dict(
                base,
                image_file=fn,
                image_url=image_url,
                canvas_label=canvas_label,
                alt_text=None,
                status="error",
                tokens_total=None,
                duration_s=None,
                error="no alt-text produced (describe step failed)",
            ))
    return rows


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate YCBA alt-text with chai + Gemini.")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N catalog URLs (default: all)")
    parser.add_argument("--label", default=RUN_LABEL,
                        help=f"output folder label (default: {RUN_LABEL})")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"{timestamp}-{args.label}")
    downloads_root = os.path.join(out_dir, "downloads")
    os.makedirs(downloads_root, exist_ok=True)

    catalog_urls = read_catalog_urls(URLS_CSV, limit=args.limit)
    print(f"Loaded {len(catalog_urls)} catalog URLs from {URLS_CSV}")
    print(f"Model: {MODEL}  |  Vertex project: {PROJECT_ID}  |  location: {LOCATION}")
    print(f"Workers: {WORKERS}")
    print(f"Output dir: {out_dir}\n")

    run_config = {
        "timestamp": timestamp,
        "model": MODEL,
        "vertex_project": PROJECT_ID,
        "location": LOCATION,
        "workers": WORKERS,
        "urls_csv": URLS_CSV,
        "limit": args.limit,
        "first_image_only": True,
        "num_catalog_urls": len(catalog_urls),
        "prompt": PROMPT,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)

    all_rows = []
    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(process_manifest, url, downloads_root): url for url in catalog_urls
        }
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                rows = [{
                    "catalog_url": url, "obj_id": None, "manifest_url": None,
                    "image_file": None, "image_url": None, "canvas_label": None,
                    "alt_text": None, "status": "error", "tokens_total": None,
                    "duration_s": None, "error": f"{type(e).__name__}: {e}",
                }]
            all_rows.extend(rows)
            done += 1
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"[{done}/{len(catalog_urls)}] {url} -> {ok}/{len(rows)} image(s) described")

    # Stable order: by object id, then image file
    all_rows.sort(key=lambda r: ((r.get("obj_id") or ""), (r.get("image_file") or "")))

    fields = ["catalog_url", "obj_id", "manifest_url", "image_file", "image_url",
              "canvas_label", "alt_text", "status", "tokens_total", "duration_s", "error"]
    csv_path = os.path.join(out_dir, "alt_text.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fields})

    json_path = os.path.join(out_dir, "alt_text.json")
    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)

    elapsed = time.time() - started
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    n_err = sum(1 for r in all_rows if r["status"] == "error")

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    for r in all_rows:
        print(f"\nObject {r['obj_id']}  ({r['catalog_url']})")
        print(f"  image: {r['image_file']}  |  {r['canvas_label']}")
        if r["status"] == "ok":
            print(f"  alt-text: {r['alt_text']}")
        else:
            print(f"  ERROR: {r['error']}")

    print("\n" + "=" * 78)
    print("Done.")
    print(f"  Images described OK : {n_ok}")
    print(f"  Errors              : {n_err}")
    print(f"  Elapsed             : {elapsed:.1f}s")
    print(f"  CSV                 : {csv_path}")
    print(f"  JSON                : {json_path}")


if __name__ == "__main__":
    main()
