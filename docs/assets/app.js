"use strict";

// Simple, dependency-free viewer for AI-generated alt-text test runs.
// To add another run: drop its alt_text.json into data/runs/ and add an
// entry to data/runs.json. Nothing else needs to change.

const state = {
  runs: [],
  rows: [],
  filtered: [],
  page: 1,
  perPage: 20,
};

const el = (id) => document.getElementById(id);

/** Turn a full IIIF image URL (.../full/full/0/default.jpg) into a sized derivative. */
function iiifSize(url, size) {
  if (!url) return "";
  if (url.includes("/full/full/")) {
    return url.replace("/full/full/", `/full/!${size},${size}/`);
  }
  return url;
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadRuns() {
  const res = await fetch("data/runs.json", { cache: "no-cache" });
  const cfg = await res.json();
  state.runs = cfg.runs || [];

  const sel = el("run-select");
  sel.innerHTML = "";
  state.runs.forEach((run, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${run.label} (${run.date})`;
    sel.appendChild(opt);
  });

  if (state.runs.length) {
    await loadRun(0);
  } else {
    el("grid-body").innerHTML =
      '<tr><td colspan="4">No runs found.</td></tr>';
  }
}

async function loadRun(index) {
  const run = state.runs[index];
  if (!run) return;
  el("grid-body").innerHTML = '<tr><td colspan="4">Loading…</td></tr>';
  const res = await fetch(run.file, { cache: "no-cache" });
  state.rows = await res.json();
  state.page = 1;
  el("footer-model").textContent = run.model || "—";
  el("run-meta").innerHTML =
    `Model <strong>${escapeHtml(run.model || "—")}</strong> · ` +
    `${state.rows.length} images · ${escapeHtml(run.date || "")}`;
  applyFilter();
}

function applyFilter() {
  const q = el("search").value.trim().toLowerCase();
  state.filtered = !q
    ? state.rows.slice()
    : state.rows.filter((r) => String(r.obj_id || "").toLowerCase().includes(q));
  state.page = 1;
  render();
}

function totalPages() {
  return Math.max(1, Math.ceil(state.filtered.length / state.perPage));
}

function render() {
  const tp = totalPages();
  if (state.page > tp) state.page = tp;
  const start = (state.page - 1) * state.perPage;
  const pageRows = state.filtered.slice(start, start + state.perPage);

  const body = el("grid-body");
  body.innerHTML = "";

  if (!pageRows.length) {
    body.innerHTML = '<tr><td colspan="4">No matching objects.</td></tr>';
  }

  pageRows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.addEventListener("click", () => openModal(r));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openModal(r);
      }
    });

    const thumbUrl = iiifSize(r.image_url, 150);
    const thumbCell = thumbUrl
      ? `<img class="thumb" loading="lazy" src="${escapeHtml(thumbUrl)}" alt="Thumbnail for object ${escapeHtml(r.obj_id)}" />`
      : `<div class="thumb placeholder">no image</div>`;

    const altCell =
      r.status === "ok"
        ? `<div class="alt-preview">${escapeHtml(r.alt_text)}</div>`
        : `<span class="badge-err">error</span>`;

    tr.innerHTML =
      `<td class="col-thumb">${thumbCell}</td>` +
      `<td class="col-id"><span class="obj-id">${escapeHtml(r.obj_id)}</span></td>` +
      `<td class="col-label">${escapeHtml(r.canvas_label || "")}</td>` +
      `<td class="col-alt">${altCell}</td>`;
    body.appendChild(tr);
  });

  const end = Math.min(start + state.perPage, state.filtered.length);
  el("page-info").textContent = state.filtered.length
    ? `${start + 1}–${end} of ${state.filtered.length} · page ${state.page}/${tp}`
    : "0 results";
  el("prev").disabled = state.page <= 1;
  el("next").disabled = state.page >= tp;
}

function openModal(r) {
  el("modal-img").src = iiifSize(r.image_url, 1000);
  el("modal-img").alt = `Image for object ${r.obj_id}`;
  el("modal-title").textContent = `Object ${r.obj_id}`;
  el("modal-view").textContent = r.canvas_label || "";
  el("modal-alt").textContent =
    r.status === "ok" ? r.alt_text : `No alt-text produced (${r.error || "error"}).`;

  const links = [];
  if (r.catalog_url)
    links.push(`<a href="${escapeHtml(r.catalog_url)}" target="_blank" rel="noopener">Catalog record ↗</a>`);
  if (r.manifest_url)
    links.push(`<a href="${escapeHtml(r.manifest_url)}" target="_blank" rel="noopener">IIIF manifest ↗</a>`);
  el("modal-links").innerHTML = links.join(" &nbsp;·&nbsp; ");

  el("modal").hidden = false;
  document.body.style.overflow = "hidden";
}

function closeModal() {
  el("modal").hidden = true;
  el("modal-img").src = "";
  document.body.style.overflow = "";
}

function wireEvents() {
  el("run-select").addEventListener("change", (e) => loadRun(Number(e.target.value)));
  el("per-page").addEventListener("change", (e) => {
    state.perPage = Number(e.target.value);
    state.page = 1;
    render();
  });
  el("search").addEventListener("input", applyFilter);
  el("prev").addEventListener("click", () => {
    if (state.page > 1) { state.page--; render(); }
  });
  el("next").addEventListener("click", () => {
    if (state.page < totalPages()) { state.page++; render(); }
  });
  document.querySelectorAll("[data-close]").forEach((n) =>
    n.addEventListener("click", closeModal)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !el("modal").hidden) closeModal();
  });
}

wireEvents();
loadRuns().catch((err) => {
  el("grid-body").innerHTML =
    `<tr><td colspan="4">Failed to load data: ${escapeHtml(err.message)}</td></tr>`;
});
