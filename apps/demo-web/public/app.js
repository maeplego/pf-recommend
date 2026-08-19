const apiBase = window.location.port === "3008" || window.location.port === "80" || window.location.port === ""
  ? ""
  : "http://localhost:8098";

const namespaceEl = document.getElementById("namespace");
const userEl = document.getElementById("user");
const kEl = document.getElementById("k");
const recsEl = document.getElementById("recs");
const similarEl = document.getElementById("similar");
const metricsEl = document.getElementById("metrics");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("model-meta");

let models = [];

function fmt(value) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toFixed(4);
}

function fillSelect(select, values) {
  select.replaceChildren();
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = String(value);
    opt.textContent = String(value);
    select.appendChild(opt);
  }
}

function appendCells(tr, values) {
  for (const value of values) {
    const td = document.createElement("td");
    td.textContent = String(value);
    tr.appendChild(td);
  }
}

async function fetchJson(path) {
  const res = await fetch(`${apiBase}${path}`);
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.error?.message || res.statusText);
  }
  return body;
}

function renderUsers() {
  const ns = namespaceEl.value;
  const model = models.find((row) => row.namespace === ns);
  const users = model ? [...model.sample_user_ids, model.cold_start_user_id] : ["new-user"];
  fillSelect(userEl, users);
}

function renderMetrics() {
  const ns = namespaceEl.value;
  const model = models.find((row) => row.namespace === ns);
  metricsEl.replaceChildren();
  if (!model) return;
  for (const [name, row] of Object.entries(model.metrics || {})) {
    const tr = document.createElement("tr");
    appendCells(tr, [
      name,
      fmt(row.recall_at_k),
      fmt(row.ndcg_at_k),
      row.n_eval_users,
      row.n_cold_start_users,
    ]);
    metricsEl.appendChild(tr);
  }
  metaEl.textContent = `${model.namespace} ${model.version} · cutoff ${model.cutoff} · ${model.n_users} users / ${model.n_items} items`;
}

function renderList(target, items, onClick) {
  target.replaceChildren();
  for (const item of items) {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "link";
    btn.type = "button";
    btn.textContent = item.title ? `${item.title} (${item.item_id})` : item.item_id;
    btn.addEventListener("click", () => onClick(item));
    const meta = document.createElement("div");
    meta.className = "hint";
    meta.textContent = `${item.reason || ""} score=${Number(item.score).toFixed(3)}`;
    li.appendChild(btn);
    li.appendChild(meta);
    target.appendChild(li);
  }
}

async function loadSimilar(item) {
  const ns = namespaceEl.value;
  const k = kEl.value;
  statusEl.textContent = `similar-items namespace=${ns} item_id=${item.item_id}`;
  const body = await fetchJson(`/v1/similar-items?namespace=${encodeURIComponent(ns)}&item_id=${encodeURIComponent(item.item_id)}&k=${k}`);
  renderList(similarEl, body.items, loadSimilar);
}

async function recommend() {
  const ns = namespaceEl.value;
  const user = userEl.value;
  const k = kEl.value;
  statusEl.textContent = "loading…";
  const body = await fetchJson(`/v1/recommend?namespace=${encodeURIComponent(ns)}&user_id=${encodeURIComponent(user)}&k=${k}`);
  statusEl.textContent = `${body.model}${body.fallback ? " (cold-start fallback)" : ""} · ${body.version}`;
  renderList(recsEl, body.items, loadSimilar);
  similarEl.replaceChildren();
}

async function boot() {
  const payload = await fetchJson("/v1/models");
  models = payload.models || [];
  fillSelect(namespaceEl, models.map((row) => row.namespace));
  if (!models.length) {
    statusEl.textContent = "no trained models yet";
    return;
  }
  renderUsers();
  renderMetrics();
  await recommend();
}

namespaceEl.addEventListener("change", () => {
  renderUsers();
  renderMetrics();
  recommend().catch((err) => {
    statusEl.textContent = err.message;
  });
});
document.getElementById("run").addEventListener("click", () => {
  recommend().catch((err) => {
    statusEl.textContent = err.message;
  });
});

boot().catch((err) => {
  statusEl.textContent = err.message;
});
