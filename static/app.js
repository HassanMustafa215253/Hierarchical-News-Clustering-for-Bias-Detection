const backendUrlInput = document.querySelector("#backendUrl");
const searchForm = document.querySelector("#searchForm");
const queryInput = document.querySelector("#query");
const statusBox = document.querySelector("#status");
const overview = document.querySelector("#overview");
const groups = document.querySelector("#groups");

backendUrlInput.value = localStorage.getItem("colabBackendUrl") || "";

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(message, type = "info") {
  statusBox.className = `status ${type}`;
  statusBox.textContent = message;
}

function clearStatus() {
  statusBox.className = "status hidden";
  statusBox.textContent = "";
}

function endpoint() {
  const backendUrl = backendUrlInput.value.trim().replace(/\/$/, "");
  if (backendUrl) localStorage.setItem("colabBackendUrl", backendUrl);
  return "/api/search";
}

function renderResults(data) {
  clearStatus();
  overview.classList.remove("hidden");
  overview.innerHTML = `
    <article><span>Query</span><strong>${escapeHtml(data.query)}</strong></article>
    <article><span>Results</span><strong>${data.total_results ?? 0}</strong></article>
    <article><span>Axis</span><strong>${escapeHtml(data.axis?.positive || "Pro A")} ↔ ${escapeHtml(data.axis?.negative || "Pro B")}</strong></article>
  `;

  groups.innerHTML = (data.groups || []).map((group) => `
    <article class="group ${escapeHtml(group.kind)}">
      <header>
        <div>
          <p>${escapeHtml(group.kind)}</p>
          <h2>${escapeHtml(group.label)}</h2>
        </div>
        <span>${group.count} result${group.count === 1 ? "" : "s"}</span>
      </header>
      <div class="result-list">
        ${(group.items || []).map((item) => `
          <div class="result">
            <span class="rank">#${item.rank}</span>
            <div>
              <h3>${escapeHtml(item.title)}</h3>
              <p>Cluster ${escapeHtml(item.cluster)} · relevance ${Number(item.relevance || 0).toFixed(3)}</p>
            </div>
          </div>
        `).join("") || "<p class='empty-line'>No results in this group.</p>"}
      </div>
    </article>
  `).join("");
}

async function runSearch(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  groups.innerHTML = "";
  overview.classList.add("hidden");
  setStatus("Searching in Colab… this can take a little while.", "loading");

  try {
    const response = await fetch(endpoint(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: 20,
        backend_url: backendUrlInput.value.trim(),
      }),
    });
    const data = await response.json();
    if (!response.ok || data.error) {
      throw new Error(data.error || `Request failed with status ${response.status}`);
    }
    renderResults(data);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

searchForm.addEventListener("submit", runSearch);
