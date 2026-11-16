/* Tiện ích dùng chung: Chart.js defaults, API, định dạng, tab trang. */

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const CLS = ["Dropout", "Enrolled", "Graduate"];
const CLS_VI = { Dropout: "Bỏ học", Enrolled: "Còn đang học", Graduate: "Tốt nghiệp" };
const clsColor = (c) => css("--" + c.toLowerCase());
const MODEL_SHORT = { knn: "K-NN", nb: "Naive Bayes", dt: "Decision Tree", dummy: "Dummy" };
const SCEN_VI = { admission: "Lúc nhập học", sem1: "Sau học kỳ 1", full: "Sau năm thứ nhất" };

const fmt = {
  pct: (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d) + "%"),
  num: (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d)),
  int: (v) => (v == null ? "—" : Number(v).toLocaleString("vi-VN")),
};

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { const err = new Error(data.error || "Lỗi máy chủ"); err.data = data; err.status = res.status; throw err; }
  return data;
}

function toast(msg, ms = 2800) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove("show"), ms);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function applyChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = '"Be Vietnam Pro", system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--border");
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.tooltip.backgroundColor = css("--text");
  Chart.defaults.plugins.tooltip.titleColor = css("--bg");
  Chart.defaults.plugins.tooltip.bodyColor = css("--bg");
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.maintainAspectRatio = false;
}

const charts = {};
function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return null;
  charts[id] = new Chart(el, config);
  charts[id]._cfg = config;
  return charts[id];
}

function confusionHTML(cm) {
  const rowsum = cm.map((r) => r.reduce((a, b) => a + b, 0));
  let h = '<div class="cm"><div class="h"></div>' + CLS.map((c) => `<div class="h">Dự đoán<br>${CLS_VI[c]}</div>`).join("");
  cm.forEach((row, i) => {
    h += `<div class="h" style="justify-content:flex-end">${CLS_VI[CLS[i]]}</div>`;
    row.forEach((v, j) => {
      const share = rowsum[i] ? v / rowsum[i] : 0;
      const base = i === j ? "43,154,102" : "224,62,62";
      h += `<div class="c" style="background:rgba(${base},${0.08 + share * 0.8});color:${share > 0.5 ? "#fff" : "inherit"}">${v}<small>${fmt.pct(share, 0)}</small></div>`;
    });
  });
  return h + "</div>";
}

document.addEventListener("DOMContentLoaded", () => {
  applyChartDefaults();
  // Tab cấp trang: <div class="page-tabs" data-tabs="x"><button data-tab="a">…  và  <section data-pane="a">…
  document.querySelectorAll(".page-tabs").forEach((bar) => {
    const buttons = [...bar.querySelectorAll("button[data-tab]")];
    const show = (key) => {
      buttons.forEach((b) => b.classList.toggle("active", b.dataset.tab === key));
      document.querySelectorAll("[data-pane]").forEach((p) => p.classList.toggle("active", p.dataset.pane === key));
      // Biểu đồ tạo khi tab đang ẩn có kích thước 0, nên vẽ lại khi tab hiện ra.
      Object.entries(charts).forEach(([id, c]) => {
        if (c && c.canvas.offsetParent && !c.canvas.width) makeChart(id, c._cfg);
      });
      try { history.replaceState(null, "", "#" + key); } catch (e) {}
    };
    buttons.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
    const fromHash = location.hash.slice(1);
    show(buttons.some((b) => b.dataset.tab === fromHash) ? fromHash : buttons[0].dataset.tab);
  });
  document.querySelectorAll("[data-nav-toggle]").forEach((b) =>
    b.addEventListener("click", () => document.body.classList.toggle("nav-open"))
  );
});
