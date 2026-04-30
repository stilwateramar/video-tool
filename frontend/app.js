const state = {
  uploadId: null,
  analysis: null,
  sampleId: null,
  sampleProfile: null,
  options: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

async function pollJob(jobId, onUpdate) {
  while (true) {
    const j = await api(`/api/job/${jobId}`);
    onUpdate(j);
    if (j.status === "done" || j.status === "error") return j;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function setProgress(el, pct, msg) {
  el.hidden = false;
  el.style.setProperty("--p", `${Math.round(pct * 100)}%`);
  el.querySelector(".bar").style.setProperty("--p", `${Math.round(pct * 100)}%`);
  el.querySelector(".msg").textContent = msg || "";
}

function fmtTime(s) {
  s = Math.max(0, s);
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, "0")}`;
}

// ---------- options bootstrap ----------

async function loadOptions() {
  state.options = await api("/api/options");
}

function fillSelect(sel, items, current) {
  sel.innerHTML = "";
  for (const it of items) {
    const opt = document.createElement("option");
    if (typeof it === "string") {
      opt.value = it;
      opt.textContent = it.replace(/_/g, " ");
    } else {
      opt.value = it.id;
      opt.textContent = it.label;
    }
    if (current && opt.value === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

// ---------- class video ----------

const classFile = $("#class-file");
const classBtn = $("#class-upload-btn");
const classProgress = $("#class-progress");
const classSummary = $("#class-summary");
const poseList = $("#pose-list");

classFile.addEventListener("change", () => {
  classBtn.disabled = !classFile.files.length;
});

classBtn.addEventListener("click", async () => {
  const f = classFile.files[0];
  if (!f) return;
  classBtn.disabled = true;
  setProgress(classProgress, 0.02, "Uploading...");
  const fd = new FormData();
  fd.append("file", f);
  const { job_id, upload_id } = await api("/api/upload", { method: "POST", body: fd });
  state.uploadId = upload_id;
  await pollJob(job_id, (j) => setProgress(classProgress, j.progress, j.message));
  const a = await api(`/api/analysis/${upload_id}`);
  state.analysis = a;
  showAnalysis(a);
  classBtn.disabled = false;
  $("#generate-btn").disabled = false;
});

function showAnalysis(a) {
  classSummary.hidden = false;
  const m = a.meta;
  classSummary.innerHTML = `<dl>
    <dt>File</dt><dd>${a.filename}</dd>
    <dt>Duration</dt><dd>${fmtTime(m.duration)}</dd>
    <dt>Resolution</dt><dd>${m.width}&times;${m.height}</dd>
    <dt>Aspect</dt><dd>${m.aspect_ratio.toFixed(2)}</dd>
    <dt>Held poses found</dt><dd>${a.segments.length}</dd>
  </dl>`;
  if (a.segments.length) {
    poseList.hidden = false;
    const rows = a.segments.map((s) => `
      <tr>
        <td>${fmtTime(s.start)}</td>
        <td>${fmtTime(s.end)}</td>
        <td>${s.duration.toFixed(1)}s</td>
        <td>${s.english}${s.sanskrit ? ` <span style="color:#9a7a52">(${s.sanskrit})</span>` : ""}</td>
        <td>${(s.confidence * 100).toFixed(0)}%</td>
      </tr>`).join("");
    poseList.innerHTML = `<table>
      <thead><tr><th>Start</th><th>End</th><th>Hold</th><th>Pose</th><th>Conf.</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }
}

// ---------- sample video ----------

const sampleFile = $("#sample-file");
const sampleBtn = $("#sample-upload-btn");
const sampleSummary = $("#sample-summary");

sampleFile.addEventListener("change", () => {
  sampleBtn.disabled = !sampleFile.files.length;
});

sampleBtn.addEventListener("click", async () => {
  const f = sampleFile.files[0];
  if (!f) return;
  sampleBtn.disabled = true;
  const fd = new FormData();
  fd.append("file", f);
  const { sample_id, profile } = await api("/api/sample", { method: "POST", body: fd });
  state.sampleId = sample_id;
  state.sampleProfile = profile;
  sampleSummary.hidden = false;
  sampleSummary.innerHTML = `<dl>
    <dt>Format</dt><dd>${profile.target_format}</dd>
    <dt>Aspect ratio</dt><dd>${profile.aspect_ratio.toFixed(2)}</dd>
    <dt>Avg shot length</dt><dd>${profile.avg_shot_seconds.toFixed(1)}s</dd>
    <dt>Suggested speed</dt><dd>${profile.suggested_speed.toFixed(2)}&times;</dd>
    <dt>Suggested preset</dt><dd>${profile.suggested_preset}</dd>
    <dt>Brightness / saturation</dt>
    <dd>${profile.avg_brightness.toFixed(2)} / ${profile.avg_saturation.toFixed(2)}</dd>
  </dl>`;
  sampleBtn.disabled = false;
  // Re-apply defaults to all reel rows that haven't been customized.
  $$("#reel-list .reel-cfg").forEach((row) => applySampleDefaults(row));
});

// ---------- reel rows ----------

const reelList = $("#reel-list");
const tmpl = $("#reel-template");

function newReelRow(name) {
  const node = tmpl.content.firstElementChild.cloneNode(true);
  fillSelect(node.querySelector(".r-kind"), state.options.kinds, "sequence");
  fillSelect(node.querySelector(".r-format"), state.options.formats, "reel");
  fillSelect(node.querySelector(".r-preset"), state.options.presets, "natural");
  fillSelect(node.querySelector(".r-audio"), state.options.audio_cleanup, "light");
  fillSelect(node.querySelector(".r-music"), state.options.music, "none");
  node.querySelector(".r-name").value = name;
  node.querySelector(".r-kind").addEventListener("change", (e) => {
    const focus = node.querySelector(".pose-focus");
    focus.hidden = e.target.value !== "pose_focus";
  });
  node.querySelector(".r-remove").addEventListener("click", () => node.remove());
  applySampleDefaults(node);
  return node;
}

function applySampleDefaults(row) {
  if (!state.sampleProfile) return;
  if (!$("#apply-style-toggle").checked) return;
  const p = state.sampleProfile;
  row.querySelector(".r-format").value = p.target_format;
  row.querySelector(".r-preset").value = p.suggested_preset;
  row.querySelector(".r-speed").value = p.suggested_speed.toFixed(2);
}

$("#add-reel-btn").addEventListener("click", () => {
  const n = reelList.children.length + 1;
  reelList.appendChild(newReelRow(`reel_${n}`));
});

// ---------- generate ----------

$("#generate-btn").addEventListener("click", async () => {
  if (!state.uploadId) return;
  const rows = $$("#reel-list .reel-cfg");
  const reels = [];
  rows.forEach((row, i) => {
    const kind = row.querySelector(".r-kind").value;
    const r = {
      name: row.querySelector(".r-name").value || `reel_${i + 1}`,
      kind,
      target_format: row.querySelector(".r-format").value,
      target_seconds: parseInt(row.querySelector(".r-secs").value, 10),
      color_preset: row.querySelector(".r-preset").value,
      audio_cleanup: row.querySelector(".r-audio").value,
      music: row.querySelector(".r-music").value,
      speed: parseFloat(row.querySelector(".r-speed").value),
      add_pose_overlay: row.querySelector(".r-overlay").checked,
      add_breath_cue: row.querySelector(".r-breath").checked,
      add_progress_bar: row.querySelector(".r-progress").checked,
      apply_style: $("#apply-style-toggle").checked,
    };
    if (kind === "pose_focus") {
      const list = row.querySelector(".r-poses").value
        .split(",").map((s) => s.trim()).filter(Boolean);
      r.pose_filter = list;
    }
    reels.push(r);
  });
  if (!reels.length) {
    alert("Add at least one reel");
    return;
  }
  setProgress($("#gen-progress"), 0.02, "Submitting...");
  const { job_id } = await api("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      upload_id: state.uploadId,
      sample_id: state.sampleId,
      reels,
    }),
  });
  const final = await pollJob(job_id, (j) =>
    setProgress($("#gen-progress"), j.progress, j.message)
  );
  if (final.status === "error") {
    setProgress($("#gen-progress"), 1, `Error: ${final.error}`);
    return;
  }
  showResults(final.result.reels);
});

function showResults(reels) {
  const root = $("#results");
  root.innerHTML = "";
  for (const r of reels) {
    const div = document.createElement("div");
    div.className = "result";
    if (r.error) {
      div.innerHTML = `<h3>${r.name}</h3><div class="meta">${r.error}</div>`;
    } else {
      div.innerHTML = `
        <video src="${r.url}" controls playsinline></video>
        <h3>${r.name}</h3>
        <div class="meta">
          ${r.kind} &middot; ${r.format} &middot; ${r.duration.toFixed(1)}s
          &middot; ${r.color_preset} &middot; ${r.speed.toFixed(2)}&times;
        </div>
        <div class="meta">
          <a href="${r.url}" download>Download</a>
        </div>`;
    }
    root.appendChild(div);
  }
}

// ---------- bootstrap ----------

(async () => {
  await loadOptions();
  reelList.appendChild(newReelRow("reel_1"));
})();
