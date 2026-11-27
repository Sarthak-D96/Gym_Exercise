// Live workout page: camera capture -> /frames API -> overlay + stats.
(function () {
  const { api, fmtDuration } = window.Gym;
  const cfg = window.GYM;
  const $ = (id) => document.getElementById(id);

  const SKELETON = [
    ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"],
    ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["nose", "left_eye"], ["nose", "right_eye"], ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ];
  const FRAME_INTERVAL_MS = 150;  // ~6-7 fps upper bound; next frame waits for the previous response
  const CAPTURE_WIDTH = 640;  // same resolution as the training videos

  const video = $("video");
  const overlay = $("overlay");
  const ctx = overlay.getContext("2d");
  const capture = document.createElement("canvas");

  const summary = JSON.parse($("session-data").textContent);
  const state = {
    status: summary.status,
    source: "camera",            // "camera" | "simulation"
    stream: null,
    activeMs: summary.duration_seconds * 1000,
    resumedAt: null,             // performance.now() when last resumed
    loopTimer: null,
    inFlight: false,
    lastResult: null,
    lastExercise: null,
    totalBase: summary.total_reps,
    frames: summary.frames_processed,
    logEmpty: true,
  };

  // ------------------------------------------------------------------ setup
  async function loadExercises() {
    try {
      const data = await api(cfg.urls.exercises);
      data.exercises.forEach((e) => { cfg.exercises[e.slug] = e; });
    } catch (_) { /* icons are cosmetic */ }
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("Camera API not available");
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false,
    });
    video.srcObject = state.stream;
    await video.play().catch(() => {});
  }

  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
  }

  async function setSource(source, { silent = false } = {}) {
    if (source === "camera") {
      try {
        await startCamera();
      } catch (err) {
        if (!silent) logEvent({ level: "info", message: `Camera unavailable (${err.message}) — using simulation mode.` });
        source = "simulation";
      }
    } else {
      stopCamera();
    }
    state.source = source;
    $("src-camera").classList.toggle("on", source === "camera");
    $("src-sim").classList.toggle("on", source === "simulation");
    $("source-badge").textContent = source === "camera" ? "📷 Camera" : "🧪 Simulation";
    video.hidden = source !== "camera";
  }

  // ------------------------------------------------------------------ session state
  function currentActiveMs() {
    return state.activeMs + (state.resumedAt ? performance.now() - state.resumedAt : 0);
  }

  function renderStatus() {
    const s = state.status;
    const labels = { ready: "Ready", active: "Workout active", paused: "Paused", completed: "Completed" };
    $("status-text").textContent = labels[s] || s;
    const badge = $("status-badge");
    badge.className = "badge " + ({ active: "badge-good", paused: "badge-warn", completed: "badge-info" }[s] || "");
    $("rec-badge").classList.toggle("live-on", s === "active");
    $("btn-begin").hidden = s !== "ready";
    $("btn-pause").hidden = s !== "active";
    $("btn-resume").hidden = s !== "paused";
    const ov = $("stage-overlay");
    if (s === "active") {
      ov.hidden = true;
    } else {
      ov.hidden = false;
      $("overlay-title").textContent = s === "paused" ? "Paused" : "Ready when you are";
      $("overlay-text").innerHTML = s === "paused"
        ? "Press <strong>Resume</strong> (or Space) to continue."
        : "Stand so your whole body is visible, then press <strong>Begin</strong>.";
    }
  }

  async function transition(event) {
    const url = cfg.urls.transition.replace("EVENT", event);
    try {
      const data = await api(url, { method: "POST" });
      if (state.status === "active" && state.resumedAt) {
        state.activeMs += performance.now() - state.resumedAt;
        state.resumedAt = null;
      }
      state.status = data.status;
      if (state.status === "active") {
        state.resumedAt = performance.now();
        scheduleNext(0);
      } else {
        clearTimeout(state.loopTimer);
      }
      renderStatus();
      return data;
    } catch (err) {
      logEvent({ level: "warning", message: err.message });
      throw err;
    }
  }

  // ------------------------------------------------------------------ frame loop
  function scheduleNext(delay) {
    clearTimeout(state.loopTimer);
    state.loopTimer = setTimeout(tick, delay);
  }

  function grabFrame() {
    if (state.source !== "camera" || !video.videoWidth) return null;
    const scale = CAPTURE_WIDTH / video.videoWidth;
    capture.width = CAPTURE_WIDTH;
    capture.height = Math.round(video.videoHeight * scale);
    capture.getContext("2d").drawImage(video, 0, 0, capture.width, capture.height);
    return capture.toDataURL("image/jpeg", 0.6);
  }

  async function tick() {
    if (state.status !== "active" || state.inFlight) return;
    state.inFlight = true;
    const started = performance.now();
    try {
      const result = await api(cfg.urls.frame, {
        method: "POST",
        body: { frame: grabFrame(), t: Math.round(currentActiveMs()) },
      });
      render(result);
    } catch (err) {
      if (err.status === 409) {
        // Session changed state elsewhere (e.g. ended in another tab).
        logEvent({ level: "warning", message: err.message });
        clearTimeout(state.loopTimer);
        state.inFlight = false;
        return;
      }
      logEvent({ level: "warning", message: `Frame failed: ${err.message}` });
    } finally {
      state.inFlight = false;
    }
    if (state.status === "active") {
      scheduleNext(Math.max(0, FRAME_INTERVAL_MS - (performance.now() - started)));
    }
  }

  // ------------------------------------------------------------------ rendering
  function render(r) {
    state.lastResult = r;
    $("latency-badge").textContent = `${r.latency_ms} ms`;
    $("frames").textContent = `${r.session.frames_processed} frames`;
    $("ml-badge").textContent = r.simulated ? "ML · placeholder" : "ML";
    $("ml-badge").className = "badge " + (r.simulated ? "badge-warn" : "badge-good");

    // Exercise + confidence
    if (r.exercise) {
      const ex = cfg.exercises[r.exercise] || {};
      $("ex-name").textContent = r.exercise_name || ex.name || r.exercise;
      $("ex-icon").textContent = ex.icon || "🏋️";
    } else {
      $("ex-name").textContent = r.pose_detected ? "Recognising…" : "No person detected";
      $("ex-icon").textContent = "✨";
    }
    const conf = Math.round((r.confidence || 0) * 100);
    $("conf-text").textContent = r.exercise ? `${conf}%` : "–";
    $("conf-bar").style.width = `${r.exercise ? conf : 0}%`;
    if (r.exercise_changed && r.exercise) {
      logEvent({ level: "info", message: `Recognised ${r.exercise_name || r.exercise}.` });
    }

    // Reps
    $("reps").textContent = r.reps;
    $("good-reps").textContent = r.good_reps;
    $("rejected-reps").textContent = r.rejected_reps || 0;
    if (r.last_rep) {
      const lr = $("last-rep");
      lr.textContent = r.last_rep.counted ? `${r.last_rep.score}/100` : "not counted";
      lr.className = "mono " + { good: "text-good", minor: "text-info", major: "text-warn" }[r.last_rep.severity];
    }
    $("angle").textContent = r.angle == null ? "–" : `${r.angle}°`;
    $("total-reps").textContent = r.session.total_reps;
    $("rom-bar").style.height = `${Math.round(r.progress * 100)}%`;
    if (r.rep_completed) {
      const el = $("reps");
      el.classList.add("pulse");
      setTimeout(() => el.classList.remove("pulse"), 220);
    }

    // Feedback
    const live = $("feedback-live");
    live.innerHTML = "";
    const items = r.live_feedback.length ? r.live_feedback
      : r.exercise ? [{ level: "good", message: "Form looks good — keep going!" }] : [];
    items.forEach((f) => {
      const div = document.createElement("div");
      div.className = `fb fb-${f.level}`;
      div.textContent = f.message;
      live.appendChild(div);
    });
    r.events.forEach(logEvent);

    drawSkeleton(r.keypoints);
  }

  function logEvent(e) {
    const log = $("event-log");
    if (state.logEmpty) { log.innerHTML = ""; state.logEmpty = false; }
    const li = document.createElement("li");
    const badge = { good: "badge-good", warning: "badge-warn", info: "badge-info" }[e.level] || "";
    const time = document.createElement("time");
    time.textContent = fmtDuration(currentActiveMs());
    const b = document.createElement("span");
    b.className = `badge ${badge}`;
    b.textContent = e.level;
    const msg = document.createElement("span");
    msg.textContent = e.message;
    li.append(time, b, msg);
    log.prepend(li);
    while (log.children.length > 60) log.lastChild.remove();
  }

  function drawSkeleton(kps) {
    const rect = overlay.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (overlay.width !== Math.round(rect.width * dpr)) {
      overlay.width = Math.round(rect.width * dpr);
      overlay.height = Math.round(rect.height * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    if (!kps || !Object.keys(kps).length) return;
    const P = (k) => [kps[k].x * rect.width, kps[k].y * rect.height];
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(143, 209, 106, 0.9)";
    SKELETON.forEach(([a, b]) => {
      if (!kps[a] || !kps[b]) return;
      ctx.beginPath();
      ctx.moveTo(...P(a));
      ctx.lineTo(...P(b));
      ctx.stroke();
    });
    ctx.fillStyle = "#ffffff";
    Object.keys(kps).forEach((k) => {
      const [x, y] = P(k);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // ------------------------------------------------------------------ UI wiring
  function startTimerDisplay() {
    setInterval(() => { $("timer").textContent = fmtDuration(currentActiveMs()); }, 250);
  }

  $("btn-begin").addEventListener("click", () => transition("begin"));
  $("btn-pause").addEventListener("click", () => transition("pause"));
  $("btn-resume").addEventListener("click", () => transition("resume"));
  $("btn-end").addEventListener("click", async () => {
    if (!confirm("End this workout and save it to your history?")) return;
    try {
      await transition("end");
      stopCamera();
      window.location.href = cfg.urls.detail;
    } catch (_) { /* logged */ }
  });
  $("src-camera").addEventListener("click", () => setSource("camera"));
  $("src-sim").addEventListener("click", () => setSource("simulation"));
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.target.matches("input, textarea, button")) return;
    e.preventDefault();
    if (state.status === "active") transition("pause");
    else if (state.status === "paused") transition("resume");
    else if (state.status === "ready") transition("begin");
  });
  window.addEventListener("beforeunload", stopCamera);

  (async function init() {
    await loadExercises();
    await setSource("camera", { silent: false });
    renderStatus();
    startTimerDisplay();
    // A reload during an active workout: pause so time isn't counted while the page was gone.
    if (state.status === "active") {
      try { await transition("pause"); } catch (_) { /* ignore */ }
    }
  })();
})();
