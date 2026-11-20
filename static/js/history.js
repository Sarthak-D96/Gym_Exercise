// History / Progress charts (UC-02).
(function () {
  const { api, cssVar } = window.Gym;
  const charts = {};

  function baseOptions() {
    const grid = cssVar("--border"), text = cssVar("--muted");
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: text, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
        y: { grid: { color: grid }, ticks: { color: text, precision: 0 }, beginAtZero: true },
      },
    };
  }

  function draw(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), config);
  }

  const label = (iso) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });

  async function load(days) {
    const data = await api(`${window.GYM.urls.history}?days=${days}`);
    const accent = cssVar("--accent");
    const labels = data.daily.map((d) => label(d.date));

    draw("chart-reps", {
      type: "bar",
      data: { labels, datasets: [{ data: data.daily.map((d) => d.reps), backgroundColor: accent, borderRadius: 4, maxBarThickness: 28 }] },
      options: baseOptions(),
    });

    const formOpts = baseOptions();
    formOpts.scales.y.max = 100;
    formOpts.spanGaps = true;
    draw("chart-form", {
      type: "line",
      data: { labels, datasets: [{ data: data.daily.map((d) => d.form_score), borderColor: accent, backgroundColor: accent, tension: 0.3, pointRadius: 3 }] },
      options: formOpts,
    });

    const exOpts = baseOptions();
    exOpts.indexAxis = "y";
    exOpts.scales.y.grid = { display: false };
    exOpts.scales.x.grid = { color: cssVar("--border") };
    draw("chart-ex", {
      type: "bar",
      data: {
        labels: data.by_exercise.map((e) => e.exercise__name),
        datasets: [{ data: data.by_exercise.map((e) => e.reps), backgroundColor: accent, borderRadius: 4, maxBarThickness: 22 }],
      },
      options: exOpts,
    });
  }

  document.querySelectorAll("[data-days]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-days]").forEach((b) => b.classList.toggle("on", b === btn));
      load(Number(btn.dataset.days));
    });
  });
  load(30);
})();
