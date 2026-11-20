// Admin usage report charts (US-06).
(function () {
  const { api, cssVar } = window.Gym;

  function opts() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      spanGaps: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: cssVar("--muted"), maxTicksLimit: 8, maxRotation: 0 } },
        y: { grid: { color: cssVar("--border") }, ticks: { color: cssVar("--muted") }, beginAtZero: true },
      },
    };
  }

  api(window.GYM.urls.data).then((d) => {
    const accent = cssVar("--accent");
    const labels = d.days.map((iso) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }));
    new Chart(document.getElementById("chart-sessions"), {
      type: "bar",
      data: { labels, datasets: [{ data: d.sessions, backgroundColor: accent, borderRadius: 4, maxBarThickness: 24 }] },
      options: opts(),
    });
    new Chart(document.getElementById("chart-latency"), {
      type: "line",
      data: { labels, datasets: [{ data: d.latency, borderColor: cssVar("--info"), backgroundColor: cssVar("--info"), tension: 0.3, pointRadius: 3 }] },
      options: opts(),
    });
  });
})();
