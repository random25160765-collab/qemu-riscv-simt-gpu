import { connect } from "./channel.js";
import { createStore } from "./store.js";
import { createTracePanel } from "./panels/trace.js";
import { createStatsPanel } from "./panels/stats.js";

const vpuId = parseInt(new URLSearchParams(location.search).get("vpu") || "0", 10);

const store = createStore();

createTracePanel(store, document.getElementById("slow-records"), {
    ringType: "slow", maxItems: 500,
});
createTracePanel(store, document.getElementById("fast-records"), {
    ringType: "fast", maxItems: 200,
});

const stats = createStatsPanel(store, {
    slowEl: document.getElementById("slow-count"),
    fastEl: document.getElementById("fast-count"),
    statusEl: document.getElementById("status-dot"),
});

connect("/events", store.append, stats.setStatus);

document.getElementById("vpu-label").textContent = `VPU-${vpuId}`;

const slider = document.getElementById("period-slider");
const periodVal = document.getElementById("period-value");
slider.addEventListener("input", () => {
    periodVal.textContent = slider.value + "ms";
});
slider.addEventListener("change", async () => {
    const ms = parseInt(slider.value, 10);
    try {
        await fetch("/set-period", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ period_ms: ms }),
        });
        periodVal.textContent = ms + "ms";
    } catch (e) { console.error("set-period failed", e); }
});
