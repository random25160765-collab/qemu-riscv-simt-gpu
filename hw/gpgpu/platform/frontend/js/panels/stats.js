export function createStatsPanel(store, { slowEl, fastEl, statusEl }) {
    let sc = 0, fc = 0;

    function setStatus(ok) {
        statusEl.className = `status ${ok ? "on" : "off"}`;
        statusEl.title = ok ? "connected" : "disconnected";
    }

    store.on("record", (rec) => {
        if (rec.ring_type === "slow") { sc++; slowEl.textContent = sc; }
        else                          { fc++; fastEl.textContent = fc; }
    });

    return { setStatus };
}
