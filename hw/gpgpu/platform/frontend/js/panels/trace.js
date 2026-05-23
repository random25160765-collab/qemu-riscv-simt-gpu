import { formatRecord } from "../format.js";

export function createTracePanel(store, container, { ringType, maxItems = 500 }) {
    store.on("record", (rec) => {
        if (rec.ring_type !== ringType) return;

        const text = formatRecord(rec);
        const div = document.createElement("div");
        div.className = "record";
        div.textContent = text;
        container.prepend(div);

        while (container.children.length > maxItems) {
            container.lastChild.remove();
        }
    });
}
