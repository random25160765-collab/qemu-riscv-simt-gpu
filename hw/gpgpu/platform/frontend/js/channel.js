export function connect(url, onRecord, onStatus) {
    const es = new EventSource(url);

    es.onopen = () => onStatus(true);
    es.onerror = () => onStatus(false);

    es.onmessage = (e) => {
        try {
            const rec = JSON.parse(e.data);
            onRecord(rec);
        } catch (err) { console.error("SSE parse error", err, e.data); }
    };

    return es;
}
