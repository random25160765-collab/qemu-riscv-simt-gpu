export function createStore() {
    const listeners = {};

    function on(evt, fn) {
        (listeners[evt] = listeners[evt] || []).push(fn);
    }

    function emit(evt, data) {
        (listeners[evt] || []).forEach(fn => fn(data));
    }

    function append(rec) {
        emit("record", rec);
    }

    return { on, append };
}
