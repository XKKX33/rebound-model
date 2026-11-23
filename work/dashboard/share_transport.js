/* Carry the existing SSE protocol through the sharing gateway's WebSocket. */
(() => {
  "use strict";
  const NativeEventSource = window.EventSource;
  const nativeFetch = window.fetch.bind(window);
  const encoder = new TextEncoder();
  const endpoint = () => `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/__share/events`;

  class SharedEventSource extends EventTarget {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 2;
    CONNECTING = 0;
    OPEN = 1;
    CLOSED = 2;

    constructor(url, options = {}) {
      super();
      this.url = new URL(url, location.href).href;
      this.withCredentials = Boolean(options.withCredentials);
      this.readyState = 0;
      this.onopen = this.onmessage = this.onerror = null;
      this.buffer = "";
      this.lastEventId = "";
      this.retry = 3000;
      this.closed = false;
      this.connect();
    }

    emit(type, data) {
      const event = data === undefined
        ? new Event(type)
        : new MessageEvent(type, { data, origin: location.origin, lastEventId: this.lastEventId });
      this.dispatchEvent(event);
      const handler = this[`on${type}`];
      if (typeof handler === "function") handler.call(this, event);
    }

    connect() {
      if (this.closed) return;
      this.readyState = 0;
      this.buffer = "";
      const socket = this.socket = new WebSocket(endpoint());
      socket.onopen = () => socket.send(JSON.stringify({
        path: new URL(this.url).pathname + new URL(this.url).search,
        method: "GET",
      }));
      socket.onmessage = event => {
        const frame = JSON.parse(event.data);
        if (frame.type === "open") {
          this.readyState = 1;
          this.emit("open");
        } else if (frame.type === "data") {
          this.feed(frame.text);
        } else if (frame.type === "http_error" || frame.type === "error") {
          socket.close();
        }
      };
      socket.onclose = () => {
        if (this.closed) return;
        this.readyState = 0;
        this.emit("error");
        this.timer = setTimeout(() => this.connect(), this.retry);
      };
      socket.onerror = () => {};
    }

    feed(text) {
      this.buffer += text.replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = this.buffer.indexOf("\n\n")) >= 0) {
        const frame = this.buffer.slice(0, boundary);
        this.buffer = this.buffer.slice(boundary + 2);
        let type = "message";
        const data = [];
        for (const line of frame.split("\n")) {
          if (!line || line.startsWith(":")) continue;
          const colon = line.indexOf(":");
          const field = colon < 0 ? line : line.slice(0, colon);
          const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") type = value;
          else if (field === "data") data.push(value);
          else if (field === "id" && !value.includes("\0")) this.lastEventId = value;
          else if (field === "retry" && /^\d+$/.test(value)) this.retry = Math.max(1000, Number(value));
        }
        if (data.length) this.emit(type, data.join("\n"));
      }
    }

    close() {
      this.closed = true;
      this.readyState = 2;
      clearTimeout(this.timer);
      this.socket?.close();
    }
  }

  function GatewayEventSource(url, options) {
    const target = new URL(url, location.href);
    if (target.origin === location.origin && target.pathname === "/debate/api/stream") {
      return new SharedEventSource(url, options);
    }
    return new NativeEventSource(url, options);
  }
  GatewayEventSource.CONNECTING = 0;
  GatewayEventSource.OPEN = 1;
  GatewayEventSource.CLOSED = 2;
  GatewayEventSource.prototype = SharedEventSource.prototype;
  window.EventSource = GatewayEventSource;

  async function streamChat(input, options, target) {
    const signal = options.signal || (input instanceof Request ? input.signal : null);
    if (signal?.aborted) throw new DOMException("请求已取消", "AbortError");
    const body = options.body !== undefined
      ? options.body
      : input instanceof Request ? await input.clone().text() : "";
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(endpoint());
      let controller = null;
      let settled = false;
      let ended = false;
      const finish = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
      };
      const fail = error => {
        if (ended) return;
        ended = true;
        finish();
        if (!settled) reject(error);
        else controller?.error(error);
        socket.close();
      };
      const abort = () => fail(new DOMException("请求已取消", "AbortError"));
      const timer = setTimeout(() => fail(new TypeError("共享连接超时，请重试")), 30000);
      signal?.addEventListener("abort", abort, { once: true });
      socket.onopen = () => socket.send(JSON.stringify({
        path: target.pathname + target.search,
        method: "POST",
        body: typeof body === "string" ? body : String(body || ""),
      }));
      socket.onmessage = event => {
        const frame = JSON.parse(event.data);
        if (frame.type === "open") {
          clearTimeout(timer);
          const stream = new ReadableStream({
            start(value) { controller = value; },
            cancel() {
              ended = true;
              finish();
              socket.close();
            },
          });
          settled = true;
          resolve(new Response(stream, { status: frame.status, headers: frame.headers }));
        } else if (frame.type === "data" && controller && !ended) {
          controller.enqueue(encoder.encode(frame.text));
        } else if (frame.type === "end" && !ended) {
          ended = true;
          finish();
          controller?.close();
        } else if (frame.type === "http_error" && !settled) {
          settled = ended = true;
          finish();
          resolve(new Response(frame.body, {
            status: frame.status,
            headers: { "Content-Type": "application/json" },
          }));
        } else if (frame.type === "error") {
          fail(new TypeError(frame.message || "共享连接中断，请重试"));
        }
      };
      socket.onerror = () => fail(new TypeError("共享连接中断，请重试"));
      socket.onclose = () => {
        if (!ended) fail(new TypeError("共享连接中断，请重试"));
      };
    });
  }

  window.fetch = (input, options = {}) => {
    const target = new URL(input instanceof Request ? input.url : String(input), location.href);
    const method = String(options.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
    if (target.origin === location.origin && target.pathname === "/debate/api/chat/stream" && method === "POST") {
      return streamChat(input, options, target);
    }
    return nativeFetch(input, options);
  };
})();
