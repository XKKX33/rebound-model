"""Password-protected gateway for a temporary classroom or team demonstration."""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import secrets
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from paths import PROJECT
sys.path.insert(0, str(PROJECT / "work/runtime"))
sys.path.insert(0, str(PROJECT / "work/sharing/runtime/python"))

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

ACCESS_FILE = Path(__file__).with_name("share_access.json")
UPSTREAM = os.environ.get("DASHBOARD_SHARE_UPSTREAM", "http://127.0.0.1:8875").rstrip("/")


def access_settings():
    password = os.environ.get("DASHBOARD_SHARE_PASSWORD", "").strip()
    if password:
        return {"username": "team", "password": password}
    if ACCESS_FILE.is_file():
        return json.loads(ACCESS_FILE.read_text(encoding="utf-8"))
    settings = {"username": "team", "password": secrets.token_urlsafe(15)}
    ACCESS_FILE.write_text(json.dumps(settings), encoding="utf-8")
    return settings


ACCESS = access_settings()
EXPECTED = "Basic " + base64.b64encode(f"{ACCESS['username']}:{ACCESS['password']}".encode()).decode()
SESSION = secrets.token_urlsafe(32)
SESSION_COOKIE = "workbench_share_session"
TRANSPORT = Path(__file__).with_name("share_transport.js")
DROP_HEADERS = {
    "host", "connection", "transfer-encoding", "content-length", "authorization",
    "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade",
}
STREAM_ROUTES = {("GET", "/debate/api/stream"), ("POST", "/debate/api/chat/stream")}


@asynccontextmanager
async def lifespan(application):
    application.state.client = httpx.AsyncClient(base_url=UPSTREAM, timeout=None, trust_env=False)
    try:
        yield
    finally:
        await application.state.client.aclose()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


def authenticated(connection):
    return (
        hmac.compare_digest(connection.headers.get("authorization", ""), EXPECTED)
        or hmac.compare_digest(connection.cookies.get(SESSION_COOKIE, ""), SESSION)
    )


def unauthorized():
    return JSONResponse(
        {"detail": "请输入小组共享访问密码"}, status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="A-share workbench"', "Cache-Control": "no-store"},
    )


@app.get("/__share/transport.js")
async def transport_script(request: Request):
    if not authenticated(request):
        return unauthorized()
    return Response(
        TRANSPORT.read_bytes(), media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/__share/health")
async def gateway_health(request: Request):
    if not authenticated(request):
        return unauthorized()
    try:
        response = await request.app.state.client.get("/api/health", timeout=5)
        ready = response.status_code == 200 and response.json().get("status") == "ready"
    except (httpx.RequestError, ValueError):
        ready = False
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "transport": "websocket-v1"},
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


@app.websocket("/__share/events")
async def stream_socket(socket: WebSocket):
    origin = socket.headers.get("origin")
    allowed_origins = {f"http://{socket.headers.get('host')}", f"https://{socket.headers.get('host')}"}
    if not authenticated(socket) or (origin and origin not in allowed_origins):
        await socket.close(code=1008)
        return
    await socket.accept()
    upstream = None
    tasks = []
    send_lock = asyncio.Lock()

    async def send(payload):
        async with send_lock:
            await socket.send_json(payload)

    try:
        instruction = await asyncio.wait_for(socket.receive_json(), timeout=10)
        method = str(instruction.get("method", "GET")).upper()
        path = str(instruction.get("path", ""))
        if (method, path) not in STREAM_ROUTES:
            await send({"type": "http_error", "status": 400, "body": '{"error":"不支持的流式请求"}'})
            return
        body = str(instruction.get("body") or "").encode("utf-8")
        if len(body) > 1_048_576:
            await send({"type": "http_error", "status": 413, "body": '{"error":"请求内容过大"}'})
            return
        client = socket.app.state.client
        outgoing = client.build_request(
            method, path, content=body,
            headers={"Accept": "text/event-stream", "Accept-Encoding": "identity", "Content-Type": "application/json"},
        )
        upstream = await client.send(outgoing, stream=True)
        if upstream.status_code >= 400:
            content = (await upstream.aread()).decode("utf-8", errors="replace")[:4096]
            await send({"type": "http_error", "status": upstream.status_code, "body": content})
            return
        if "text/event-stream" not in upstream.headers.get("content-type", ""):
            await send({"type": "error", "message": "流式接口未返回事件数据"})
            return
        await send({
            "type": "open", "status": upstream.status_code,
            "headers": {"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache"},
        })

        async def forward_events():
            async for text in upstream.aiter_text():
                if text:
                    await send({"type": "data", "text": text})
            await send({"type": "end"})

        async def wait_disconnect():
            while True:
                message = await socket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("text") == "abort":
                    return

        async def keepalive():
            while True:
                await asyncio.sleep(15)
                await send({"type": "heartbeat"})

        tasks = [
            asyncio.create_task(forward_events()),
            asyncio.create_task(wait_disconnect()),
            asyncio.create_task(keepalive()),
        ]
        completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in completed:
            task.result()
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except (httpx.RequestError, ValueError, TypeError, TimeoutError):
        try:
            await send({"type": "error", "message": "共享连接中断，请重试"})
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if upstream is not None:
            await upstream.aclose()
        try:
            await socket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.api_route("/{route:path}", methods=["GET", "POST", "DELETE"])
async def forward(route: str, request: Request):
    if not authenticated(request):
        return unauthorized()
    content = await request.body()
    if len(content) > 1_048_576:
        return JSONResponse({"detail": "请求内容过大"}, status_code=413)
    target = "/" + route + ("?" + request.url.query if request.url.query else "")
    client = request.app.state.client
    try:
        outgoing = client.build_request(
            request.method, target, content=content,
            headers={key: value for key, value in request.headers.items() if key.lower() not in DROP_HEADERS},
        )
        response = await client.send(outgoing, stream=True)
    except httpx.RequestError:
        return JSONResponse({"detail": "本机看板未启动"}, status_code=503)
    headers = {key: value for key, value in response.headers.items() if key.lower() not in DROP_HEADERS}
    if response.status_code == 200 and "text/html" in response.headers.get("content-type", ""):
        html = (await response.aread()).decode("utf-8")
        html = html.replace("</head>", '<script src="/__share/transport.js"></script></head>', 1)
        await response.aclose()
        headers = {key: value for key, value in headers.items() if key.lower() not in {"content-encoding", "etag"}}
        headers["Cache-Control"] = "no-store"
        result = Response(html, status_code=response.status_code, headers=headers)
        result.set_cookie(
            SESSION_COOKIE, SESSION, httponly=True, samesite="strict",
            secure=request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https",
        )
        return result
    return StreamingResponse(
        response.aiter_raw(), status_code=response.status_code,
        headers=headers,
        background=BackgroundTask(response.aclose),
    )


if __name__ == "__main__":
    import uvicorn

    print(f"共享网关将转发 {UPSTREAM}；访问凭据保存在 {ACCESS_FILE.name}。", flush=True)
    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("DASHBOARD_SHARE_PORT", "8880")),
        ws="wsproto", log_level="warning",
    )
