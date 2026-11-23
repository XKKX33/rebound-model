"""Password-protected gateway for a temporary classroom or team demonstration."""
from __future__ import annotations

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

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
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
DROP_HEADERS = {"host", "connection", "transfer-encoding", "content-length", "authorization"}


@asynccontextmanager
async def lifespan(application):
    application.state.client = httpx.AsyncClient(base_url=UPSTREAM, timeout=None, trust_env=False)
    try:
        yield
    finally:
        await application.state.client.aclose()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.api_route("/{route:path}", methods=["GET", "POST", "DELETE"])
async def forward(route: str, request: Request):
    if not hmac.compare_digest(request.headers.get("authorization", ""), EXPECTED):
        return JSONResponse(
            {"detail": "请输入小组共享访问密码"}, status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="A-share workbench"', "Cache-Control": "no-store"},
        )
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
    return StreamingResponse(
        response.aiter_raw(), status_code=response.status_code,
        headers={key: value for key, value in response.headers.items() if key.lower() not in DROP_HEADERS},
        background=BackgroundTask(response.aclose),
    )


if __name__ == "__main__":
    import uvicorn

    print(f"共享网关将转发 {UPSTREAM}；访问凭据保存在 {ACCESS_FILE.name}。", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DASHBOARD_SHARE_PORT", "8880")), log_level="warning")
