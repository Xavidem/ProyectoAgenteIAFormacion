from __future__ import annotations
import os
import json
import asyncio
import re
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, List, Optional

import requests
from requests.cookies import RequestsCookieJar
from urllib.parse import quote
from playwright.async_api import async_playwright

# ============================================================
#  CONFIGURACIÓN BÁSICA DEL SITIO Y DOCROOT
# ============================================================

STORAGE_STATE = Path(os.getenv("SP_STORAGE_STATE", ".sp_storage_state.json")).resolve()

SP_SITE_HOST = "everisgroup.sharepoint.com"
SP_SITE_PATH = "/sites/FormacionesCertificaciones"

# relativa al sitio
SP_DOC_ROOT = "/Documentos compartidos/Documentacion General/DOCUMENTACIÓN"

SP_LOGIN_URL = (
    "https://everisgroup.sharepoint.com/sites/FormacionesCertificaciones/"
    "Documentos%20compartidos/Forms/AllItems.aspx"
    "?id=%2Fsites%2FFormacionesCertificaciones%2FDocumentos%20compartidos"
    "%2FDocumentacion%20General%2FDOCUMENTACI%C3%93N"
)

BASE_WEB  = f"https://{SP_SITE_HOST}"
BASE_SITE = f"{BASE_WEB}{SP_SITE_PATH}"
BASE_API  = f"{BASE_SITE}/_api"
NAV_TIMEOUT_MS = int(os.getenv("SP_NAV_TIMEOUT_MS", "600000"))
REQ_TIMEOUT_MS = int(os.getenv("SP_REQ_TIMEOUT_MS", "600000"))


FEDAUTH_NAMES = {"FedAuth", "rtFa"}


# ============================================================
#  INICIO: HELPERS GENERALES
# ============================================================

def _normalize_server_relative(path: str) -> str:
    s = str(path or "")
    s = s.replace("\\", "/")
    s = re.sub(r"^https?://[^/]+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"/{2,}", "/", s)

    if not s.startswith("/"):
        s = "/" + s

    site_prefix = SP_SITE_PATH.rstrip("/")
    if s.lower().startswith((site_prefix + "/").lower()):
        s = s[len(site_prefix):] 

    s = re.sub(r"/{2,}", "/", s)

    return s


def _to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        for e in x:
            if isinstance(e, str) and e:
                return e
        return str(x[0]) if x else ""
    if isinstance(x, (bool, int, float)):
        return str(x)
    return str(x)


def _normalize_domain(domain) -> str:
    if domain is None:
        return ""
    if isinstance(domain, (list, tuple)):
        for d in domain:
            if isinstance(d, str) and d:
                return d
        return str(domain[0]) if domain else ""
    return str(domain)


def _normalize_path(path) -> str:
    if not path:
        return "/"
    if isinstance(path, (list, tuple)):
        for p in path:
            if isinstance(p, str) and p:
                return p
        return "/"
    return str(path)

def _no_browser_login() -> bool:
    return os.getenv("SP_NO_BROWSER_LOGIN", "")

# ============================================================
#  FIN: HELPERS GENERALES
# ============================================================
# ============================================================
#  INICIO: GESTIÓN DE STORAGE_STATE Y COOKIES
# ============================================================

def _load_cookiejar_from_storage() -> RequestsCookieJar:
    """
    Carga cookies de Playwright storage_state.json y las mapea a RequestsCookieJar.
    Filtra solo dominios relevantes (sharepoint, microsoft).
    """
    data = json.loads(STORAGE_STATE.read_text(encoding="utf-8"))
    jar = RequestsCookieJar()
    cookies = data.get("cookies", [])
    if not isinstance(cookies, list):
        raise RuntimeError("storage_state.json inválido: campo cookies no es una lista")

    for c in cookies:
        name = _to_str(c.get("name")).strip()
        value = _to_str(c.get("value"))
        domain_raw = c.get("domain")
        path_raw = c.get("path", "/")

        if not name or value is None:
            continue

        domain = _normalize_domain(domain_raw)
        path = _normalize_path(path_raw)
        if not domain:
            continue

        dom = domain.lstrip(".").lower()
        if not (
            dom.endswith(SP_SITE_HOST.lower())
            or dom.endswith("sharepoint.com")
            or dom.endswith("microsoftonline.com")
            or dom.endswith("microsoft.com")
        ):
            continue

        jar.set(name=name, value=value, domain=domain, path=path)
    return jar


async def _has_sp_auth_cookies(ctx, host: str) -> bool:
    cookies: List[Dict] = await ctx.cookies()
    host_l = host.lower()
    found = set()
    for c in cookies:
        name = str(c.get("name", ""))
        domain = str(c.get("domain", "")).lstrip(".").lower()
        if name in FEDAUTH_NAMES and (domain.endswith(host_l) or domain.endswith("sharepoint.com")):
            found.add(name)
    return len(found) == 2

# ============================================================
#  FIN: GESTIÓN DE STORAGE_STATE Y COOKIES
# ============================================================

# ============================================================
#  INICIO: LOGIN INTERACTIVO CON PLAYWRIGHT
# ============================================================

async def _ensure_login_and_storage_async():
    """
    Abre un navegador Chromium (canal msedge/chrome si se indica),
    deja que el usuario se autentique en SharePoint y guarda storage_state.json
    con las cookies FedAuth/rtFa.
    """
    channel = os.getenv("SP_BROWSER_CHANNEL", "msedge").strip()   # "msedge" | "chrome" | ""
    exe = os.getenv("SP_EXECUTABLE_PATH", "").strip()       # opcional: ruta binario concreto

    async with async_playwright() as p:
        launch_kwargs = {"headless": False}
        if channel:
            launch_kwargs["channel"] = channel
        if exe:
            launch_kwargs["executable_path"] = exe

        browser = await p.chromium.launch(**launch_kwargs)
        try:
            ctx = await browser.new_context()
            page = await ctx.new_page()
            page.set_default_timeout(NAV_TIMEOUT_MS)

            print(f"[SP] Abriendo navegador para login en: {SP_LOGIN_URL}")
            await page.goto(SP_LOGIN_URL, wait_until="load", timeout=NAV_TIMEOUT_MS)

            # Damos tiempo a que el usuario complete SSO/MFA.
            # Esperamos a ver FedAuth + rtFa hasta ~5 minutos (60 * 5s)
            for i in range(60):
                if await _has_sp_auth_cookies(ctx, SP_SITE_HOST):
                    print("[SP] Cookies FedAuth/rtFa detectadas, guardando storage_state...")
                    break
                await asyncio.sleep(5.0)
            else:
                print("[SP][WARN] No se han encontrado cookies FedAuth/rtFa tras 5 minutos. Se guarda igualmente el estado actual.")

            await ctx.storage_state(path=str(STORAGE_STATE))
        finally:
            await browser.close()


def _run_coro_in_thread(coro):
    result_container = {}
    err_container = {}

    def runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_container["result"] = loop.run_until_complete(coro)
        except BaseException as e:
            err_container["err"] = e
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "err" in err_container:
        raise err_container["err"]
    return result_container.get("result")


def _ensure_login_and_storage(force: bool = False) -> None:
    if STORAGE_STATE.exists() and not force:
        return
    _run_coro_in_thread(_ensure_login_and_storage_async())

# ============================================================
#  FIN: LOGIN INTERACTIVO CON PLAYWRIGHT
# ============================================================


# ============================================================
#  INICIO: SESIONES REQUESTS + REST HELPER
# ============================================================

def _new_requests_session() -> requests.Session:
    if not STORAGE_STATE.exists():
        if os.getenv("SP_DISABLE_LOGIN", "0") == "1":
            raise RuntimeError(
                "STORAGE_STATE no existe y SP_DISABLE_LOGIN=1. "
                "Genera .sp_storage_state.json fuera del contenedor antes de usar este proceso."
            )
        _ensure_login_and_storage(force=True)

    s = requests.Session()
    s.cookies = _load_cookiejar_from_storage()
    s.headers.update({
        "Accept": "application/json;odata=verbose",
        "Content-Type": "application/json;odata=verbose",
    })
    return s



def _rest_get(session: requests.Session, rel_url: str, params: dict | None = None, allow_403_retry: bool = True) -> dict:
    if rel_url.startswith("/_api"):
        url = f"{BASE_SITE}{rel_url}"
    else:
        url = f"{BASE_SITE}/_api{rel_url}"

    r = session.get(url, params=params, timeout=REQ_TIMEOUT_MS / 1000.0)
    if (
        r.status_code in (401, 403)
        and allow_403_retry
        and os.getenv("SP_DISABLE_LOGIN", "0") != "1"
    ):
        print(f"[SP][WARN] _rest_get {rel_url} status={r.status_code}, reintentando con login forzado...")
        _ensure_login_and_storage(force=True)
        session.cookies = _load_cookiejar_from_storage()
        r = session.get(url, params=params, timeout=REQ_TIMEOUT_MS / 1000.0)

    r.raise_for_status()
    data = r.json()

    if "value" not in data:
        d = data.get("d")
        if isinstance(d, dict):
            if "results" in d and isinstance(d["results"], list):
                data = {"value": d["results"]}
            else:
                data = {"value": [d]}

    return data

# ============================================================
#  FIN: SESIONES REQUESTS + REST HELPER
# ============================================================

# ============================================================
#  INICIO: LISTADO RECURSIVO DE CARPETAS Y FICHEROS
# ============================================================

def _list_folder(session: requests.Session, server_relative_folder: str):
    srv_web = _normalize_server_relative(server_relative_folder)
    srv_site = SP_SITE_PATH.rstrip("/") + srv_web

    print(f"[DBG] _list_folder srv_web={srv_web}")
    print(f"[DBG] _list_folder srv_site={srv_site}")

    # GetFolderByServerRelativeUrl(srv_site)
    try:
        files_resp = _rest_get(
            session,
            f"/web/GetFolderByServerRelativeUrl('{srv_site}')/Files",
            params={
                "$select": "Name,ServerRelativeUrl,TimeLastModified,UniqueId,Length,Author/Title",
                "$expand": "Author",
            },
        )
        folders_resp = _rest_get(
            session,
            f"/web/GetFolderByServerRelativeUrl('{srv_site}')/Folders",
            params={
                "$select": "Name,ServerRelativeUrl,TimeLastModified,UniqueId",
            },
        )

        files = files_resp.get("value", [])
        folders = folders_resp.get("value", [])

        # Si aquí ya tenemos algo, devolvemos sin más
        if files or folders:
            return files, folders

    except Exception as e:
        print(f"[SP][WARN] _list_folder Url('{srv_site}') falló: {e}")

    # GetFolderByServerRelativePath(decodedurl=@a1)
    try:
        files_resp = _rest_get(
            session,
            "/web/GetFolderByServerRelativePath(decodedurl=@a1)/Files",
            params={
                "@a1": f"'{srv_site}'",
                "$select": "Name,ServerRelativeUrl,TimeLastModified,UniqueId,Length,Author/Title",
                "$expand": "Author",
            },
        )
        folders_resp = _rest_get(
            session,
            "/web/GetFolderByServerRelativePath(decodedurl=@a1)/Folders",
            params={
                "@a1": f"'{srv_site}'",
                "$select": "Name,ServerRelativeUrl,TimeLastModified,UniqueId",
            },
        )

        files = files_resp.get("value", [])
        folders = folders_resp.get("value", [])
        return files, folders

    except Exception as e:
        print(f"[SP][WARN] _list_folder Path(decodedurl=@a1) falló para {srv_site}: {e}")
        return [], []


def list_all_files_under_docroot() -> Iterable[Dict]:
    session = _new_requests_session()

    root = _normalize_server_relative(SP_DOC_ROOT)
    print(f"[DBG] ROOT={root}")

    files0, folders0 = _list_folder(session, root)
    print(f"[DBG] Primer nivel ROOT -> Files={len(files0)}, Folders={len(folders0)}")
    if files0:
        print("[DBG] Files ROOT:", [f.get("Name") for f in files0[:10]])
    if folders0:
        print("[DBG] Folders ROOT:", [d.get("Name") for d in folders0[:10]])

    stack = [root]
    total_levels = 0
    total_files_seen = 0
    total_folders_seen = 0

    visited = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        total_levels += 1
        files, folders = _list_folder(session, current)
        print(f"[DBG] Nivel {total_levels} -> {current}")
        print(f"[DBG] files: {len(files)} ; folders: {len(folders)}")

        if files:
            print("[DBG] Ejemplos files:", [f.get("Name") for f in files[:3]])
        if folders:
            print("[DBG] Ejemplos folders:", [d.get("Name") for d in folders[:3]])

        total_files_seen += len(files)
        total_folders_seen += len(folders)

        for f in files:
            yield f
        for d in folders:
            child = d.get("ServerRelativeUrl")
            if child and child != current:
                stack.append(child)

    print(f"[DBG] Totales -> levels={total_levels}, files={total_files_seen}, folders={total_folders_seen}")

# ============================================================
#  FIN: SESIONES REQUESTS + REST HELPER
# ============================================================

# ============================================================
#  INICIO: DESCARGA BINARIA DE UN FICHERO
# ============================================================

def download_bytes(server_relative_url: str) -> bytes:
    session = _new_requests_session()
    path = _normalize_server_relative(server_relative_url) 

    if path.startswith(SP_SITE_PATH):
        srv_full = path
    else:
        srv_full = SP_SITE_PATH.rstrip("/") + path

    api_rel = f"/web/GetFileByServerRelativeUrl('{srv_full}')/$value"
    url = f"{BASE_SITE}/_api{api_rel}"

    r = session.get(url, timeout=REQ_TIMEOUT_MS / 1000.0)
    r.raise_for_status()
    return r.content

# ============================================================
#  FIN: DESCARGA BINARIA DE UN FICHERO
# ============================================================

# ============================================================
#  INICIO: URL CANÓNICA PARA CLICKAR EN EL RAG
# ============================================================

def build_canonical_url(server_relative_url: str) -> str:
    if not server_relative_url:
        return ""

    s = str(server_relative_url)
    s = re.sub(r"^https?://[^/]+", "", s, flags=re.IGNORECASE)
    s = s.replace("\\", "/")
    if not s.startswith("/"):
        s = "/" + s

    parts = [quote(part, safe="") for part in s.split("/")]
    encoded_path = "/".join(parts)

    return f"{BASE_WEB}{encoded_path}?web=1"
