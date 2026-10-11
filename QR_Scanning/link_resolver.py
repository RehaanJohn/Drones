"""Fetch readable text from QR URLs without blocking camera decoding."""
from collections import OrderedDict
import ipaddress
import json
import queue
import socket
import threading
import time
from urllib.parse import urlsplit
import urllib.request

from bs4 import BeautifulSoup

MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CONTENT_CHARS = 4000


def is_web_url(value):
    try:
        parts = urlsplit(value.strip())
        return parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
    except ValueError:
        return False


def validate_public_url(url):
    """Only fetch public HTTP(S) destinations, including every redirect."""
    parts = urlsplit(url)
    if (parts.scheme.lower() not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None):
        raise ValueError("Only public HTTP(S) links without embedded credentials are supported")
    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    addresses = socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("Local/private network links are not fetched automatically")


class PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    max_repeats = 2

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def extract_page_text(html):
    """Prefer main content, excluding scripts, navigation, and generator footers."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup.select("script, style, noscript, template, head, nav, footer, header, aside, svg, form, [hidden], [aria-hidden='true']"):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if tag.attrs is None:
            continue
        style = tag.get("style", "").lower().replace(" ", "")
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()
    for tag in soup.select("main, article, [role='main'], .container"):
        text = tag.get_text(" ", strip=True)
        if text:
            return " ".join(text.split())
    text = (soup.body or soup).get_text(" ", strip=True)
    return " ".join(text.split()) or title


def resolve_url(url, timeout=5.0):
    """Return text or a concise error; no browser, cookies, or JavaScript execution."""
    url = url.strip()
    result = {"url": url}
    try:
        validate_public_url(url)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), PublicRedirectHandler())
        request = urllib.request.Request(url, headers={
            "User-Agent": "QRScanner/1.0",
            "Accept": "text/html, text/plain, application/json, application/xhtml+xml",
            "Accept-Encoding": "identity",
        })
        with opener.open(request, timeout=timeout) as response:
            result["url"] = response.url
            content_type = response.headers.get_content_type()
            if not (content_type.startswith("text/") or content_type in ("application/json", "application/xhtml+xml")):
                raise ValueError(f"Unsupported content type: {content_type}")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Page is larger than the 1 MB limit")
            encoding = response.headers.get_content_charset() or "utf-8"
            body = raw.decode(encoding, errors="replace")
        if content_type in ("text/html", "application/xhtml+xml"):
            content = extract_page_text(body)
        elif content_type == "application/json":
            content = json.dumps(json.loads(body), ensure_ascii=False)
        else:
            content = body.strip()
        if not content:
            raise ValueError("No readable text found; the page may require JavaScript or login")
        result["content"] = content[:MAX_CONTENT_CHARS]
        if len(content) > MAX_CONTENT_CHARS:
            result["truncated"] = True
    except Exception as exc:
        # A bad/offline URL must not take down the scanner or the worker.
        result["error"] = str(exc) or type(exc).__name__
    return result


class LinkResolver:
    """Bounded daemon workers; repeated links share requests and a 60-second cache.

    submit/poll run on the scanner thread; workers only publish result objects.
    No worker writes to the terminal or a file that may be closed on shutdown.
    """

    def __init__(self, fetch=resolve_url):
        self.fetch = fetch
        self.requests = queue.Queue(maxsize=16)
        self.results = queue.Queue(maxsize=18)
        self.pending = {}
        self.cache = OrderedDict()
        self.stop = threading.Event()
        self.workers = [threading.Thread(target=self._worker, daemon=True, name=f"qr-link-{i}")
                        for i in range(2)]
        for worker in self.workers:
            worker.start()

    def _worker(self):
        while not self.stop.is_set():
            try:
                url = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.stop.is_set():
                break
            try:
                result = self.fetch(url)
            except Exception as exc:
                result = {"url": url, "error": str(exc) or type(exc).__name__}
            while not self.stop.is_set():
                try:
                    self.results.put((url, result), timeout=0.1)
                    break
                except queue.Full:
                    continue

    def submit(self, url, captured_at):
        url = url.strip()
        cached = self.cache.get(url)
        if cached and cached[0] > time.monotonic():
            self.cache.move_to_end(url)
            return [(url, captured_at, cached[1])]
        if url in self.pending:
            return []
        if len(self.pending) >= 18:
            return [(url, captured_at, {"url": url, "error": "Link lookup queue is full"})]
        try:
            self.requests.put_nowait(url)
        except queue.Full:
            return [(url, captured_at, {"url": url, "error": "Link lookup queue is full"})]
        self.pending[url] = captured_at
        return []

    def poll(self):
        ready = []
        while True:
            try:
                url, result = self.results.get_nowait()
            except queue.Empty:
                break
            captured_at = self.pending.pop(url)
            ready.append((url, captured_at, result))
            ttl = 5 if "error" in result else 60
            self.cache[url] = (time.monotonic() + ttl, result)
            self.cache.move_to_end(url)
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        return ready

    def close(self):
        self.stop.set()
        # Daemon workers can finish a timed request without holding up camera exit.
        for worker in self.workers:
            worker.join(timeout=0.1)
