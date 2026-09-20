"""
Self-hosted chat UI for the drone MCP agent. Runs on the Raspberry Pi.

    python3 app.py

Then open http://<pi-ip>:5000 from your phone or laptop on the same network.
"""
import asyncio
import os
import threading

from flask import Flask, jsonify, render_template_string, request, redirect
from dotenv import load_dotenv, set_key

# Load environment variables from .env file (if it exists)
load_dotenv()

from chat_agent import DroneChatAgent

app = Flask(__name__)
agent = DroneChatAgent()

# Run one persistent asyncio event loop in a background thread
_loop = asyncio.new_event_loop()


def _start_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_start_loop, daemon=True).start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


SETUP_PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Setup - Drone Chat</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #121212; color: #fff; }
    .card { background: #1e1e1e; padding: 20px; border-radius: 8px; }
    h3 { margin-top: 0; color: #00ff00; }
    label { display: block; margin-bottom: 8px; color: #aaa; }
    input { width: 100%; padding: 10px; margin-bottom: 20px; box-sizing: border-box; background: #333; color: white; border: 1px solid #444; border-radius: 4px; }
    button { padding: 10px 20px; font-size: 16px; background: #00ff00; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
    button:hover { background: #00cc00; }
  </style>
</head>
<body>
  <div class="card">
    <h3>Drone AI Setup</h3>
    <p>Please enter your API keys below. The keys will be saved locally to a <code>.env</code> file so you only have to do this once.</p>
    <form action="/setup" method="POST">
      <label for="groq_key">Groq API Key (Recommended, limits bypassed):</label>
      <input type="password" id="groq_key" name="groq_key" placeholder="gsk_..." />
      
      <label for="gemini_key">Gemini API Key (Optional Fallback):</label>
      <input type="password" id="gemini_key" name="gemini_key" placeholder="AIza..." />
      
      <button type="submit">Save and Continue</button>
    </form>
  </div>
</body>
</html>
"""

CHAT_PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone Control</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: sans-serif; max-width: 700px; margin: 0 auto; padding: 12px; background: #0d0d0d; color: #fff; display: flex; flex-direction: column; gap: 12px; }
    h3 { color: #00ff00; font-size: 1.1em; }

    /* Camera feed */
    #video-feed {
      width: 100%; border-radius: 8px; border: 1px solid #222;
      background: #000; display: block; min-height: 60px;
    }
    #cam-status { font-size: 0.75em; color: #555; text-align: right; margin-top: 2px; }

    /* QR results panel */
    #qr-panel { background: #111; border: 1px solid #222; border-radius: 8px; padding: 10px; }
    #qr-panel h3 { margin-bottom: 8px; }
    #qr-list { display: flex; flex-direction: column; gap: 6px; max-height: 160px; overflow-y: auto; }
    .qr-item { background: #1a1a1a; border-left: 4px solid #00ff00; border-radius: 4px; padding: 8px 10px; }
    .qr-time { font-size: 0.72em; color: #666; }
    .qr-data { font-size: 0.95em; font-weight: bold; color: #00ff00; word-break: break-all; margin-top: 2px; }
    #qr-empty { color: #444; font-size: 0.85em; }
    #scanner-status { font-size: 0.72em; color: #555; margin-top: 6px; }

    /* Chat */
    #chat-panel { background: #111; border: 1px solid #222; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
    #log { display: flex; flex-direction: column; gap: 6px; max-height: 260px; overflow-y: auto; }
    .msg { padding: 8px 12px; border-radius: 8px; white-space: pre-wrap; font-size: 0.9em; max-width: 90%; }
    .user { background: #1e3a8a; align-self: flex-end; }
    .assistant { background: #1e1e1e; align-self: flex-start; border: 1px solid #2a2a2a; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 10px; font-size: 15px; background: #1a1a1a; color: white; border: 1px solid #333; border-radius: 4px; }
    button { padding: 10px 16px; font-size: 15px; background: #00ff00; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
    button:hover { background: #00cc00; }
  </style>
</head>
<body>
  <h3>🚁 Drone Control</h3>

  <!-- Camera Feed -->
  <div>
    <img id="video-feed" alt="Connecting to camera..." />
    <div id="cam-status">Camera: connecting...</div>
  </div>

  <!-- QR Results -->
  <div id="qr-panel">
    <h3>📷 QR Scanner</h3>
    <div id="qr-list"><span id="qr-empty">No QR codes scanned yet.</span></div>
    <div id="scanner-status">Scanner: connecting to localhost:5001...</div>
  </div>

  <!-- Chat -->
  <div id="chat-panel">
    <h3>💬 AI Chat</h3>
    <div id="log"></div>
    <form id="form">
      <input id="input" autocomplete="off" placeholder="e.g. takeoff to 10m, land, rotate 90°" />
      <button type="submit">Send</button>
    </form>
  </div>

  <script>
    // ── Camera feed (scanner.py on port 5001) ──────────────────
    const videoFeed = document.getElementById('video-feed');
    const camStatus = document.getElementById('cam-status');
    const scannerBase = window.location.protocol + '//' + window.location.hostname + ':5001';

    videoFeed.src = scannerBase + '/video_feed';
    videoFeed.onload = () => camStatus.textContent = 'Camera: live ✓';
    videoFeed.onerror = () => camStatus.textContent = 'Camera: scanner.py not running on port 5001';

    // ── QR scan results (polls scanner.py /api/scans) ──────────
    const qrList = document.getElementById('qr-list');
    const qrEmpty = document.getElementById('qr-empty');
    const scannerStatus = document.getElementById('scanner-status');
    let lastQrCount = 0;

    async function fetchQR() {
      try {
        const res = await fetch(scannerBase + '/api/scans');
        if (!res.ok) throw new Error('bad response');
        const scans = await res.json();
        scannerStatus.textContent = 'Scanner: connected ✓  |  Total scans: ' + scans.length;

        if (scans.length === 0) {
          qrList.innerHTML = '<span id="qr-empty">No QR codes scanned yet.</span>';
          return;
        }

        if (scans.length !== lastQrCount) {
          lastQrCount = scans.length;
          qrList.innerHTML = scans.map(s => `
            <div class="qr-item">
              <div class="qr-time">${s.time}</div>
              <div class="qr-data">${s.data}</div>
            </div>
          `).join('');
          qrList.scrollTop = 0; // newest on top (API returns reversed)
        }
      } catch (e) {
        scannerStatus.textContent = 'Scanner: not reachable on port 5001';
      }
    }

    fetchQR();
    setInterval(fetchQR, 1000);

    // ── Chat ───────────────────────────────────────────────────
    const log = document.getElementById('log');
    const form = document.getElementById('form');
    const input = document.getElementById('input');

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      log.appendChild(div);
      div.scrollIntoView({ behavior: 'smooth' });
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      addMsg(text, 'user');
      input.value = '';
      const placeholder = document.createElement('div');
      placeholder.className = 'msg assistant';
      placeholder.textContent = '...';
      log.appendChild(placeholder);
      placeholder.scrollIntoView({ behavior: 'smooth' });
      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        placeholder.textContent = data.reply || data.error || '(error)';
      } catch (err) {
        placeholder.textContent = 'Request failed: ' + err;
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return render_template_string(SETUP_PAGE)
    return render_template_string(CHAT_PAGE)


@app.route("/setup", methods=["POST"])
def setup():
    groq_key = request.form.get("groq_key", "").strip()
    gemini_key = request.form.get("gemini_key", "").strip()
    
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_file):
        open(env_file, 'a').close()
        
    if groq_key:
        set_key(env_file, "GROQ_API_KEY", groq_key)
        os.environ["GROQ_API_KEY"] = groq_key
        
    if gemini_key:
        set_key(env_file, "GEMINI_API_KEY", gemini_key)
        os.environ["GEMINI_API_KEY"] = gemini_key
        
    return redirect("/")


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    try:
        reply = run_async(agent.chat(message))
        return jsonify({"reply": reply})
    except Exception as exc:  # surface errors to the UI instead of a 500 wall
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    run_async(agent.connect())
    app.run(host="0.0.0.0", port=5000)