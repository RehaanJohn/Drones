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
  <title>Drone Chat</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 12px; background: #121212; color: #fff; }
    #log { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
    .msg { padding: 8px 12px; border-radius: 8px; white-space: pre-wrap; }
    .user { background: #1e3a8a; align-self: flex-end; color: #fff; }
    .assistant { background: #1e1e1e; align-self: flex-start; border: 1px solid #333; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 10px; font-size: 16px; background: #333; color: white; border: 1px solid #444; border-radius: 4px; }
    button { padding: 10px 16px; font-size: 16px; background: #00ff00; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
  </style>
</head>
<body>
  <h3 style="color: #00ff00; margin-top: 0;">Drone Control Chat</h3>
  <div id="log"></div>
  <form id="form">
    <input id="input" autocomplete="off" placeholder="e.g. connect and takeoff to 10m" />
    <button type="submit">Send</button>
  </form>
  <script>
    const log = document.getElementById('log');
    const form = document.getElementById('form');
    const input = document.getElementById('input');

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      log.appendChild(div);
      div.scrollIntoView();
    }

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      addMsg(text, 'user');
      input.value = '';
      addMsg('...', 'assistant');
      const placeholder = log.lastChild;
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