"""
Self-hosted chat UI for the drone MCP agent. Runs on the Raspberry Pi.

    python3 app.py

Then open http://<pi-ip>:5000 from your phone or laptop on the same network.

Requires ANTHROPIC_API_KEY set in the environment (get one at
console.anthropic.com). Billed separately/per-token from any Claude
subscription — keep that in mind if you leave this running.
"""
import asyncio
import os
import threading

from flask import Flask, jsonify, render_template_string, request

from chat_agent import DroneChatAgent

app = Flask(__name__)
agent = DroneChatAgent()

# Run one persistent asyncio event loop in a background thread, since the
# MCP client session and Anthropic tool-use loop are async but Flask's
# request handlers here are sync.
_loop = asyncio.new_event_loop()


def _start_loop():
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


threading.Thread(target=_start_loop, daemon=True).start()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone chat</title>
  <style>
    body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 12px; }
    #log { display: flex; flex-direction: column; gap: 8px; margin-bottom: 12px; }
    .msg { padding: 8px 12px; border-radius: 8px; white-space: pre-wrap; }
    .user { background: #dbeafe; align-self: flex-end; }
    .assistant { background: #f1f5f9; align-self: flex-start; }
    form { display: flex; gap: 8px; }
    input { flex: 1; padding: 10px; font-size: 16px; }
    button { padding: 10px 16px; font-size: 16px; }
  </style>
</head>
<body>
  <h3>Drone chat</h3>
  <div id="log"></div>
  <form id="form">
    <input id="input" autocomplete="off" placeholder="e.g. connect and check telemetry" />
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
    return render_template_string(PAGE)


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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before starting app.py")
    run_async(agent.connect())
    app.run(host="0.0.0.0", port=5000)