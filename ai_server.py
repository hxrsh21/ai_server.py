#!/usr/bin/env python3
"""
BrainHack Maze Robot AI Server
Run on RPi: python3 ai_server.py
ESP32 connects to this for smart decisions
"""
from flask import Flask, request, jsonify
import requests, re, time

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

decision_history = []

def ask_ollama(sensor_data, total_decisions):
    """Ask Ollama AI for best robot movement decision."""
    prompt = f"""You are the AI brain of a maze solving robot.

SENSOR DATA:
{sensor_data}

HISTORY: {total_decisions} total moves made so far.

Based on sensor distances in cm:
- Values close to 999 = open space (no obstacle)
- Values below 35 = obstacle (wall/object)
- FL = Front Left sensor
- FR = Front Right sensor  
- L = Left scan distance
- R = Right scan distance
- B = Back distance

RULES:
- Reply with ONLY one letter: L, R, B, or F
- L = turn left (if left has more space)
- R = turn right (if right has more space)
- B = reverse (if front and sides blocked but back clear)
- F = go forward (if path somehow clear)
- Choose the direction with MOST space available
- If L and R similar, prefer L

Reply with ONLY the single letter decision. Nothing else."""

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 5,
                "num_ctx": 128,
                "temperature": 0.1,
                "num_thread": 4
            }
        }, timeout=5)

        reply = r.json().get("response", "").strip()
        # Extract just the letter
        for char in reply.upper():
            if char in ['L', 'R', 'B', 'F']:
                return char
        return None
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def local_fallback(sensor_data):
    """Fast local decision if Ollama too slow."""
    import re
    l = float(re.search(r'L:([\d.]+)', sensor_data).group(1)) if re.search(r'L:([\d.]+)', sensor_data) else 999
    r = float(re.search(r'R:([\d.]+)', sensor_data).group(1)) if re.search(r'R:([\d.]+)', sensor_data) else 999
    b = float(re.search(r'B:([\d.]+)', sensor_data).group(1)) if re.search(r'B:([\d.]+)', sensor_data) else 999

    if l > r and l > 30:   return 'L'
    if r > 30:             return 'R'
    if b > 35:             return 'B'
    return 'L' if l >= r else 'R'

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "message": "RPi AI Server ready!"})

@app.route('/decide', methods=['POST'])
def decide():
    data = request.json
    sensor_data   = data.get('sensors', '')
    total_decisions = data.get('decisions', 0)
    ai_decisions  = data.get('ai_decisions', 0)

    print(f"\n[{time.strftime('%H:%M:%S')}] Sensors: {sensor_data}")

    # Try Ollama first
    t_start = time.time()
    action = ask_ollama(sensor_data, total_decisions)
    t_end = time.time()

    reason = "Ollama AI"
    if not action:
        # Fallback to local logic
        action = local_fallback(sensor_data)
        reason = "Local fallback"

    # Save to history
    decision_history.append({
        "sensors": sensor_data,
        "action": action,
        "reason": reason,
        "time_ms": int((t_end - t_start) * 1000)
    })

    print(f"[{time.strftime('%H:%M:%S')}] Decision: {action} ({reason}) in {int((t_end-t_start)*1000)}ms")

    return jsonify({
        "action": action,
        "reason": reason,
        "time_ms": int((t_end - t_start) * 1000),
        "total_decisions": len(decision_history)
    })

@app.route('/history', methods=['GET'])
def history():
    return jsonify({
        "total": len(decision_history),
        "decisions": decision_history[-10:]  # last 10
    })

@app.route('/stats', methods=['GET'])
def stats():
    if not decision_history:
        return jsonify({"message": "No decisions yet"})
    actions = [d['action'] for d in decision_history]
    return jsonify({
        "total": len(decision_history),
        "left":  actions.count('L'),
        "right": actions.count('R'),
        "back":  actions.count('B'),
        "forward": actions.count('F')
    })

if __name__ == '__main__':
    print("╔══════════════════════════════════════╗")
    print("║  Maze Robot AI Server — Starting    ║")
    print("╚══════════════════════════════════════╝")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print("Endpoints:")
    print("  GET  /ping    — test connection")
    print("  POST /decide  — get AI decision")
    print("  GET  /history — see past decisions")
    print("  GET  /stats   — see decision stats")
    print("\nStarting server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
