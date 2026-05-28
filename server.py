"""
Agent Dashboard Server — FastAPI + WebSocket
Wraps the autonomous-agent Supervisor with real-time monitoring.
"""

import asyncio
import json
import os
import sys
import threading
import time
import queue
from datetime import datetime
from typing import Optional

# Add parent's autonomous-agent to path
_sys_path = os.path.join(os.path.dirname(__file__), "..", "autonomous-agent")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import Config
from llm_client import create_clients
from worker import WorkerAgent
from critic import CriticAgent
from supervisor import Supervisor, Milestone, MilestoneResult, RunState

app = FastAPI(title="Agent Dashboard")

# ── Global State ──
_supervisor: Optional[Supervisor] = None
_run_state: Optional[RunState] = None
_event_queue: queue.Queue = queue.Queue()
_active_ws: set[WebSocket] = set()
_run_thread: Optional[threading.Thread] = None
_run_config = {"task": "", "dag": False, "total_milestones": 0}
_run_results: list[MilestoneResult] = []
_is_running = False


def event_callback(event: str, data: dict):
    """Forward supervisor events to WebSocket clients"""
    payload = json.dumps({"type": event, "data": data, "ts": datetime.now().isoformat()}, ensure_ascii=False)
    _event_queue.put(payload)


# ── WebSocket ──

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _active_ws.add(ws)
    try:
        # Drain event queue
        while True:
            try:
                payload = _event_queue.get(timeout=0.1)
                for client in list(_active_ws):
                    try:
                        await client.send_text(payload)
                    except Exception:
                        _active_ws.discard(client)
            except queue.Empty:
                pass
            # Check for client messages
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        _active_ws.discard(ws)


# ── REST API ──

@app.get("/api/status")
async def get_status():
    global _is_running, _run_state, _run_config
    total = _run_state.total_count if _run_state else _run_config.get("total_milestones", 0)
    if _run_state:
        return {
            "running": _is_running,
            "task": _run_config["task"],
            "dag": _run_config["dag"],
            "progress": _run_state.progress,
            "completed": _run_state.completed_count,
            "total": max(total, _run_state.total_count),
            "total_iterations": _run_state.total_iterations,
            "total_cost": _run_state.total_cost,
            "total_tokens": _run_state.total_tokens,
            "started_at": _run_state.started_at,
        }
    return {"running": _is_running, "progress": 0, "completed": 0, "total": total}


@app.get("/api/results")
async def get_results():
    global _run_results
    return [
        {
            "name": r.milestone.name,
            "score": r.score,
            "passed": r.passed,
            "retries": r.retries,
            "duration": r.duration_seconds,
            "cost": r.cost,
            "tokens": r.input_tokens + r.output_tokens,
            "issues": r.issues[:5],
            "tier": "powerful" if r.milestone.use_powerful_model else "cheap",
        }
        for r in _run_results
    ]


@app.get("/api/checkpoints")
async def get_checkpoints():
    ckpt_dir = ".checkpoints"
    if not os.path.exists(ckpt_dir):
        return []
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".json")], reverse=True)
    result = []
    for f in files[:10]:
        path = os.path.join(ckpt_dir, f)
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            result.append({
                "file": f,
                "task": data.get("task", "")[:80],
                "completed": len(data.get("completed", [])),
                "total_cost": data.get("total_cost", 0),
            })
        except Exception:
            pass
    return result


@app.post("/api/run")
async def start_run(data: dict):
    global _is_running, _run_thread, _run_state, _run_results

    if _is_running:
        return {"error": "Already running"}

    task = data.get("task", "")
    dag = data.get("dag", False)
    if not task:
        return {"error": "Task is required"}

    # Build milestones from JSON
    milestones_data = data.get("milestones", [])
    if milestones_data:
        milestones = [
            Milestone(
                name=m.get("name", f"M{i}"),
                description=m.get("description", ""),
                validation_hint=m.get("validation_hint", ""),
                rules=m.get("rules", []),
                depends_on=m.get("depends_on", []),
                use_powerful_model=m.get("use_powerful_model", False),
            )
            for i, m in enumerate(milestones_data)
        ]
    else:
        # Single milestone mode
        milestones = [Milestone(name="Task", description=task, validation_hint="Complete, accurate, specific")]

    cfg = Config.from_env()
    if not cfg.cheap_model.api_key:
        cfg.cheap_model.api_key = os.getenv("LLM_API_KEY", "")

    def worker_factory(use_powerful=False):
        tier_client, _ = create_clients(cfg, use_powerful=use_powerful)
        return WorkerAgent(tier_client)

    def critic_factory(use_powerful=False):
        _, tier_client = create_clients(cfg, use_powerful=use_powerful)
        return CriticAgent(tier_client)

    _run_config = {"task": task, "dag": dag, "total_milestones": len(milestones)}
    _run_results = []
    _is_running = True

    def _run():
        global _is_running, _run_state, _run_results, _supervisor
        try:
            print(f"[Dashboard] Starting run: task='{task[:60]}...', milestones={len(milestones)}, dag={dag}")
            _supervisor = Supervisor(cfg, worker_factory, critic_factory)
            _supervisor.on_event(event_callback)
            results = _supervisor.run(task, milestones,
                                      resume_from_checkpoint=False,
                                      use_dag=dag and len(milestones) > 1)
            _run_state = _supervisor.state
            _run_results = results
            print(f"[Dashboard] Run complete: {len(results)} results")
        except Exception as e:
            import traceback
            traceback.print_exc()
            event_callback("error", {"message": str(e)})
        finally:
            _is_running = False
            event_callback("run_complete", {})

    _run_thread = threading.Thread(target=_run, daemon=True)
    _run_thread.start()
    return {"status": "started", "milestones": len(milestones), "dag": dag}


@app.post("/api/stop")
async def stop_run():
    global _is_running
    _is_running = False
    return {"status": "stopping"}


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
