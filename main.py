import asyncio
import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LEGACY_DATA_FILE = DATA_DIR / "experiments.json"
LEGACY_BACKUP_FILE = DATA_DIR / "experiments.legacy.json"
EXPERIMENT_FILE_NAME = "experiments.json"
STATIC_DIR = BASE_DIR / "static"
STORE_LOCK = RLock()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

app = FastAPI(title="系统提示词 A/B 试炼场")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ExperimentIn(BaseModel):
    id: str | None = None
    name: str = Field(default="未命名实验")
    goal: str = ""
    promptA: str = ""
    promptB: str = ""
    inputPrompt: str = ""
    model: str | None = None
    temperature: float = 0.7
    maxTokens: int = 1000


class RoundIn(BaseModel):
    experimentId: str
    promptA: str
    promptB: str
    inputPrompt: str


class GenerateIn(BaseModel):
    experimentId: str
    roundId: str | None = None
    inputPrompt: str | None = None


class RunPatchIn(BaseModel):
    humanChoice: str = ""
    humanNote: str = ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_for_dir(value: str | None = None) -> str:
    parsed: datetime | None = None
    if value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    parsed = parsed or datetime.now(timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")


def safe_dir_name(name: str, timestamp: str) -> str:
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    base = re.sub(r"\s+", " ", base).strip(" .")
    if not base:
        base = "未命名实验"
    base = base[:80].rstrip(" .") or "未命名实验"
    return f"{base}-{timestamp}"


def unique_storage_dir(name: str, timestamp: str) -> str:
    base = safe_dir_name(name, timestamp)
    candidate = base
    index = 2
    while (DATA_DIR / candidate).exists():
        candidate = f"{base}-{index}"
        index += 1
    return candidate


def experiment_path(storage_dir: str) -> Path:
    path = DATA_DIR / storage_dir / EXPERIMENT_FILE_NAME
    resolved = path.resolve()
    if DATA_DIR.resolve() not in resolved.parents:
        raise HTTPException(status_code=400, detail="Invalid experiment storage path")
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def load_experiment_file(path: Path) -> dict[str, Any]:
    with STORE_LOCK:
        experiment = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(experiment, dict) or not experiment.get("id"):
        raise ValueError(f"Invalid experiment file: {path}")
    experiment.setdefault("storageDir", path.parent.name)
    return experiment


def save_experiment(experiment: dict[str, Any], path: Path | None = None) -> Path:
    ensure_store()
    storage_dir = experiment.get("storageDir")
    if not storage_dir:
        storage_dir = unique_storage_dir(
            experiment.get("name", "未命名实验"),
            timestamp_for_dir(experiment.get("createdAt")),
        )
        experiment["storageDir"] = storage_dir
    target = path or experiment_path(storage_dir)
    with STORE_LOCK:
        atomic_write_json(target, experiment)
    return target


def next_legacy_backup_path() -> Path:
    if not LEGACY_BACKUP_FILE.exists():
        return LEGACY_BACKUP_FILE
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DATA_DIR / f"experiments.legacy.{timestamp}.json"


def migrate_legacy_store() -> None:
    ensure_store()
    if not LEGACY_DATA_FILE.exists():
        return
    with STORE_LOCK:
        try:
            store = json.loads(LEGACY_DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        experiments = store.get("experiments") if isinstance(store, dict) else None
        if not isinstance(experiments, list):
            return
        for experiment in experiments:
            if not isinstance(experiment, dict):
                continue
            experiment.setdefault("id", new_id("exp"))
            experiment.setdefault("createdAt", now_iso())
            experiment.setdefault("updatedAt", experiment.get("createdAt", now_iso()))
            experiment.setdefault("rounds", [])
            experiment["storageDir"] = experiment.get(
                "storageDir"
            ) or unique_storage_dir(
                experiment.get("name", "未命名实验"),
                timestamp_for_dir(experiment.get("createdAt")),
            )
            atomic_write_json(experiment_path(experiment["storageDir"]), experiment)
        LEGACY_DATA_FILE.replace(next_legacy_backup_path())


def list_experiment_files() -> list[Path]:
    ensure_store()
    migrate_legacy_store()
    with STORE_LOCK:
        return sorted(DATA_DIR.glob(f"*/{EXPERIMENT_FILE_NAME}"))


def load_all_experiments() -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    for path in list_experiment_files():
        try:
            experiments.append(load_experiment_file(path))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return experiments


def find_experiment_location(experiment_id: str) -> tuple[dict[str, Any], Path]:
    for path in list_experiment_files():
        try:
            experiment = load_experiment_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if experiment.get("id") == experiment_id:
            return experiment, path
    raise HTTPException(status_code=404, detail="Experiment not found")


def find_round(experiment: dict[str, Any], round_id: str) -> dict[str, Any]:
    for round_record in experiment.get("rounds", []):
        if round_record.get("roundId") == round_id:
            return round_record
    raise HTTPException(status_code=404, detail="Round not found")


def find_run_location(
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    for path in list_experiment_files():
        try:
            experiment = load_experiment_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for round_record in experiment.get("rounds", []):
            for run in round_record.get("runs", []):
                if run.get("runId") == run_id:
                    return experiment, round_record, run, path
    raise HTTPException(status_code=404, detail="Run not found")


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Missing OPENAI_API_KEY. Copy .env.example to .env and configure your OpenAI-compatible provider.",
        )
    return AsyncOpenAI(
        api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    )


async def call_model(
    messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int
) -> str:
    client = get_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Model call failed: {exc}"
        ) from exc
    return extract_model_content(response)


async def stream_model(
    messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int
):
    client = get_client()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Model stream failed: {exc}"
        ) from exc

    if isinstance(response, (str, dict)):
        content = extract_model_content(response)
        if content:
            yield content
        return

    if not hasattr(response, "__aiter__"):
        content = extract_model_content(response)
        if content:
            yield content
        return

    async for chunk in response:
        content = extract_stream_delta(chunk)
        if content:
            yield content


def extract_stream_delta(chunk: Any) -> str:
    if isinstance(chunk, str):
        sse_content = extract_sse_content(chunk)
        return sse_content or chunk

    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                delta = first.get("delta")
                if isinstance(delta, dict) and delta.get("content") is not None:
                    return str(delta.get("content") or "")
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return str(message.get("content") or "")
                if first.get("text") is not None:
                    return str(first.get("text") or "")
        if chunk.get("content") is not None:
            return str(chunk.get("content") or "")
        if chunk.get("text") is not None:
            return str(chunk.get("text") or "")
        return ""

    choices = getattr(chunk, "choices", None) or []
    if choices:
        first = choices[0]
        delta = getattr(first, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if content is not None:
                return str(content or "")
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return str(content or "")
        text = getattr(first, "text", None)
        if text is not None:
            return str(text or "")

    content = getattr(chunk, "content", None)
    if content is not None:
        return str(content or "")
    text = getattr(chunk, "text", None)
    if text is not None:
        return str(text or "")
    return ""


def extract_model_content(response: Any) -> str:
    if isinstance(response, str):
        sse_content = extract_sse_content(response)
        if sse_content:
            return sse_content
        parsed = extract_json(response)
        if "choices" not in parsed:
            return response
        response = parsed

    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                delta = first.get("delta")
                if isinstance(delta, dict):
                    return str(delta.get("content") or "")
                if first.get("text") is not None:
                    return str(first.get("text") or "")
        if response.get("content") is not None:
            return str(response.get("content") or "")
        if response.get("text") is not None:
            return str(response.get("text") or "")
        return json.dumps(response, ensure_ascii=False)

    choices = getattr(response, "choices", None) or []
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return str(getattr(message, "content", "") or "")
        delta = getattr(first, "delta", None)
        if delta is not None:
            return str(getattr(delta, "content", "") or "")
        text = getattr(first, "text", None)
        if text is not None:
            return str(text or "")

    content = getattr(response, "content", None)
    if content is not None:
        return str(content or "")
    text = getattr(response, "text", None)
    if text is not None:
        return str(text or "")
    return str(response)


def extract_sse_content(text: str) -> str:
    parts: list[str] = []
    saw_sse = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        saw_sse = True
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = extract_stream_delta(parsed)
        if delta:
            parts.append(delta)
    if saw_sse:
        return "".join(parts)
    return ""


def build_prompt_messages(prompt: str, user_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.strip()},
        {"role": "user", "content": user_input.strip()},
    ]


def run_input(round_record: dict[str, Any], run: dict[str, Any]) -> str:
    value = run.get("inputPrompt")
    if value is None:
        value = run.get("userInput")
    if value is None:
        value = round_record.get("inputPrompt", "")
    return str(value or "")


def run_index(run: dict[str, Any]) -> int:
    try:
        return int(run.get("index", 0))
    except (TypeError, ValueError):
        return 0


def build_round_messages(
    prompt: str, round_record: dict[str, Any], run: dict[str, Any], side: str
) -> list[dict[str, str]]:
    output_key = "outputA" if side == "A" else "outputB"
    current_index = run_index(run)
    messages = [{"role": "system", "content": prompt.strip()}]

    for previous_run in sorted(round_record.get("runs", []), key=run_index):
        if previous_run.get("runId") == run.get("runId"):
            continue
        if current_index and run_index(previous_run) >= current_index:
            continue

        previous_input = run_input(round_record, previous_run).strip()
        previous_output = str(previous_run.get(output_key) or "").strip()
        if previous_input and previous_output:
            messages.append({"role": "user", "content": previous_input})
            messages.append({"role": "assistant", "content": previous_output})

    current_input = run_input(round_record, run).strip()
    messages.append({"role": "user", "content": current_input})
    return messages


def evaluation_turns(
    round_record: dict[str, Any], current_run: dict[str, Any]
) -> list[dict[str, Any]]:
    current_index = run_index(current_run)
    turns: list[dict[str, Any]] = []
    for run in sorted(round_record.get("runs", []), key=run_index):
        if current_index and run_index(run) > current_index:
            continue
        turns.append(
            {
                "runId": run.get("runId"),
                "index": run.get("index"),
                "status": run.get("status", ""),
                "inputPrompt": run_input(round_record, run),
                "outputA": run.get("outputA", ""),
                "outputB": run.get("outputB", ""),
            }
        )
    return turns


def extract_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {"parseError": True, "raw": text}


def normalize_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    evaluation.setdefault("winner", "")
    evaluation.setdefault("scoreA", "")
    evaluation.setdefault("scoreB", "")
    evaluation.setdefault("dimensions", {})
    evaluation.setdefault("summary", "")
    evaluation.setdefault("suggestions", [])
    return evaluation


def make_round(
    experiment: dict[str, Any], prompt_a: str, prompt_b: str, input_prompt: str
) -> dict[str, Any]:
    timestamp = now_iso()
    rounds = experiment.setdefault("rounds", [])
    round_record = {
        "roundId": new_id("round"),
        "index": len(rounds) + 1,
        "promptA": prompt_a,
        "promptB": prompt_b,
        "inputPrompt": input_prompt,
        "createdAt": timestamp,
        "updatedAt": timestamp,
        "runs": [],
    }
    rounds.append(round_record)
    experiment["promptA"] = prompt_a
    experiment["promptB"] = prompt_b
    experiment["inputPrompt"] = input_prompt
    experiment["updatedAt"] = timestamp
    return round_record


def get_or_create_round(
    experiment: dict[str, Any],
    experiment_file: Path,
    round_id: str | None,
    input_prompt: str | None = None,
) -> dict[str, Any]:
    rounds = experiment.setdefault("rounds", [])
    if round_id:
        return find_round(experiment, round_id)
    if rounds:
        return rounds[-1]
    round_record = make_round(
        experiment,
        experiment.get("promptA", ""),
        experiment.get("promptB", ""),
        input_prompt if input_prompt is not None else experiment.get("inputPrompt", ""),
    )
    save_experiment(experiment, experiment_file)
    return round_record


def make_run(
    experiment: dict[str, Any],
    round_record: dict[str, Any],
    input_prompt: str | None = None,
) -> dict[str, Any]:
    timestamp = now_iso()
    runs = round_record.setdefault("runs", [])
    current_input = (
        input_prompt
        if input_prompt is not None
        else round_record.get("inputPrompt", "")
    )
    current_input = str(current_input or "")
    if not runs:
        round_record["inputPrompt"] = current_input
    experiment["inputPrompt"] = current_input
    run = {
        "runId": new_id("run"),
        "index": len(runs) + 1,
        "inputPrompt": current_input,
        "outputA": "",
        "outputB": "",
        "aiEvaluation": {},
        "humanChoice": "",
        "humanNote": "",
        "errors": [],
        "model": experiment.get("model", DEFAULT_MODEL),
        "params": deepcopy(experiment.get("params", {})),
        "status": "running",
        "startedAt": timestamp,
        "completedAt": "",
        "updatedAt": timestamp,
    }
    runs.append(run)
    round_record["updatedAt"] = timestamp
    experiment["updatedAt"] = timestamp
    return run


def complete_run(
    run_id: str, output_a: str, output_b: str, errors: list[str]
) -> dict[str, Any]:
    experiment, round_record, run, experiment_file = find_run_location(run_id)
    timestamp = now_iso()
    run["outputA"] = output_a
    run["outputB"] = output_b
    run["errors"] = errors
    run["status"] = "completed_with_errors" if errors else "completed"
    run["completedAt"] = timestamp
    run["updatedAt"] = timestamp
    round_record["updatedAt"] = timestamp
    experiment["updatedAt"] = timestamp
    save_experiment(experiment, experiment_file)
    return deepcopy(run)


def export_payload(experiment: dict[str, Any]) -> dict[str, Any]:
    rounds = deepcopy(
        sorted(experiment.get("rounds", []), key=lambda item: item.get("index", 0))
    )
    for round_record in rounds:
        round_record["runs"] = sorted(
            round_record.get("runs", []), key=lambda item: item.get("index", 0)
        )
        for run in round_record["runs"]:
            run.setdefault("inputPrompt", run_input(round_record, run))
            run["conversationTurns"] = evaluation_turns(round_record, run)
    return {
        "experiment": {
            "id": experiment.get("id"),
            "storageDir": experiment.get("storageDir"),
            "name": experiment.get("name"),
            "goal": experiment.get("goal"),
            "model": experiment.get("model"),
            "params": experiment.get("params", {}),
            "promptA": experiment.get("promptA", ""),
            "promptB": experiment.get("promptB", ""),
            "inputPrompt": experiment.get("inputPrompt", ""),
            "createdAt": experiment.get("createdAt"),
            "updatedAt": experiment.get("updatedAt"),
        },
        "rounds": rounds,
    }


def markdown_code_block(value: Any, language: str = "text") -> list[str]:
    text = str(value or "")
    if not text:
        text = "（空）"
    max_ticks = max(
        (len(match.group(0)) for match in re.finditer(r"`+", text)), default=0
    )
    fence = "`" * max(3, max_ticks + 1)
    return [f"{fence}{language}", text, fence]


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
async def get_config() -> dict[str, str]:
    return {"defaultModel": DEFAULT_MODEL}


@app.get("/api/experiments")
async def list_experiments() -> dict[str, Any]:
    summaries = []
    for experiment in load_all_experiments():
        rounds = experiment.get("rounds", [])
        run_count = sum(len(round_record.get("runs", [])) for round_record in rounds)
        summaries.append(
            {
                "id": experiment["id"],
                "storageDir": experiment.get("storageDir", ""),
                "name": experiment.get("name", ""),
                "goal": experiment.get("goal", ""),
                "model": experiment.get("model", DEFAULT_MODEL),
                "roundCount": len(rounds),
                "runCount": run_count,
                "createdAt": experiment.get("createdAt", ""),
                "updatedAt": experiment.get("updatedAt", ""),
            }
        )
    summaries.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
    return {"experiments": summaries}


@app.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict[str, Any]:
    experiment, _ = find_experiment_location(experiment_id)
    return {
        "experiment": export_payload(experiment)["experiment"],
        "rounds": export_payload(experiment)["rounds"],
    }


@app.post("/api/experiments")
async def upsert_experiment(payload: ExperimentIn) -> dict[str, Any]:
    timestamp = now_iso()
    experiment_id = payload.id or new_id("exp")
    incoming = {
        "id": experiment_id,
        "name": payload.name.strip() or "未命名实验",
        "goal": payload.goal.strip(),
        "model": payload.model or DEFAULT_MODEL,
        "params": {
            "temperature": payload.temperature,
            "maxTokens": payload.maxTokens,
            "baseUrl": os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        },
        "promptA": payload.promptA,
        "promptB": payload.promptB,
        "inputPrompt": payload.inputPrompt,
        "updatedAt": timestamp,
    }

    if payload.id:
        try:
            experiment, experiment_file = find_experiment_location(experiment_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
        else:
            incoming["createdAt"] = experiment.get("createdAt", timestamp)
            incoming["rounds"] = experiment.get("rounds", [])
            incoming["storageDir"] = experiment.get(
                "storageDir", experiment_file.parent.name
            )
            save_experiment(incoming, experiment_file)
            return {
                "experiment": export_payload(incoming)["experiment"],
                "rounds": export_payload(incoming)["rounds"],
            }

    incoming["createdAt"] = timestamp
    incoming["rounds"] = []
    incoming["storageDir"] = unique_storage_dir(
        incoming["name"], timestamp_for_dir(incoming["createdAt"])
    )
    save_experiment(incoming)
    return {"experiment": export_payload(incoming)["experiment"], "rounds": []}


@app.post("/api/rounds")
async def create_round(payload: RoundIn) -> dict[str, Any]:
    experiment, experiment_file = find_experiment_location(payload.experimentId)
    round_record = make_round(
        experiment, payload.promptA, payload.promptB, payload.inputPrompt
    )
    save_experiment(experiment, experiment_file)
    return {
        "round": deepcopy(round_record),
        "experiment": export_payload(experiment)["experiment"],
        "rounds": export_payload(experiment)["rounds"],
    }


@app.post("/api/generate")
async def generate(payload: GenerateIn) -> dict[str, Any]:
    experiment, experiment_file = find_experiment_location(payload.experimentId)
    round_record = get_or_create_round(
        experiment, experiment_file, payload.roundId, payload.inputPrompt
    )
    run = make_run(experiment, round_record, payload.inputPrompt)
    save_experiment(experiment, experiment_file)

    model = run.get("model") or DEFAULT_MODEL
    params = run.get("params", {})
    temperature = float(params.get("temperature", 0.7))
    max_tokens = int(params.get("maxTokens", 1000))
    output_a, output_b = await asyncio.gather(
        call_model(
            build_round_messages(
                round_record.get("promptA", ""), round_record, run, "A"
            ),
            model,
            temperature,
            max_tokens,
        ),
        call_model(
            build_round_messages(
                round_record.get("promptB", ""), round_record, run, "B"
            ),
            model,
            temperature,
            max_tokens,
        ),
    )
    completed = complete_run(run["runId"], output_a, output_b, [])
    return {"roundId": round_record["roundId"], "run": completed}


@app.post("/api/generate/stream")
async def generate_stream(payload: GenerateIn) -> StreamingResponse:
    experiment, experiment_file = find_experiment_location(payload.experimentId)
    round_record = get_or_create_round(
        experiment, experiment_file, payload.roundId, payload.inputPrompt
    )
    run = make_run(experiment, round_record, payload.inputPrompt)
    save_experiment(experiment, experiment_file)

    model = run.get("model") or DEFAULT_MODEL
    params = run.get("params", {})
    temperature = float(params.get("temperature", 0.7))
    max_tokens = int(params.get("maxTokens", 1000))
    prompt_a = round_record.get("promptA", "")
    prompt_b = round_record.get("promptB", "")
    run_id = run["runId"]
    round_id = round_record["roundId"]

    async def stream_events():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        output_a = ""
        output_b = ""
        errors: list[str] = []

        async def run_side(side: str, prompt: str) -> None:
            try:
                messages = build_round_messages(prompt, round_record, run, side)
                async for delta in stream_model(
                    messages, model, temperature, max_tokens
                ):
                    await queue.put({"side": side, "delta": delta})
                await queue.put({"side": side, "done": True})
            except HTTPException as exc:
                await queue.put({"side": side, "error": exc.detail, "done": True})
            except Exception as exc:
                await queue.put(
                    {"side": side, "error": f"Model stream failed: {exc}", "done": True}
                )

        tasks = [
            asyncio.create_task(run_side("A", prompt_a)),
            asyncio.create_task(run_side("B", prompt_b)),
        ]
        finished_sides: set[str] = set()

        try:
            yield (
                json.dumps(
                    {
                        "runId": run_id,
                        "roundId": round_id,
                        "startedAt": run.get("startedAt"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            while len(finished_sides) < 2:
                event = await queue.get()
                side = event.get("side")
                if side == "A" and event.get("delta"):
                    output_a += str(event["delta"])
                elif side == "B" and event.get("delta"):
                    output_b += str(event["delta"])
                if side and event.get("error"):
                    errors.append(f"{side}: {event['error']}")
                if event.get("done") and side:
                    finished_sides.add(str(side))
                yield json.dumps(event, ensure_ascii=False) + "\n"

            await asyncio.gather(*tasks, return_exceptions=True)
            completed = complete_run(run_id, output_a, output_b, errors)
            yield (
                json.dumps(
                    {
                        "done": True,
                        "runId": run_id,
                        "roundId": round_id,
                        "run": completed,
                        "generatedAt": completed.get("completedAt"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    return StreamingResponse(stream_events(), media_type="application/x-ndjson")


@app.post("/api/runs/{run_id}/evaluate")
async def evaluate_run(run_id: str) -> dict[str, Any]:
    experiment, round_record, run, experiment_file = find_run_location(run_id)
    params = run.get("params", {})
    model = run.get("model") or experiment.get("model") or DEFAULT_MODEL
    temperature = 0.2
    max_tokens = min(int(params.get("maxTokens", 1000)), 1500)

    experiment_goal = (experiment.get("goal") or "").strip() or "未填写"
    judge_prompt = f"""系统提示词 A/B 测试评审。请只输出 JSON，不要输出 Markdown。
实验目标：{experiment_goal}
评审时必须优先依据实验目标判断哪一侧更好；如果实验目标与通用质量维度冲突，以实验目标为准。
按 task_completion、accuracy、completeness、clarity、style_match、format_following、risk 七个维度评价 A/B。
JSON 字段必须包含 winner、scoreA、scoreB、dimensions、summary、suggestions。
winner 只能是 "A"、"B"、"tie" 或 "neither"。scoreA 和 scoreB 是 1-10 的数字。"""
    user_content = json.dumps(
        {
            "experimentGoal": experiment.get("goal", ""),
            "inputPrompt": run_input(round_record, run),
            "conversationTurns": evaluation_turns(round_record, run),
            "systemPromptA": round_record.get("promptA", ""),
            "systemPromptB": round_record.get("promptB", ""),
            "outputA": run.get("outputA", ""),
            "outputB": run.get("outputB", ""),
        },
        ensure_ascii=False,
    )
    raw = await call_model(
        [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_content},
        ],
        model,
        temperature,
        max_tokens,
    )

    timestamp = now_iso()
    evaluation = normalize_evaluation(extract_json(raw))
    run["aiEvaluation"] = evaluation
    run["updatedAt"] = timestamp
    run["evaluatedAt"] = timestamp
    round_record["updatedAt"] = timestamp
    experiment["updatedAt"] = timestamp
    save_experiment(experiment, experiment_file)
    return {"run": deepcopy(run), "evaluation": evaluation, "raw": raw}


@app.patch("/api/runs/{run_id}")
async def update_run(run_id: str, payload: RunPatchIn) -> dict[str, Any]:
    experiment, round_record, run, experiment_file = find_run_location(run_id)
    timestamp = now_iso()
    run["humanChoice"] = payload.humanChoice
    run["humanNote"] = payload.humanNote
    run["updatedAt"] = timestamp
    round_record["updatedAt"] = timestamp
    experiment["updatedAt"] = timestamp
    save_experiment(experiment, experiment_file)
    return {"run": deepcopy(run)}


@app.get("/api/export/{experiment_id}.json")
async def export_json(experiment_id: str) -> JSONResponse:
    experiment, _ = find_experiment_location(experiment_id)
    return JSONResponse(
        export_payload(experiment),
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.json"'},
    )


@app.get("/api/export/{experiment_id}.md")
async def export_markdown(experiment_id: str) -> PlainTextResponse:
    experiment, _ = find_experiment_location(experiment_id)
    payload = export_payload(experiment)
    info = payload["experiment"]
    lines = [
        f"# 系统提示词对比实验报告：{info.get('name') or experiment_id}",
        "",
        "## 实验信息",
        f"- 实验 ID：{info.get('id')}",
        f"- 实验目标：{info.get('goal') or '未填写'}",
        f"- 模型：{info.get('model')}",
        f"- 参数：`{json.dumps(info.get('params', {}), ensure_ascii=False)}`",
        f"- 创建时间：{info.get('createdAt')}",
        f"- 更新时间：{info.get('updatedAt')}",
        "",
    ]

    for round_record in payload["rounds"]:
        lines.extend(
            [
                f"## 第 {round_record.get('index')} 轮：系统 Prompt 版本",
                "",
                "### 首条输入 Prompt",
                round_record.get("inputPrompt", ""),
                "",
                "### 系统 Prompt A",
                round_record.get("promptA", ""),
                "",
                "### 系统 Prompt B",
                round_record.get("promptB", ""),
                "",
            ]
        )
        runs = round_record.get("runs", [])
        if not runs:
            lines.extend(["### 本轮暂无生成记录", ""])
            continue
        for run in runs:
            evaluation = run.get("aiEvaluation") or {}
            lines.extend(
                [
                    f"### 第 {run.get('index')} 次生成",
                    "",
                    f"- Run ID：{run.get('runId')}",
                    f"- 状态：{run.get('status', '')}",
                    f"- 开始时间：{run.get('startedAt', '')}",
                    f"- 完成时间：{run.get('completedAt', '')}",
                    f"- 本次输入：{run_input(round_record, run)}",
                    "",
                    "#### 轮内对话记录（截至本次生成）",
                    "",
                ]
            )
            for turn in run.get("conversationTurns", []):
                lines.extend(
                    [
                        f"##### 第 {turn.get('index')} 次往返",
                        "",
                        "**User**",
                        *markdown_code_block(turn.get("inputPrompt", "")),
                        "",
                        "**Assistant A**",
                        *markdown_code_block(turn.get("outputA", "")),
                        "",
                        "**Assistant B**",
                        *markdown_code_block(turn.get("outputB", "")),
                        "",
                    ]
                )
            lines.extend(
                [
                    "#### 输出 A",
                    run.get("outputA", ""),
                    "",
                    "#### 输出 B",
                    run.get("outputB", ""),
                    "",
                    "#### AI 评价",
                    f"- 优胜方：{evaluation.get('winner', '')}",
                    f"- A 得分：{evaluation.get('scoreA', '')}",
                    f"- B 得分：{evaluation.get('scoreB', '')}",
                    f"- 摘要：{evaluation.get('summary', '')}",
                    "",
                    "#### 人工选择",
                    f"- 选择：{run.get('humanChoice', '')}",
                    f"- 备注：{run.get('humanNote', '')}",
                    "",
                ]
            )
            if run.get("errors"):
                lines.extend(
                    [
                        "#### 错误",
                        *[f"- {error}" for error in run.get("errors", [])],
                        "",
                    ]
                )

    return PlainTextResponse(
        "\n".join(lines),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{experiment_id}.md"'},
    )
