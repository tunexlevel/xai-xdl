

import json
import re
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

from helper.utils import strip_atom_mapping_labels, strip_atom_mapping


WHEEL_ACTIONS = {
    "transfer",
    "transfer_waste",
    "wash",
    "timed_stir",
    "wait",
    "measure_spectrum",
    "monitor_reaction_spectrum",
    "capture_reference_spectrum",
    "capture_dark_spectrum",
    "capture_image",
    "start_video",
    "stop_video",
}


def _strip_protocol_atom_maps(value: Any) -> Any:
    if isinstance(value, str):
        return strip_atom_mapping(value, canonical=False)
    if isinstance(value, list):
        return [_strip_protocol_atom_maps(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_protocol_atom_maps(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _strip_protocol_atom_maps(item)
            for key, item in value.items()
        }
    return value


def read_config() -> dict[str, Any]:
    """Return protocol AI configuration when no backend is configured."""
    return {}


def _legacy_command_available(command: Any) -> bool:
    return bool(command)


def run_ai(prompt: str, timeout_sec: int = 8) -> str:
    raise RuntimeError("No local AI backend is configured.")


def recipe_to_protocol(recipe_text: str) -> list[dict[str, Any]]:
    """Parse recipe lines locally, without requiring an AI backend."""
    if not isinstance(recipe_text, str) or not recipe_text.strip():
        raise ValueError("Recipe text must be a non-empty string.")

    protocol: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for line in recipe_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _ai_wheel_line(line)
        if parsed:
            protocol.extend(parsed)
        else:
            unsupported.append(line)

    if unsupported:
        raise ValueError(
            "Could not parse recipe line(s): " + "; ".join(unsupported)
        )
    protocol = _strip_protocol_atom_maps(protocol)
    validate_protocol(protocol)
    return protocol


def reaction_to_draft_protocol(
    reactant_smiles: str,
    product_smiles: str = "",
    volume_ml: float = 1.0,
    stir_duration_sec: int = 60,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create a reviewable protocol draft when no experimental recipe exists."""
    if not isinstance(reactant_smiles, str) or not reactant_smiles.strip():
        raise ValueError("Reactant SMILES must be a non-empty string.")

    reactant_smiles = strip_atom_mapping(reactant_smiles, canonical=False)
    components = [component.strip() for component in reactant_smiles.split(".")]
    components = [component for component in components if component]
    if not components:
        raise ValueError("Reactant SMILES contains no components.")

    volume = float(volume_ml)
    duration = int(stir_duration_sec)
    if volume <= 0 or duration <= 0:
        raise ValueError("Draft volume and stir duration must be positive.")

    steps: list[dict[str, Any]] = []
    for component in components:
        steps.append({
            "action": "transfer",
            "reagent": component,
            "volume_ml": volume,
            "destination": "vial_1",
            "source_text": f"add {component} {volume:g}ml to vial 1",
        })
    steps.extend([
        {
            "action": "timed_stir",
            "vials": ["vial_1"],
            "duration_sec": duration,
            "speed": "medium",
            "source_text": f"stir vial 1 for {duration}s at medium",
        },
        {
            "action": "measure_spectrum",
            "vial": "vial_1",
            "integration_time_ms": 50,
            "led_brightness": 255,
            "source_text": "measure spectrum vial 1",
        },
    ])
    steps = _strip_protocol_atom_maps(steps)
    validate_protocol(steps)
    warnings = [
        "Draft protocol generated from SMILES because recipe_text was not supplied.",
        "Each reactant component was assigned 1.0 mL; verify stoichiometry and concentration.",
        "Stirring was assigned 60 seconds at medium speed; verify reaction conditions.",
        "Temperature, solvent, catalyst, atmosphere, and work-up are unknown.",
    ]
    if product_smiles:
        warnings.append(
            "The predicted product is included for review only; it does not validate the procedure."
        )
    return steps, warnings


def validate_protocol(
    protocol: list[dict[str, Any]],
    platform: str = "wheel_platform",
) -> None:
    """Reject incomplete or unsupported steps before XDL generation."""
    if not isinstance(protocol, list) or not protocol:
        raise ValueError("Protocol must contain at least one step.")
    if platform != "wheel_platform":
        raise ValueError(f"Unsupported protocol platform: {platform}")

    for index, step in enumerate(protocol, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {index} must be an object.")
        action = str(step.get("action", "")).strip().lower()
        if action not in WHEEL_ACTIONS:
            raise ValueError(f"Step {index} has unsupported action: {action!r}")

        if action in {"transfer", "wash"}:
            _require_text(step, "reagent", index)
            _require_positive(step, "volume_ml", index)
            _require_vial(step.get("destination"), index, "destination")
        elif action == "transfer_waste":
            _require_positive(step, "volume_ml", index)
            _require_vial(step.get("source_vial"), index, "source_vial")
        elif action == "timed_stir":
            vials = _normalize_vials(step.get("vials"))
            if not vials:
                raise ValueError(f"Step {index} requires at least one vial.")
            for vial in vials:
                _require_vial(vial, index, "vials")
            _require_positive(step, "duration_sec", index)
            _require_text(step, "speed", index)
        elif action == "wait":
            _require_positive(step, "duration_sec", index)
        elif action in {"measure_spectrum", "monitor_reaction_spectrum"}:
            _require_vial(step.get("vial"), index, "vial")
            if action == "monitor_reaction_spectrum":
                _require_positive(step, "duration_sec", index)
                _require_positive(step, "interval_sec", index)
        elif action == "capture_reference_spectrum":
            if step.get("vial"):
                _require_vial(step.get("vial"), index, "vial")
        elif action in {"capture_image", "start_video"}:
            _require_text(step, "filename", index)


def _require_text(step: dict[str, Any], field: str, index: int) -> str:
    value = str(step.get(field, "")).strip()
    if not value:
        raise ValueError(f"Step {index} requires {field!r}.")
    return value


def _require_positive(step: dict[str, Any], field: str, index: int) -> float:
    try:
        value = float(step.get(field))
    except (TypeError, ValueError):
        raise ValueError(f"Step {index} requires numeric {field!r}.") from None
    if value <= 0:
        raise ValueError(f"Step {index} requires positive {field!r}.")
    return value


def _require_vial(value: Any, index: int, field: str) -> str:
    try:
        vial = _normalize_vial(value)
    except (RuntimeError, ValueError):
        raise ValueError(f"Step {index} has invalid {field!r}: {value!r}.") from None
    number = int(vial.split("_")[1])
    if number < 1 or number > 24:
        raise ValueError(f"Step {index} has vial outside vial_1..vial_24.")
    return vial


def ai_available() -> bool:
    config = read_config()
    model = str(config.get("ollama_model") or "qwen2.5:1.5b").strip() or "qwen2.5:1.5b"
    backend = str(config.get("backend") or "auto").strip().lower()
    if backend in {"portable_ollama", "ollama"}:
        return False
    command = config.get("command", [])
    return _legacy_command_available(command)

def ai_recipe_available() -> bool:
    return ai_available()

def ai_recipe_to_protocol(recipe_text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in recipe_text.splitlines() if line.strip()]
    protocol: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for line in lines:
        direct = _ai_wheel_line(line)
        if direct:
            protocol.extend(direct)
        else:
            unresolved.append(line)

    if unresolved:
        config = _read_config()
        if not ai_available():
            raise RuntimeError(
                "AI Assist is not configured for line(s): "
                + "; ".join(unresolved)
            )
        timeout = max(3, min(int(config.get("timeout_sec", 8) or 8), 8))
        for line in unresolved:
            prompt = _build_prompt(line)
            payload = _run_ai_text(prompt, timeout)
            protocol.extend(_normalize_protocol(payload, line))

    protocol = _strip_protocol_atom_maps(protocol)
    validate_protocol(protocol)
    return protocol



def _clean_media_filename(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(?:of|as|named)\s+", "", text, flags=re.IGNORECASE).strip()
    return text.strip("\"' ")

def _run_ai_text(prompt: str, timeout: int) -> Any:
    last_error = "Local AI command failed."
    for _ in range(2):
        try:
            return _extract_json(run_ai(prompt, timeout_sec=timeout))
        except Exception as exc:
            last_error = str(exc).strip() or "Local AI command failed."
    raise RuntimeError(last_error)


def _read_config() -> dict[str, Any]:
    return read_config()


def _build_prompt(recipe_text: str, platform = "wheel_platform") -> str:
    action_specs = {
        "wheel_platform": [
            {"action": "transfer", "fields": ["reagent", "volume_ml", "destination"]},
            {"action": "transfer_waste", "fields": ["source_vial", "volume_ml"]},
            {"action": "wash", "fields": ["reagent", "volume_ml", "destination", "cycles"]},
            {"action": "timed_stir", "fields": ["vials", "duration_sec", "speed"]},
            {"action": "wait", "fields": ["duration_sec"]},
            {"action": "measure_spectrum", "fields": ["vial", "integration_time_ms", "led_brightness"]},
            {"action": "monitor_reaction_spectrum", "fields": ["vial", "duration_sec", "interval_sec"]},
            {"action": "capture_reference_spectrum", "fields": ["integration_time_ms", "led_brightness"]},
            {"action": "capture_dark_spectrum", "fields": ["integration_time_ms"]},
            {"action": "capture_image", "fields": ["filename"]},
            {"action": "start_video", "fields": ["filename"]},
            {"action": "stop_video", "fields": []},
        ],
    }
    actions = action_specs.get(platform, action_specs["wheel_platform"])
    example = [
        {
            "action": "transfer",
            "reagent": "Aniline",
            "volume_ml": 1.0,
            "destination": "vial_1",
            "source_text": "transfer 1 ml of aniline to vial 1",
        },
        {
            "action": "wait",
            "duration_sec": 5,
            "source_text": "pause 5s",
        },
    ]
    return (
        "Convert this single recipe line into JSON protocol steps.\n"
        f"Platform: {platform}\n"
        "Return JSON only. Do not add prose, markdown, or code fences.\n"
        "Return either one JSON object or a JSON array.\n"
        "Each item must contain an action field and only the fields needed by that action.\n"
        f"Allowed actions: {json.dumps(actions)}\n"
        "Use vial_1..vial_24 for wheel vials.\n"
        "Use integers for duration_sec and floats for volume_ml.\n"
        "Normalize typos like 'via 18' to 'vial_18'.\n"
        "Keep source_text equal to the input line that produced the step.\n"
        f"Example output: {json.dumps(example)}\n"
        f"Recipe line:\n{recipe_text.strip()}\n"
    )


def _extract_json(text: str) -> Any:
    match = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1).strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            chunk = text[start : end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def _normalize_protocol(payload: Any,  recipe_text: str, platform = "wheel_platform") -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "action" in payload:
            payload = [payload]
        else:
            payload = payload.get("steps", [])
    if not isinstance(payload, list):
        raise RuntimeError("AI Assist did not return a step list.")
    normalizers = {
        "wheel_platform": _normalize_wheel_steps,
        "robot_arm_platform": _normalize_robot_steps,
        "8pumps_platform": _normalize_pump_steps,
        "syringe_platform": _normalize_syringe_steps,
    }
    normalizer = normalizers.get(platform, _normalize_wheel_steps)
    lines = [line.strip() for line in recipe_text.splitlines() if line.strip()]
    return normalizer(payload, lines)


def _normalize_wheel_steps(payload: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise RuntimeError("AI Assist returned an invalid wheel step.")
        action = str(raw.get("action", "")).strip().lower()
        source = str(raw.get("source_text") or _line_at(lines, index))
        if action == "transfer":
            out.append(
                {
                    "action": "transfer",
                    "pump_id": str(raw.get("pump_id", "")),
                    "reagent": _title(raw.get("reagent")),
                    "volume_ml": _to_float(raw.get("volume_ml")),
                    "destination": _normalize_vial(raw.get("destination")),
                    "source_text": source,
                }
            )
            continue
        if action == "transfer_waste":
            out.append(
                {
                    "action": "transfer_waste",
                    "source_vial": _normalize_vial(raw.get("source_vial")),
                    "volume_ml": _to_float(raw.get("volume_ml")),
                    "source_text": source,
                }
            )
            continue
        if action == "wash":
            out.append(
                {
                    "action": "wash",
                    "pump_id": str(raw.get("pump_id", "P2")),
                    "reagent": _title(raw.get("reagent", "Water")),
                    "volume_ml": _to_float(raw.get("volume_ml")),
                    "destination": _normalize_vial(raw.get("destination")),
                    "cycles": max(1, int(raw.get("cycles", 1) or 1)),
                    "source_text": source,
                }
            )
            continue
        if action == "timed_stir":
            duration = _to_int(raw.get("duration_sec"))
            speed = str(raw.get("speed", "90")).strip()
            if re.search(r"(s|sec|secs|second|seconds)\s*$", speed.lower()):
                swapped_duration = _to_int(speed)
                speed = str(duration)
                duration = swapped_duration
            out.append(
                {
                    "action": "timed_stir",
                    "vials": _normalize_vials(raw.get("vials")),
                    "duration_sec": duration,
                    "speed": speed,
                    "source_text": source,
                }
            )
            continue
        if action == "wait":
            out.append({"action": "wait", "duration_sec": _to_int(raw.get("duration_sec")), "source_text": source})
            continue
        raise RuntimeError(f"Unsupported AI wheel action: {action}")
    return out


def _system_wheel_line(line: str) -> list[dict[str, Any]]:
    raw = line.strip()
    cleaned = _repair_wheel_line(raw)
    source = raw
    raw = cleaned
    text = raw.lower()
    m = re.fullmatch(
        r"(?:add|transfer|dispense|dose|deliver|load|pour|inject)\s+(.+?)\s+([0-9]*\.?[0-9]+\s*ml?)\s+(?:to|into)\s+via?l?\s+(\d+)",
        text,
    )
    if not m:
        m = re.fullmatch(
            r"(?:add|transfer|dispense|dose|deliver|load|pour|inject)\s+([0-9]*\.?[0-9]+\s*ml?)\s+(?:of\s+)?(.+?)\s+(?:to|into)\s+via?l?\s+(\d+)",
            text,
        )
        if m:
            return [
                {
                    "action": "transfer",
                    "pump_id": "",
                    "reagent": _title(m.group(2)),
                    "volume_ml": _to_float(m.group(1)),
                    "destination": _normalize_vial(m.group(3)),
                    "source_text": source,
                }
            ]
    if m:
        return [
            {
                "action": "transfer",
                "pump_id": "",
                "reagent": _title(m.group(1)),
                "volume_ml": _to_float(m.group(2)),
                "destination": _normalize_vial(m.group(3)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:pour|inject)\s+(?:into\s+)?vial\s+(\d+)\s+([0-9]*\.?[0-9]+\s*ml?)\s+(?:of\s+)?(.+)",
        text,
    )
    if m:
        return [
            {
                "action": "transfer",
                "pump_id": "",
                "reagent": _title(m.group(3)),
                "volume_ml": _to_float(m.group(2)),
                "destination": _normalize_vial(m.group(1)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:in\s+vial\s+(\d+),\s*)?(?:waste|remove|discard|drain)\s+waste\s+of\s+([0-9]*\.?[0-9]+\s*ml?)\s+from\s+vial\s+(\d+)",
        text,
    )
    if m:
        target = m.group(3) or m.group(1)
        return [
            {
                "action": "transfer_waste",
                "source_vial": _normalize_vial(target),
                "volume_ml": _to_float(m.group(2)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:waste|remove|discard|drain)\s+([0-9]*\.?[0-9]+\s*ml?)\s+(?:of\s+)?waste\s+from\s+vial\s+(\d+)",
        text,
    )
    if m:
        return [
            {
                "action": "transfer_waste",
                "source_vial": _normalize_vial(m.group(2)),
                "volume_ml": _to_float(m.group(1)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:in\s+vial\s+(\d+),\s*)?(?:waste|remove|discard|drain)\s+([0-9]*\.?[0-9]+\s*ml?)\s+(?:of\s+)?waste",
        text,
    )
    if m:
        return [
            {
                "action": "transfer_waste",
                "source_vial": _normalize_vial(m.group(1)),
                "volume_ml": _to_float(m.group(2)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(r"(?:waste|remove|discard|drain)\s+([0-9]*\.?[0-9]+\s*ml?)\s+from\s+vial\s+(\d+)", text)
    if m:
        return [
            {
                "action": "transfer_waste",
                "source_vial": _normalize_vial(m.group(2)),
                "volume_ml": _to_float(m.group(1)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:wash|rinse|clean)\s+vial\s+(\d+)\s+with\s+(.+?)\s+([0-9]*\.?[0-9]+\s*ml?)(?:\s+x(\d+))?",
        text,
    )
    if m:
        return [
            {
                "action": "wash",
                "pump_id": "P2",
                "reagent": _title(m.group(2)),
                "volume_ml": _to_float(m.group(3)),
                "destination": _normalize_vial(m.group(1)),
                "cycles": max(1, int(m.group(4) or 1)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:wash out|rinse out)\s+vial\s+(\d+)\s+with\s+(.+?)\s+([0-9]*\.?[0-9]+\s*ml?)(?:\s+x(\d+))?",
        text,
    )
    if m:
        return [
            {
                "action": "wash",
                "pump_id": "P2",
                "reagent": _title(m.group(2)),
                "volume_ml": _to_float(m.group(3)),
                "destination": _normalize_vial(m.group(1)),
                "cycles": max(1, int(m.group(4) or 1)),
                "source_text": source,
            }
        ]
    m = re.fullmatch(
        r"(?:stir|mix|agitate)\s+(.+?)\s+for\s+([0-9]*\.?[0-9]+\s*(?:s|sec|secs|second|seconds)?)\s+at\s+([a-z0-9_]+)",
        text,
    )
    if m:
        duration = _to_int(m.group(2))
        speed = m.group(3).strip()
        if re.search(r"(s|sec|secs|second|seconds)\s*$", speed.lower()):
            swapped_duration = _to_int(speed)
            speed = str(duration)
            duration = swapped_duration
        return [
            {
                "action": "timed_stir",
                "vials": _normalize_stir_targets(m.group(1)),
                "duration_sec": duration,
                "speed": speed,
                "source_text": raw,
            }
        ]
    m = re.fullmatch(
        r"(?:wait|hold|pause|leave(?: it)?|leave\s+vial\s+\d+|let\s+vial\s+\d+\s+sit)\s+(?:for\s+)?([0-9]*\.?[0-9]+\s*(?:h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds)?)",
        text,
    )
    if m:
            return [{"action": "wait", "duration_sec": _to_int(m.group(1)), "source_text": source}]
    semantic = _semantic_wheel_line(cleaned)
    if semantic:
        semantic["source_text"] = source
    return [semantic] if semantic else []


def _ai_wheel_line(line: str) -> list[dict[str, Any]]:
    direct = _system_wheel_line(line)
    if direct:
        return direct
    repaired = _repair_wheel_line(line)
    if repaired != line:
        direct = _system_wheel_line(repaired)
        if direct:
            return direct
    semantic = _semantic_wheel_line(repaired)
    return [semantic] if semantic else []


def _repair_wheel_line(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\bvia\b", "vial", text, flags=re.IGNORECASE)
    text = re.sub(r"\binto vial\b", "to vial", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin vial\s+(\d+),\s*add\b", r"add", text, flags=re.IGNORECASE)
    text = re.sub(r"\bin vial\s+(\d+),\s*(remove|discard|drain|waste)\b", r"\2", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:then|now|next|please|kindly|after that)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\beverything\b", "all", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _semantic_wheel_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    lower = text.lower()
    if not lower:
        return None
    if any(word in lower for word in ("wait", "hold", "pause", "leave it", "let it sit")):
        duration = _find_duration(lower)
        if duration is not None:
            return {"action": "wait", "duration_sec": duration, "source_text": text}
    if any(word in lower for word in ("capture image", "take image", "capture photo", "take photo", "snap image", "snap photo")):
        item = {"action": "capture_image", "source_text": text}
        filename = _find_filename(lower)
        if filename:
            item["filename"] = filename
        return item
    if any(word in lower for word in ("start video", "record video", "start recording")):
        item = {"action": "start_video", "source_text": text}
        filename = _find_filename(lower)
        if filename:
            item["filename"] = filename
        return item
    if any(word in lower for word in ("stop video", "stop recording")):
        return {"action": "stop_video", "source_text": text}
    if any(word in lower for word in ("reference spectrum", "blank spectrum")):
        item = {"action": "capture_reference_spectrum", "source_text": text}
        vial = _find_first_vial(lower)
        if vial:
            item["vial"] = vial
        return item
    if "dark spectrum" in lower:
        return {"action": "capture_dark_spectrum", "source_text": text}
    if any(word in lower for word in ("monitor spectrum", "monitor reaction spectrum")):
        return {
            "action": "monitor_reaction_spectrum",
            "vial": _find_first_vial(lower) or "vial_1",
            "duration_sec": _find_duration(lower) or 3600,
            "interval_sec": _find_interval(lower) or 60,
            "source_text": text,
        }
    if any(word in lower for word in ("measure spectrum", "read spectrum", "scan spectrum", "take spectrum")):
        return {
            "action": "measure_spectrum",
            "vial": _find_first_vial(lower) or "",
            "integration_time_ms": _find_integration_ms(lower) or 50,
            "led_brightness": _find_led_brightness(lower) or 255,
            "source_text": text,
        }
    if any(word in lower for word in ("stir", "mix", "agitate")):
        vials = _find_vials(lower)
        duration = _find_duration(lower)
        speed = _find_stir_speed(lower)
        if vials and duration is not None and speed:
            return {
                "action": "timed_stir",
                "vials": vials,
                "duration_sec": duration,
                "speed": speed,
                "source_text": text,
            }
    if any(word in lower for word in ("wash", "rinse", "clean", "wash out", "rinse out")):
        vial = _find_first_vial(lower)
        volume = _find_volume(lower)
        reagent = _find_reagent(lower)
        cycles = _find_cycles(lower) or 1
        if vial and volume is not None and reagent:
            return {
                "action": "wash",
                "pump_id": "P2",
                "reagent": reagent,
                "volume_ml": volume,
                "destination": vial,
                "cycles": cycles,
                "source_text": text,
            }
    if any(word in lower for word in ("waste", "remove", "discard", "drain", "empty")):
        vial = _find_first_vial(lower)
        volume = _find_volume(lower)
        if vial and volume is not None:
            return {
                "action": "transfer_waste",
                "source_vial": vial,
                "volume_ml": volume,
                "source_text": text,
            }
    if any(word in lower for word in ("add", "transfer", "dispense", "dose", "deliver", "load", "put", "send", "drop", "pour", "inject")):
        vial = _find_first_vial(lower)
        volume = _find_volume(lower)
        reagent = _find_reagent(lower)
        if vial and volume is not None and reagent:
            return {
                "action": "transfer",
                "pump_id": "",
                "reagent": reagent,
                "volume_ml": volume,
                "destination": vial,
                "source_text": text,
            }
    return None


def _find_volume(text: str) -> float | None:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*ml?\b", text)
    return float(match.group(1)) if match else None


def _find_duration(text: str) -> int | None:
    match = re.search(
        r"([0-9]*\.?[0-9]+)\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds)\b",
        text,
    )
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"h", "hr", "hrs", "hour", "hours"}:
            return int(value * 3600)
        if unit in {"min", "mins", "minute", "minutes"}:
            return int(value * 60)
        return int(value)
    plain = re.search(r"\bfor\s+([0-9]*\.?[0-9]+)\b", text)
    return int(float(plain.group(1))) if plain else None


def _find_cycles(text: str) -> int | None:
    match = re.search(r"\bx\s*(\d+)\b", text)
    return int(match.group(1)) if match else None


def _find_first_vial(text: str) -> str | None:
    match = re.search(r"via?l?\s+(\d+)", text)
    return f"vial_{int(match.group(1))}" if match else None


def _find_vials(text: str) -> list[str]:
    if any(flag in text for flag in ("all vials", "all reactors", "every vial", "everything")) or re.search(r"\ball\b", text):
        return [f"vial_{i}" for i in range(1, 25)]
    items = [f"vial_{int(number)}" for number in re.findall(r"via?l?\s+(\d+)", text)]
    return list(dict.fromkeys(items))


def _find_stir_speed(text: str) -> str | None:
    at_match = re.search(r"\bat\s+([a-z0-9_]+)", text)
    if at_match:
        speed = at_match.group(1).strip()
        if re.search(r"(s|sec|secs|second|seconds)\s*$", speed):
            for_match = re.search(r"\bfor\s+([0-9]*\.?[0-9]+)\b", text)
            if for_match:
                return for_match.group(1)
        return speed
    numbers = [value for value in re.findall(r"([0-9]*\.?[0-9]+)", text)]
    if len(numbers) >= 2:
        return numbers[0]
    return "90"



def _find_syringe_pump(text: str) -> str | None:
    match = re.search(r"\bpump\s*([abcde])\b|\b([abcde])\s*pump\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def _find_syringe_rate(text: str) -> float | None:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*(?:u\s*l|ul|µl|microlitre|microliter)s?\s*/?\s*min", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _find_percent(text: str) -> float | None:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*(?:%|percent)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _find_fan_word_percent(text: str) -> float | None:
    words = {
        "off": 0.0,
        "low": 25.0,
        "medium": 50.0,
        "mid": 50.0,
        "high": 75.0,
        "full": 100.0,
        "max": 100.0,
    }
    lower = str(text or "").lower()
    for word, percent in words.items():
        if re.search(rf"\b{word}\b", lower):
            return percent
    return None


def _find_number_after_word(text: str, words: tuple[str, ...]) -> float | None:
    pattern = "|".join(re.escape(word) for word in words)
    match = re.search(rf"(?:{pattern})\s*([0-9]*\.?[0-9]+)", str(text or ""), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _to_fan_percent(value: Any) -> float:
    mapped = _find_fan_word_percent(str(value or ""))
    if mapped is not None:
        return mapped
    return _to_float(value)


def _find_temperature(text: str) -> float | None:
    match = re.search(r"([0-9]*\.?[0-9]+)\s*(?:c|celsius)\b", text)
    return float(match.group(1)) if match else None


def _find_pwm(text: str) -> int | None:
    match = re.search(r"(?:pwm\s*)?([0-9]{1,3})\b", text)
    return int(match.group(1)) if match else None


def _find_integration_ms(text: str) -> int | None:
    match = re.search(r"(?:integration|int)\s*([0-9]*\.?[0-9]+)\s*ms\b", text)
    return int(float(match.group(1))) if match else None


def _find_led_brightness(text: str) -> int | None:
    match = re.search(r"(?:led|brightness)\s*([0-9]{1,3})\b", text)
    return max(0, min(255, int(match.group(1)))) if match else None


def _find_interval(text: str) -> int | None:
    match = re.search(r"(?:every|interval)\s+([0-9]*\.?[0-9]+\s*(?:h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds)?)", text)
    return _find_duration(match.group(1)) if match else None


def _find_filename(text: str) -> str | None:
    match = re.search(r"(?:named|as)\s+([a-z0-9_.-]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([a-z0-9_.-]+\.(?:png|jpg|jpeg|tif|tiff|bmp|mp4|avi|mov|mkv))\b", text)
    return match.group(1) if match else None


def _find_reagent(text: str) -> str | None:
    working = f" {text} "
    working = re.sub(r"[,;:]", " ", working)
    working = re.sub(r"via?l?\s+\d+", " ", working)
    working = re.sub(r"\bp\s*[1-8]\b", " ", working)
    working = re.sub(r"[0-9]*\.?[0-9]+\s*ml?\b", " ", working)
    working = re.sub(r"[0-9]*\.?[0-9]+\s*(?:s|sec|secs|second|seconds)\b", " ", working)
    working = re.sub(r"\bx\s*\d+\b", " ", working)
    for token in [
        "add",
        "transfer",
        "dispense",
        "dose",
        "deliver",
        "load",
        "put",
        "send",
        "drop",
        "pour",
        "inject",
        "waste",
        "remove",
        "discard",
        "drain",
        "take",
        "out",
        "empty",
        "using",
        "via",
        "through",
        "flush",
        "wash",
        "rinse",
        "clean",
        "warm",
        "cool",
        "stir",
        "mix",
        "agitate",
        "wait",
        "hold",
        "pause",
        "then",
        "after",
        "that",
        "now",
        "next",
        "please",
        "kindly",
        "leave",
        "left",
        "let",
        "sit",
        "everything",
        "keep",
        "temp",
        "temperature",
        "snap",
        "image",
        "photo",
        "from",
        "to",
        "into",
        "with",
        "of",
        "in",
        "for",
        "at",
        "and",
        "x",
    ]:
        working = re.sub(rf"\b{token}\b", " ", working)
    working = re.sub(r"\bwaste\b", " ", working)
    parts = [part for part in re.split(r"\s+", working.strip()) if part]
    if not parts:
        return None
    return _title(" ".join(parts))


def _normalize_stir_targets(value: str) -> list[str]:
    text = str(value or "").strip().lower()
    if text in {"all", "all vials", "every vial", "all reactors", "everything"}:
        return [f"vial_{i}" for i in range(1, 25)]
    cleaned = re.sub(r"\band\b", ",", text)
    items = []
    for token in [part.strip() for part in cleaned.split(",") if part.strip()]:
        items.append(_normalize_vial(token))
    return [item for item in dict.fromkeys(items) if item]


def _ai_syringe_line(line: str) -> list[dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(line or "").strip())
    text = re.sub(r"^\s*(?:then|now|next|please|kindly|after that)\s+", "", text, flags=re.IGNORECASE)
    lower = text.lower().strip(" ,.")
    if not lower:
        return []
    pump = _find_syringe_pump(lower)
    volume = _find_volume(lower)
    rate = _find_syringe_rate(lower)
    duration = _find_duration(lower)
    source = str(line or "").strip()
    if "aspirat" in lower and pump and volume is not None:
        item = {"action": "aspirate", "pump": pump, "volume_ml": volume, "source_text": source}
        if rate is not None:
            item["rate_ul_min"] = rate
        return [item]
    if any(word in lower for word in ("dispense", "add", "dose", "deliver", "pump out", "inject")) and pump and volume is not None:
        item = {"action": "dispense", "pump": pump, "volume_ml": volume, "source_text": source}
        if rate is not None:
            item["rate_ul_min"] = rate
        return [item]
    if "rate" in lower and pump and rate is not None:
        return [{"action": "set_rate", "pump": pump, "rate_ul_min": rate, "source_text": source}]
    if any(word in lower for word in ("start", "run", "continuous flow")) and pump:
        item = {"action": "start", "pump": pump, "source_text": source}
        if rate is not None:
            item["rate_ul_min"] = rate
        return [item]
    if "stop" in lower:
        return [{"action": "stop", "pump": pump or "ALL", "source_text": source}]
    if any(word in lower for word in ("fan", "stir", "stire", "mix")):
        percent = _find_percent(lower)
        if percent is None:
            percent = _find_fan_word_percent(lower)
        if percent is None:
            percent = _find_number_after_word(lower, ("at", "to", "="))
        if percent is not None:
            return [{"action": "fan", "percent": percent, "source_text": source}]
    if any(word in lower for word in ("heat", "temp", "temperature")):
        temp = _find_temperature(lower) or _find_pwm(lower)
        if temp is None:
            temp = _find_number_after_word(lower, ("at", "to", "="))
        if temp is not None:
            return [{"action": "heat", "temp_c": float(temp), "source_text": source}]
    if any(word in lower for word in ("wait", "hold", "pause")) and duration is not None:
        return [{"action": "wait", "seconds": duration, "source_text": source}]
    return []


def _normalize_syringe_steps(payload: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise RuntimeError("AI Assist returned an invalid syringe step.")
        action = str(raw.get("action", "")).strip().lower()
        source = str(raw.get("source_text") or _line_at(lines, index))
        if action in {"set_flow_rate", "setrate"}:
            action = "set_rate"
        if action in {"start_pump", "run"}:
            action = "start"
        if action == "stop_pump":
            action = "stop"
        if action in {"stir", "set_fan"}:
            action = "fan"
        if action in {"set_temperature", "heat_chill_to_temp", "start_heat_chill"}:
            action = "heat"
        item = {"action": action, "source_text": source}
        if action in {"set_rate", "start", "stop", "aspirate", "dispense"}:
            item["pump"] = str(raw.get("pump") or raw.get("pump_id") or "ALL").strip().upper()
        if action in {"set_rate", "start"} and raw.get("rate_ul_min") is not None:
            item["rate_ul_min"] = _to_float(raw.get("rate_ul_min"))
        if action in {"aspirate", "dispense"}:
            item["volume_ml"] = _to_float(raw.get("volume_ml") or raw.get("volume"))
            if raw.get("rate_ul_min") is not None:
                item["rate_ul_min"] = _to_float(raw.get("rate_ul_min"))
        elif action == "heat":
            item["temp_c"] = _to_float(raw.get("temp_c") or raw.get("temperature") or raw.get("temperature_c"))
        elif action == "fan":
            item["percent"] = _to_fan_percent(raw.get("percent") or raw.get("speed"))
        elif action == "wait":
            item["seconds"] = _to_int(raw.get("seconds") or raw.get("duration_sec"))
        if action not in {"set_rate", "start", "stop", "aspirate", "dispense", "heat", "fan", "wait"}:
            raise RuntimeError(f"Unsupported AI syringe action: {action}")
        out.append(item)
    return out


def _normalize_robot_steps(payload: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise RuntimeError("AI Assist returned an invalid robot-arm step.")
        action = str(raw.get("action", "")).strip().lower()
        source = str(raw.get("source_text") or _line_at(lines, index))
        if action in {"home", "open_gripper", "close_gripper"}:
            out.append({"action": action, "source_text": source})
            continue
        if action == "move_axis":
            out.append(
                {
                    "action": "move_axis",
                    "axis": str(raw.get("axis", "")).strip(),
                    "position": _to_int(raw.get("position")),
                    "source_text": source,
                }
            )
            continue
        if action in {"go_to", "move_to", "move_pose"}:
            out.append({"action": "go_to", "pose": str(raw.get("pose", "")).strip(), "source_text": source})
            continue
        if action == "pick":
            out.append(
                {
                    "action": "pick",
                    "pickup_pose": str(raw.get("pickup_pose", "")).strip(),
                    "carry_pose": str(raw.get("carry_pose", "carry")).strip(),
                    "source_text": source,
                }
            )
            continue
        if action == "place":
            out.append(
                {
                    "action": "place",
                    "drop_pose": str(raw.get("drop_pose", "")).strip(),
                    "retreat_pose": str(raw.get("retreat_pose", "carry")).strip(),
                    "source_text": source,
                }
            )
            continue
        if action == "wait":
            out.append({"action": "wait", "duration_sec": _to_int(raw.get("duration_sec")), "source_text": source})
            continue
        raise RuntimeError(f"Unsupported AI robot-arm action: {action}")
    return out


def _normalize_pump_steps(payload: list[dict[str, Any]], lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise RuntimeError("AI Assist returned an invalid 8-pumps step.")
        action = str(raw.get("action", "")).strip().lower()
        source = str(raw.get("source_text") or _line_at(lines, index))
        item = dict(raw)
        item["action"] = action
        item["source_text"] = source
        if action == "dispense":
            action = "add"
            item["action"] = "add"
        elif action == "remove_waste":
            action = "transfer"
            item["action"] = "transfer"
            item["from_vessel"] = item.get("from_vessel", "reactor")
            item["to_vessel"] = item.get("to_vessel", "waste")
        elif action == "stir":
            speed = str(item.get("speed", "medium")).strip() or "medium"
            duration = _to_int(item.get("duration_sec"))
            out.extend(
                [
                    {"action": "start_stir", "vessel": str(item.get("vessel", "reactor")), "speed": speed, "source_text": source},
                    {"action": "wait", "duration_sec": duration, "source_text": source},
                    {"action": "stop_stir", "vessel": str(item.get("vessel", "reactor")), "source_text": source},
                ]
            )
            continue
        elif action == "set_heat":
            action = "heat_chill"
            item["action"] = "heat_chill"
            item["vessel"] = item.get("vessel", "reactor")
            item["temp_c"] = item.get("temp_c", item.get("pwm"))
            item["heat_mode"] = "pwm"
        elif action == "hold_temperature":
            action = "heat_chill"
            item["action"] = "heat_chill"
            item["vessel"] = item.get("vessel", "reactor")
            item["temp_c"] = item.get("temp_c", item.get("temperature", item.get("temperature_c")))
            item["heat_mode"] = "temperature"
        if "reagent" in item:
            item["reagent"] = _title(item.get("reagent"))
        if "volume_ml" in item:
            item["volume_ml"] = _to_float(item.get("volume_ml"))
        if "duration_sec" in item:
            item["duration_sec"] = _to_int(item.get("duration_sec"))
        if "temperature_c" in item:
            item["temperature_c"] = _to_float(item.get("temperature_c"))
        if "temp_c" in item:
            item["temp_c"] = _to_float(item.get("temp_c"))
        if "pwm" in item:
            item["pwm"] = _to_int(item.get("pwm"))
        if action == "add":
            item["vessel"] = str(item.get("vessel", "reactor") or "reactor")
        if action == "transfer":
            item["from_vessel"] = str(item.get("from_vessel", "reactor") or "reactor")
            item["to_vessel"] = str(item.get("to_vessel", "waste") or "waste")
        if action in {"start_stir", "stop_stir", "heat_chill", "heat_chill_to_temp", "start_heat_chill", "stop_heat_chill"}:
            item["vessel"] = str(item.get("vessel", "reactor") or "reactor")
        out.append(item)
    return out


def _line_at(lines: list[str], index: int) -> str:
    return lines[index] if 0 <= index < len(lines) else ""


def _title(value: Any) -> str:
    return str(value or "").strip().replace("_", " ").title()


def _normalize_vial(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise RuntimeError(f"Invalid vial value: {value}")
    return f"vial_{int(digits)}"


def _normalize_vials(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    items = [_normalize_vial(item) for item in raw_values if str(item or "").strip()]
    return list(dict.fromkeys(items))


def _to_float(value: Any) -> float:
    text = str(value or "").strip().lower().replace("ml", "").strip()
    if not text:
        raise RuntimeError(f"Invalid numeric value: {value}")
    return float(text)


def _to_int(value: Any) -> int:
    text = str(value or "").strip().lower()
    digits = re.findall(r"[0-9]*\.?[0-9]+", text)
    if not digits:
        raise RuntimeError(f"Invalid numeric value: {value}")
    amount = float(digits[0])
    unit_match = re.search(r"\b(h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds)\b", text)
    unit = unit_match.group(1) if unit_match else ""
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return int(amount * 3600)
    if unit in {"min", "mins", "minute", "minutes"}:
        return int(amount * 60)
    return int(amount)



def _parse_vials_any(value: Any) -> list[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = [value]
    items = []
    for raw in raw_values:
        vial = _normalize_vial(str(raw))
        if vial:
            items.append(vial)
    return list(dict.fromkeys(items))



def _infer_reagents(protocol: list[dict[str, Any]]) -> list[str]:
    seen = []
    for step in protocol:
        reagent = str(step.get("reagent", "")).strip()
        if reagent and reagent not in seen:
            seen.append(reagent)
    return seen


def validate_xdl(xdl: str) -> None:
    """Validate the local XDL structure before it is handed to a runner."""
    if not isinstance(xdl, str) or not xdl.strip():
        raise ValueError("XDL must be a non-empty string.")

    try:
        root = ElementTree.fromstring(xdl)
    except ElementTree.ParseError as error:
        raise ValueError(f"XDL is not well-formed XML: {error}") from error

    if root.tag != "XDL":
        raise ValueError("XDL document must have an XDL root element.")

    synthesis = root.find("Synthesis")
    if synthesis is None:
        raise ValueError("XDL document requires a Synthesis element.")
    if synthesis.find("Metadata") is None:
        raise ValueError("XDL document requires Metadata.")

    hardware = synthesis.find("Hardware")
    if hardware is None:
        raise ValueError("XDL document requires Hardware.")
    component_ids = {
        component.get("id")
        for component in hardware.findall("Component")
    }
    required_components = {f"vial_{number}" for number in range(1, 25)} | {"waste"}
    missing_components = required_components - component_ids
    if missing_components:
        missing = ", ".join(sorted(missing_components))
        raise ValueError(f"XDL hardware is missing component(s): {missing}")

    reagents = synthesis.find("Reagents")
    if reagents is None:
        raise ValueError("XDL document requires Reagents.")
    for reagent in reagents.findall("Reagent"):
        if not str(reagent.get("name", "")).strip():
            raise ValueError("Every XDL reagent requires a name.")

    procedure = synthesis.find("Procedure")
    if procedure is None:
        raise ValueError("XDL document requires Procedure.")
    required_attributes = {
        "Add": ("vessel", "reagent", "volume"),
        "Transfer": ("from_vessel", "to_vessel", "volume"),
        "CleanVessel": ("vessel", "solvent", "volume", "repeats"),
        "StartStir": ("vessel", "stir_speed"),
        "StopStir": ("vessel",),
        "Wait": ("time",),
        "MeasureSpectrum": ("vial",),
        "MonitorReactionSpectrum": ("vial", "time", "interval"),
        "CaptureImage": ("filename",),
        "StartVideo": ("filename",),
    }
    allowed_tags = set(required_attributes) | {
        "HeatChill",
        "HeatChillToTemp",
        "StartHeatChill",
        "StopHeatChill",
        "CaptureReferenceSpectrum",
        "CaptureDarkSpectrum",
        "StopVideo",
    }
    for step in procedure:
        if step.tag not in allowed_tags:
            raise ValueError(f"Unsupported XDL procedure element: {step.tag}")
        for attribute in required_attributes.get(step.tag, ()):
            if not str(step.get(attribute, "")).strip():
                raise ValueError(
                    f"XDL element {step.tag} requires attribute {attribute!r}."
                )



def protocol_to_xdl(
    protocol: list[dict[str, Any]],
    run_name: str = "from_gui",
    include_source_comments: bool = True,
) -> str:
    protocol = _strip_protocol_atom_maps(protocol)
    validate_protocol(protocol)
    reagents = _infer_reagents(protocol)
    lines: list[str] = ["<XDL>", "  <Synthesis>", f'    <Metadata description="{run_name}" />', "    <Hardware>"]
    for vial in [f"vial_{i}" for i in range(1, 25)]:
            lines.append(f'      <Component id="{vial}" type="reactor" />')
    lines.append('      <Component id="waste" type="waste" />')
    if any(str(step.get("action", "")).lower() in {"capture_image", "start_video", "stop_video"} for step in protocol):
        lines.append('      <Component id="camera" type="camera" />')
    if any(str(step.get("action", "")).lower() in {"measure_spectrum", "monitor_reaction_spectrum", "capture_reference_spectrum", "capture_dark_spectrum"} for step in protocol):
        lines.append('      <Component id="uvvis" type="uvvis" />')
    lines.extend(["    </Hardware>", "    <Reagents>"])
    for reagent in reagents:
        lines.append(f'      <Reagent name="{reagent}" role="reagent" />')
    lines.extend(["    </Reagents>", "    <Procedure>"])
    last_source = ""
    for step in protocol:
        source_text = str(step.get("source_text", "")).strip()
        if include_source_comments and source_text and source_text != last_source:
            lines.append(f"      <!-- {source_text} -->")
            last_source = source_text
        action = str(step.get("action", "")).lower()
        if action == "transfer":
            reagent = str(step.get("reagent", "")).strip()
            destination = _normalize_vial(step.get("destination", ""))
            volume = step.get("volume_ml", 0)
            if destination and reagent:
                lines.append(f'      <Add vessel="{destination}" reagent="{reagent}" volume="{volume} mL" />')
        elif action == "transfer_waste":
            source = _normalize_vial(step.get("source_vial", ""))
            volume = step.get("volume_ml", 0)
            if source:
                lines.append(f'      <Transfer from_vessel="{source}" to_vessel="waste" volume="{volume} mL" />')
        elif action == "wash":
            destination = _normalize_vial(step.get("destination", ""))
            reagent = str(step.get("reagent", "Water")).strip() or "Water"
            volume = step.get("volume_ml", 0)
            cycles = int(step.get("cycles", 1) or 1)
            if destination:
                lines.append(
                    f'      <CleanVessel vessel="{destination}" solvent="{reagent}" volume="{volume} mL" repeats="{max(1, cycles)}" />'
                )
        elif action == "timed_stir":
            stir_vials = _parse_vials_any(step.get("vials", []))
            speed = step.get("speed", "60")
            duration = int(step.get("duration_sec", 0) or 0)
            for vessel in stir_vials:
                lines.append(f'      <StartStir vessel="{vessel}" stir_speed="{speed}" />')
            if stir_vials:
                lines.append(f'      <Wait time="{duration} s" />')
            for vessel in stir_vials:
                lines.append(f'      <StopStir vessel="{vessel}" />')
        elif action == "start_stir":
            vessel = _normalize_vial(step.get("vial", ""))
            if vessel:
                lines.append(f'      <StartStir vessel="{vessel}" stir_speed="{step.get("speed", "60")}" />')
        elif action == "stop_stir":
            vessel = _normalize_vial(step.get("vial", ""))
            if vessel:
                lines.append(f'      <StopStir vessel="{vessel}" />')
        elif action == "wait":
            lines.append(f'      <Wait time="{int(step.get("duration_sec", 0) or 0)} s" />')
        elif action == "heat_chill":
            lines.append(
                f'      <HeatChill vessel="{_normalize_vial(step.get("vial", ""))}" temp="{step.get("temp_c", 25)} C" time="{int(step.get("duration_sec", 0) or 0)} s" />'
            )
        elif action == "heat_chill_to_temp":
            lines.append(
                f'      <HeatChillToTemp vessel="{_normalize_vial(step.get("vial", ""))}" temp="{step.get("temp_c", 25)} C" />'
            )
        elif action == "start_heat_chill":
            lines.append(
                f'      <StartHeatChill vessel="{_normalize_vial(step.get("vial", ""))}" temp="{step.get("temp_c", 25)} C" />'
            )
        elif action == "stop_heat_chill":
            lines.append(f'      <StopHeatChill vessel="{_normalize_vial(step.get("vial", ""))}" />')
        elif action == "measure_spectrum":
            attrs = [f'vial="{_normalize_vial(step.get("vial", "")) or step.get("vial", "")}"']
            if step.get("integration_time_ms"):
                attrs.append(f'integration_time_ms="{step.get("integration_time_ms")}"')
            if step.get("led_brightness"):
                attrs.append(f'led_brightness="{step.get("led_brightness")}"')
            lines.append(f'      <MeasureSpectrum {" ".join(attrs)} />')
        elif action == "monitor_reaction_spectrum":
            lines.append(
                f'      <MonitorReactionSpectrum vial="{_normalize_vial(step.get("vial", ""))}" time="{int(step.get("duration_sec", 0) or 0)} s" interval="{int(step.get("interval_sec", 60) or 60)} s" />'
            )
        elif action == "capture_reference_spectrum":
            vial = _normalize_vial(step.get("vial", ""))
            attr = f' vial="{vial}"' if vial else ""
            lines.append(f"      <CaptureReferenceSpectrum{attr} />")
        elif action == "capture_dark_spectrum":
            lines.append("      <CaptureDarkSpectrum />")
        elif action == "capture_image":
            lines.append(f'      <CaptureImage filename="{step.get("filename", "")}" />')
        elif action == "start_video":
            lines.append(f'      <StartVideo filename="{step.get("filename", "")}" />')
        elif action == "stop_video":
            lines.append("      <StopVideo />")
    lines.extend(["    </Procedure>", "  </Synthesis>", "</XDL>"])
    xdl = "\n".join(lines) + "\n"
    validate_xdl(xdl)
    return xdl


if __name__ == "__main__":
    sample_recipe = """\
add aniline 1.0ml to vial 1
wash vial 1 with water 2ml x2
stir vial 1, vial 2 for 10s at medium
capture dark spectrum
capture reference spectrum vial 24
measure spectrum vial 3
"""
    sample_protocol = recipe_to_protocol(sample_recipe)
    print(json.dumps(sample_protocol, indent=2))
    print(protocol_to_xdl(sample_protocol, run_name="console_test"))

