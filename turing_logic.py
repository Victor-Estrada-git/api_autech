# -*- coding: utf-8 -*-
"""
Turing Machine logic — no Flet, no matplotlib.
Includes: simulation, graph JSON, FSAM export/import, JFF (JFLAP) export/import.
"""
import json
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Transition parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_transitions(text: str) -> dict:
    """
    Format: state,read -> newState,write,direction (L/R/S)
    Returns: {state: {read: [newState, write, direction]}}
    """
    transitions = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '->' not in line:
            continue
        left, right = line.split('->', 1)
        left_parts  = [p.strip() for p in left.split(',')]
        right_parts = [p.strip() for p in right.split(',')]
        if len(left_parts) < 2 or len(right_parts) < 3:
            continue
        state, read            = left_parts[0],  left_parts[1]
        new_state, write, direction = right_parts[0], right_parts[1], right_parts[2].upper()
        transitions.setdefault(state, {})[read] = [new_state, write, direction]
    return transitions


# ══════════════════════════════════════════════════════════════════════════════
#  Simulator
# ══════════════════════════════════════════════════════════════════════════════

def simulate_turing(
    states: List[str],
    transitions: dict,
    initial_state: str,
    accept_states: List[str],
    tape_input: str,
    head_pos: int = 0,
    max_steps: int = 1000,
) -> dict:
    """
    Simulates a Turing Machine step-by-step.

    Returns:
        {steps: [...], result: 'ACCEPTED' | 'REJECTED' | 'TIMEOUT'}
    """
    tape = list(tape_input) if tape_input else ['_']
    while len(tape) <= head_pos:
        tape.append('_')

    current_state = initial_state
    accept_set    = set(accept_states)

    def _step(n, state, tape, head, msg, accepted=False, rejected=False,
              prev=None, sym=None, trans=None):
        return {
            "step": n, "state": state, "tape": list(tape), "headPos": head,
            "message": msg, "isAccepted": accepted, "isRejected": rejected,
            "prevState": prev, "symbolRead": sym, "transitionTaken": trans,
        }

    steps = [_step(0, current_state, tape, head_pos,
                   f"Inicio: estado={current_state}")]

    for i in range(1, max_steps + 1):
        if current_state in accept_set:
            steps.append(_step(i, current_state, tape, head_pos,
                               f"✅ Cadena ACEPTADA en estado {current_state}",
                               accepted=True, prev=current_state))
            return {"steps": steps, "result": "ACCEPTED"}

        # Extend tape
        while head_pos < 0:
            tape.insert(0, '_'); head_pos = 0
        while head_pos >= len(tape):
            tape.append('_')

        symbol_read = tape[head_pos]
        prev_state  = current_state
        trans       = transitions.get(current_state, {}).get(symbol_read)

        if trans is None:
            if current_state in accept_set:
                steps.append(_step(i, current_state, tape, head_pos,
                                   f"✅ Cadena ACEPTADA en estado {current_state}",
                                   accepted=True, prev=prev_state, sym=symbol_read))
                return {"steps": steps, "result": "ACCEPTED"}
            steps.append(_step(i, current_state, tape, head_pos,
                               f"❌ Sin transición para ({current_state}, {symbol_read}) — RECHAZADA",
                               rejected=True, prev=prev_state, sym=symbol_read))
            return {"steps": steps, "result": "REJECTED"}

        new_state, write_sym, direction = trans
        tape[head_pos] = write_sym
        current_state  = new_state
        prev_head      = head_pos

        if direction == 'R':
            head_pos += 1
        elif direction == 'L':
            head_pos -= 1

        if head_pos < 0:
            tape.insert(0, '_'); head_pos = 0
        while head_pos >= len(tape):
            tape.append('_')

        label = f"δ({prev_state},{symbol_read})=({new_state},{write_sym},{direction})"
        steps.append(_step(i, current_state, tape, head_pos,
                           f"{label}  cabeza: {prev_head}→{head_pos}",
                           prev=prev_state, sym=symbol_read,
                           trans=[new_state, write_sym, direction]))

        if current_state in accept_set:
            steps.append(_step(i + 1, current_state, tape, head_pos,
                               f"✅ Cadena ACEPTADA en estado {current_state}",
                               accepted=True, prev=prev_state))
            return {"steps": steps, "result": "ACCEPTED"}

    steps.append(_step(max_steps + 1, current_state, tape, head_pos,
                       f"⚠️ Límite de {max_steps} pasos alcanzado — posible bucle infinito",
                       rejected=True, prev=current_state))
    return {"steps": steps, "result": "TIMEOUT"}


# ══════════════════════════════════════════════════════════════════════════════
#  Graph JSON (for Flutter AutomatonCanvas)
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_json(states: List[str], transitions: dict, initial: str,
                     accept_states: List[str]) -> dict:
    """Convert TM definition to Flutter AutomatonCanvas JSON."""
    states_json = [
        {"id": s, "isInitial": s == initial, "isAccepting": s in accept_states}
        for s in states
    ]
    grouped: Dict[tuple, List[str]] = {}
    for state, trans in transitions.items():
        for read, (new_state, write, direction) in trans.items():
            key   = (state, new_state)
            label = f"{read}→{write},{direction}"
            grouped.setdefault(key, []).append(label)

    edges_json = [
        {"from": frm, "to": to, "label": "\n".join(labels)}
        for (frm, to), labels in grouped.items()
    ]
    return {"states": states_json, "edges": edges_json}


# ══════════════════════════════════════════════════════════════════════════════
#  FSAM  (JSON format — own application format)
# ══════════════════════════════════════════════════════════════════════════════

def export_tm_fsam(
    states: List[str],
    transitions: dict,
    initial: str,
    accept_states: List[str],
) -> str:
    """
    Serialises a Turing Machine to the FSAM (JSON) format.

    The transitions are re-encoded as multi-line text
    (``state,read -> newState,write,direction``) so the resulting file can be
    loaded back by ``import_tm_fsam`` *and* pasted directly into the UI form.
    """
    lines = []
    for state in sorted(transitions):
        for read in sorted(transitions[state]):
            ns, wr, dr = transitions[state][read]
            lines.append(f"{state},{read} -> {ns},{wr},{dr}")

    payload = {
        "fsam_version": "1.0",
        "type": "TM",
        "states":       ",".join(states),
        "initial":      initial,
        "acceptStates": ",".join(accept_states),
        "transitions":  "\n".join(lines),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_tm_fsam(json_str: str) -> dict:
    """
    Parse an FSAM (JSON) string for a Turing Machine.

    Returns a dict with keys ``states``, ``initial``, ``acceptStates``,
    ``transitions`` (text) — ready to pass to the Flutter API definition map
    or to ``parse_transitions``.

    Raises ``ValueError`` on malformed input.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido: {e}") from e

    if data.get("fsam_version") not in ("1.0",):
        raise ValueError("Versión FSAM no reconocida.")
    if data.get("type") != "TM":
        raise ValueError(f"Tipo esperado 'TM', se encontró '{data.get('type')}'.")

    required = ("states", "initial", "acceptStates", "transitions")
    missing  = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Campos faltantes en FSAM: {missing}")

    return {
        "states":       data["states"],
        "initial":      data["initial"],
        "acceptStates": data["acceptStates"],
        "transitions":  data["transitions"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JFF  (JFLAP XML format)
# ══════════════════════════════════════════════════════════════════════════════

def export_tm_jff(
    states: List[str],
    transitions: dict,
    initial: str,
    accept_states: List[str],
) -> str:
    """
    Serialises a Turing Machine to the JFF (JFLAP XML) format.

    The resulting file can be imported into JFLAP 7 (turing machine editor)
    and loaded back by ``import_tm_jff``.
    """
    accept_set = set(accept_states)
    state_ids  = {s: str(i) for i, s in enumerate(states)}

    # Lay states out in a row
    spacing = 120.0
    def _pos(i):
        cols = max(1, math.ceil(math.sqrt(len(states))))
        return (spacing + (i % cols) * spacing * 2,
                spacing + (i // cols) * spacing * 2)

    root      = ET.Element("structure")
    ET.SubElement(root, "type").text = "turing"
    automaton = ET.SubElement(root, "automaton")

    for idx, s in enumerate(states):
        x, y  = _pos(idx)
        node  = ET.SubElement(automaton, "state",
                               id=state_ids[s], name=s)
        ET.SubElement(node, "x").text = f"{x:.1f}"
        ET.SubElement(node, "y").text = f"{y:.1f}"
        if s == initial:
            ET.SubElement(node, "initial")
        if s in accept_set:
            ET.SubElement(node, "final")

    for state in sorted(transitions):
        for read_sym in sorted(transitions[state]):
            ns, wr, dr = transitions[state][read_sym]
            if ns not in state_ids:
                continue
            t = ET.SubElement(automaton, "transition")
            ET.SubElement(t, "from").text  = state_ids[state]
            ET.SubElement(t, "to").text    = state_ids[ns]
            ET.SubElement(t, "read").text  = read_sym if read_sym != '_' else ''
            ET.SubElement(t, "write").text = wr       if wr       != '_' else ''
            ET.SubElement(t, "move").text  = dr

    _indent_xml(root)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(root, encoding="unicode"))


def import_tm_jff(xml_str: str) -> dict:
    """
    Parse a JFF (JFLAP XML) string for a Turing Machine.

    Returns a dict with keys ``states``, ``initial``, ``acceptStates``,
    ``transitions`` (text).  Raises ``ValueError`` on errors.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        raise ValueError(f"XML inválido: {e}") from e

    t = root.findtext("type", "").strip().lower()
    if t not in ("turing", "tm"):
        raise ValueError(f"Tipo JFF no reconocido: '{t}'. Se esperaba 'turing'.")

    automaton  = root.find("automaton")
    if automaton is None:
        raise ValueError("Elemento <automaton> no encontrado.")

    id_to_name = {}
    states     = []
    initial    = None
    accept_set = []

    for node in automaton.findall("state"):
        sid  = node.get("id", "")
        name = node.get("name") or f"q{sid}"
        id_to_name[sid] = name
        states.append(name)
        if node.find("initial") is not None:
            initial = name
        if node.find("final") is not None:
            accept_set.append(name)

    if not initial and states:
        initial = states[0]

    lines = []
    for tr in automaton.findall("transition"):
        frm   = id_to_name.get(tr.findtext("from", ""), "")
        to    = id_to_name.get(tr.findtext("to",   ""), "")
        read  = tr.findtext("read",  "") or "_"
        write = tr.findtext("write", "") or "_"
        move  = (tr.findtext("move",  "R") or "R").upper()
        if frm and to:
            lines.append(f"{frm},{read} -> {to},{write},{move}")

    return {
        "states":       ",".join(states),
        "initial":      initial or "",
        "acceptStates": ",".join(accept_set),
        "transitions":  "\n".join(lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print indentation to an ElementTree in-place."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():   # noqa: F821 (last child)
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad