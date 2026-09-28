# -*- coding: utf-8 -*-
"""
Turing Machine logic — no Flet, no matplotlib.
Includes: simulation and graph JSON.
"""
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Transition parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_transitions(text: str, errors: Optional[List[str]] = None) -> dict:
    """
    Format: state,read -> newState,write,direction (L/R/S)
    Returns: {state: {read: [newState, write, direction]}}

    Si se pasa `errors`, se agrega ahí un aviso por cada línea mal escrita
    (antes se ignoraban sin avisar). Las líneas inválidas se siguen omitiendo
    para no romper la simulación del resto de la máquina.
    """
    transitions = {}
    for n, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '->' not in line:
            if errors is not None:
                errors.append(f"Línea {n}: falta '->' en «{line}»; se ignoró.")
            continue
        left, right = line.split('->', 1)
        left_parts  = [p.strip() for p in left.split(',')]
        right_parts = [p.strip() for p in right.split(',')]
        if len(left_parts) < 2 or len(right_parts) < 3:
            if errors is not None:
                errors.append(
                    f"Línea {n}: formato inválido en «{line}» "
                    f"(use estado,lee -> nuevoEstado,escribe,L/R/S); se ignoró.")
            continue
        state, read            = left_parts[0],  left_parts[1]
        new_state, write, direction = right_parts[0], right_parts[1], right_parts[2].upper()
        if errors is not None:
            if direction not in ('L', 'R', 'S'):
                errors.append(
                    f"Línea {n}: dirección '{right_parts[2]}' inválida "
                    f"(use L, R o S); la cabeza no se moverá.")
            if read in transitions.get(state, {}):
                errors.append(
                    f"Línea {n}: ya había una transición para ({state}, {read}); "
                    f"se usa la última.")
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
    # Cabeza inicial a la izquierda de la entrada: se rellena con blancos a
    # la izquierda para que el paso 0 ya registre un índice válido (antes
    # quedaba negativo y sólo se corregía a partir del paso 1).
    if head_pos < 0:
        tape = ['_'] * (-head_pos) + tape
        head_pos = 0
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
