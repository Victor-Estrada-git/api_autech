# -*- coding: utf-8 -*-
"""
PDA logic — no Flet.
Includes: PDA class, PDASimulator, factored CFG conversion,
          FSAM export/import, JFF (JFLAP) export/import.
"""
import json
import math
import re
import itertools
import xml.etree.ElementTree as ET
from collections import deque
from typing import Dict, Set, Tuple, List, Optional, Any

EPSILON = 'ε'


# ══════════════════════════════════════════════════════════════════════════════
#  PDA class
# ══════════════════════════════════════════════════════════════════════════════

class PDA:
    def __init__(self):
        self.states:         Set[str]  = set()
        self.input_alphabet: Set[str]  = set()
        self.stack_alphabet: Set[str]  = set()
        self.transitions:    Dict[Tuple[str, str, str],
                                   Set[Tuple[str, Tuple[str, ...]]]] = {}
        self.start_state:    Optional[str] = None
        self.start_symbol:   Optional[str] = None
        self.accept_states:  Set[str]  = set()

    def clear(self):
        self.__init__()

    # ── parsing ────────────────────────────────────────────────────────────────

    def parse(self, states_str, input_alpha_str, stack_alpha_str,
              start_state_str, start_symbol_str, accept_states_str,
              transitions_str):
        """Parse PDA definition from string fields."""
        self.clear()
        errors = []

        def clean_split(text: str) -> Set[str]:
            return set(s.strip() for s in text.split(',') if s.strip())

        self.states          = clean_split(states_str)
        self.input_alphabet  = clean_split(input_alpha_str)
        self.input_alphabet.discard(EPSILON)
        self.stack_alphabet  = clean_split(stack_alpha_str)
        self.start_state     = start_state_str.strip()  if start_state_str  else None
        self.start_symbol    = start_symbol_str.strip() if start_symbol_str else None
        self.accept_states   = clean_split(accept_states_str)

        if not self.states:
            errors.append("El conjunto de estados (Q) no puede estar vacío.")
        if not self.stack_alphabet:
            errors.append("El alfabeto de pila (Γ) no puede estar vacío.")
        if not self.start_state:
            errors.append("Debe definir un estado inicial (q₀).")
        if not self.start_symbol:
            errors.append("Debe definir un símbolo inicial de pila (Z₀).")
        if self.start_state and self.start_state not in self.states:
            errors.append(f"El estado inicial '{self.start_state}' no está en Q.")
        if self.start_symbol and self.start_symbol not in self.stack_alphabet:
            self.stack_alphabet.add(self.start_symbol)

        invalid_accept = self.accept_states - self.states
        if invalid_accept:
            errors.append(f"Estados de aceptación inválidos: {invalid_accept}")

        pattern = re.compile(
            r"^\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*->\s*([^,]+)\s*,\s*(.*)\s*$"
        )
        processed = {}
        for i, line in enumerate(transitions_str.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = pattern.match(line)
            if not m:
                errors.append(f"Línea {i}: Formato inválido. Use: 'origen,entrada,pop -> destino,push'")
                continue
            q_r, inp, pop, q_w, push_str = [x.strip() for x in m.groups()]
            if q_r not in self.states:
                errors.append(f"Línea {i}: Estado '{q_r}' no está en Q.")
            if q_w not in self.states:
                errors.append(f"Línea {i}: Estado '{q_w}' no está en Q.")
            if inp != EPSILON and inp not in self.input_alphabet:
                errors.append(f"Línea {i}: Símbolo '{inp}' no está en Σ.")
            if pop != EPSILON and pop not in self.stack_alphabet:
                errors.append(f"Línea {i}: Símbolo '{pop}' no está en Γ.")
            push_tup = self._parse_stack_push(push_str, i, errors)
            if push_tup is None:
                continue
            key = (q_r, inp, pop)
            processed.setdefault(key, set()).add((q_w, push_tup))

        if errors:
            raise ValueError("\n".join(errors))
        self.transitions = processed

    def parse_from_ui(self, *args, **kwargs):
        """Alias for parse() — compatibility."""
        return self.parse(*args, **kwargs)

    def parse_from_dict(self, data: dict):
        return self.parse(
            data.get('states', ''), data.get('inputAlphabet', ''),
            data.get('stackAlphabet', ''), data.get('startState', ''),
            data.get('startSymbol', ''), data.get('acceptStates', ''),
            data.get('transitions', ''),
        )

    def _parse_stack_push(self, push_str, line_num, errors):
        if push_str == EPSILON:
            return tuple()
        syms   = sorted(self.stack_alphabet, key=len, reverse=True)
        result = []
        tmp    = push_str
        while tmp:
            found = False
            for sym in syms:
                if tmp.startswith(sym):
                    result.append(sym)
                    tmp  = tmp[len(sym):]
                    found = True
                    break
            if not found:
                errors.append(f"Línea {line_num}: Símbolo inválido en '{push_str}'")
                return None
        return tuple(result)

    # ── graph JSON ─────────────────────────────────────────────────────────────

    def to_graph_json(self) -> dict:
        states_list = [
            {"id": s, "isInitial": s == self.start_state, "isAccepting": s in self.accept_states}
            for s in self.states
        ]
        grouped: Dict[Tuple[str, str], List[str]] = {}
        for (q, a, z), results in self.transitions.items():
            for p, gamma in results:
                push_str = "".join(gamma) if gamma else EPSILON
                label = f"{a},{z}/{push_str}"
                grouped.setdefault((q, p), []).append(label)
        edges_list = [
            {"from": frm, "to": to, "label": "\n".join(labels)}
            for (frm, to), labels in grouped.items()
        ]
        return {"states": states_list, "edges": edges_list}

    # ── text serialisation ─────────────────────────────────────────────────────

    def to_text_fields(self) -> dict:
        """Return all fields as strings (suitable for FSAM payload)."""
        lines = []
        for (q, a, z), results in sorted(self.transitions.items()):
            for (p, gamma) in sorted(results):
                push_str = "".join(gamma) if gamma else EPSILON
                lines.append(f"{q},{a},{z} -> {p},{push_str}")
        return {
            "states":        ",".join(sorted(self.states)),
            "inputAlphabet": ",".join(sorted(self.input_alphabet)),
            "stackAlphabet": ",".join(sorted(self.stack_alphabet)),
            "startState":    self.start_state  or "",
            "startSymbol":   self.start_symbol or "",
            "acceptStates":  ",".join(sorted(self.accept_states)),
            "transitions":   "\n".join(lines),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  PDA Simulator
# ══════════════════════════════════════════════════════════════════════════════

class PDASimulator:
    def __init__(self, pda: PDA):
        self.pda        = pda
        self.max_steps  = 2000
        self.max_stack_size = 200

    def simulate(self, input_string: str) -> Dict[str, Any]:
        if not self.pda.states or not self.pda.start_state or not self.pda.start_symbol:
            return {'accepted': False, 'error': 'PDA incompleto', 'trace': [], 'steps': 0}

        initial = (self.pda.start_state, input_string, [self.pda.start_symbol])
        queue   = deque([initial])
        visited = {(self.pda.start_state, len(input_string), (self.pda.start_symbol,))}
        trace   = [self._fmt(initial, 0, "Configuración inicial")]
        step    = 0

        while queue and step < self.max_steps:
            step += 1
            state, inp, stack = queue.popleft()
            if not inp and state in self.pda.accept_states:
                trace.append(f"✅ Aceptada en el paso {step}. Estado: {state}")
                return {'accepted': True, 'acceptance_type': 'final_state',
                        'trace': trace, 'steps': step}
            self._explore(state, EPSILON, inp,       stack, queue, visited, trace, step)
            if inp:
                self._explore(state, inp[0], inp[1:], stack, queue, visited, trace, step)

        if step >= self.max_steps:
            trace.append(f"⚠️ Límite de {self.max_steps} pasos alcanzado.")
        trace.append("❌ Cadena RECHAZADA")
        return {'accepted': False, 'trace': trace, 'steps': step}

    def _explore(self, state, char, inp_after, stack, queue, visited, trace, step):
        top = stack[-1] if stack else None
        if top is not None:
            self._apply(state, char, top, inp_after, stack[:-1], queue, visited, trace, step)
        self._apply(state, char, EPSILON, inp_after, stack, queue, visited, trace, step)

    def _apply(self, state, char, sym_pop, inp_after, stack_after_pop,
               queue, visited, trace, step):
        key = (state, char, sym_pop)
        if key in self.pda.transitions:
            for next_state, push_syms in self.pda.transitions[key]:
                new_stack = list(stack_after_pop)
                new_stack.extend(reversed(push_syms))
                if len(new_stack) <= self.max_stack_size:
                    cfg_key = (next_state, len(inp_after), tuple(new_stack))
                    if cfg_key not in visited:
                        visited.add(cfg_key)
                        new_cfg = (next_state, inp_after, new_stack)
                        queue.append(new_cfg)
                        push_s = "".join(push_syms) or EPSILON
                        move   = f"Leer '{char}'" if char != EPSILON else "ε-movimiento"
                        trace.append(self._fmt(new_cfg, step,
                                               f"{move}, Pop '{sym_pop}', Push '{push_s}'"))

    def _fmt(self, cfg, step, move=None):
        state, inp, stack = cfg
        stack_s  = "".join(reversed(stack)) if stack else EPSILON
        inp_d    = f"'{inp}'" if inp else EPSILON
        move_i   = f"  ({move})" if move else ""
        return f"Paso {step}: Estado={state}, Entrada={inp_d}, Pila={stack_s}{move_i}"


# ══════════════════════════════════════════════════════════════════════════════
#  CFG helpers  (for the factored conversion)
# ══════════════════════════════════════════════════════════════════════════════

def _is_cfg_nt(sym: str) -> bool:
    """A symbol is a non-terminal if it is 'S' or matches the [q,A,p] pattern."""
    return sym == 'S' or (sym.startswith('[') and sym.endswith(']'))


def _find_generating_nts(productions: Dict[str, List[List[str]]]) -> Set[str]:
    """
    Bottom-up fixpoint: a NT is *generating* if it has at least one production
    where every symbol is a terminal or an already-generating NT.
    """
    generating: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for nt, prods in productions.items():
            if nt in generating:
                continue
            for prod in prods:
                if all(
                    (not _is_cfg_nt(s) or s in generating)
                    for s in prod
                    if s != 'ε'
                ):
                    generating.add(nt)
                    changed = True
                    break
    return generating


def _find_reachable_nts(start: str,
                         productions: Dict[str, List[List[str]]]) -> Set[str]:
    """BFS from *start* symbol to collect all reachable NTs."""
    reachable: Set[str] = {start}
    queue = deque([start])
    while queue:
        nt = queue.popleft()
        for prod in productions.get(nt, []):
            for sym in prod:
                if _is_cfg_nt(sym) and sym not in reachable:
                    reachable.add(sym)
                    queue.append(sym)
    return reachable


def _left_factor_cfg(
    productions: Dict[str, List[List[str]]],
    new_nt_counter: Optional[List[int]] = None,
) -> Dict[str, List[List[str]]]:
    """
    Apply one pass of left-factoring to every NT group.

    For a group  A → α β₁ | α β₂ | … | α βₙ | γ₁ | …
    introduce    A' → β₁ | β₂ | … | βₙ
    and rewrite  A → α A' | γ₁ | …

    Only the longest common prefix shared by ≥2 alternatives is factored.
    """
    if new_nt_counter is None:
        new_nt_counter = [0]

    result:  Dict[str, List[List[str]]] = {}
    pending: Dict[str, List[List[str]]] = dict(productions)

    while pending:
        nt, prods = next(iter(pending.items()))
        del pending[nt]

        if len(prods) <= 1:
            result[nt] = prods
            continue

        # Group by first symbol
        groups: Dict[str, Tuple[str, List[List[str]]]] = {}
        order:  List[str] = []
        singletons: List[List[str]] = []

        for prod in prods:
            first = prod[0] if prod else 'ε'
            if first == 'ε' or len(prod) == 0:
                singletons.append(prod)
            else:
                if first not in groups:
                    groups[first] = (first, [])
                    order.append(first)
                groups[first][1].append(prod)

        new_prods: List[List[str]] = list(singletons)
        for first in order:
            _, group = groups[first]
            if len(group) == 1:
                new_prods.append(group[0])
            else:
                # Find longest common prefix across this group
                prefix = _longest_common_prefix(group)
                if len(prefix) <= 1:
                    # Only one-symbol common prefix — only worthwhile if ≥2 alts
                    new_nt_counter[0] += 1
                    new_name = f"{nt}'{new_nt_counter[0]}"
                    tails = [p[len(prefix):] or ['ε'] for p in group]
                    new_prods.append(prefix + [new_name])
                    pending[new_name] = tails
                else:
                    new_nt_counter[0] += 1
                    new_name = f"{nt}'{new_nt_counter[0]}"
                    tails = [p[len(prefix):] or ['ε'] for p in group]
                    new_prods.append(prefix + [new_name])
                    pending[new_name] = tails

        result[nt] = new_prods

    return result


def _longest_common_prefix(prods: List[List[str]]) -> List[str]:
    """Return the longest common prefix of a list of symbol sequences."""
    if not prods:
        return []
    prefix = []
    for syms in zip(*prods):
        if len(set(syms)) == 1:
            prefix.append(syms[0])
        else:
            break
    return prefix


# ══════════════════════════════════════════════════════════════════════════════
#  Factored CFG conversion
# ══════════════════════════════════════════════════════════════════════════════

def convert_pda_to_cfg(pda: PDA) -> str:
    """
    Convert the PDA to an equivalent Context-Free Grammar using the
    standard [q,A,p]-construction, then:

      1. Remove non-generating non-terminals.
      2. Remove unreachable non-terminals.
      3. Deduplicate productions.
      4. Left-factor where possible.
      5. Format with readable alignment.
    """
    if not pda.states or not pda.start_state or not pda.start_symbol:
        raise ValueError("PDA incompleto para conversión.")

    START = "S"
    accept_set = pda.accept_states if pda.accept_states else pda.states
    # ── Step 1: build raw productions ─────────────────────────────────────────
    raw: Dict[str, List[List[str]]] = {}

    def add(lhs, rhs):
        raw.setdefault(lhs, []).append(rhs)

    for qf in sorted(accept_set):
        add(START, [f"[{pda.start_state},{pda.start_symbol},{qf}]"])

    # Base case for acceptance-by-final-state: when in an accept state q,
    # any remaining stack symbol A can be "erased" (ε-derivation).
    # This is equivalent to pre-converting the PDA to accept by empty stack.
    for q in sorted(accept_set):
        for A in sorted(pda.stack_alphabet):
            add(f"[{q},{A},{q}]", ['ε'])

    for (q, a, Z), results in pda.transitions.items():
        a_sym = [a] if a != EPSILON else []
        for (r, gamma) in results:
            if not gamma:
                # [q,Z,r] → a  (or ε)
                lhs = f"[{q},{Z},{r}]"
                add(lhs, a_sym if a_sym else ['ε'])
            else:
                k = len(gamma)
                for inter in itertools.product(pda.states, repeat=k - 1):
                    all_st = [r] + list(inter)
                    for p_k in pda.states:
                        lhs      = f"[{q},{Z},{p_k}]"
                        cur_st   = all_st + [p_k]
                        rhs_syms = list(a_sym) + [
                            f"[{cur_st[i]},{gamma[i]},{cur_st[i+1]}]"
                            for i in range(k)
                        ]
                        add(lhs, rhs_syms)

    # ── Step 2: remove non-generating NTs ─────────────────────────────────────
    generating = _find_generating_nts(raw)
    filtered: Dict[str, List[List[str]]] = {}
    for nt, prods in raw.items():
        if nt not in generating:
            continue
        valid = [
            prod for prod in prods
            if all(not _is_cfg_nt(s) or s in generating
                   for s in prod if s != 'ε')
        ]
        if valid:
            filtered[nt] = valid

    # ── Step 3: remove unreachable NTs ────────────────────────────────────────
    reachable = _find_reachable_nts(START, filtered)
    useful    = {nt: prods for nt, prods in filtered.items() if nt in reachable}

    # ── Step 4: deduplicate ────────────────────────────────────────────────────
    deduped: Dict[str, List[List[str]]] = {}
    for nt, prods in useful.items():
        seen: Set[tuple] = set()
        unique = []
        for p in prods:
            k = tuple(p)
            if k not in seen:
                seen.add(k)
                unique.append(p)
        deduped[nt] = unique

    # ── Step 5: left-factor ────────────────────────────────────────────────────
    counter = [0]
    factored = _left_factor_cfg(deduped, counter)

    # ── Step 6: format output ──────────────────────────────────────────────────
    return _format_cfg_output(factored, START, pda.input_alphabet,
                               pda.stack_alphabet, pda.states)


def _format_cfg_output(
    productions: Dict[str, List[List[str]]],
    start: str,
    input_alphabet: Set[str],
    stack_alphabet: Set[str],
    states: Set[str],
) -> str:
    if not productions:
        return "La GIC equivalente está vacía (no hay cadenas aceptadas)."

    # Collect all non-terminals in BFS order from S
    ordered_nts: List[str] = []
    visited_nts: Set[str]  = set()
    bfs_q = deque([start])
    visited_nts.add(start)
    while bfs_q:
        nt = bfs_q.popleft()
        if nt in productions:
            ordered_nts.append(nt)
            for prod in productions[nt]:
                for sym in prod:
                    if _is_cfg_nt(sym) and sym not in visited_nts:
                        visited_nts.add(sym)
                        bfs_q.append(sym)

    # Add any NTs not reached (shouldn't happen after cleanup, but be safe)
    for nt in productions:
        if nt not in visited_nts:
            ordered_nts.append(nt)

    # Align arrows
    max_lhs   = max((len(nt) for nt in ordered_nts), default=1)
    arrow     = " → "
    separator = " | "
    cont_pad  = " " * max_lhs + "   "   # indent for continuation lines

    lines_out = []
    for nt in ordered_nts:
        prods = productions[nt]
        if not prods:
            continue
        # Sort productions: terminals first, then by length
        def _sort_key(p):
            has_nt = any(_is_cfg_nt(s) for s in p if s != 'ε')
            return (int(has_nt), len(p), " ".join(p))
        prods_sorted = sorted(prods, key=_sort_key)

        # Group all into one line (or split if very long)
        rhs_parts = [" ".join(p) for p in prods_sorted]
        full_line = f"{nt.ljust(max_lhs)}{arrow}{separator.join(rhs_parts)}"
        if len(full_line) <= 100:
            lines_out.append(full_line)
        else:
            # Multi-line
            lines_out.append(f"{nt.ljust(max_lhs)}{arrow}{rhs_parts[0]}")
            for rp in rhs_parts[1:]:
                lines_out.append(f"{cont_pad}{separator.strip()} {rp}")

    num_prods = sum(len(v) for v in productions.values())
    num_nts   = len(productions)

    sep = "-" * 55
    header = (
        f"Gramática Libre de Contexto equivalente al APD\n{sep}\n\n"
        f"No terminales:  {num_nts}   "
        f"Producciones útiles: {num_prods}\n"
        f"Símbolo inicial: {start}\n"
        f"Terminales: {{{', '.join(sorted(input_alphabet)) or 'ε'}}}\n\n"
        f"Producciones:\n{sep}"
    )
    return header + "\n" + "\n".join(lines_out) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  FSAM  (JSON format)
# ══════════════════════════════════════════════════════════════════════════════

def export_pda_fsam(pda: PDA) -> str:
    """
    Serialise a PDA to the FSAM (JSON) format.

    The ``transitions`` field uses the same plain-text format as the UI form
    so the content can be pasted directly back.
    """
    payload = {"fsam_version": "1.0", "type": "PDA"}
    payload.update(pda.to_text_fields())
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_pda_fsam(json_str: str) -> dict:
    """
    Parse an FSAM (JSON) string for a PDA.

    Returns a dict with the same keys as ``PDA.parse_from_dict`` expects.
    Raises ``ValueError`` on malformed input.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON inválido: {e}") from e

    if data.get("fsam_version") not in ("1.0",):
        raise ValueError("Versión FSAM no reconocida.")
    if data.get("type") != "PDA":
        raise ValueError(f"Tipo esperado 'PDA', se encontró '{data.get('type')}'.")

    required = ("states", "inputAlphabet", "stackAlphabet",
                "startState", "startSymbol", "acceptStates", "transitions")
    missing  = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Campos faltantes en FSAM: {missing}")

    return {k: data[k] for k in required}


# ══════════════════════════════════════════════════════════════════════════════
#  JFF  (JFLAP XML format)
# ══════════════════════════════════════════════════════════════════════════════

def export_pda_jff(pda: PDA) -> str:
    """
    Serialise a PDA to the JFF (JFLAP XML) format (``<type>pda</type>``).
    The output can be opened in JFLAP 7.
    """
    states_list = sorted(pda.states)
    state_ids   = {s: str(i) for i, s in enumerate(states_list)}
    spacing     = 130.0

    def _pos(i):
        cols = max(1, math.ceil(math.sqrt(len(states_list))))
        return (spacing + (i % cols) * spacing * 2,
                spacing + (i // cols) * spacing * 2)

    root      = ET.Element("structure")
    ET.SubElement(root, "type").text = "pda"
    automaton = ET.SubElement(root, "automaton")

    for idx, s in enumerate(states_list):
        x, y = _pos(idx)
        node = ET.SubElement(automaton, "state", id=state_ids[s], name=s)
        ET.SubElement(node, "x").text = f"{x:.1f}"
        ET.SubElement(node, "y").text = f"{y:.1f}"
        if s == pda.start_state:
            ET.SubElement(node, "initial")
        if s in pda.accept_states:
            ET.SubElement(node, "final")

    for (q, a, z), results in sorted(pda.transitions.items()):
        for (p, gamma) in sorted(results):
            if p not in state_ids:
                continue
            push_str = "".join(gamma) if gamma else ""
            t = ET.SubElement(automaton, "transition")
            ET.SubElement(t, "from").text  = state_ids[q]
            ET.SubElement(t, "to").text    = state_ids[p]
            _jff_opt(t, "read",  a if a != EPSILON else "")
            _jff_opt(t, "pop",   z if z != EPSILON else "")
            _jff_opt(t, "push",  push_str)

    _indent_xml(root)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + ET.tostring(root, encoding="unicode"))


def import_pda_jff(xml_str: str) -> dict:
    """
    Parse a JFF (JFLAP XML) string for a PDA.

    Returns a dict suitable for ``PDA.parse_from_dict``.
    Raises ``ValueError`` on errors.
    """
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        raise ValueError(f"XML inválido: {e}") from e

    t = root.findtext("type", "").strip().lower()
    if t != "pda":
        raise ValueError(f"Tipo JFF no reconocido: '{t}'. Se esperaba 'pda'.")

    automaton = root.find("automaton")
    if automaton is None:
        raise ValueError("Elemento <automaton> no encontrado.")

    id_to_name: Dict[str, str] = {}
    states_list: List[str]     = []
    initial      = None
    accept_list: List[str]     = []

    for node in automaton.findall("state"):
        sid  = node.get("id", "")
        name = node.get("name") or f"q{sid}"
        id_to_name[sid] = name
        states_list.append(name)
        if node.find("initial") is not None:
            initial = name
        if node.find("final") is not None:
            accept_list.append(name)

    if not initial and states_list:
        initial = states_list[0]

    input_set: Set[str]  = set()
    stack_set: Set[str]  = set()
    lines: List[str]     = []

    for tr in automaton.findall("transition"):
        frm  = id_to_name.get(tr.findtext("from", ""), "")
        to   = id_to_name.get(tr.findtext("to",   ""), "")
        read = tr.findtext("read",  "") or EPSILON
        pop  = tr.findtext("pop",   "") or EPSILON
        push = tr.findtext("push",  "") or EPSILON
        if not frm or not to:
            continue
        if read != EPSILON:
            input_set.add(read)
        if pop  != EPSILON:
            stack_set.add(pop)
        for ch in (push if push != EPSILON else ""):
            stack_set.add(ch)
        lines.append(f"{frm},{read},{pop} -> {to},{push}")

    # If no stack symbols found, add a default Z
    if not stack_set:
        stack_set.add("Z")
    # Start symbol: first stack symbol popped from initial state, or 'Z'
    start_sym = next(iter(stack_set), "Z")

    return {
        "states":        ",".join(states_list),
        "inputAlphabet": ",".join(sorted(input_set)),
        "stackAlphabet": ",".join(sorted(stack_set)),
        "startState":    initial or "",
        "startSymbol":   start_sym,
        "acceptStates":  ",".join(accept_list),
        "transitions":   "\n".join(lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Internal XML helpers
# ══════════════════════════════════════════════════════════════════════════════

def _jff_opt(parent: ET.Element, tag: str, text: str) -> ET.Element:
    """Add a sub-element; leave text empty (not None) for epsilon in JFLAP."""
    el = ET.SubElement(parent, tag)
    el.text = text
    return el


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():   # noqa: F821
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad