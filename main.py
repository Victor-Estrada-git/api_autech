"""
FastAPI Backend for Automata Simulator
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from pda_logic import PDA, PDASimulator, convert_pda_to_cfg
from regex_logic import (
    regex_to_min_dfa_json,
    parse_regex, build_nfa, nfa_to_dfa, minimize_dfa,
    automaton_to_json, operation_union, operation_intersection,
)
from turing_logic import parse_transitions, simulate_turing, build_graph_json

app = FastAPI(title="Automata Simulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# PDA ENDPOINTS
# ═══════════════════════════════════════════════════════════

class PDADefinition(BaseModel):
    states: str
    inputAlphabet: str
    stackAlphabet: str
    startState: str
    startSymbol: str
    acceptStates: str
    transitions: str


class PDASimulateRequest(BaseModel):
    definition: PDADefinition
    inputString: str


@app.post("/api/pda/validate")
def pda_validate(req: PDADefinition):
    """Validate PDA and return graph JSON for Flutter canvas."""
    pda = PDA()
    try:
        pda.parse(
            req.states, req.inputAlphabet, req.stackAlphabet,
            req.startState, req.startSymbol, req.acceptStates,
            req.transitions,
        )
        return {"valid": True, "graph": pda.to_graph_json(), "message": "PDA válido."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pda/simulate")
def pda_simulate(req: PDASimulateRequest):
    """Simulate PDA on input string."""
    pda = PDA()
    try:
        d = req.definition
        pda.parse(d.states, d.inputAlphabet, d.stackAlphabet,
                  d.startState, d.startSymbol, d.acceptStates, d.transitions)
        sim = PDASimulator(pda)
        result = sim.simulate(req.inputString)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pda/to_cfg")
def pda_to_cfg(req: PDADefinition):
    """Convert PDA to Context-Free Grammar."""
    pda = PDA()
    try:
        pda.parse(
            req.states, req.inputAlphabet, req.stackAlphabet,
            req.startState, req.startSymbol, req.acceptStates, req.transitions,
        )
        cfg_text = convert_pda_to_cfg(pda)
        return {"cfg": cfg_text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════
# REGEX ENDPOINTS
# ═══════════════════════════════════════════════════════════

class RegexRequest(BaseModel):
    regex: str


class AutomatonData(BaseModel):
    states: List[str]
    transitions: Dict[str, Dict[str, str]]
    initial: str
    accepting: List[str]
    alphabet: List[str]


class OperationRequest(BaseModel):
    automaton1: AutomatonData
    automaton2: Optional[AutomatonData] = None
    operation: str  # kleene, positive, union, intersection, difference, reverse, power
    n: Optional[int] = None  # for power operation


@app.post("/api/regex/to_automaton")
def regex_to_automaton(req: RegexRequest):
    """Convert regex to minimized DFA, return Flutter-compatible graph JSON."""
    try:
        graph = regex_to_min_dfa_json(req.regex)
        return graph
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.post("/api/regex/from_automaton")
def automaton_to_regex(req: AutomatonData):
    """Convert DFA to regex using state elimination."""
    try:
        result = _dfa_to_regex(req.states, req.transitions, req.initial, set(req.accepting))
        return {"regex": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _dfa_to_regex(states, transitions, initial, accepting):
    """Simple state elimination for DFA → regex."""
    # Add new start and accept states
    states = list(states)
    trans = {s: dict(t) for s, t in transitions.items()}

    new_init = "__INIT__"
    new_accept = "__ACCEPT__"
    trans[new_init] = {None: initial}  # ε-transition

    for s in accepting:
        if s not in trans: trans[s] = {}
        trans[s][None] = new_accept  # ε-transition

    all_states = [new_init] + states + [new_accept]

    # State elimination (simplified)
    # For now, return a descriptive message
    if not accepting:
        return "∅ (lenguaje vacío)"
    return f"Autómata con {len(states)} estado(s) — use la función de conversión"


@app.post("/api/regex/operation")
def perform_operation(req: OperationRequest):
    """Perform language operation on automaton(a)."""
    try:
        a1 = req.automaton1
        s1 = a1.states
        t1 = a1.transitions
        i1 = a1.initial
        acc1 = set(a1.accepting)
        alph = set(a1.alphabet)

        if req.operation == "kleene":
            result = _kleene(s1, t1, i1, acc1, alph)
        elif req.operation == "positive":
            result = _positive(s1, t1, i1, acc1, alph)
        elif req.operation == "reverse":
            result = _reverse(s1, t1, i1, acc1, alph)
        elif req.operation == "complement":
            result = _complement(s1, t1, i1, acc1, alph)
        elif req.operation in ("union", "intersection", "difference"):
            if not req.automaton2:
                raise HTTPException(status_code=400, detail="Se requiere un segundo autómata")
            a2 = req.automaton2
            s2, t2, i2, acc2 = a2.states, a2.transitions, a2.initial, set(a2.accepting)
            combined_alph = alph | set(a2.alphabet)
            if req.operation == "union":
                result = operation_union(s1, t1, i1, acc1, s2, t2, i2, acc2, combined_alph)
            elif req.operation == "intersection":
                result = operation_intersection(s1, t1, i1, acc1, s2, t2, i2, acc2, combined_alph)
            else:  # difference
                acc2_comp = set(s2) - acc2
                result = operation_intersection(s1, t1, i1, acc1, s2, t2, i2, acc2_comp, combined_alph)
        elif req.operation == "power":
            n = req.n or 1
            result = _power(s1, t1, i1, acc1, alph, n)
        else:
            raise HTTPException(status_code=400, detail=f"Operación desconocida: {req.operation}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _kleene(states, transitions, initial, accepting, alphabet):
    """Kleene closure: add ε-transitions from accept states back to initial."""
    new_start = "KL_START"
    new_states = list(states) + [new_start]
    new_trans = {s: dict(t) for s, t in transitions.items()}
    # New start → old initial, also accepting
    new_trans[new_start] = dict(transitions.get(initial, {}))
    new_accepting = set(accepting) | {new_start}
    return automaton_to_json(new_states, new_trans, new_start, new_accepting, alphabet)


def _positive(states, transitions, initial, accepting, alphabet):
    """Positive closure: same as Kleene but initial is not accepting."""
    new_start = "PL_START"
    new_states = list(states) + [new_start]
    new_trans = {s: dict(t) for s, t in transitions.items()}
    new_trans[new_start] = dict(transitions.get(initial, {}))
    return automaton_to_json(new_states, new_trans, new_start, set(accepting), alphabet)


def _reverse(states, transitions, initial, accepting, alphabet):
    """Reverse: swap initial/accepting, reverse transitions."""
    new_start = "REV_START"
    new_states = list(states) + [new_start]
    rev_trans: Dict[str, Dict[str, str]] = {s: {} for s in new_states}
    for s, trans in transitions.items():
        for sym, dest in trans.items():
            rev_trans[dest][sym] = s
    # New start ε-connects to all old accept states
    for s in accepting:
        rev_trans[new_start][f"ε_{s}"] = s  # pseudo ε
    new_accepting = {initial}
    return automaton_to_json(new_states, rev_trans, new_start, new_accepting, alphabet)


def _complement(states, transitions, initial, accepting, alphabet):
    """Complement: swap accepting / non-accepting."""
    new_accepting = set(states) - accepting
    return automaton_to_json(list(states), transitions, initial, new_accepting, alphabet)


def _power(states, transitions, initial, accepting, alphabet, n: int):
    """L^n: concatenate automaton n times."""
    if n <= 0:
        # Only ε
        s = "EPS"
        return automaton_to_json([s], {s: {}}, s, {s}, alphabet)
    if n == 1:
        return automaton_to_json(list(states), transitions, initial, accepting, alphabet)

    # Build product for concatenation
    # Simple approach: chain copies
    all_states = []
    all_trans = {}
    all_accepting = set()
    prev_accepting = None

    for k in range(n):
        prefix = f"P{k}_"
        k_states = [prefix + s for s in states]
        all_states += k_states
        for s, trans in transitions.items():
            all_trans[prefix + s] = {sym: prefix + dest for sym, dest in trans.items()}
        k_accepting = {prefix + s for s in accepting}

        if k == n - 1:
            all_accepting = k_accepting

        # Connect previous accepting to this initial
        if prev_accepting:
            for pa in prev_accepting:
                for sym, dest in transitions.get(initial, {}).items():
                    all_trans[pa][sym] = prefix + dest

        prev_accepting = k_accepting

    k_initial = "P0_" + initial
    return automaton_to_json(all_states, all_trans, k_initial, all_accepting, alphabet)


# ═══════════════════════════════════════════════════════════
# TURING MACHINE ENDPOINTS
# ═══════════════════════════════════════════════════════════

class TuringDefinition(BaseModel):
    states: str          # comma-separated
    initial: str
    acceptStates: str    # comma-separated
    transitions: str     # multiline format


class TuringSimulateRequest(BaseModel):
    definition: TuringDefinition
    tape: str
    headPos: int = 0
    maxSteps: int = 500


@app.post("/api/turing/graph")
def turing_graph(req: TuringDefinition):
    """Parse TM definition and return graph JSON for Flutter canvas."""
    states = [s.strip() for s in req.states.split(',') if s.strip()]
    accept_states = [s.strip() for s in req.acceptStates.split(',') if s.strip()]
    transitions = parse_transitions(req.transitions)
    graph = build_graph_json(states, transitions, req.initial.strip(), accept_states)
    return graph


@app.post("/api/turing/simulate")
def turing_simulate(req: TuringSimulateRequest):
    """Simulate Turing Machine and return all steps."""
    d = req.definition
    states = [s.strip() for s in d.states.split(',') if s.strip()]
    accept_states = [s.strip() for s in d.acceptStates.split(',') if s.strip()]
    transitions = parse_transitions(d.transitions)

    try:
        result = simulate_turing(
            states=states,
            transitions=transitions,
            initial_state=d.initial.strip(),
            accept_states=accept_states,
            tape_input=req.tape,
            head_pos=req.headPos,
            max_steps=req.maxSteps,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)