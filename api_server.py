from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from pda_logic import PDA, PDASimulator, convert_pda_to_cfg
from regex_logic import (
    regex_to_min_dfa_json,
    automaton_to_regex,
    parse_regex,
    build_nfa,
    nfa_to_dfa as rl_nfa_to_dfa,   # regex_logic version (para otros endpoints)
    minimize_dfa,
    automaton_to_json,
)
from lenguajes_regulares import (
    NFA  as LR_NFA,
    DFA  as LR_DFA,
    nfa_to_dfa as lr_nfa_to_dfa,   # lenguajes_regulares version
    op_union,
    op_concat,
    op_kleene,
    op_complement,
    op_intersection,
    op_difference,
    op_reverse,
    op_homomorphism,
    op_right_quotient,
)
from turing_logic import simulate_turing, parse_transitions, build_graph_json

app = FastAPI(title="Automata API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── PDA ──────────────────────────────────────────────────────────────────────

class PDARequest(BaseModel):
    states: str
    input_alpha: str
    stack_alpha: str
    start_state: str
    start_symbol: str
    accept_states: str
    transitions: str


class PDASimRequest(BaseModel):
    states: str
    input_alpha: str
    stack_alpha: str
    start_state: str
    start_symbol: str
    accept_states: str
    transitions: str
    input_string: str = ""


def _build_pda(data: PDARequest) -> PDA:
    pda = PDA()
    pda.parse(
        data.states, data.input_alpha, data.stack_alpha,
        data.start_state, data.start_symbol, data.accept_states,
        data.transitions,
    )
    return pda


@app.post("/pda/validate")
async def pda_validate(data: PDARequest):
    """
    Valida el PDA y retorna el grafo para AutomatonCanvas.
    Respuesta: { "graph": { "states": [...], "edges": [...] } }
    """
    try:
        pda = _build_pda(data)
        return {"graph": pda.to_graph_json(), "valid": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pda/simulate")
async def pda_simulate(data: PDASimRequest):
    """
    Simula el PDA sobre una cadena de entrada.
    Respuesta: { "accepted": bool, "trace": [...], "steps": int }
    """
    try:
        pda = PDA()
        pda.parse(
            data.states, data.input_alpha, data.stack_alpha,
            data.start_state, data.start_symbol, data.accept_states,
            data.transitions,
        )
        sim = PDASimulator(pda)
        result = sim.simulate(data.input_string)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pda/to-cfg")
async def pda_to_cfg(data: PDARequest):
    """
    Convierte el PDA a una Gramática Libre de Contexto (CFG).
    Respuesta: { "cfg": "<texto>" }
    """
    try:
        pda = _build_pda(data)
        resultado = convert_pda_to_cfg(pda)
        return {"cfg": resultado}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Regex ────────────────────────────────────────────────────────────────────

@app.get("/regex/to-automaton")
async def regex_to_automaton(exp: str):
    """
    Convierte una expresión regular al DFA minimizado.
    Retorna: { "states": [...], "edges": [...], "alphabet": [...] }
    """
    try:
        result = regex_to_min_dfa_json(exp)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AutomatonToRegexRequest(BaseModel):
    states: List[str]
    transitions: Dict[str, Dict[str, str]]
    initial: str
    accepting: List[str]
    alphabet: Optional[List[str]] = None


@app.post("/regex/automaton-to-regex")
async def automaton_to_regex_endpoint(data: AutomatonToRegexRequest):
    """
    Convierte un autómata (DFA/NFA) a expresión regular por eliminación de estados.
    Respuesta: { "regex": "<expresión regular>" }
    """
    try:
        alphabet = set(data.alphabet) if data.alphabet else None
        # If alphabet not provided, infer from transitions
        if not alphabet:
            alphabet = set()
            for trans in data.transitions.values():
                for label in trans.keys():
                    for sym in label.split(','):
                        sym = sym.strip()
                        if sym and sym != 'ε':
                            alphabet.add(sym)

        regex = automaton_to_regex(
            states=data.states,
            transitions=data.transitions,
            initial=data.initial,
            accepting=set(data.accepting),
            alphabet=alphabet,
        )
        return {"regex": regex}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LanguageOperationRequest(BaseModel):
    operation: str          # "union" | "intersection" | "kleene" | "complement"
    regex1: Optional[str] = None
    regex2: Optional[str] = None   # only for binary ops



# ─── Helpers: lenguajes_regulares ↔ automaton_to_json ─────────────────────────

def _build_lr_dfa(regex: str) -> LR_DFA:
    """
    Construye un LR_DFA desde una regex usando el parser de regex_logic
    y lo envuelve en la clase DFA de lenguajes_regulares.
    """
    tokens = parse_regex(regex)
    nfa_s, nfa_t, nfa_i, nfa_f = build_nfa(tokens)
    dfa_s, dfa_t, dfa_i, dfa_a, alpha = rl_nfa_to_dfa(nfa_s, nfa_t, nfa_i, nfa_f)
    min_s, min_t, min_i, min_a = minimize_dfa(dfa_s, dfa_t, dfa_i, dfa_a, alpha)
    return LR_DFA(set(min_s), alpha, min_t, min_i, set(min_a))


def _lr_to_json(automaton) -> dict:
    """
    Convierte un NFA o DFA de lenguajes_regulares al formato de automaton_to_json.
    Los estados pueden ser objetos State (con .name) o strings.
    """
    if isinstance(automaton, LR_NFA):
        automaton = lr_nfa_to_dfa(automaton)

    def sname(s):
        return s.name if hasattr(s, "name") else str(s)

    states      = [sname(s) for s in automaton.states]
    initial     = sname(automaton.start)
    accepting   = {sname(s) for s in automaton.accept}
    alphabet    = sorted(automaton.alphabet)
    transitions = {
        sname(s): {sym: sname(tgt) for sym, tgt in t.items()}
        for s, t in automaton.transitions.items()
    }
    return automaton_to_json(states, transitions, initial, accepting, alphabet)


@app.post("/regex/operation")
async def regex_operation(data: LanguageOperationRequest):
    """
    Aplica una operación de lenguaje sobre uno o dos DFAs construidos desde regex.

    Todas las operaciones se resuelven en el servidor usando lenguajes_regulares.py:
      Unarias  : kleene | complement | reverse
      Binarias : union | intersection | difference | concat
      Con param: homomorphism (mapping: dict) | rightquotient (symbol: str)

    Respuesta: { "states": [...], "edges": [...], "alphabet": [...] }
    """
    try:
        if not data.regex1:
            raise ValueError("Se requiere al menos regex1.")

        op   = data.operation.lower()
        dfa1 = _build_lr_dfa(data.regex1)

        # ── Operaciones unarias ────────────────────────────────────────────────
        if op == "kleene":
            result = op_kleene(dfa1)

        elif op == "complement":
            result = op_complement(dfa1)

        elif op == "reverse":
            result = op_reverse(dfa1)

        elif op == "homomorphism":
            if not data.mapping:
                raise ValueError("Se requiere el campo 'mapping' para homomorfismo.")
            result = op_homomorphism(dfa1, data.mapping)

        elif op == "rightquotient":
            if not data.symbol:
                raise ValueError("Se requiere el campo 'symbol' para cociente derecho.")
            result = op_right_quotient(dfa1, data.symbol)

        # ── Operaciones binarias ───────────────────────────────────────────────
        elif op in ("union", "intersection", "difference", "concat"):
            if not data.regex2:
                raise ValueError(f"La operacion '{op}' requiere regex2.")
            dfa2 = _build_lr_dfa(data.regex2)

            if op == "union":
                result = op_union(dfa1, dfa2)
            elif op == "intersection":
                result = op_intersection(dfa1, dfa2)
            elif op == "difference":
                result = op_difference(dfa1, dfa2)
            else:  # concat
                result = op_concat(dfa1, dfa2)

        else:
            raise ValueError(
                f"Operacion desconocida: '{op}'. "
                "Soportadas: union, intersection, kleene, complement, "
                "difference, concat, reverse, homomorphism, rightquotient."
            )

        return _lr_to_json(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Turing ───────────────────────────────────────────────────────────────────

class TuringRequest(BaseModel):
    states: str
    initial: str
    accepts: str
    transitions: str
    cinta: str
    head_pos: int = 0
    max_steps: int = 1000


@app.post("/turing/simulate")
async def turing_simulate(data: TuringRequest):
    """
    Simula una Máquina de Turing paso a paso.
    Retorna: { "steps": [...], "result": "ACCEPTED"|"REJECTED"|"TIMEOUT" }
    """
    try:
        estados = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        trans_dict = parse_transitions(data.transitions)

        resultado = simulate_turing(
            states=estados,
            transitions=trans_dict,
            initial_state=data.initial.strip(),
            accept_states=aceptados,
            tape_input=data.cinta,
            head_pos=data.head_pos,
            max_steps=data.max_steps,
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/turing/graph")
async def turing_graph(data: TuringRequest):
    """
    Devuelve el grafo de la MT para AutomatonCanvas.
    Retorna: { "states": [...], "edges": [...] }
    """
    try:
        estados = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        trans_dict = parse_transitions(data.transitions)

        graph = build_graph_json(
            states=estados,
            transitions=trans_dict,
            initial=data.initial.strip(),
            accept_states=aceptados,
        )
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "endpoints": [
            "POST /pda/validate",
            "POST /pda/simulate",
            "POST /pda/to-cfg",
            "GET  /regex/to-automaton?exp=<regex>",
            "POST /regex/automaton-to-regex",
            "POST /regex/operation  (union|intersection|kleene|complement|difference|concat|reverse|homomorphism|rightquotient)",
            "POST /turing/simulate",
            "POST /turing/graph",
        ],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)