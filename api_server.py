from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from pda_logic import PDA, PDASimulator, convert_pda_to_cfg
from regex_logic import (
    regex_to_min_dfa_json,
    automaton_to_regex,
    operation_union,
    operation_intersection,
    operation_kleene,
    parse_regex,
    build_nfa,
    nfa_to_dfa,
    minimize_dfa,
    automaton_to_json,
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


@app.post("/regex/operation")
async def regex_operation(data: LanguageOperationRequest):
    """
    Aplica una operación de lenguaje sobre uno o dos AFDs generados desde regex.
    Operaciones soportadas: union, intersection, kleene, complement.
    Respuesta: { "states": [...], "edges": [...], "alphabet": [...] }
    """
    try:
        if not data.regex1:
            raise ValueError("Se requiere al menos regex1.")

        # Build first automaton
        tokens1 = parse_regex(data.regex1)
        nfa_s1, nfa_t1, nfa_i1, nfa_f1 = build_nfa(tokens1)
        dfa_s1, dfa_t1, dfa_i1, dfa_a1, alpha1 = nfa_to_dfa(nfa_s1, nfa_t1, nfa_i1, nfa_f1)
        s1, t1, i1, a1 = minimize_dfa(dfa_s1, dfa_t1, dfa_i1, dfa_a1, alpha1)

        op = data.operation.lower()

        if op == "kleene":
            result = operation_kleene(s1, t1, i1, a1, alpha1)

        elif op == "complement":
            # Complement: flip accepting and non-accepting states
            all_states = set(s1)
            new_accepting = all_states - set(a1)
            result = automaton_to_json(s1, t1, i1, new_accepting, alpha1)

        elif op in ("union", "intersection"):
            if not data.regex2:
                raise ValueError(f"La operación '{op}' requiere regex2.")

            tokens2 = parse_regex(data.regex2)
            nfa_s2, nfa_t2, nfa_i2, nfa_f2 = build_nfa(tokens2)
            # Unify alphabets
            alpha = alpha1
            dfa_s2, dfa_t2, dfa_i2, dfa_a2, alpha2 = nfa_to_dfa(nfa_s2, nfa_t2, nfa_i2, nfa_f2)
            alpha = alpha1 | alpha2
            s2, t2, i2, a2 = minimize_dfa(dfa_s2, dfa_t2, dfa_i2, dfa_a2, alpha)

            if op == "union":
                result = operation_union(s1, t1, i1, a1, s2, t2, i2, a2, alpha)
            else:
                result = operation_intersection(s1, t1, i1, a1, s2, t2, i2, a2, alpha)
        else:
            raise ValueError(f"Operación desconocida: '{op}'. Use: union, intersection, kleene, complement.")

        return result

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
            "POST /regex/operation",
            "POST /turing/simulate",
            "POST /turing/graph",
        ],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)