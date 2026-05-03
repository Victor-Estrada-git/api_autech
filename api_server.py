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
    nfa_to_dfa as rl_nfa_to_dfa,
    minimize_dfa,
    automaton_to_json,
)
from lenguajes_regulares import (
    NFA  as LR_NFA,
    DFA  as LR_DFA,
    nfa_to_dfa   as lr_nfa_to_dfa,
    minimize_dfa as lr_minimize_dfa,
    dfa_to_regex as lr_dfa_to_regex,
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

app = FastAPI(title="Automata API", version="2.1")

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
        # FIX 1: validar que el estado inicial exista en la lista de estados
        if data.initial not in data.states:
            raise HTTPException(
                status_code=400,
                detail=f"El estado inicial '{data.initial}' no existe en la lista de estados."
            )

        # FIX 2: autómata sin estados de aceptación → lenguaje vacío (∅), no es un error
        if not data.accepting:
            return {"regex": "∅"}

        alphabet = set(data.alphabet) if data.alphabet else None
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
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LanguageOperationRequest(BaseModel):
    operation: str
    regex1: Optional[str] = None
    regex2: Optional[str] = None
    mapping: Optional[Dict[str, str]] = None
    symbol: Optional[str] = None


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
    Aplica una operación de lenguaje sobre uno o dos DFAs construidos desde regex,
    y devuelve la EXPRESIÓN REGULAR del lenguaje resultante (no el autómata).

    Operaciones soportadas:
      Unarias  : kleene | complement | reverse
      Binarias : union | intersection | difference | concat
      Con param: homomorphism (mapping: dict) | rightquotient (symbol: str)

    Respuesta: { "regex": "<expresión regular del lenguaje resultante>" }
    """
    try:
        if not data.regex1:
            raise ValueError("Se requiere al menos regex1.")

        op   = data.operation.lower()
        dfa1 = _build_lr_dfa(data.regex1)

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
            else:
                result = op_concat(dfa1, dfa2)
        else:
            raise ValueError(
                f"Operacion desconocida: '{op}'. "
                "Soportadas: union, intersection, kleene, complement, "
                "difference, concat, reverse, homomorphism, rightquotient."
            )

        # Normalizar el resultado a un AFD minimizado y obtener la regex
        final_dfa = result if isinstance(result, LR_DFA) else lr_nfa_to_dfa(result)
        final_dfa = lr_minimize_dfa(final_dfa)
        regex_result = lr_dfa_to_regex(final_dfa)

        return {"regex": regex_result}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Minimización de AFD ──────────────────────────────────────────────────────

class MinimizeRequest(BaseModel):
    """
    Recibe un AFD en formato de grafo (idéntico al que devuelve AutomatonCanvas)
    y devuelve el AFD mínimo equivalente.

    Campos:
      states     : lista de { "id": str, "initial": bool, "accepting": bool }
      edges      : lista de { "from": str, "to": str, "label": str }
                   (label puede ser "a,b" para múltiples símbolos en una arista)
      alphabet   : lista de símbolos (opcional; se infiere si no se pasa)
    """
    states: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    alphabet: Optional[List[str]] = None


def _graph_to_lr_dfa(
    states: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    alphabet: Optional[List[str]],
) -> LR_DFA:
    """
    Convierte el formato de grafo (states + edges) en un LR_DFA.
    Primero construye un NFA (para soportar entradas NFA-ε) y luego
    lo convierte a DFA mediante construcción de subconjuntos.
    """
    from lenguajes_regulares import State as LRState

    # Crear objetos State
    state_map = {s["id"]: LRState(s["id"]) for s in states}

    # Determinar estado inicial y estados de aceptación
    initial_id   = next((s["id"] for s in states
                         if s.get("initial") or s.get("is_initial")), None)
    accept_ids   = {s["id"] for s in states
                    if s.get("accepting") or s.get("is_accepting")}

    if initial_id is None and states:
        initial_id = states[0]["id"]

    # Construir transiciones NFA (soporta múltiples destinos y ε)
    nfa_trans: dict = {}
    alpha_inferred: set[str] = set()

    for e in edges:
        frm   = e.get("from", "")
        to    = e.get("to", "")
        label = e.get("label", "")
        syms  = [s.strip() for s in label.split(",") if s.strip()]
        for sym in syms:
            src_obj = state_map.get(frm)
            dst_obj = state_map.get(to)
            if src_obj is None or dst_obj is None:
                continue
            nfa_trans.setdefault(src_obj, {}).setdefault(sym, set()).add(dst_obj)
            if sym != "ε":
                alpha_inferred.add(sym)

    # Alfabeto explícito tiene prioridad
    if alphabet:
        alpha_set = set(alphabet)
    else:
        alpha_set = alpha_inferred

    start_obj   = state_map.get(initial_id)
    accept_objs = {state_map[aid] for aid in accept_ids if aid in state_map}

    nfa = LR_NFA(
        set(state_map.values()),
        alpha_set,
        nfa_trans,
        start_obj,
        accept_objs,
    )

    return lr_nfa_to_dfa(nfa)


@app.post("/automaton/minimize")
async def automaton_minimize(data: MinimizeRequest):
    """
    Minimiza un AFD (o AFN, que se convierte primero a AFD).

    Entrada : { "states": [...], "edges": [...], "alphabet": [...] }
    Salida  : {
                "states": [...],   ← grafo del AFD mínimo
                "edges":  [...],
                "alphabet": [...],
                "stats": {
                  "before": N,     ← estados antes de minimizar
                  "after":  M,     ← estados después
                  "saved":  N-M    ← estados eliminados
                }
              }
    """
    try:
        if not data.states:
            raise ValueError("El autómata está vacío.")

        # Convertir grafo → LR_DFA
        dfa_before = _graph_to_lr_dfa(data.states, data.edges, data.alphabet)
        before_count = len(dfa_before.states)

        # Minimizar
        dfa_min = lr_minimize_dfa(dfa_before)
        after_count = len(dfa_min.states)

        # Convertir resultado → formato de grafo
        result = _lr_to_json(dfa_min)
        result["stats"] = {
            "before": before_count,
            "after":  after_count,
            "saved":  before_count - after_count,
        }
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
        estados   = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        initial   = data.initial.strip()

        # FIX 3: validar que el estado inicial exista en la lista de estados
        if initial not in estados:
            raise HTTPException(
                status_code=400,
                detail=f"El estado inicial '{initial}' no existe en la lista de estados."
            )

        trans_dict = parse_transitions(data.transitions)
        resultado = simulate_turing(
            states=estados,
            transitions=trans_dict,
            initial_state=initial,
            accept_states=aceptados,
            tape_input=data.cinta,
            head_pos=data.head_pos,
            max_steps=data.max_steps,
        )
        return resultado
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/turing/graph")
async def turing_graph(data: TuringRequest):
    """
    Devuelve el grafo de la MT para AutomatonCanvas.
    Retorna: { "states": [...], "edges": [...] }
    """
    try:
        estados   = [s.strip() for s in data.states.split(',') if s.strip()]
        aceptados = [s.strip() for s in data.accepts.split(',') if s.strip()]
        initial   = data.initial.strip()

        # Misma validación para consistencia
        if initial not in estados:
            raise HTTPException(
                status_code=400,
                detail=f"El estado inicial '{initial}' no existe en la lista de estados."
            )

        trans_dict = parse_transitions(data.transitions)
        graph = build_graph_json(
            states=estados,
            transitions=trans_dict,
            initial=initial,
            accept_states=aceptados,
        )
        return graph
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "2.1",
        "endpoints": [
            "POST /pda/validate",
            "POST /pda/simulate",
            "POST /pda/to-cfg",
            "GET  /regex/to-automaton?exp=<regex>",
            "POST /regex/automaton-to-regex",
            "POST /regex/operation  → { regex } del lenguaje resultante (union|intersection|kleene|complement|difference|concat|reverse|homomorphism|rightquotient)",
            "POST /automaton/minimize  ← NUEVO: minimiza cualquier AFD/AFN",
            "POST /turing/simulate",
            "POST /turing/graph",
        ],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)