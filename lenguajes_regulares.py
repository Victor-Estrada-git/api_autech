"""
╔══════════════════════════════════════════════════════════════╗
║       Operaciones sobre Lenguajes Regulares                  ║
║  Construcción de Thompson + Algoritmo de Subconjuntos        ║
║  + Minimización de AFD (llenado de tabla / Hopcroft)         ║
╚══════════════════════════════════════════════════════════════╝

Operaciones implementadas:
  DEFINICIÓN    : Unión, Concatenación, Cierre de Kleene (L*)
  CERRADURA     : Intersección, Complemento, Diferencia
  TRANSFORMACIÓN: Reversa, Homomorfismo, Cociente por la derecha
  MINIMIZACIÓN  : minimize_dfa  (tabla de distinguibilidad + Union-Find)
"""

from collections import deque

# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════

EPSILON = 'ε'


# ══════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════

class State:
    """Representa un estado de un autómata."""
    _counter = 0

    def __init__(self, name=None):
        if name is None:
            State._counter += 1
            self.name = f"s{State._counter}"
        else:
            self.name = name

    def __repr__(self):  return self.name
    def __str__(self):   return self.name
    def __hash__(self):  return hash(self.name)
    def __eq__(self, o): return isinstance(o, State) and self.name == o.name
    def __lt__(self, o): return self.name < o.name


def _new_state(prefix="s"):
    State._counter += 1
    return State(f"{prefix}{State._counter}")


# ──────────────────────────────────────────────────────────────

class NFA:
    """AFN-ε  (Autómata Finito No Determinista con transiciones épsilon)."""

    def __init__(self, states, alphabet, transitions, start, accept):
        self.states      = set(states)
        self.alphabet    = set(alphabet) - {EPSILON}
        self.transitions = transitions          # {state: {symbol: set(states)}}
        self.start       = start
        self.accept      = set(accept)

    def epsilon_closure(self, states):
        closure = set(states)
        stack   = list(states)
        while stack:
            s = stack.pop()
            for t in self.transitions.get(s, {}).get(EPSILON, set()):
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
        return frozenset(closure)

    def move(self, states, symbol):
        result = set()
        for s in states:
            result |= self.transitions.get(s, {}).get(symbol, set())
        return result

    def accepts(self, string):
        current = self.epsilon_closure({self.start})
        for c in string:
            current = self.epsilon_closure(self.move(current, c))
        return bool(current & self.accept)

    def print_automaton(self, title="AFN-ε"):
        print(f"\n{'═'*56}")
        print(f"  {title}")
        print(f"{'═'*56}")
        print(f"  Estados           : {sorted(self.states)}")
        print(f"  Alfabeto          : {sorted(self.alphabet)}")
        print(f"  Estado inicial    : {self.start}")
        print(f"  Estados aceptación: {sorted(self.accept)}")
        print(f"  Transiciones:")
        for state in sorted(self.states):
            trans = self.transitions.get(state, {})
            for sym in sorted(trans.keys(), key=str):
                targets = sorted(trans[sym])
                print(f"    δ({state}, {sym!s:4}) = {{ {', '.join(str(t) for t in targets)} }}")
        print(f"{'═'*56}\n")


# ──────────────────────────────────────────────────────────────

class DFA:
    """AFD  (Autómata Finito Determinista)."""

    def __init__(self, states, alphabet, transitions, start, accept):
        self.states      = set(states)
        self.alphabet    = set(alphabet)
        self.transitions = transitions          # {state: {symbol: state}}
        self.start       = start
        self.accept      = set(accept)

    def accepts(self, string):
        current = self.start
        for c in string:
            current = self.transitions.get(current, {}).get(c)
            if current is None:
                return False
        return current in self.accept

    def complete(self):
        """Devuelve un AFD completo agregando estado muerto si es necesario."""
        DEAD = "∅"
        new_trans  = {s: dict(t) for s, t in self.transitions.items()}
        needs_dead = False
        for state in list(self.states):
            for sym in self.alphabet:
                if sym not in new_trans.get(state, {}):
                    new_trans.setdefault(state, {})[sym] = DEAD
                    needs_dead = True
        if needs_dead:
            new_trans[DEAD] = {sym: DEAD for sym in self.alphabet}
            return DFA(self.states | {DEAD}, self.alphabet, new_trans,
                       self.start, self.accept)
        return DFA(self.states, self.alphabet, new_trans, self.start, self.accept)

    def print_automaton(self, title="AFD"):
        print(f"\n{'═'*56}")
        print(f"  {title}")
        print(f"{'═'*56}")
        print(f"  Estados           : {sorted(str(s) for s in self.states)}")
        print(f"  Alfabeto          : {sorted(self.alphabet)}")
        print(f"  Estado inicial    : {self.start}")
        print(f"  Estados aceptación: {sorted(str(s) for s in self.accept)}")
        print(f"  Transiciones:")
        for state in sorted(str(s) for s in self.states):
            trans = self.transitions.get(state, {})
            for sym in sorted(self.alphabet):
                target = trans.get(sym, "—")
                print(f"    δ({state}, {sym}) = {target}")
        print(f"{'═'*56}\n")


# ══════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE THOMPSON  (Regex → AFN-ε)
# ══════════════════════════════════════════════════════════════

def _basic(symbol):
    s0, s1 = _new_state(), _new_state()
    trans  = {s0: {symbol: {s1}}}
    alpha  = set() if symbol == EPSILON else {symbol}
    return NFA({s0, s1}, alpha, trans, s0, {s1})


def thompson_union(n1, n2):
    s0, sf = _new_state(), _new_state()
    trans  = {}
    for s, t in n1.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    for s, t in n2.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    trans[s0] = {EPSILON: {n1.start, n2.start}}
    for acc in n1.accept | n2.accept:
        trans.setdefault(acc, {}).setdefault(EPSILON, set()).add(sf)
    return NFA(n1.states | n2.states | {s0, sf},
               n1.alphabet | n2.alphabet, trans, s0, {sf})


def thompson_concat(n1, n2):
    trans = {}
    for s, t in n1.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    for s, t in n2.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    for acc in n1.accept:
        trans.setdefault(acc, {}).setdefault(EPSILON, set()).add(n2.start)
    return NFA(n1.states | n2.states,
               n1.alphabet | n2.alphabet, trans, n1.start, n2.accept)


def thompson_star(n):
    s0, sf = _new_state(), _new_state()
    trans  = {}
    for s, t in n.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    trans[s0] = {EPSILON: {n.start, sf}}
    for acc in n.accept:
        trans.setdefault(acc, {}).setdefault(EPSILON, set()).update({n.start, sf})
    return NFA(n.states | {s0, sf}, n.alphabet, trans, s0, {sf})


def thompson_plus(n):
    return thompson_concat(n, thompson_star(n))


def thompson_optional(n):
    return thompson_union(n, _basic(EPSILON))


# ──────────────────────────────────────────────────────────────
# Parser de expresiones regulares
# ──────────────────────────────────────────────────────────────

class _RegexParser:
    """
    Parsea una regex e invoca Thompson en cada nodo.

    Sintaxis soportada:
      Caracteres : a-z, A-Z, 0-9, y otros no especiales
      Unión      : |
      Kleene     : *   (postfijo)
      Una o más  : +   (postfijo)
      Opcional   : ?   (postfijo)
      Agrupación : ( )
      Épsilon    : ε  o  @
      Escape     : \\x  (usa 'x' literalmente)
    """

    def __init__(self, regex):
        self.s   = regex
        self.pos = 0

    def parse(self):
        nfa = self._union()
        if self.pos != len(self.s):
            raise ValueError(f"Carácter inesperado '{self.s[self.pos]}' en pos {self.pos}")
        return nfa

    def _peek(self):
        return self.s[self.pos] if self.pos < len(self.s) else None

    def _consume(self, expected=None):
        c = self._peek()
        if expected and c != expected:
            raise ValueError(f"Se esperaba '{expected}', se obtuvo '{c}'")
        self.pos += 1
        return c

    def _union(self):
        left = self._concat()
        while self._peek() == '|':
            self._consume('|')
            left = thompson_union(left, self._concat())
        return left

    def _concat(self):
        left = self._quantifier()
        while self._peek() not in (None, ')', '|'):
            left = thompson_concat(left, self._quantifier())
        return left

    def _quantifier(self):
        base = self._atom()
        while self._peek() in ('*', '+', '?'):
            op = self._consume()
            if   op == '*': base = thompson_star(base)
            elif op == '+': base = thompson_plus(base)
            elif op == '?': base = thompson_optional(base)
        return base

    def _atom(self):
        c = self._peek()
        if c is None:
            raise ValueError("Fin inesperado de la expresión regular")
        if c == '(':
            self._consume('(')
            nfa = self._union()
            self._consume(')')
            return nfa
        if c in ('ε', '@'):
            self._consume(); return _basic(EPSILON)
        if c == '\\':
            self._consume('\\'); return _basic(self._consume())
        if c not in (')', '|', '*', '+', '?'):
            self._consume(); return _basic(c)
        raise ValueError(f"Carácter inesperado '{c}' en pos {self.pos}")


def regex_to_nfa(regex, reset_counter=False):
    """Convierte una expresión regular a un AFN-ε (Construcción de Thompson)."""
    if reset_counter:
        State._counter = 0
    return _RegexParser(regex).parse()


# ══════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE SUBCONJUNTOS  (AFN-ε → AFD)
# ══════════════════════════════════════════════════════════════

def nfa_to_dfa(nfa):
    """
    Convierte un AFN-ε en un AFD (Algoritmo de Construcción de Subconjuntos).
    """
    ctr       = [0]
    state_map = {}

    def dfa_name(fs):
        if fs not in state_map:
            state_map[fs] = f"D{ctr[0]}"; ctr[0] += 1
        return state_map[fs]

    start_set = nfa.epsilon_closure({nfa.start})
    dfa_start = dfa_name(start_set)
    queue     = deque([start_set])
    visited   = set()
    dfa_trans = {}
    dfa_acc   = set()

    while queue:
        cur = queue.popleft()
        if cur in visited: continue
        visited.add(cur)
        cname = dfa_name(cur)

        if cur & nfa.accept:
            dfa_acc.add(cname)

        for sym in sorted(nfa.alphabet):
            nxt = nfa.epsilon_closure(nfa.move(cur, sym))
            if not nxt: continue
            nname = dfa_name(nxt)
            dfa_trans.setdefault(cname, {})[sym] = nname
            if nxt not in visited:
                queue.append(nxt)

    return DFA(set(state_map.values()), nfa.alphabet, dfa_trans, dfa_start, dfa_acc)


# ══════════════════════════════════════════════════════════════
# MINIMIZACIÓN DE AFD  (llenado de tabla + Union-Find)
# ══════════════════════════════════════════════════════════════

def minimize_dfa(dfa):
    """
    Minimiza un AFD usando el algoritmo de llenado de tabla
    (equivalente a Myhill-Nerode / Hopcroft simplificado).

    Pasos:
      1. Completar el AFD (agregar estado muerto si falta).
      2. Eliminar estados inalcanzables.
      3. Marcar pares distinguibles:
         - Base : (aceptación, no-aceptación)
         - Iteración : δ(p,a) y δ(q,a) ya marcados → marcar (p,q)
      4. Unir estados indistinguibles (Union-Find).
      5. Construir el AFD mínimo.
      6. Eliminar estado muerto resultante (si existe).

    Devuelve un DFA minimizado.
    """
    if not dfa.states:
        return dfa

    # ── 1. Completar ─────────────────────────────────────────
    d = dfa.complete()

    # ── 2. Estados alcanzables ────────────────────────────────
    reachable = {d.start}
    stack     = [d.start]
    while stack:
        s = stack.pop()
        for sym in d.alphabet:
            t = d.transitions.get(s, {}).get(sym)
            if t and t not in reachable:
                reachable.add(t)
                stack.append(t)

    states = sorted(str(s) for s in reachable)
    n      = len(states)
    if n == 0:
        return dfa

    idx = {s: i for i, s in enumerate(states)}

    # ── 3. Tabla de distinguibilidad ─────────────────────────
    # marked[i][j] = True  ⟺  states[i] y states[j] distinguibles (i < j)
    marked = [[False] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if (states[i] in d.accept) != (states[j] in d.accept):
                marked[i][j] = True

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if marked[i][j]:
                    continue
                for sym in d.alphabet:
                    ti = d.transitions.get(states[i], {}).get(sym)
                    tj = d.transitions.get(states[j], {}).get(sym)
                    if ti is None or tj is None:
                        continue
                    ti, tj = str(ti), str(tj)
                    ii = idx.get(ti, -1)
                    jj = idx.get(tj, -1)
                    if ii < 0 or jj < 0 or ii == jj:
                        continue
                    p, q = (ii, jj) if ii < jj else (jj, ii)
                    if marked[p][q]:
                        marked[i][j] = True
                        changed = True
                        break

    # ── 4. Union-Find ─────────────────────────────────────────
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if not marked[i][j]:
                union(i, j)

    # ── 5. Construir grupos ───────────────────────────────────
    groups: dict[int, list[str]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(states[i])

    group_name = {root: f"M{gi}" for gi, root in enumerate(sorted(groups))}
    state_to_group = {s: group_name[find(idx[s])] for s in states}

    min_trans: dict[str, dict[str, str]] = {}
    min_accept: set[str] = set()

    for root, members in groups.items():
        gname = group_name[root]
        rep   = members[0]
        min_trans[gname] = {}
        for sym in d.alphabet:
            t = d.transitions.get(rep, {}).get(sym)
            if t is not None:
                tg = state_to_group.get(str(t))
                if tg:
                    min_trans[gname][sym] = tg
        if rep in d.accept:
            min_accept.add(gname)

    min_start  = state_to_group[str(d.start)]
    min_states = set(group_name.values())

    # ── 6. Eliminar estado muerto ─────────────────────────────
    dead_states = {
        s for s in min_states
        if s not in min_accept
        and all(min_trans.get(s, {}).get(sym) == s for sym in d.alphabet)
    }

    final_states = min_states - dead_states
    final_trans  = {
        s: {sym: t for sym, t in (min_trans.get(s) or {}).items()
            if t not in dead_states}
        for s in final_states
    }

    return DFA(
        final_states,
        d.alphabet,
        final_trans,
        min_start,
        min_accept - dead_states,
    )


# ══════════════════════════════════════════════════════════════
# CONVERSIÓN DFA → NFA  (trivial)
# ══════════════════════════════════════════════════════════════

def dfa_to_nfa(dfa):
    trans = {s: {sym: {tgt} for sym, tgt in t.items()}
             for s, t in dfa.transitions.items()}
    return NFA(dfa.states, dfa.alphabet, trans, dfa.start, dfa.accept)


# ══════════════════════════════════════════════════════════════
# OPERACIONES SOBRE LENGUAJES REGULARES
# ══════════════════════════════════════════════════════════════

def _rename_nfa(nfa, prefix):
    """
    Devuelve una copia del NFA con todos los estados renombrados con
    un prefijo único. Necesario porque dos DFAs creados con nfa_to_dfa
    en llamadas independientes usan los mismos nombres ('D0','D1',...)
    y colisionarían al combinarse con Thompson.
    """
    mapping = {s: State(f"{prefix}{str(s)}") for s in nfa.states}
    new_trans = {}
    for s, t in nfa.transitions.items():
        if s not in mapping:
            continue
        new_trans[mapping[s]] = {
            sym: {mapping[d] for d in dsts if d in mapping}
            for sym, dsts in t.items()
        }
    return NFA(
        set(mapping.values()),
        set(nfa.alphabet),
        new_trans,
        mapping[nfa.start],
        {mapping[a] for a in nfa.accept if a in mapping},
    )


# ── 1. OPERACIONES DE DEFINICIÓN ──────────────────────────────

def op_union(a1, a2):
    """L1 ∪ L2 — todas las cadenas en L1 o en L2."""
    n1 = a1 if isinstance(a1, NFA) else dfa_to_nfa(a1)
    n2 = a2 if isinstance(a2, NFA) else dfa_to_nfa(a2)
    # Renombrar para garantizar estados únicos antes de combinar
    n1 = _rename_nfa(n1, "U1_")
    n2 = _rename_nfa(n2, "U2_")
    return thompson_union(n1, n2)

def op_concat(a1, a2):
    """L1 · L2 — concatenación de cadenas."""
    n1 = a1 if isinstance(a1, NFA) else dfa_to_nfa(a1)
    n2 = a2 if isinstance(a2, NFA) else dfa_to_nfa(a2)
    n1 = _rename_nfa(n1, "C1_")
    n2 = _rename_nfa(n2, "C2_")
    return thompson_concat(n1, n2)

def op_kleene(a):
    """L* — cero o más repeticiones."""
    n = a if isinstance(a, NFA) else dfa_to_nfa(a)
    n = _rename_nfa(n, "K_")
    return thompson_star(n)


# ── 2. PROPIEDADES DE CERRADURA ───────────────────────────────

def _product_dfa(a1, a2, accept_pred):
    """
    Producto cartesiano de dos AFD.  Recorre solo los estados alcanzables
    y usa accept_pred(b1, b2) para decidir qué pares (s1,s2) son finales.
    Usado por intersección y diferencia.
    """
    d1 = (a1 if isinstance(a1, DFA) else nfa_to_dfa(a1)).complete()
    d2 = (a2 if isinstance(a2, DFA) else nfa_to_dfa(a2)).complete()
    alpha = d1.alphabet | d2.alphabet
    name  = lambda p: f"({p[0]},{p[1]})"

    start   = (d1.start, d2.start)
    queue   = deque([start])
    visited = set()
    trans   = {}
    acc     = set()

    while queue:
        (s1, s2) = queue.popleft()
        if (s1, s2) in visited:
            continue
        visited.add((s1, s2))
        if accept_pred(s1 in d1.accept, s2 in d2.accept):
            acc.add(name((s1, s2)))
        for sym in sorted(alpha):
            t1 = d1.transitions.get(s1, {}).get(sym, "∅")
            t2 = d2.transitions.get(s2, {}).get(sym, "∅")
            nxt = (t1, t2)
            trans.setdefault(name((s1, s2)), {})[sym] = name(nxt)
            if nxt not in visited:
                queue.append(nxt)

    return DFA({name(p) for p in visited}, alpha, trans, name(start), acc)


def op_intersection(a1, a2):
    """L1 ∩ L2 — producto cartesiano (acepta cuando ambos aceptan)."""
    return _product_dfa(a1, a2, lambda b1, b2: b1 and b2)


def op_complement(a):
    """L̄ — Complemento: cadenas sobre Σ que NO están en L."""
    d = (a if isinstance(a, DFA) else nfa_to_dfa(a)).complete()
    return DFA(d.states, d.alphabet, d.transitions, d.start, d.states - d.accept)


def op_difference(a1, a2):
    """L1 − L2 — producto cartesiano (acepta cuando L1 sí y L2 no)."""
    return _product_dfa(a1, a2, lambda b1, b2: b1 and not b2)


# ── 3. OPERACIONES DE TRANSFORMACIÓN ─────────────────────────

def op_reverse(a):
    """L^R — Reversa: cadenas de L escritas al revés."""
    dfa  = a if isinstance(a, DFA) else nfa_to_dfa(a)
    rev  = {}
    for state, t in dfa.transitions.items():
        for sym, tgt in t.items():
            rev.setdefault(tgt, {}).setdefault(sym, set()).add(state)
    new_start = _new_state("r")
    rev[new_start] = {EPSILON: set(dfa.accept)}
    nfa = NFA(dfa.states | {new_start}, dfa.alphabet, rev, new_start, {dfa.start})
    return nfa_to_dfa(nfa)


def op_homomorphism(a, h):
    """
    Homomorfismo: sustituye cada símbolo del alfabeto por una cadena.
    h : dict  { 'a' -> 'xy', 'b' -> 'z', ... }
    """
    dfa = a if isinstance(a, DFA) else nfa_to_dfa(a)
    so  = {s: State(f"h_{s}") for s in dfa.states}
    all_s = set(so.values())
    trans = {}
    for state, t in dfa.transitions.items():
        for sym, tgt in t.items():
            img = h.get(sym, sym)
            src, dst = so[state], so[tgt]
            if not img:
                trans.setdefault(src, {}).setdefault(EPSILON, set()).add(dst)
            else:
                prev = src
                for i, c in enumerate(img):
                    nxt = dst if i == len(img) - 1 else _new_state("h")
                    if i != len(img) - 1: all_s.add(nxt)
                    trans.setdefault(prev, {}).setdefault(c, set()).add(nxt)
                    prev = nxt
    new_alpha = {c for v in h.values() for c in v}
    nfa = NFA(all_s, new_alpha, trans, so[dfa.start], {so[s] for s in dfa.accept})
    return nfa_to_dfa(nfa)


def op_right_quotient(a, symbol):
    """
    L/a — Cociente por la derecha con símbolo 'a'.
    L/a = { w | wa ∈ L }
    """
    dfa = a if isinstance(a, DFA) else nfa_to_dfa(a)
    new_acc = {
        s for s in dfa.states
        if dfa.transitions.get(s, {}).get(symbol) in dfa.accept
    }
    return DFA(dfa.states, dfa.alphabet, dfa.transitions, dfa.start, new_acc)


# ══════════════════════════════════════════════════════════════
# CONVERSIÓN AFD → EXPRESIÓN REGULAR  (Eliminación de estados)
# ══════════════════════════════════════════════════════════════
#
# Usado por las operaciones sobre lenguajes para entregar el
# RESULTADO como expresión regular (no como autómata).  Seguimos
# calculando con autómatas, pero al final colapsamos el AFD a una
# regex equivalente mediante el algoritmo clásico de eliminación
# de estados (state-elimination generalizado / Brzozowski).
# ══════════════════════════════════════════════════════════════

def _rx_has_toplevel_union(r):
    """¿La regex r contiene un '|' fuera de paréntesis?"""
    if r is None or len(r) <= 1:
        return False
    depth = 0
    for c in r:
        if   c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == '|' and depth == 0:
            return True
    return False


def _rx_is_single_group(r):
    """¿r es de la forma (…) donde el paréntesis externo cubre TODO r?"""
    if r is None or len(r) < 2 or r[0] != '(' or r[-1] != ')':
        return False
    depth = 0
    for i, c in enumerate(r):
        if   c == '(': depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0 and i < len(r) - 1:
                return False
    return True


def _rx_star(r):
    """Aplica clausura de Kleene:  r → r*  (añade paréntesis si hacen falta)."""
    if r is None or r == '' or r == EPSILON:
        return EPSILON
    if r.endswith('*'):                      # (r*)* = r*
        return r
    if len(r) == 1 or _rx_is_single_group(r):
        return r + '*'
    return f'({r})*'


def _rx_concat(a, b):
    """Concatena dos regex, parentizando operandos con '|' en el nivel superior."""
    if a is None or b is None:
        return None
    if a == EPSILON: return b
    if b == EPSILON: return a
    la = f'({a})' if _rx_has_toplevel_union(a) else a
    lb = f'({b})' if _rx_has_toplevel_union(b) else b
    return la + lb


def _rx_union(a, b):
    """Unión de dos regex (sin paréntesis externos)."""
    if a is None: return b
    if b is None: return a
    if a == b:    return a
    return f'{a}|{b}'


def dfa_to_regex(dfa):
    """
    Convierte un AFD a una expresión regular equivalente mediante el
    algoritmo de ELIMINACIÓN DE ESTADOS.

    Procedimiento:
      1. Se añade un estado inicial ficticio S y uno final ficticio E.
         S ──ε──► inicio_real       cada_aceptador ──ε──► E
      2. Si hay varios símbolos entre dos estados se agrupan con '|'.
      3. Se eliminan los estados reales uno a uno, componiendo:
             R(i,j)  ∪=  R(i,q) · R(q,q)* · R(q,j)
      4. La regex final es la etiqueta de S → E.

    Siempre devuelve una cadena:
      · lenguaje vacío  → '∅'
      · lenguaje {ε}    → 'ε'
    """
    if not dfa.states or not dfa.accept:
        return '∅'

    state_names = {str(s) for s in dfa.states}
    initial     = str(dfa.start)
    accept_set  = {str(s) for s in dfa.accept}

    if initial not in state_names:
        return '∅'

    # Agrupar (src,dst) → lista de símbolos
    trans_map: dict = {}
    for src, t in dfa.transitions.items():
        src_s = str(src)
        if src_s not in state_names:
            continue
        for sym, tgt in t.items():
            tgt_s = str(tgt)
            if tgt_s in state_names:
                trans_map.setdefault((src_s, tgt_s), []).append(sym)

    NEW_START, NEW_END = '__S__', '__E__'
    nodes = [NEW_START] + sorted(state_names) + [NEW_END]
    R = {i: {j: None for j in nodes} for i in nodes}

    # Transiciones ficticias
    R[NEW_START][initial] = EPSILON
    for acc in accept_set:
        R[acc][NEW_END] = EPSILON

    # Transiciones reales (varios símbolos → unidos con '|')
    for (src_s, tgt_s), syms in trans_map.items():
        syms = sorted(set(syms))
        R[src_s][tgt_s] = syms[0] if len(syms) == 1 else '|'.join(syms)

    # Eliminar estados reales uno por uno
    for q in sorted(state_names):
        loop = R[q][q]

        for i in nodes:
            if i == q or R[i][q] is None:
                continue
            riq = R[i][q]
            for j in nodes:
                if j == q or R[q][j] is None:
                    continue
                rqj    = R[q][j]
                middle = riq if loop is None else _rx_concat(riq, _rx_star(loop))
                new_rx = _rx_concat(middle, rqj)
                R[i][j] = _rx_union(R[i][j], new_rx)

        nodes.remove(q)   # q ya no se consulta más

    result = R[NEW_START][NEW_END]
    return result if result else '∅'


# ══════════════════════════════════════════════════════════════
# INGRESO MANUAL DE UN AFN
# ══════════════════════════════════════════════════════════════

def input_nfa_manual():
    """Permite al alumno definir un AFN o AFN-ε desde la consola."""
    print("\n─── Ingresar AFN / AFN-ε manualmente ───────────────────")
    raw_states = input("Estados (separados por coma, ej: q0,q1,q2): ").strip().split(',')
    states     = [s.strip() for s in raw_states]
    state_objs = {s: State(s) for s in states}

    raw_alpha  = input("Alfabeto (separado por coma, ej: a,b): ").strip().split(',')
    alphabet   = [a.strip() for a in raw_alpha]

    start_in   = input("Estado inicial: ").strip()
    raw_acc    = input("Estados de aceptación (separados por coma): ").strip().split(',')
    accept_in  = [s.strip() for s in raw_acc]

    print("\nIngresa las transiciones. Formato:  estado,símbolo,destinos")
    print("  • Usa 'ε' o '@' para épsilon")
    print("  • Para múltiples destinos sepáralos con ';'  (ej: q0,a,q1;q2)")
    print("  • Escribe 'fin' para terminar.\n")

    transitions = {}
    while True:
        line = input("  Transición: ").strip()
        if line.lower() == 'fin':
            break
        parts = line.split(',')
        if len(parts) != 3:
            print("  [!] Formato incorrecto. Usa:  estado,símbolo,destinos"); continue
        src_s, sym, dsts_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
        sym = EPSILON if sym in ('@', 'epsilon', 'eps', 'ε') else sym
        dst_list = [d.strip() for d in dsts_s.split(';')]
        src_obj  = state_objs.get(src_s)
        if not src_obj:
            print(f"  [!] Estado '{src_s}' no reconocido."); continue
        dst_objs = set()
        for d in dst_list:
            if d in state_objs: dst_objs.add(state_objs[d])
            else: print(f"  [!] Estado destino '{d}' no reconocido.")
        transitions.setdefault(src_obj, {}).setdefault(sym, set()).update(dst_objs)

    return NFA(
        {state_objs[s] for s in states},
        set(alphabet),
        transitions,
        state_objs[start_in],
        {state_objs[s] for s in accept_in}
    )


# ══════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL
# ══════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       Operaciones sobre Lenguajes Regulares                  ║
║  Construcción de Thompson  +  Algoritmo de Subconjuntos      ║
║  +  Minimización de AFD                                      ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
─── MENÚ PRINCIPAL ───────────────────────────────────────────
  [1]  Regex → AFN-ε         (Construcción de Thompson)
  [2]  AFN-ε → AFD           (Construcción de Subconjuntos)
  [3]  Ingresar AFN manualmente

  ── Operaciones de Definición ──────────────────────────────
  [4]  Unión          L1 ∪ L2
  [5]  Concatenación  L1 · L2
  [6]  Cierre Kleene  L*

  ── Propiedades de Cerradura ───────────────────────────────
  [7]  Intersección   L1 ∩ L2    (producto cartesiano)
  [8]  Complemento    L̄           (swap estados finales/no-finales)
  [9]  Diferencia     L1 − L2    (= L1 ∩ L̄2)

  ── Operaciones de Transformación ──────────────────────────
  [10] Reversa        L^R
  [11] Homomorfismo
  [12] Cociente der.  L / a

  ── Minimización ────────────────────────────────────────────
  [13] Minimizar AFD  (tabla de distinguibilidad)

  ── Utilidades ─────────────────────────────────────────────
  [14] Probar cadena en un autómata
  [15] Mostrar autómata guardado
  [0]  Salir
──────────────────────────────────────────────────────────────
"""


def main():
    print(BANNER)
    stored: dict[str, object] = {}

    def save(aut, default="A"):
        name = input(f"  Nombre para guardar [{default}]: ").strip() or default
        stored[name] = aut
        kind = "AFN-ε" if isinstance(aut, NFA) else "AFD"
        print(f"  ✓ {kind} guardado como '{name}'")
        return name

    def load(prompt="  Selecciona autómata"):
        if not stored:
            print("  [!] No hay autómatas guardados."); return None, None
        print(f"  Disponibles: {list(stored.keys())}")
        name = input(f"{prompt}: ").strip()
        aut  = stored.get(name)
        if aut is None: print(f"  [!] '{name}' no encontrado.")
        return aut, name

    def ensure_dfa(a, name):
        if isinstance(a, DFA): return a
        print(f"  ('{name}' es AFN-ε, convirtiéndolo a AFD automáticamente…)")
        return nfa_to_dfa(a)

    while True:
        print(MENU)
        choice = input("Opción: ").strip()

        if choice == '0':
            print("¡Hasta luego!"); break

        elif choice == '1':
            print("\nSintaxis: a·b→ab, unión→|, estrella→*, más→+, opc→?")
            print("Épsilon: ε o @     Agrupación: ()")
            regex = input("Expresión regular: ").strip()
            try:
                nfa = regex_to_nfa(regex)
                nfa.print_automaton(f"AFN-ε  ← '{regex}'  [Thompson]")
                save(nfa, f"nfa_{regex[:8]}")
            except Exception as e:
                print(f"  [Error] {e}")

        elif choice == '2':
            aut, name = load("  AFN-ε a convertir")
            if aut is None: continue
            if isinstance(aut, DFA):
                print("  [!] Ya es un AFD."); continue
            dfa = nfa_to_dfa(aut)
            dfa.print_automaton(f"AFD  ← '{name}'  [Subconjuntos]")
            save(dfa, f"dfa_{name}")

        elif choice == '3':
            nfa = input_nfa_manual()
            nfa.print_automaton("AFN ingresado manualmente")
            save(nfa, "nfa_manual")

        elif choice == '4':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_union(a1, a2)
            result.print_automaton(f"AFN-ε  L1∪L2  ({n1} ∪ {n2})")
            save(result, f"union_{n1}_{n2}")

        elif choice == '5':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_concat(a1, a2)
            result.print_automaton(f"AFN-ε  L1·L2  ({n1} · {n2})")
            save(result, f"concat_{n1}_{n2}")

        elif choice == '6':
            aut, name = load("  Autómata para L*")
            if aut is None: continue
            result = op_kleene(aut)
            result.print_automaton(f"AFN-ε  L*  ({name}*)")
            save(result, f"star_{name}")

        elif choice == '7':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_intersection(a1, a2)
            result.print_automaton(f"AFD  L1∩L2  ({n1} ∩ {n2})")
            save(result, f"inter_{n1}_{n2}")

        elif choice == '8':
            aut, name = load()
            if aut is None: continue
            result = op_complement(aut)
            result.print_automaton(f"AFD  Complemento  (¬{name})")
            save(result, f"comp_{name}")

        elif choice == '9':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_difference(a1, a2)
            result.print_automaton(f"AFD  L1−L2  ({n1} − {n2})")
            save(result, f"diff_{n1}_{n2}")

        elif choice == '10':
            aut, name = load()
            if aut is None: continue
            result = op_reverse(aut)
            result.print_automaton(f"AFD  Reversa  ({name}^R)")
            save(result, f"rev_{name}")

        elif choice == '11':
            aut, name = load()
            if aut is None: continue
            d = ensure_dfa(aut, name)
            print(f"  Alfabeto: {sorted(d.alphabet)}")
            print("  Define el homomorfismo  h(símbolo) = cadena")
            h = {}
            for sym in sorted(d.alphabet):
                img = input(f"    h({sym}) = ").strip()
                h[sym] = img
            result = op_homomorphism(d, h)
            result.print_automaton(f"AFD  Homomorfismo de '{name}'")
            save(result, f"hom_{name}")

        elif choice == '12':
            aut, name = load()
            if aut is None: continue
            d = ensure_dfa(aut, name)
            sym = input(f"  Símbolo  a  para  L/a  (alfabeto: {sorted(d.alphabet)}): ").strip()
            if sym not in d.alphabet:
                print(f"  [!] '{sym}' no está en el alfabeto."); continue
            result = op_right_quotient(d, sym)
            result.print_automaton(f"AFD  Cociente  {name}/{sym}")
            save(result, f"quot_{name}_{sym}")

        elif choice == '13':
            aut, name = load("  AFD a minimizar")
            if aut is None: continue
            d = ensure_dfa(aut, name)
            before = len(d.states)
            result = minimize_dfa(d)
            after  = len(result.states)
            saved  = before - after
            result.print_automaton(
                f"AFD Mínimo  ← '{name}'  "
                f"[{before} → {after} estados, −{saved} eliminados]"
            )
            save(result, f"min_{name}")

        elif choice == '14':
            aut, name = load()
            if aut is None: continue
            cadena = input("  Cadena a probar (Enter = cadena vacía ε): ")
            ok = aut.accepts(cadena)
            marca = "✓  ACEPTADA" if ok else "✗  RECHAZADA"
            print(f"\n  '{cadena if cadena else 'ε'}' → {marca} por '{name}'\n")

        elif choice == '15':
            aut, name = load()
            if aut is None: continue
            aut.print_automaton(name)

        else:
            print("  [!] Opción no válida.")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()