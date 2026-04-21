"""
╔══════════════════════════════════════════════════════════════╗
║       Operaciones sobre Lenguajes Regulares                  ║
║  Construcción de Thompson + Algoritmo de Subconjuntos        ║
╚══════════════════════════════════════════════════════════════╝

Operaciones implementadas:
  DEFINICIÓN   : Unión, Concatenación, Cierre de Kleene (L*)
  CERRADURA    : Intersección, Complemento, Diferencia
  TRANSFORMACIÓN: Reversa, Homomorfismo, Cociente por la derecha
"""

from collections import deque

# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════

EPSILON = 'ε'  # símbolo épsilon


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

    # ── Operaciones sobre conjuntos de estados ──────────────────

    def epsilon_closure(self, states):
        """Clausura-ε de un conjunto de estados."""
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
        """Estados alcanzables leyendo 'symbol' desde 'states'."""
        result = set()
        for s in states:
            result |= self.transitions.get(s, {}).get(symbol, set())
        return result

    def accepts(self, string):
        """Devuelve True si el AFN acepta 'string'."""
        current = self.epsilon_closure({self.start})
        for c in string:
            current = self.epsilon_closure(self.move(current, c))
        return bool(current & self.accept)

    # ── Visualización ───────────────────────────────────────────

    def print_automaton(self, title="AFN-ε"):
        print(f"\n{'═'*56}")
        print(f"  {title}")
        print(f"{'═'*56}")
        print(f"  Estados          : {sorted(self.states)}")
        print(f"  Alfabeto         : {sorted(self.alphabet)}")
        print(f"  Estado inicial   : {self.start}")
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
        """Devuelve True si el AFD acepta 'string'."""
        current = self.start
        for c in string:
            current = self.transitions.get(current, {}).get(c)
            if current is None:
                return False
        return current in self.accept

    def complete(self):
        """Devuelve un AFD completo agregando estado muerto si es necesario."""
        DEAD = "∅"
        new_trans   = {s: dict(t) for s, t in self.transitions.items()}
        needs_dead  = False
        for state in list(self.states):
            for sym in self.alphabet:
                if sym not in new_trans.get(state, {}):
                    new_trans.setdefault(state, {})[sym] = DEAD
                    needs_dead = True
        if needs_dead:
            new_trans[DEAD] = {sym: DEAD for sym in self.alphabet}
            return DFA(self.states | {DEAD}, self.alphabet, new_trans, self.start, self.accept)
        return DFA(self.states, self.alphabet, new_trans, self.start, self.accept)

    def print_automaton(self, title="AFD"):
        print(f"\n{'═'*56}")
        print(f"  {title}")
        print(f"{'═'*56}")
        print(f"  Estados          : {sorted(str(s) for s in self.states)}")
        print(f"  Alfabeto         : {sorted(self.alphabet)}")
        print(f"  Estado inicial   : {self.start}")
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
    """AFN-ε para un símbolo (o ε)."""
    s0, s1 = _new_state(), _new_state()
    trans  = {s0: {symbol: {s1}}}
    alpha  = set() if symbol == EPSILON else {symbol}
    return NFA({s0, s1}, alpha, trans, s0, {s1})


def thompson_union(n1, n2):
    """Thompson: AFN-ε para L1 ∪ L2."""
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
    """Thompson: AFN-ε para L1 · L2."""
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
    """Thompson: AFN-ε para L*  (Cierre de Kleene)."""
    s0, sf = _new_state(), _new_state()
    trans  = {}
    for s, t in n.transitions.items():
        trans[s] = {sym: set(sts) for sym, sts in t.items()}
    trans[s0] = {EPSILON: {n.start, sf}}
    for acc in n.accept:
        trans.setdefault(acc, {}).setdefault(EPSILON, set()).update({n.start, sf})
    return NFA(n.states | {s0, sf}, n.alphabet, trans, s0, {sf})


def thompson_plus(n):
    """Thompson: AFN-ε para L+  (una o más repeticiones)."""
    return thompson_concat(n, thompson_star(n))


def thompson_optional(n):
    """Thompson: AFN-ε para L?  (cero o una vez)."""
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
    """
    Convierte una expresión regular a un AFN-ε usando la Construcción de Thompson.

    Ejemplos de regex válidas:
      (a|b)*       → cadenas sobre {a,b}
      ab*c         → 'a' seguido de 0+ 'b' y una 'c'
      (0|1)+       → cadenas binarias no vacías
      a?(b|c)*     → opcional 'a' seguido de b/c repetidos

    reset_counter: pone a 0 el contador de estados (útil solo para demostraciones aisladas).
    """
    if reset_counter:
        State._counter = 0
    return _RegexParser(regex).parse()


# ══════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE SUBCONJUNTOS  (AFN-ε → AFD)
# ══════════════════════════════════════════════════════════════

def nfa_to_dfa(nfa):
    """
    Convierte un AFN-ε en un AFD mediante el Algoritmo de Construcción de Subconjuntos.

    Cada estado del AFD representa un subconjunto de estados del AFN.
    """
    ctr       = [0]
    state_map = {}              # frozenset(NFA states) → nombre DFA

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
# CONVERSIÓN DFA → NFA  (trivial, para reutilizar operaciones)
# ══════════════════════════════════════════════════════════════

def dfa_to_nfa(dfa):
    trans = {s: {sym: {tgt} for sym, tgt in t.items()}
             for s, t in dfa.transitions.items()}
    return NFA(dfa.states, dfa.alphabet, trans, dfa.start, dfa.accept)


# ══════════════════════════════════════════════════════════════
# OPERACIONES SOBRE LENGUAJES REGULARES
# ══════════════════════════════════════════════════════════════

# ── 1. OPERACIONES DE DEFINICIÓN ──────────────────────────────
# (Thompson ya implementa unión, concatenación y Kleene sobre NFAs)

def op_union(a1, a2):
    """L1 ∪ L2 — todas las cadenas en L1 o en L2."""
    n1 = a1 if isinstance(a1, NFA) else dfa_to_nfa(a1)
    n2 = a2 if isinstance(a2, NFA) else dfa_to_nfa(a2)
    return thompson_union(n1, n2)

def op_concat(a1, a2):
    """L1 · L2 — concatenación de cadenas."""
    n1 = a1 if isinstance(a1, NFA) else dfa_to_nfa(a1)
    n2 = a2 if isinstance(a2, NFA) else dfa_to_nfa(a2)
    return thompson_concat(n1, n2)

def op_kleene(a):
    """L* — cero o más repeticiones."""
    n = a if isinstance(a, NFA) else dfa_to_nfa(a)
    return thompson_star(n)


# ── 2. PROPIEDADES DE CERRADURA ───────────────────────────────

def op_intersection(a1, a2):
    """
    L1 ∩ L2 — Producto cartesiano de estados.

    Se obtiene un AFD cuyos estados son pares (q1, q2) del producto cartesiano.
    Un par es de aceptación solo si AMBOS componentes lo son.
    """
    d1 = (a1 if isinstance(a1, DFA) else nfa_to_dfa(a1)).complete()
    d2 = (a2 if isinstance(a2, DFA) else nfa_to_dfa(a2)).complete()
    alpha = d1.alphabet | d2.alphabet

    start = (d1.start, d2.start)
    queue = deque([start]); visited = set()
    trans = {}; acc = set()

    name = lambda p: f"({p[0]},{p[1]})"

    while queue:
        (s1, s2) = queue.popleft()
        if (s1, s2) in visited: continue
        visited.add((s1, s2))
        if s1 in d1.accept and s2 in d2.accept:
            acc.add(name((s1, s2)))
        for sym in sorted(alpha):
            t1 = d1.transitions.get(s1, {}).get(sym, "∅")
            t2 = d2.transitions.get(s2, {}).get(sym, "∅")
            nxt = (t1, t2)
            trans.setdefault(name((s1, s2)), {})[sym] = name(nxt)
            if nxt not in visited:
                queue.append(nxt)

    return DFA({name(p) for p in visited}, alpha, trans, name(start), acc)


def op_complement(a):
    """
    L̄  — Complemento: todas las cadenas sobre Σ que NO están en L.

    Se intercambian los estados finales por no finales en el AFD completo.
    """
    d = (a if isinstance(a, DFA) else nfa_to_dfa(a)).complete()
    return DFA(d.states, d.alphabet, d.transitions, d.start, d.states - d.accept)


def op_difference(a1, a2):
    """
    L1 − L2 = L1 ∩ L̄2  — cadenas en L1 pero no en L2.
    """
    return op_intersection(a1, op_complement(a2))


# ── 3. OPERACIONES DE TRANSFORMACIÓN ─────────────────────────

def op_reverse(a):
    """
    L^R — Reversa: todas las cadenas de L escritas al revés.

    Se invierten las transiciones, el antiguo estado inicial se vuelve
    el nuevo estado de aceptación, y los antiguos estados de aceptación
    se conectan vía ε a un nuevo estado inicial.
    """
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
    Homomorfismo: sustituye cada símbolo del alfabeto por una cadena de otro.

    h : dict  { 'a' -> 'xy', 'b' -> 'z', ... }
    """
    dfa = a if isinstance(a, DFA) else nfa_to_dfa(a)
    so  = {s: State(f"h_{s}") for s in dfa.states}     # renombrado
    all_s = set(so.values())
    trans = {}

    for state, t in dfa.transitions.items():
        for sym, tgt in t.items():
            img = h.get(sym, sym)
            src, dst = so[state], so[tgt]
            if not img:                                  # h(a) = ε
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

    Los nuevos estados de aceptación son aquellos desde los que leer
    'symbol' lleva a un estado de aceptación del AFD original.
    """
    dfa = a if isinstance(a, DFA) else nfa_to_dfa(a)
    new_acc = {
        s for s in dfa.states
        if dfa.transitions.get(s, {}).get(symbol) in dfa.accept
    }
    return DFA(dfa.states, dfa.alphabet, dfa.transitions, dfa.start, new_acc)


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

        src_obj = state_objs.get(src_s)
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

  ── Utilidades ─────────────────────────────────────────────
  [13] Probar cadena en un autómata
  [14] Mostrar autómata guardado
  [0]  Salir
──────────────────────────────────────────────────────────────
"""


def main():
    print(BANNER)
    stored: dict[str, object] = {}     # nombre → NFA | DFA

    # ── helpers ──────────────────────────────────────────────

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

    # ── bucle ─────────────────────────────────────────────────

    while True:
        print(MENU)
        choice = input("Opción: ").strip()

        # ── 0: Salir ─────────────────────────────────────────
        if choice == '0':
            print("¡Hasta luego!"); break

        # ── 1: Regex → AFN-ε ─────────────────────────────────
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

        # ── 2: AFN-ε → AFD ───────────────────────────────────
        elif choice == '2':
            aut, name = load("  AFN-ε a convertir")
            if aut is None: continue
            if isinstance(aut, DFA):
                print("  [!] Ya es un AFD."); continue
            dfa = nfa_to_dfa(aut)
            dfa.print_automaton(f"AFD  ← '{name}'  [Subconjuntos]")
            save(dfa, f"dfa_{name}")

        # ── 3: Ingreso manual ─────────────────────────────────
        elif choice == '3':
            nfa = input_nfa_manual()
            nfa.print_automaton("AFN ingresado manualmente")
            save(nfa, "nfa_manual")

        # ── 4: Unión ─────────────────────────────────────────
        elif choice == '4':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_union(a1, a2)
            result.print_automaton(f"AFN-ε  L1∪L2  ({n1} ∪ {n2})")
            save(result, f"union_{n1}_{n2}")

        # ── 5: Concatenación ──────────────────────────────────
        elif choice == '5':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_concat(a1, a2)
            result.print_automaton(f"AFN-ε  L1·L2  ({n1} · {n2})")
            save(result, f"concat_{n1}_{n2}")

        # ── 6: Cierre Kleene ──────────────────────────────────
        elif choice == '6':
            aut, name = load("  Autómata para L*")
            if aut is None: continue
            result = op_kleene(aut)
            result.print_automaton(f"AFN-ε  L*  ({name}*)")
            save(result, f"star_{name}")

        # ── 7: Intersección ───────────────────────────────────
        elif choice == '7':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_intersection(a1, a2)
            result.print_automaton(f"AFD  L1∩L2  ({n1} ∩ {n2})")
            save(result, f"inter_{n1}_{n2}")

        # ── 8: Complemento ────────────────────────────────────
        elif choice == '8':
            aut, name = load()
            if aut is None: continue
            result = op_complement(aut)
            result.print_automaton(f"AFD  Complemento  (¬{name})")
            save(result, f"comp_{name}")

        # ── 9: Diferencia ─────────────────────────────────────
        elif choice == '9':
            print("  Primer autómata (L1):"); a1, n1 = load()
            if a1 is None: continue
            print("  Segundo autómata (L2):"); a2, n2 = load()
            if a2 is None: continue
            result = op_difference(a1, a2)
            result.print_automaton(f"AFD  L1−L2  ({n1} − {n2})")
            save(result, f"diff_{n1}_{n2}")

        # ── 10: Reversa ───────────────────────────────────────
        elif choice == '10':
            aut, name = load()
            if aut is None: continue
            result = op_reverse(aut)
            result.print_automaton(f"AFD  Reversa  ({name}^R)")
            save(result, f"rev_{name}")

        # ── 11: Homomorfismo ──────────────────────────────────
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

        # ── 12: Cociente por la derecha ───────────────────────
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

        # ── 13: Probar cadena ─────────────────────────────────
        elif choice == '13':
            aut, name = load()
            if aut is None: continue
            cadena = input("  Cadena a probar (Enter = cadena vacía ε): ")
            ok = aut.accepts(cadena)
            marca = "✓  ACEPTADA" if ok else "✗  RECHAZADA"
            print(f"\n  '{cadena if cadena else 'ε'}' → {marca} por '{name}'\n")

        # ── 14: Mostrar autómata ──────────────────────────────
        elif choice == '14':
            aut, name = load()
            if aut is None: continue
            aut.print_automaton(name)

        else:
            print("  [!] Opción no válida.")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    main()