# -*- coding: utf-8 -*-

from collections import defaultdict, deque
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  Regex Parser  (string → postfix tokens)
# ══════════════════════════════════════════════════════════════════════════════

def parse_regex(regex: str) -> list:
    if not isinstance(regex, str):
        raise ValueError("La expresión regular debe ser una cadena de texto.")
    regex = regex.strip()
    if not regex:
        raise ValueError("La expresión regular no puede estar vacía.")
    regex = ''.join(regex.split())
    if '\\' in regex:
        # Antes se descartaba en silencio y 'ab\' se evaluaba como 'ab'.
        raise ValueError("El carácter de escape '\\' no está soportado en la expresión.")

    if regex[0] in '.|*+?' or regex[-1] in '.|(':
        raise ValueError(
            f"Expresión inválida: empieza con '{regex[0]}' o termina con '{regex[-1]}'")

    # Busca el grupo "(...)" que cierra al final de `processed`. Se busca
    # sobre la cadena YA procesada (no sobre la original) porque las
    # expansiones previas cambian su longitud: con la versión anterior
    # (a?b)? o (a+)+ recortaban caracteres equivocados y se rechazaban.
    def _grupo_final(processed: str, op: str) -> int:
        balance, j = 0, len(processed) - 1
        while j >= 0:
            if processed[j] == ')':
                balance += 1
            elif processed[j] == '(':
                balance -= 1
                if balance == 0:
                    return j
            j -= 1
        raise ValueError(f"Operador '{op}' sin grupo de apertura válido.")

    # a+ → aa*
    processed, i = '', 0
    while i < len(regex):
        char = regex[i]
        if char == '+' and i > 0:
            if processed.endswith(')'):
                j = _grupo_final(processed, '+')
                grp = processed[j:]
                processed = processed[:j] + grp + grp + '*'
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev = regex[i - 1]
                processed = processed[:-1]
                processed += prev + prev + '*'
            else:
                raise ValueError(f"Operador '+' inválido después de '{regex[i - 1]}'")
        else:
            processed += char
        i += 1
    regex = processed

    # a? → (a|ε)
    processed, i = '', 0
    while i < len(regex):
        char = regex[i]
        if char == '?' and i > 0:
            if processed.endswith(')'):
                j = _grupo_final(processed, '?')
                grp = processed[j:]
                processed = processed[:j] + f"({grp}|ε)"
            elif regex[i - 1].isalnum() or regex[i - 1] == 'ε':
                prev = regex[i - 1]
                processed = processed[:-1]
                processed += f"({prev}|ε)"
            else:
                raise ValueError(f"Operador '?' inválido después de '{regex[i - 1]}'")
        else:
            processed += char
        i += 1
    regex = processed

    # Implicit concatenation
    new_regex = ''
    alnum_eps = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ε'
    for i in range(len(regex)):
        new_regex += regex[i]
        if i + 1 < len(regex):
            c1, c2 = regex[i], regex[i + 1]
            if (c1 in alnum_eps or c1 in ')*+?') and (c2 in alnum_eps or c2 == '('):
                new_regex += '.'
    regex = new_regex

    # Validate
    paren_count, last_char = 0, None
    for char in regex:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
            if paren_count < 0:
                raise ValueError("Paréntesis desbalanceados.")
        if last_char is not None:
            if last_char in '.|' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}'")
            if last_char in '*+?' and char in '*+?(':
                raise ValueError(f"Operador inválido '{char}' después de '{last_char}'")
            if last_char == '(' and char in '.|*+?)':
                raise ValueError(f"Operador inválido '{char}' después de '('")
        last_char = char
    if paren_count != 0:
        raise ValueError("Paréntesis desbalanceados.")

    # Shunting-yard → postfix
    def precedence(op):
        return {'|': 1, '.': 2}.get(op, 3 if op in '*+?' else 0)

    tokens, stack = [], []
    for char in regex:
        if char.isalnum() or char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            tokens.append(('SYMBOL', char))
        elif char == 'ε':
            tokens.append(('EPSILON', char))
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                tokens.append(('OPERATOR', stack.pop()))
            if stack: stack.pop()
        elif char in '.|*+?':
            while stack and stack[-1] != '(' and precedence(stack[-1]) >= precedence(char):
                tokens.append(('OPERATOR', stack.pop()))
            stack.append(char)
    while stack:
        tokens.append(('OPERATOR', stack.pop()))
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
#  NFA Builder (Thompson construction)
# ══════════════════════════════════════════════════════════════════════════════

def build_nfa(tokens: list) -> tuple:
    state_counter = 0
    states        = set()
    transitions   = defaultdict(lambda: defaultdict(set))

    def new_state():
        nonlocal state_counter
        s = f"q{state_counter}"
        states.add(s)
        state_counter += 1
        return s

    stack = []
    for token_type, value in tokens:
        if token_type in ('SYMBOL', 'EPSILON'):
            start, end = new_state(), new_state()
            transitions[start][value if value != 'ε' else 'ε'].add(end)
            stack.append({'start': start, 'end': end})
        elif token_type == 'OPERATOR':
            if value == '.':
                r, l = stack.pop(), stack.pop()
                transitions[l['end']]['ε'].add(r['start'])
                stack.append({'start': l['start'], 'end': r['end']})
            elif value == '|':
                r, l = stack.pop(), stack.pop()
                ns, ne = new_state(), new_state()
                transitions[ns]['ε'].update([l['start'], r['start']])
                transitions[l['end']]['ε'].add(ne)
                transitions[r['end']]['ε'].add(ne)
                stack.append({'start': ns, 'end': ne})
            elif value == '*':
                nfa = stack.pop()
                ns, ne = new_state(), new_state()
                transitions[ns]['ε'].update([nfa['start'], ne])
                transitions[nfa['end']]['ε'].update([nfa['start'], ne])
                stack.append({'start': ns, 'end': ne})

    if len(stack) != 1:
        raise ValueError("Expresión regular inválida.")
    final = stack[0]
    return states, transitions, final['start'], final['end']


# ══════════════════════════════════════════════════════════════════════════════
#  NFA → DFA  (subset construction)
# ══════════════════════════════════════════════════════════════════════════════

def nfa_to_dfa(nfa_states, nfa_transitions, nfa_initial, nfa_final) -> tuple:
    eps_cache = {}

    def epsilon_closure(state_set):
        key = frozenset(state_set)
        if key in eps_cache:
            return eps_cache[key]
        closure = set(state_set)
        q = deque(state_set)
        while q:
            s = q.popleft()
            for ns in nfa_transitions.get(s, {}).get('ε', set()):
                if ns not in closure:
                    closure.add(ns)
                    q.append(ns)
        result = frozenset(closure)
        eps_cache[key] = result
        return result

    alphabet = set()
    for s in nfa_states:
        for sym in nfa_transitions.get(s, {}):
            if sym != 'ε':
                alphabet.add(sym)

    dfa_map, dfa_transitions, dfa_accepting = {}, {}, set()
    init_closure = epsilon_closure({nfa_initial})
    dfa_map[init_closure] = "D0"
    dfa_initial  = "D0"
    worklist     = deque([init_closure])
    if nfa_final in init_closure:
        dfa_accepting.add("D0")
    counter = 1

    while worklist:
        cur_set  = worklist.popleft()
        cur_name = dfa_map[cur_set]
        dfa_transitions[cur_name] = {}
        for sym in sorted(alphabet):
            next_direct = set()
            for s in cur_set:
                next_direct.update(nfa_transitions.get(s, {}).get(sym, set()))
            if not next_direct:
                continue
            next_closure = epsilon_closure(next_direct)
            if not next_closure:
                continue
            if next_closure not in dfa_map:
                new_name = f"D{counter}"
                counter += 1
                dfa_map[next_closure] = new_name
                worklist.append(next_closure)
                if nfa_final in next_closure:
                    dfa_accepting.add(new_name)
            dfa_transitions[cur_name][sym] = dfa_map[next_closure]

    dfa_states = list(dfa_map.values())
    trap = f"D{counter}"
    has_trap = False
    for s in list(dfa_states):
        if s not in dfa_transitions:
            dfa_transitions[s] = {}
        for sym in alphabet:
            if sym not in dfa_transitions[s]:
                if not has_trap:
                    dfa_states.append(trap)
                    dfa_transitions[trap] = {a: trap for a in alphabet}
                    has_trap = True
                dfa_transitions[s][sym] = trap

    return dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet


# ══════════════════════════════════════════════════════════════════════════════
#  DFA Minimizer (Hopcroft)
# ══════════════════════════════════════════════════════════════════════════════

def minimize_dfa(dfa_states, dfa_transitions, dfa_initial, dfa_accepting, alphabet) -> tuple:
    if not dfa_states:
        return [], {}, None, set()

    states       = set(dfa_states)
    accepting    = set(dfa_accepting)
    non_accepting = states - accepting
    partitions   = []
    if accepting:    partitions.append(accepting)
    if non_accepting: partitions.append(non_accepting)
    worklist = deque(partitions[:])

    while worklist:
        part = worklist.popleft()
        if not part:
            continue
        for sym in alphabet:
            preds = {s for s in states if dfa_transitions.get(s, {}).get(sym) in part}
            new_parts, changed = [], False
            for P in partitions:
                inter = P & preds
                diff  = P - preds
                if inter and diff:
                    new_parts += [inter, diff]
                    changed = True
                    if P in worklist:
                        worklist.remove(P)
                        worklist += [inter, diff]
                    else:
                        worklist.append(inter if len(inter) <= len(diff) else diff)
                else:
                    new_parts.append(P)
            if changed:
                partitions = new_parts

    min_states, state_reps, min_state_map = [], {}, {}
    min_accepting, min_initial = set(), None

    for i, partition in enumerate(partitions):
        if not partition:
            continue
        name = f"Min{i}"
        min_states.append(name)
        rep = next(iter(partition))
        state_reps[name] = rep
        for s in partition:
            min_state_map[s] = name
            if s == dfa_initial:
                min_initial = name
            if s in accepting:
                min_accepting.add(name)

    min_transitions = {}
    for name in min_states:
        rep = state_reps[name]
        min_transitions[name] = {}
        for sym in alphabet:
            dest = dfa_transitions.get(rep, {}).get(sym)
            if dest and dest in min_state_map:
                min_transitions[name][sym] = min_state_map[dest]

    if min_initial is None:
        return [], {}, None, set()
    reachable = {min_initial}
    q = deque([min_initial])
    while q:
        s = q.popleft()
        for sym in min_transitions.get(s, {}):
            ns = min_transitions[s][sym]
            if ns not in reachable:
                reachable.add(ns)
                q.append(ns)

    final_states      = [s for s in min_states if s in reachable]
    final_transitions = {s: {sym: d for sym, d in min_transitions[s].items()
                              if d in reachable}
                         for s in final_states}
    final_accepting   = {s for s in min_accepting if s in reachable}
    final_initial     = min_initial
    rename            = {old: f"M{i}" for i, old in enumerate(sorted(final_states))}
    return (
        list(rename.values()),
        {rename[s]: {sym: rename[d] for sym, d in final_transitions[s].items()
                      if d in rename}
         for s in final_states},
        rename.get(final_initial),
        {rename[s] for s in final_accepting if s in rename},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Automaton → JSON (Flutter)
# ══════════════════════════════════════════════════════════════════════════════

def automaton_to_json(states, transitions, initial, accepting, alphabet=None) -> dict:
    states_json = [
        {"id": s, "isInitial": s == initial, "isAccepting": s in accepting}
        for s in states
    ]
    edges_json = []
    for frm, trans in transitions.items():
        grouped: Dict[str, List[str]] = {}
        for sym, to in trans.items():
            grouped.setdefault(to, []).append(sym)
        for to, syms in grouped.items():
            edges_json.append({"from": frm, "to": to, "label": ",".join(sorted(syms))})
    return {
        "states": states_json,
        "edges":  edges_json,
        "alphabet": sorted(list(alphabet)) if alphabet else [],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Regex → Minimized DFA  (all-in-one)
# ══════════════════════════════════════════════════════════════════════════════

def regex_to_min_dfa_json(regex: str) -> dict:
    tokens    = parse_regex(regex)
    nfa_s, nfa_t, nfa_i, nfa_f = build_nfa(tokens)
    dfa_s, dfa_t, dfa_i, dfa_a, alpha = nfa_to_dfa(nfa_s, nfa_t, nfa_i, nfa_f)
    min_s, min_t, min_i, min_a = minimize_dfa(dfa_s, dfa_t, dfa_i, dfa_a, alpha)
    return automaton_to_json(min_s, min_t, min_i, min_a, alpha)


# ══════════════════════════════════════════════════════════════════════════════
#  ── Regex string helpers (GNFA state-elimination building blocks) ──────────
#
#  These produce cleaner intermediate expressions than the original code by:
#    • flattening nested unions    (a|(b|c)) → (a|b|c)
#    • avoiding double-wrapping    ((a|b))* → (a|b)*
#    • stripping trivial parens    (r)  →  r  when r is not a top-level union
# ══════════════════════════════════════════════════════════════════════════════

def _has_outer_parens(s: str) -> bool:
    """True if *s* is wrapped by matching outer parentheses."""
    if not (s.startswith('(') and s.endswith(')')):
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0:
            return i == len(s) - 1
    return False


def _top_level_split(s: str, sep: str) -> List[str]:
    """Split *s* by *sep* only at depth 0 (not inside parentheses)."""
    parts, depth, cur = [], 0, ''
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += c
    if cur or parts:
        parts.append(cur)
    return parts


def _get_union_parts(r: str) -> List[str]:
    """If *r* is a top-level union ``(a|b|c)``, return ``['a','b','c']``; else ``[r]``."""
    if _has_outer_parens(r):
        inner = r[1:-1]
        parts = _top_level_split(inner, '|')
        if len(parts) > 1:
            return parts
    return [r]


def _wrap(r: str) -> str:
    """
    Wrap *r* in parentheses only when needed inside a *concatenation* context,
    i.e. when *r* is a top-level union.  Strips redundant outer parens otherwise.
    """
    if not r or r == 'ε' or len(r) == 1:
        return r
    # If r has outer parens, check whether the inner content is a union
    if _has_outer_parens(r):
        inner = r[1:-1]
        if len(_top_level_split(inner, '|')) > 1:
            return r          # Union inside parens — keep them
        return inner          # Not a union — strip parens
    # No outer parens: if r is a union at top level, wrap it
    if len(_top_level_split(r, '|')) > 1:
        return f'({r})'
    return r


def _concat(r1: Optional[str], r2: Optional[str]) -> Optional[str]:
    if r1 is None or r2 is None:
        return None
    if r1 == 'ε':
        return r2
    if r2 == 'ε':
        return r1
    return _wrap(r1) + _wrap(r2)


def _union(r1: Optional[str], r2: Optional[str]) -> Optional[str]:
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    if r1 == r2:
        return r1
    # Flatten nested unions
    parts1 = _get_union_parts(r1)
    parts2 = _get_union_parts(r2)
    seen   = dict.fromkeys(parts1 + parts2)   # ordered dedup
    combined = list(seen)
    if len(combined) == 1:
        return combined[0]
    return '(' + '|'.join(combined) + ')'


def _star(r: Optional[str]) -> str:
    if r is None or r == 'ε':
        return 'ε'
    if len(r) == 1:
        return f'{r}*'
    # If r already has outer parens (e.g. (a|b)), use r* directly
    if _has_outer_parens(r):
        # Check for idempotent star  (r*)* → r*
        inner = r[1:-1]
        if inner.endswith('*'):
            return r[1:-1]    # strip outer parens  → r* unchanged
        return f'{r}*'
    # If r is already starred, it's idempotent
    if r.endswith('*'):
        return r
    return f'({r})*'


# ══════════════════════════════════════════════════════════════════════════════
#  ── Regex AST for factored simplification ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

class _Sym:
    __slots__ = ('v',)
    def __init__(self, v): self.v = v

class _Eps:
    pass

class _Empty:
    pass

class _Star:
    __slots__ = ('c',)
    def __init__(self, c): self.c = c

class _Cat:
    __slots__ = ('ch',)
    def __init__(self, ch): self.ch = list(ch)

class _Alt:
    __slots__ = ('ch',)
    def __init__(self, ch): self.ch = list(ch)

_RNode = (_Sym, _Eps, _Empty, _Star, _Cat, _Alt)

# ── parser ────────────────────────────────────────────────────────────────────

def _parse_re_str(s: str):
    """Parse a regex string (GNFA output) into an AST."""
    chars = list(s)
    pos   = [0]

    def cur():
        return chars[pos[0]] if pos[0] < len(chars) else None

    def eat():
        c = chars[pos[0]]; pos[0] += 1; return c

    def parse_alt():
        alts = [parse_cat()]
        while cur() == '|':
            eat()
            alts.append(parse_cat())
        return _Alt(alts) if len(alts) > 1 else alts[0]

    def parse_cat():
        items = []
        while cur() is not None and cur() not in (')', '|'):
            items.append(parse_post())
        if not items:
            return _Eps()
        return _Cat(items) if len(items) > 1 else items[0]

    def parse_post():
        base = parse_atom()
        while cur() == '*':
            eat(); base = _Star(base)
        return base

    def parse_atom():
        c = cur()
        if c is None:
            return _Eps()
        if c == '(':
            eat(); node = parse_alt()
            if cur() == ')': eat()
            return node
        if c == 'ε':
            eat(); return _Eps()
        if c == '∅':
            eat(); return _Empty()
        eat(); return _Sym(c)

    return parse_alt()

# ── emitter ───────────────────────────────────────────────────────────────────

def _emit_re(node) -> str:
    if isinstance(node, _Sym):
        return node.v
    if isinstance(node, _Eps):
        return 'ε'
    if isinstance(node, _Empty):
        return '∅'
    if isinstance(node, _Star):
        inner = _emit_re(node.c)
        if isinstance(node.c, (_Alt, _Cat)):
            return f'({inner})*'
        return f'{inner}*'
    if isinstance(node, _Cat):
        parts = []
        for ch in node.ch:
            s = _emit_re(ch)
            if isinstance(ch, _Alt):
                s = f'({s})'
            parts.append(s)
        return ''.join(parts)
    if isinstance(node, _Alt):
        return '|'.join(_emit_re(c) for c in node.ch)
    return ''

# ── simplifier ────────────────────────────────────────────────────────────────

def _simp(node, depth: int = 0):
    """Bottom-up simplification with prefix factoring."""
    if depth > 60:
        return node

    if isinstance(node, (_Sym, _Eps, _Empty)):
        return node

    # ── Star ──────────────────────────────────────────────────────────────────
    if isinstance(node, _Star):
        c = _simp(node.c, depth + 1)
        if isinstance(c, (_Eps, _Empty)):
            return _Eps()
        if isinstance(c, _Star):
            return c                          # (r*)* → r*
        # (r|ε)* → r*
        if isinstance(c, _Alt):
            non_eps = [x for x in c.ch if not isinstance(x, _Eps)]
            has_eps = len(non_eps) < len(c.ch)
            if has_eps:
                if not non_eps:
                    return _Eps()
                core = non_eps[0] if len(non_eps) == 1 else _simp(_Alt(non_eps), depth + 1)
                return _Star(core)
        return _Star(c)

    # ── Concatenation ─────────────────────────────────────────────────────────
    if isinstance(node, _Cat):
        children = [_simp(c, depth + 1) for c in node.ch]
        flat = []
        for c in children:
            if isinstance(c, _Cat):
                flat.extend(c.ch)
            else:
                flat.append(c)
        flat = [c for c in flat if not isinstance(c, _Eps)]
        if any(isinstance(c, _Empty) for c in flat):
            return _Empty()
        if not flat:
            return _Eps()
        return flat[0] if len(flat) == 1 else _Cat(flat)

    # ── Alternation ───────────────────────────────────────────────────────────
    if isinstance(node, _Alt):
        children = [_simp(c, depth + 1) for c in node.ch]
        flat = []
        for c in children:
            if isinstance(c, _Alt):
                flat.extend(c.ch)
            else:
                flat.append(c)
        # Remove ∅
        flat = [c for c in flat if not isinstance(c, _Empty)]
        # Deduplicate by emitted string
        seen: dict = {}
        for c in flat:
            k = _emit_re(c)
            if k not in seen:
                seen[k] = c
        flat = list(seen.values())
        if not flat:
            return _Empty()
        if len(flat) == 1:
            return flat[0]
        # r·r*|ε  →  r*   (i.e. r⁺|ε = r*)
        if len(flat) == 2:
            for i in range(2):
                ci, other = flat[i], flat[1 - i]
                if isinstance(other, _Eps) and isinstance(ci, _Cat) and len(ci.ch) == 2:
                    h, t = ci.ch
                    if isinstance(t, _Star) and _emit_re(h) == _emit_re(t.c):
                        return t   # h·h* → h*
        # Prefix factoring
        flat = _factor_prefix(flat, depth)
        if len(flat) == 1:
            return flat[0]
        return _Alt(flat)

    return node


def _head(node):
    """First symbol of a node (for prefix grouping)."""
    if isinstance(node, _Cat) and node.ch:
        return node.ch[0]
    return node


def _tail(node):
    """Rest of a _Cat node after the first child, or _Eps."""
    if isinstance(node, _Cat) and len(node.ch) > 1:
        t = node.ch[1:]
        return t[0] if len(t) == 1 else _Cat(t)
    return _Eps()


def _factor_prefix(alts: list, depth: int) -> list:
    """Group alternatives by their first symbol and factor common prefixes."""
    if len(alts) <= 1 or depth > 55:
        return alts
    groups: Dict[str, tuple] = {}  # head_key → (head_node, [tail_nodes])
    order: List[str] = []
    for alt in alts:
        h    = _head(alt)
        hkey = _emit_re(h)
        if hkey not in groups:
            groups[hkey] = (h, [])
            order.append(hkey)
        groups[hkey][1].append(_tail(alt))

    if all(len(groups[k][1]) == 1 for k in order):
        return alts    # No common prefix — nothing to factor

    result = []
    for hkey in order:
        h, tails = groups[hkey]
        if len(tails) == 1:
            t = tails[0]
            result.append(h if isinstance(t, _Eps)
                          else (_simp(_Cat([h, t]), depth + 1)))
        else:
            eps_tails  = [t for t in tails if isinstance(t, _Eps)]
            real_tails = [t for t in tails if not isinstance(t, _Eps)]
            if not real_tails:
                result.append(h)
            else:
                tail_node = real_tails[0] if len(real_tails) == 1 else _simp(_Alt(real_tails), depth + 1)
                if eps_tails:
                    # h followed by optional tail
                    tail_node = _simp(_Alt([tail_node, _Eps()]), depth + 1)
                result.append(_simp(_Cat([h, tail_node]), depth + 1))

    return result


def simplify_regex(r: str) -> str:
    """
    Parse a GNFA-output regex string into an AST, apply simplification rules
    iteratively until a fixpoint, then re-emit.

    Simplifications applied:
      • Flatten nested unions/concatenations
      • Remove ∅ and ε identity elements
      • Idempotent star:  (r*)* → r*,  ε* → ε
      • Absorb epsilon in star:  (r|ε)* → r*
      • Deduplicate alternatives
      • Factor common prefixes in unions:  ab|ac → a(b|c)
    """
    if r is None or r == '∅':
        return '∅'
    try:
        ast = _parse_re_str(r)
        prev_str = ''
        for _ in range(12):
            ast      = _simp(ast)
            cur_str  = _emit_re(ast)
            if cur_str == prev_str:
                break
            prev_str = cur_str
        return prev_str or r
    except Exception:
        return r   # fallback: return original if anything goes wrong


# ══════════════════════════════════════════════════════════════════════════════
#  Automaton → Regular Expression  (state elimination / GNFA)
# ══════════════════════════════════════════════════════════════════════════════

def automaton_to_regex(states: list, transitions: dict, initial: str,
                       accepting: set, alphabet: set = None) -> str:
    """
    Convert a DFA/NFA to a regular expression using the Generalised NFA
    (state-elimination) algorithm.

    The raw GNFA result is post-processed by ``simplify_regex`` to produce
    a compact, factored expression.
    """
    if not states:
        raise ValueError("El autómata no tiene estados.")
    if not initial:
        raise ValueError("El autómata no tiene estado inicial.")
    if not accepting:
        raise ValueError("El autómata no tiene estados de aceptación.")

    # Filter unreachable states
    reachable = {initial}
    queue = deque([initial])
    while queue:
        s = queue.popleft()
        for sym, dest in transitions.get(s, {}).items():
            dests = [dest] if isinstance(dest, str) else list(dest)
            for d in dests:
                if d not in reachable:
                    reachable.add(d)
                    queue.append(d)

    states   = [s for s in states   if s in reachable]
    accepting = {s for s in accepting if s in reachable}

    QS = '__qs__'
    QA = '__qa__'
    all_states = [QS] + list(states) + [QA]
    gnfa: Dict[str, Dict[str, Optional[str]]] = {
        s: {t: None for t in all_states} for s in all_states
    }
    gnfa[QS][initial] = 'ε'
    for acc in accepting:
        gnfa[acc][QA] = _union(gnfa[acc][QA], 'ε')

    for frm, trans in transitions.items():
        if frm not in reachable or not isinstance(trans, dict):
            continue
        for sym, dest in trans.items():
            raw_syms = [s.strip() for s in sym.split(',') if s.strip()]
            dests    = [dest] if isinstance(dest, str) else list(dest)
            for rsym in raw_syms:
                sym_label = 'ε' if rsym in ('ε', '') else rsym
                for to in dests:
                    if to not in reachable:
                        continue
                    gnfa[frm][to] = _union(gnfa[frm][to], sym_label)

    for elim in list(states):
        loop     = gnfa[elim][elim]
        star_lp  = _star(loop)
        remaining = [s for s in all_states if s != elim]

        for qi in remaining:
            for qj in remaining:
                r_in  = gnfa[qi][elim]
                r_out = gnfa[elim][qj]
                if r_in is None or r_out is None:
                    continue
                mid      = _concat(star_lp if star_lp != 'ε' else None, r_out)
                new_path = _concat(r_in, mid if mid else r_out)
                gnfa[qi][qj] = _union(gnfa[qi][qj], new_path)

        all_states.remove(elim)
        del gnfa[elim]
        for s in all_states:
            if elim in gnfa[s]:
                del gnfa[s][elim]

    result = gnfa[QS][QA]
    if result is None:
        return '∅'

    # Post-process: AST-based simplification
    return simplify_regex(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    # La consola de Windows (cp1252) no imprime → ni ε.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Test automaton_to_regex
    states = ['M0', 'M1', 'M2']
    transitions = {
        'M0': {'a': 'M0', 'b': 'M1'},
        'M1': {'a': 'M2', 'b': 'M2'},
        'M2': {'a': 'M2', 'b': 'M2'},
    }
    regex = automaton_to_regex(states, transitions, 'M0', {'M1'}, {'a', 'b'})
    print(f"a*b  → {regex}")

    # Test with a*(b|c)
    result = regex_to_min_dfa_json('a*(b|c)')
    print(f"a*(b|c) DFA states: {[s['id'] for s in result['states']]}")

    # Test simplify_regex directly
    tests = [
        # simplify_regex ya no conserva los paréntesis externos redundantes.
        ('((a|b)|(c|d))', 'a|b|c|d'),
        ('(ε|a)*',        'a*'),
        ('(a*)*',         'a*'),
        ('(a)(b)',        'ab'),
    ]
    print("\nSimplification tests:")
    for inp, expected in tests:
        got = simplify_regex(inp)
        ok  = '✓' if got == expected else f'✗ (expected {expected})'
        print(f"  {inp!r:25s} → {got!r:20s} {ok}")