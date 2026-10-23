# -*- coding: utf-8 -*-
"""
Paridad backend (Python) ↔ app (Dart).

La app usa la API como primera opción y, si no responde, un respaldo local en
Dart. Este script ejecuta los MOTORES DEL BACKEND sobre una batería de casos
(ejercicios de la app, casos aleatorios con semilla fija y casos borde) y
guarda los resultados en

    autechvapis/test/fixtures/paridad_backend.json

El test autechvapis/test/paridad_backend_test.dart corre los mismos casos con
los motores locales de la app y exige resultados idénticos. Además, este
script comprueba que los resultados del backend sean CORRECTOS contra un
oráculo independiente (p. ej. el módulo `re` de Python, o la gramática
generada contra la simulación del APD) y falla si encuentra un error.

Uso (desde la raíz del repo):
    python backend/tools/paridad_app.py
"""
import asyncio
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from fastapi import HTTPException  # noqa: E402

import api_server as api  # noqa: E402
from pda_logic import PDA, PDASimulator, convert_pda_to_cfg, _is_cfg_nt  # noqa: E402

FIXTURE = ROOT / 'autechvapis' / 'test' / 'fixtures' / 'paridad_backend.json'
EJERCICIOS = ROOT / 'autechvapis' / 'lib' / 'simulador' / 'ejercicios' / 'ejercicios_simulador.dart'

ERRORES_CORRECCION = []


def fallo(msg):
    ERRORES_CORRECCION.append(msg)


def cadenas(alfabeto, max_len):
    out = []
    for n in range(max_len + 1):
        out += [''.join(p) for p in itertools.product(sorted(alfabeto), repeat=n)]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Lectura de los ejercicios de la app (ejercicios_simulador.dart)
# ══════════════════════════════════════════════════════════════════════════════

def _bloques(src, clase):
    partes = src.split(f'  {clase}(')[1:]
    return [p.split('\n  ),')[0] for p in partes]


def _campo(bloque, nombre):
    m = re.search(rf"\b{nombre}: '([^']*)'", bloque)
    return m.group(1) if m else None


def _lista(bloque, nombre):
    m = re.search(rf"\b{nombre}: \[(.*?)\]", bloque, re.S)
    return re.findall(r"'([^']*)'", m.group(1)) if m else []


def ejercicios():
    src = EJERCICIOS.read_text(encoding='utf-8')
    pda = [{
        'id': _campo(b, 'id'),
        'states': _campo(b, 'estados'),
        'inputAlphabet': _campo(b, 'alfabetoEntrada'),
        'stackAlphabet': _campo(b, 'alfabetoPila'),
        'startState': _campo(b, 'estadoInicial'),
        'startSymbol': _campo(b, 'simboloInicial'),
        'acceptStates': _campo(b, 'estadosFinales'),
        'transitions': '\n'.join(_lista(b, 'transiciones')),
        'extra': _lista(b, 'aceptadas') + _lista(b, 'rechazadas'),
    } for b in _bloques(src, 'EjercicioPda')]
    regex = [{'id': _campo(b, 'id'), 'regex': _campo(b, 'regex')}
             for b in _bloques(src, 'EjercicioRegex')]
    leng = []
    for b in _bloques(src, 'EjercicioLenguajes'):
        leng.append({
            'id': _campo(b, 'id'),
            'destino': re.search(r'DestinoLenguajes\.(\w+)', b).group(1),
            'estados': _lista(b, 'estados'),
            'finales': _lista(b, 'finales'),
            'transiciones': _lista(b, 'transiciones'),
            'operacion': _campo(b, 'operacion'),
            'l1': _campo(b, 'l1'),
            'l2': _campo(b, 'l2'),
        })
    return pda, regex, leng


# ══════════════════════════════════════════════════════════════════════════════
#  PDA
# ══════════════════════════════════════════════════════════════════════════════

def _veredicto(pda, w):
    r = PDASimulator(pda).simulate(w)
    if r['accepted']:
        return 'accepted'
    return 'inconclusive' if r.get('inconclusive') else 'rejected'


def _parse_cfg_text(text):
    prods, actual = {}, None
    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            continue
        if '→' in line and not line.startswith('|'):
            actual, rhs = line.split('→', 1)
            actual = actual.strip()
        elif line.startswith('|') and actual is not None:
            rhs = line[1:]
        else:
            continue
        for alt in rhs.split(' | '):
            syms = alt.split()
            if syms:
                prods.setdefault(actual, []).append(syms)
    return prods


def _cfg_lenguaje(prods, max_len):
    lang = {nt: set() for nt in prods}
    cambio = True
    while cambio:
        cambio = False
        for nt, alts in prods.items():
            for alt in alts:
                parcial = {''}
                for s in alt:
                    if s == 'ε':
                        continue
                    opciones = lang.get(s, set()) if _is_cfg_nt(s) else {s}
                    parcial = {p + o for p in parcial for o in opciones
                               if len(p) + len(o) <= max_len}
                    if not parcial:
                        break
                for w in parcial:
                    if w not in lang[nt]:
                        lang[nt].add(w)
                        cambio = True
    return sorted(lang.get('S', set()))


def caso_pda(cid, d, max_len, extra=(), cfg_len=4):
    pda = PDA()
    caso = {'id': cid, 'def': {k: d[k] for k in (
        'states', 'inputAlphabet', 'stackAlphabet', 'startState',
        'startSymbol', 'acceptStates', 'transitions')}}
    try:
        pda.parse(d['states'], d['inputAlphabet'], d['stackAlphabet'],
                  d['startState'], d['startSymbol'], d['acceptStates'],
                  d['transitions'])
    except ValueError as e:
        caso['errors'] = sorted(str(e).split('\n'))
        return caso
    caso['errors'] = []
    simbolos = sorted(s for s in pda.input_alphabet if len(s) == 1)
    entradas = list(dict.fromkeys(cadenas(simbolos, max_len) + list(extra)))
    caso['verdicts'] = {w: _veredicto(pda, w) for w in entradas}
    texto = convert_pda_to_cfg(pda)
    caso['cfg'] = texto
    lenguaje = _cfg_lenguaje(_parse_cfg_text(texto), cfg_len)
    caso['cfgLanguage'] = lenguaje
    # Corrección: la gramática debe generar EXACTAMENTE las cadenas que el
    # APD acepta (salvo las que la búsqueda no pudo decidir).
    for w in cadenas(simbolos, cfg_len):
        v = caso['verdicts'].get(w) or _veredicto(pda, w)
        if v == 'inconclusive':
            continue
        if (v == 'accepted') != (w in lenguaje):
            fallo(f'PDA {cid}: la CFG {"no genera" if v == "accepted" else "genera"} '
                  f'"{w}" pero el APD la {"acepta" if v == "accepted" else "rechaza"}')
    return caso


def pda_aleatorio(rng, n):
    estados = [f'q{i}' for i in range(rng.randint(1, 4))]
    pushes = ['ε', 'A', 'B', 'Z', 'AZ', 'AA', 'BA', 'AB', 'BB', 'BZ']
    reglas = []
    for _ in range(rng.randint(2, 7)):
        reglas.append(f"{rng.choice(estados)},{rng.choice(['a', 'b', 'ε'])},"
                      f"{rng.choice(['A', 'B', 'Z', 'ε'])} -> "
                      f"{rng.choice(estados)},{rng.choice(pushes)}")
    return {
        'states': ','.join(estados), 'inputAlphabet': 'a,b',
        'stackAlphabet': 'A,B,Z', 'startState': 'q0', 'startSymbol': 'Z',
        'acceptStates': ','.join(rng.sample(estados, rng.randint(1, len(estados)))),
        'transitions': '\n'.join(reglas),
    }


PDA_BORDE = [
    # λ como sinónimo de ε (entrada, pop y push)
    dict(states='q0,q1', inputAlphabet='a', stackAlphabet='Z', startState='q0',
         startSymbol='Z', acceptStates='q1', transitions='q0,a,Z -> q0,λ\nq0,λ,λ -> q1,λ'),
    # Γ vacío aunque haya Z₀
    dict(states='q0', inputAlphabet='a', stackAlphabet='', startState='q0',
         startSymbol='Z', acceptStates='q0', transitions='q0,a,Z -> q0,Z'),
    # Errores varios
    dict(states='q0', inputAlphabet='a', stackAlphabet='Z', startState='q9',
         startSymbol='Z', acceptStates='qx', transitions='q0,b,Z -> q1,Z\nbasura\nq0,a,Y -> q0,Z'),
    dict(states='q0', inputAlphabet='a', stackAlphabet='Z', startState='q0',
         startSymbol='Z', acceptStates='q0', transitions='q0,a,Z -> q0,ZX'),
    dict(states='', inputAlphabet='a', stackAlphabet='Z', startState='',
         startSymbol='', acceptStates='', transitions=''),
    # Símbolos de pila de varios caracteres, comentarios, CRLF, duplicados
    dict(states='p,q', inputAlphabet='a,b', stackAlphabet='Z0,A', startState='p',
         startSymbol='Z0', acceptStates='q',
         transitions='# comentario\r\np,a,Z0 -> p,AZ0\r\np,a,A -> p,AA\r\np,a,A -> p,AA\r\n'
                     'p,b,A -> q,ε\r\nq,b,A -> q,ε\r\nq,ε,Z0 -> q,Z0'),
    # Estado llamado q_ini (fuerza el nombre q_ini' en la CFG)
    dict(states='q_ini,q_vac', inputAlphabet='a', stackAlphabet='Z', startState='q_ini',
         startSymbol='Z', acceptStates='q_vac', transitions='q_ini,a,Z -> q_vac,Z'),
    # Sin estados de aceptación
    dict(states='q0', inputAlphabet='a', stackAlphabet='Z', startState='q0',
         startSymbol='Z', acceptStates='', transitions='q0,a,Z -> q0,Z'),
    # Pop ε con pila vacía y bucle ε que apila (límite de seguridad)
    dict(states='q0,q1', inputAlphabet='a,b', stackAlphabet='A,Z', startState='q0',
         startSymbol='Z', acceptStates='q1',
         transitions='q0,a,Z -> q0,ε\nq0,b,ε -> q1,ε\nq1,ε,ε -> q1,A'),
]


def seccion_pda(pda_ej):
    casos = []
    for e in pda_ej:
        casos.append(caso_pda(e['id'], e, 5, e['extra']))
    for i, d in enumerate(PDA_BORDE):
        casos.append(caso_pda(f'borde_{i}', d, 4))
    rng = random.Random(20261022)
    for i in range(45):
        casos.append(caso_pda(f'aleatorio_{i}', pda_aleatorio(rng, i), 4))
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  Regex → AFD mínimo
# ══════════════════════════════════════════════════════════════════════════════

def regex_a_re(r):
    """Sintaxis de la app → `re` de Python ('.' de la app no se usa aquí)."""
    return r.replace('ε', '(?:)').replace('∅', '(?!)')


def oraculo_regex(r, w):
    return re.fullmatch(regex_a_re(r), w) is not None


def acepta_grafo(g, w):
    ini = next((s['id'] for s in g['states'] if s['isInitial']), None)
    fin = {s['id'] for s in g['states'] if s['isAccepting']}
    delta = {}
    for e in g['edges']:
        for sym in e['label'].split(','):
            delta[(e['from'], sym.strip())] = e['to']
    cur = ini
    for ch in w:
        cur = delta.get((cur, ch))
        if cur is None:
            return False
    return cur in fin


def regex_aleatoria(rng, prof=3):
    if prof == 0 or rng.random() < 0.25:
        return rng.choice(['a', 'b', 'a', 'b', 'ε'])
    op = rng.choice(['cat', 'cat', 'alt', 'star', 'plus', 'opt'])
    if op == 'cat':
        return regex_aleatoria(rng, prof - 1) + regex_aleatoria(rng, prof - 1)
    if op == 'alt':
        return f'({regex_aleatoria(rng, prof - 1)}|{regex_aleatoria(rng, prof - 1)})'
    sub = regex_aleatoria(rng, prof - 1)
    sub = sub if len(sub) == 1 else f'({sub})'
    return sub + {'star': '*', 'plus': '+', 'opt': '?'}[op]


REGEX_BORDE = [
    'a', 'ε', 'ε*', 'a**', 'a*+', '', '   ', '(a', 'a)', '()', 'a|', '|a', '*a',
    'a||b', '(|a)', 'a\\b', '(a?b)?', '(a+)+', '((a|b)?)+', 'a b', 'á+é',
    '0|1(0|1)*0', 'abc', 'a.b', 'a-b', '(ab|ε)*a', 'ε|a', 'a?b*c',
    '(a|b)*abb', '(ab)+', 'a(b|c)*', '(a|b)*', 'a*b',
]


def seccion_regex(regex_ej):
    lista = [(e['id'], e['regex']) for e in regex_ej]
    lista += [(f'borde_{i}', r) for i, r in enumerate(REGEX_BORDE)]
    rng = random.Random(7)
    lista += [(f'aleatoria_{i}', regex_aleatoria(rng)) for i in range(70)]
    casos = []
    for cid, r in lista:
        caso = {'id': cid, 'regex': r}
        try:
            g = api.regex_to_min_dfa_json(r)
        except ValueError as e:
            caso['error'] = str(e)
            casos.append(caso)
            continue
        except Exception:  # noqa: BLE001 — la API respondería 500
            caso['error'] = 'ERROR_INTERNO'
            casos.append(caso)
            continue
        caso['graph'] = g
        alfa = sorted(set(g['alphabet']) | ({'a', 'b'} if not g['alphabet'] else set()))
        caso['strings'] = cadenas(alfa, 6 if len(alfa) <= 2 else 4)
        caso['accepts'] = [acepta_grafo(g, w) for w in caso['strings']]
        # Corrección contra `re` (solo si la sintaxis es traducible).
        if all(ch in 'abc01ε|*+?()' for ch in r):
            for w, ok in zip(caso['strings'], caso['accepts']):
                if ok != oraculo_regex(r, w):
                    fallo(f'regex {cid} "{r}": el AFD {"acepta" if ok else "rechaza"} "{w}"')
        casos.append(caso)
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  Autómata → regex (eliminación de estados)
# ══════════════════════════════════════════════════════════════════════════════

def _trans_como_la_app(states, edges):
    """El mismo diccionario que arma la app (automaton_regex_engine.dart)."""
    trans = {}
    for e in edges:
        for sym in [s.strip() for s in e['label'].split(',') if s.strip()]:
            trans.setdefault(e['from'], {})[sym] = e['to']
    ini = next((s['id'] for s in states if s['isInitial']), states[0]['id'])
    acc = [s['id'] for s in states if s['isAccepting']]
    return [s['id'] for s in states], trans, ini, acc


def afd_aleatorio(rng):
    n = rng.randint(1, 5)
    ids = [f'p{i}' for i in range(n)]
    acc = set(rng.sample(ids, rng.randint(0, n)))
    states = [{'id': s, 'isInitial': i == 0, 'isAccepting': s in acc}
              for i, s in enumerate(ids)]
    edges = []
    for s in ids:
        destinos = {}
        for sym in 'ab':
            if rng.random() < 0.8:
                destinos.setdefault(rng.choice(ids), []).append(sym)
        for d, syms in destinos.items():
            edges.append({'from': s, 'to': d, 'label': ','.join(syms)})
    return states, edges


def seccion_a2r(regex_casos, leng_ej):
    fuentes = []
    for c in regex_casos:
        if 'graph' in c:
            fuentes.append((f"de_regex_{c['id']}", c['graph']['states'], c['graph']['edges']))
    for e in leng_ej:
        if e['destino'] == 'simplificacion':
            finales = set(e['finales'])
            states = [{'id': s, 'isInitial': i == 0, 'isAccepting': s in finales}
                      for i, s in enumerate(e['estados'])]
            edges = []
            for t in e['transiciones']:
                f, sym, to = t.split(',')
                edges.append({'from': f, 'to': to, 'label': sym})
            fuentes.append((e['id'], states, edges))
    rng = random.Random(99)
    for i in range(60):
        st, ed = afd_aleatorio(rng)
        fuentes.append((f'aleatorio_{i}', st, ed))

    casos = []
    for cid, states, edges in fuentes:
        ids, trans, ini, acc = _trans_como_la_app(states, edges)
        alfabeto = sorted({s for t in trans.values() for s in t})
        req = api.AutomatonToRegexRequest(
            states=ids, transitions=trans, initial=ini, accepting=acc,
            alphabet=alfabeto)
        regex = asyncio.run(api.automaton_to_regex_endpoint(req))['regex']
        caso = {'id': cid, 'states': ids, 'transitions': trans, 'initial': ini,
                'accepting': acc, 'regex': regex}
        casos.append(caso)
        # Corrección: la expresión describe el mismo lenguaje que el AFD.
        sim = {'states': states, 'edges': edges}
        for w in cadenas(alfabeto or ['a'], 6 if len(alfabeto) <= 2 else 4):
            if acepta_grafo(sim, w) != oraculo_regex(regex, w):
                fallo(f'a2r {cid}: "{regex}" no coincide con el AFD en "{w}"')
                break
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  Operaciones sobre lenguajes (POST /regex/operation)
# ══════════════════════════════════════════════════════════════════════════════

def _alfabeto_regex(r):
    return {c for c in r if c.isalnum() and c != 'ε'}


def _oraculo_op(op, r1, r2, mapping, symbol, w, alfa1):
    def en(r, s):
        return oraculo_regex(r, s)
    if op == 'union':
        return en(r1, w) or en(r2, w)
    if op == 'intersection':
        return en(r1, w) and en(r2, w)
    if op == 'difference':
        return en(r1, w) and not en(r2, w)
    if op == 'concat':
        return any(en(r1, w[:i]) and en(r2, w[i:]) for i in range(len(w) + 1))
    if op == 'kleene':
        return re.fullmatch(f'(?:{regex_a_re(r1)})*', w) is not None
    if op == 'complement':
        return all(c in alfa1 for c in w) and not en(r1, w)
    if op == 'reverse':
        return en(r1, w[::-1])
    if op == 'rightquotient':
        return en(r1, w + symbol)
    if op == 'homomorphism':
        # Solo homomorfismos que no borran: |u| ≤ |h(u)|.
        for n in range(len(w) + 1):
            for u in itertools.product(sorted(alfa1), repeat=n):
                if ''.join(mapping.get(c, c) for c in u) == w and en(r1, ''.join(u)):
                    return True
        return False
    raise AssertionError(op)


def seccion_ops(leng_ej):
    lista = []
    for e in leng_ej:
        if e['destino'] == 'operaciones':
            lista.append((e['id'], e['operacion'], e['l1'], e['l2'], None, None))
    rng = random.Random(1234)
    binarias = ['union', 'intersection', 'difference', 'concat']
    for i in range(80):
        op = rng.choice(binarias + ['kleene', 'complement', 'reverse',
                                    'homomorphism', 'rightquotient'])
        r1 = regex_aleatoria(rng, 2)
        r2 = regex_aleatoria(rng, 2) if op in binarias else None
        mapping = symbol = None
        if op == 'homomorphism':
            mapping = {c: rng.choice(['a', 'b', 'ab', 'ba', 'bb', 'c'])
                       for c in sorted(_alfabeto_regex(r1)) or ['a']}
        if op == 'rightquotient':
            symbol = rng.choice(['a', 'b'])
        lista.append((f'aleatoria_{i}', op, r1, r2, mapping, symbol))
    # Bordes: errores y lenguajes vacío / {ε}
    lista += [
        ('borde_sin_r2', 'union', 'a', None, None, None),
        ('borde_regex_invalida', 'union', 'a|', 'b', None, None),
        ('borde_op_desconocida', 'shuffle', 'a', 'b', None, None),
        ('borde_vacio', 'intersection', 'a', 'b', None, None),
        ('borde_epsilon', 'kleene', 'ε', None, None, None),
        ('borde_complemento_total', 'complement', '(a|b)*', None, None, None),
        ('borde_homo_borra', 'homomorphism', 'ab', None, {'a': '', 'b': 'b'}, None),
        ('borde_cociente_sin_simbolo', 'rightquotient', 'ab', None, None, ''),
    ]
    casos = []
    for cid, op, r1, r2, mapping, symbol in lista:
        caso = {'id': cid, 'op': op, 'regex1': r1, 'regex2': r2,
                'mapping': mapping, 'symbol': symbol}
        req = api.LanguageOperationRequest(operation=op, regex1=r1, regex2=r2,
                                           mapping=mapping, symbol=symbol)
        try:
            caso['regex'] = asyncio.run(api.regex_operation(req))['regex']
        except HTTPException as e:
            caso['error'] = e.detail if e.status_code == 400 else 'ERROR_INTERNO'
            casos.append(caso)
            continue
        casos.append(caso)
        # Corrección contra `re`.
        alfa1 = _alfabeto_regex(r1)
        alfa = alfa1 | _alfabeto_regex(r2 or '') | {c for v in (mapping or {}).values() for c in v}
        alfa = alfa or {'a'}
        borra = mapping is not None and any(v == '' for v in mapping.values())
        if borra:
            continue
        res = caso['regex']
        for w in cadenas(alfa, 5 if len(alfa) <= 2 else 3):
            esperado = _oraculo_op(op, r1, r2, mapping or {}, symbol, w, alfa1)
            obtenido = res != '∅' and oraculo_regex(res, w)
            if esperado != obtenido:
                fallo(f'op {cid} {op}({r1!r}, {r2!r}) = {res!r}: "{w}" '
                      f'debería {"estar" if esperado else "no estar"}')
                break
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  AFN-ε → AFD y minimización (pestañas "AFN → AFD" y "Simplificación")
# ══════════════════════════════════════════════════════════════════════════════

def afn_aleatorio(rng):
    n = rng.randint(1, 5)
    ids = [f'q{i}' for i in range(n)]
    fin = set(rng.sample(ids, rng.randint(0, n)))
    states = [{'id': s, 'isInitial': i == 0, 'isAccepting': s in fin}
              for i, s in enumerate(ids)]
    pares = {}
    for _ in range(rng.randint(1, 3 * n)):
        f, t = rng.choice(ids), rng.choice(ids)
        pares.setdefault((f, t), set()).add(rng.choice(['a', 'b', 'a', 'b', 'ε']))
    edges = [{'from': f, 'to': t, 'label': ','.join(sorted(s))}
             for (f, t), s in pares.items()]
    return states, edges


def _afn_acepta(states, edges, w):
    ini = next((s['id'] for s in states if s['isInitial']), states[0]['id'])
    fin = {s['id'] for s in states if s['isAccepting']}
    trans = {}
    for e in edges:
        for sym in [x.strip() for x in e['label'].split(',') if x.strip()]:
            trans.setdefault((e['from'], sym), set()).add(e['to'])

    def cierre(c):
        c, pila = set(c), list(c)
        while pila:
            s = pila.pop()
            for t in trans.get((s, 'ε'), ()):
                if t not in c:
                    c.add(t)
                    pila.append(t)
        return c
    cur = cierre({ini})
    for ch in w:
        cur = cierre({t for s in cur for t in trans.get((s, ch), ())})
    return bool(cur & fin)


def seccion_automatas(leng_ej):
    fuentes = []
    for e in leng_ej:
        if e['destino'] in ('afnAfd', 'simplificacion'):
            finales = set(e['finales'])
            states = [{'id': s, 'isInitial': i == 0, 'isAccepting': s in finales}
                      for i, s in enumerate(e['estados'])]
            pares = {}
            for t in e['transiciones']:
                f, sym, to = t.split(',')
                pares.setdefault((f, to), []).append(sym)
            edges = [{'from': f, 'to': t, 'label': ','.join(s)} for (f, t), s in pares.items()]
            fuentes.append((e['id'], states, edges))
    rng = random.Random(4321)
    for i in range(60):
        st, ed = afn_aleatorio(rng)
        fuentes.append((f'aleatorio_{i}', st, ed))

    casos = []
    for cid, states, edges in fuentes:
        alfabeto = sorted({x.strip() for e in edges for x in e['label'].split(',')
                           if x.strip() and x.strip() != 'ε'})
        if not alfabeto:
            continue  # la app no convierte autómatas sin símbolos
        dfa = api._graph_to_lr_dfa(states, edges, None)
        dfa_json = api._lr_to_json(dfa)
        min_json = api._lr_to_json(api.lr_minimize_dfa(dfa))
        casos.append({'id': cid, 'states': states, 'edges': edges,
                      'dfa': dfa_json, 'min': min_json})
        for w in cadenas(alfabeto, 5 if len(alfabeto) <= 2 else 3):
            esperado = _afn_acepta(states, edges, w)
            if acepta_grafo(dfa_json, w) != esperado or acepta_grafo(min_json, w) != esperado:
                fallo(f'automata {cid}: el AFD/mínimo difiere del AFN en "{w}"')
                break
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  Máquina de Turing (POST /turing/simulate y /turing/graph)
# ══════════════════════════════════════════════════════════════════════════════

TURING_SCREEN = ROOT / 'autechvapis' / 'lib' / 'simulador' / 'modules' / 'turing_screen.dart'


def ejemplos_turing():
    """Argumentos de cada `_loadExample(...)` de turing_screen.dart."""
    src = TURING_SCREEN.read_text(encoding='utf-8')
    out = []
    for m in re.finditer(r"_loadExample\(\s*'", src):
        i, args, cur, en_literal = m.end() - 1, [], '', False
        while True:
            c = src[i]
            if c == "'":
                j = src.index("'", i + 1)
                cur += src[i + 1:j].encode().decode('unicode_escape').encode('latin-1').decode('utf-8')
                i = j + 1
                continue
            if c == ',':
                args.append(cur)
                cur = ''
            elif c == ')':
                if cur:
                    args.append(cur)
                break
            i += 1
        if len(args) == 5:
            out.append(args)
    return out


def mt_aleatoria(rng):
    n = rng.randint(1, 4)
    ids = [f'q{i}' for i in range(n)] + ['qf']
    lineas = []
    for _ in range(rng.randint(1, 7)):
        lineas.append(f"{rng.choice(ids[:-1])},{rng.choice(['a', 'b', '_', 'X'])} -> "
                      f"{rng.choice(ids)},{rng.choice(['a', 'b', '_', 'X'])},{rng.choice('LRS')}")
    return ','.join(ids), 'q0', 'qf', '\n'.join(lineas)


def _turing(states, initial, accepts, trans, cinta):
    req = api.TuringRequest(states=states, initial=initial, accepts=accepts,
                            transitions=trans, cinta=cinta, head_pos=0, max_steps=500)
    try:
        r = asyncio.run(api.turing_simulate(req))
        sim = {
            'result': r['result'],
            'warnings': r.get('warnings', []),
            'steps': [[p['step'], p['state'], ''.join(p['tape']), p['headPos'], p['message'],
                       p['isAccepted'], p['isRejected']] for p in r['steps']],
        }
    except HTTPException as e:
        sim = {'error': e.detail}
    try:
        g = asyncio.run(api.turing_graph(req))
    except HTTPException as e:
        g = {'error': e.detail}
    return sim, g


def seccion_turing():
    maquinas = [(f'ejemplo_{i}', *a[:3], a[3], a[4]) for i, a in enumerate(ejemplos_turing())]
    rng = random.Random(555)
    for i in range(30):
        st, ini, acc, tr = mt_aleatoria(rng)
        maquinas.append((f'aleatoria_{i}', st, ini, acc, tr, ''))
    maquinas += [
        ('borde_inicial_inexistente', 'q0,q1', 'qz', 'q1', 'q0,a -> q1,a,R', 'a'),
        ('borde_lineas_malas', 'q0,q1', 'q0', 'q1',
         'q0,a -> q1\nsin flecha\nq0,b -> q1,b,X\nq0,a -> q0,b,r\r\nq0,_ -> q1,_,S', 'ab'),
        ('borde_cinta_vacia', 'q0,q1', 'q0', 'q1', 'q0,_ -> q1,_,L', ''),
        ('borde_inicial_acepta', 'q0', 'q0', 'q0', 'q0,a -> q0,a,R', 'aaa'),
        ('borde_bucle', 'q0,q1', 'q0', 'q1', 'q0,a -> q0,a,S', 'a'),
    ]
    casos = []
    for cid, st, ini, acc, tr, cinta0 in maquinas:
        cintas = list(dict.fromkeys([cinta0] + cadenas('ab', 3)))[:9]
        caso = {'id': cid, 'states': st, 'initial': ini, 'accepts': acc,
                'transitions': tr, 'runs': []}
        for c in cintas:
            sim, g = _turing(st, ini, acc, tr, c)
            caso['runs'].append({'tape': c, 'sim': sim})
            caso['graph'] = g
        casos.append(caso)
    return casos


# ══════════════════════════════════════════════════════════════════════════════
#  Principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    pda_ej, regex_ej, leng_ej = ejercicios()
    datos = {'pda': seccion_pda(pda_ej)}
    datos['regex'] = seccion_regex(regex_ej)
    datos['a2r'] = seccion_a2r(datos['regex'], leng_ej)
    datos['ops'] = seccion_ops(leng_ej)
    datos['automatas'] = seccion_automatas(leng_ej)
    datos['turing'] = seccion_turing()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding='utf-8')
    total = sum(len(v) for v in datos.values())
    print(f'{total} casos escritos en {FIXTURE.relative_to(ROOT)}')
    if ERRORES_CORRECCION:
        print(f'\n{len(ERRORES_CORRECCION)} RESULTADOS INCORRECTOS EN EL BACKEND:')
        for m in ERRORES_CORRECCION[:60]:
            print('  -', m)
        sys.exit(1)
    print('Backend correcto en todos los casos verificables.')


if __name__ == '__main__':
    main()
