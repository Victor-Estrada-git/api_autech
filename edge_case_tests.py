# -*- coding: utf-8 -*-
"""
edge_case_tests.py  –  AUTECH API  (suite de casos límite)
===========================================================
Prueba que la API responda correctamente ante entradas inválidas,
extremas o inesperadas — exactamente lo que un usuario real podría mandar.

Categorías:
  [REGEX]   Expresiones regulares
  [PDA]     Autómata de pila
  [TURING]  Máquina de Turing
  [GENERAL] Payloads malformados / campos vacíos

Uso:
    pip install requests
    python edge_case_tests.py
"""

import json
import time
import requests

BASE_URL = "https://api-autech.onrender.com"
TIMEOUT  = 45

# ── Colores ANSI para la consola ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ── Contadores globales ───────────────────────────────────────────────────────
passed = 0
failed = 0
warned = 0
results_log = []


# ── Wake-up ───────────────────────────────────────────────────────────────────

def wakeup():
    print(f"\n{CYAN}  Verificando que la API esté activa...{RESET}")
    for i in range(1, 19):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
            if r.status_code == 200:
                print(f"  {GREEN}API activa.{RESET}\n")
                return True
        except Exception:
            pass
        print(f"  Intento {i}: reintentando en 5s...")
        time.sleep(5)
    print(f"  {RED}API no responde. Verifica Render.{RESET}")
    return False


# ── Función de un solo test ───────────────────────────────────────────────────

def run_test(name: str, method: str, path: str, payload,
             expect_status: int, description: str = ""):
    """
    Ejecuta un caso de prueba y evalúa si el resultado coincide con lo esperado.

    expect_status:
        200  → debe funcionar correctamente
        400  → debe rechazar con error de validación (dato inválido)
        422  → Pydantic rechaza el payload antes de llegar a la lógica
        cualquier otro → se acepta cualquier código >= 400
    """
    global passed, failed, warned

    url = f"{BASE_URL}/{path}"
    try:
        if method == "GET":
            resp = requests.get(url, params=payload, timeout=TIMEOUT)
        else:
            resp = requests.post(url, json=payload, timeout=TIMEOUT)

        code = resp.status_code
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:200]}

        # Evaluar resultado
        if code == expect_status:
            status_icon = f"{GREEN}[PASS]{RESET}"
            passed += 1
            outcome = "PASS"
        elif expect_status >= 400 and code >= 400:
            # Esperábamos error y obtuvimos error (aunque código distinto)
            status_icon = f"{YELLOW}[WARN]{RESET}"
            warned += 1
            outcome = "WARN"
        else:
            status_icon = f"{RED}[FAIL]{RESET}"
            failed += 1
            outcome = "FAIL"

        print(f"  {status_icon} {name}")
        if description:
            print(f"         {CYAN}{description}{RESET}")
        print(f"         Esperado: HTTP {expect_status}  |  Obtenido: HTTP {code}")

        # Si es FAIL, mostrar qué devolvió
        if outcome == "FAIL":
            detail = body.get("detail", body)
            print(f"         {RED}Respuesta: {str(detail)[:120]}{RESET}")

        # Si esperábamos 200, verificar que tenga contenido útil
        if expect_status == 200 and code == 200:
            _check_response_content(name, path, body)

        results_log.append({
            "name": name, "path": path, "method": method,
            "expect": expect_status, "got": code,
            "outcome": outcome, "body_preview": str(body)[:200],
        })

    except requests.exceptions.Timeout:
        print(f"  {RED}[FAIL]{RESET} {name}  →  TIMEOUT (>{TIMEOUT}s)")
        failed += 1
        results_log.append({"name": name, "outcome": "TIMEOUT"})
    except Exception as exc:
        print(f"  {RED}[FAIL]{RESET} {name}  →  Excepción: {exc}")
        failed += 1
        results_log.append({"name": name, "outcome": "EXCEPTION", "error": str(exc)})


def _check_response_content(name, path, body):
    """Verifica que la respuesta 200 tenga las claves esperadas."""
    checks = {
        "pda/validate":          ["graph"],
        "pda/simulate":          ["accepted", "trace"],
        "pda/to-cfg":            ["cfg"],
        "regex/to-automaton":    ["states", "edges"],
        "regex/automaton-to-regex": ["regex"],
        "regex/operation":       ["regex"],
        "turing/simulate":       ["result", "steps"],
        "turing/graph":          ["states", "edges"],
        "automaton/minimize":    ["states", "edges", "stats"],
    }
    expected_keys = checks.get(path, [])
    for key in expected_keys:
        if key not in body:
            print(f"         {YELLOW}⚠ Respuesta 200 pero falta la clave '{key}'{RESET}")


def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CASOS DE PRUEBA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Payloads base válidos ─────────────────────────────────────────────────────

PDA_VALID = {
    "states": "q0,q1,q2",
    "input_alpha": "a,b",
    "stack_alpha": "$,A",
    "start_state": "q0",
    "start_symbol": "$",
    "accept_states": "q2",
    "transitions": "q0,a,$->q1,A$\nq1,b,A->q2,",
}

AUTOMATON_VALID = {
    "states": ["q0", "q1", "q2"],
    "transitions": {"q0": {"a": "q1"}, "q1": {"b": "q2"}},
    "initial": "q0",
    "accepting": ["q2"],
    "alphabet": ["a", "b"],
}

TURING_VALID = {
    "states": "q0,q1,q2",
    "initial": "q0",
    "accepts": "q2",
    "transitions": "q0,1->q1,1,R\nq1,1->q2,1,R",
    "cinta": "11",
    "head_pos": 0,
    "max_steps": 50,
}


def run_all_tests():

    # ══════════════════════════════════════════════════════════════════════
    section("REGEX — Casos válidos normales")
    # ══════════════════════════════════════════════════════════════════════

    run_test("Regex simple: a",
             "GET", "regex/to-automaton", {"exp": "a"}, 200,
             "Regex de un solo símbolo")

    run_test("Regex básica: (a|b)*abb",
             "GET", "regex/to-automaton", {"exp": "(a|b)*abb"}, 200,
             "Regex clásica del libro")

    run_test("Regex con kleene anidado: (a*b*)*",
             "GET", "regex/to-automaton", {"exp": "(a*b*)*"}, 200)

    run_test("Automata -> Regex (DFA simple)",
             "POST", "regex/automaton-to-regex", AUTOMATON_VALID, 200)

    run_test("Op union: a|b",
             "POST", "regex/operation",
             {"operation": "union", "regex1": "a", "regex2": "b"}, 200)

    run_test("Op kleene: a*",
             "POST", "regex/operation",
             {"operation": "kleene", "regex1": "a"}, 200)

    run_test("Op complement",
             "POST", "regex/operation",
             {"operation": "complement", "regex1": "a"}, 200)

    run_test("Op intersection: (a|b)* ∩ a*b",
             "POST", "regex/operation",
             {"operation": "intersection", "regex1": "(a|b)*", "regex2": "a*b"}, 200)

    run_test("Op difference: a* - ab",
             "POST", "regex/operation",
             {"operation": "difference", "regex1": "a*", "regex2": "ab"}, 200)

    run_test("Op reverse: (ab)*",
             "POST", "regex/operation",
             {"operation": "reverse", "regex1": "(ab)*"}, 200)

    run_test("Op concat: a·b*",
             "POST", "regex/operation",
             {"operation": "concat", "regex1": "a", "regex2": "b*"}, 200)

    # ══════════════════════════════════════════════════════════════════════
    section("REGEX — Casos límite y entradas inválidas")
    # ══════════════════════════════════════════════════════════════════════

    run_test("Regex vacía: exp=''",
             "GET", "regex/to-automaton", {"exp": ""}, 400,
             "Cadena vacía no es regex válida")

    run_test("Regex con paréntesis sin cerrar: (a|b",
             "GET", "regex/to-automaton", {"exp": "(a|b"}, 400,
             "Paréntesis desbalanceado")

    run_test("Regex con paréntesis extra: a|b)",
             "GET", "regex/to-automaton", {"exp": "a|b)"}, 400,
             "Cierre sin apertura")

    run_test("Regex con solo operador: *",
             "GET", "regex/to-automaton", {"exp": "*"}, 400,
             "Kleene sin operando")

    run_test("Regex con solo pipe: |",
             "GET", "regex/to-automaton", {"exp": "|"}, 400,
             "Union sin operandos")

    run_test("Regex muy larga (1000 chars)",
             "GET", "regex/to-automaton",
             {"exp": "(a|b)" * 200}, 200,
             "Regex de 1000 caracteres — debe procesar o devolver 400, no colgar")

    run_test("Regex con caracteres especiales: a+b?",
             "GET", "regex/to-automaton", {"exp": "a+b?"}, 200,
             "Algunos parsers soportan + y ? — si no, debe dar 400 limpio")

    run_test("Op binaria sin regex2: union sin regex2",
             "POST", "regex/operation",
             {"operation": "union", "regex1": "a"}, 400,
             "union requiere regex2")

    run_test("Op inexistente: 'magia'",
             "POST", "regex/operation",
             {"operation": "magia", "regex1": "a"}, 400,
             "Operacion no soportada")

    run_test("Op sin operation field",
             "POST", "regex/operation",
             {"regex1": "a", "regex2": "b"}, 422,
             "Campo 'operation' requerido por Pydantic")

    run_test("Automata->Regex: initial no existe en states",
             "POST", "regex/automaton-to-regex",
             {**AUTOMATON_VALID, "initial": "qX"}, 400,
             "Estado inicial que no está en la lista de estados")

    run_test("Automata->Regex: accepting vacío",
             "POST", "regex/automaton-to-regex",
             {**AUTOMATON_VALID, "accepting": []}, 200,
             "Sin estados de aceptación → lenguaje vacío, no debe crashear")

    run_test("Automata->Regex: states vacío",
             "POST", "regex/automaton-to-regex",
             {**AUTOMATON_VALID, "states": []}, 400,
             "Lista de estados vacía")

    # ══════════════════════════════════════════════════════════════════════
    section("PDA — Casos válidos normales")
    # ══════════════════════════════════════════════════════════════════════

    run_test("PDA validate básico",
             "POST", "pda/validate", PDA_VALID, 200)

    run_test("PDA simulate: cadena aceptada (ab)",
             "POST", "pda/simulate",
             {**PDA_VALID, "input_string": "ab"}, 200,
             "La cadena 'ab' debe ser aceptada")

    run_test("PDA simulate: cadena rechazada (ba)",
             "POST", "pda/simulate",
             {**PDA_VALID, "input_string": "ba"}, 200,
             "La cadena 'ba' debe ser rechazada (accepted=false)")

    run_test("PDA simulate: cadena vacía",
             "POST", "pda/simulate",
             {**PDA_VALID, "input_string": ""}, 200,
             "Cadena vacía — debe responder, no colgar")

    run_test("PDA to-CFG",
             "POST", "pda/to-cfg", PDA_VALID, 200)

    # ══════════════════════════════════════════════════════════════════════
    section("PDA — Casos límite y entradas inválidas")
    # ══════════════════════════════════════════════════════════════════════

    run_test("PDA: start_state no está en states",
             "POST", "pda/validate",
             {**PDA_VALID, "start_state": "qX"}, 400,
             "Estado inicial que no existe")

    run_test("PDA: accept_states no están en states",
             "POST", "pda/validate",
             {**PDA_VALID, "accept_states": "qZ"}, 400,
             "Estado de aceptacion inexistente")

    run_test("PDA: states vacío",
             "POST", "pda/validate",
             {**PDA_VALID, "states": ""}, 400,
             "Campo states vacío")

    run_test("PDA: transitions vacías",
             "POST", "pda/validate",
             {**PDA_VALID, "transitions": ""}, 200,
             "PDA sin transiciones — válido pero trivial")

    run_test("PDA: transición con formato incorrecto",
             "POST", "pda/validate",
             {**PDA_VALID, "transitions": "esto_no_es_una_transicion"}, 400,
             "Formato de transición inválido")

    run_test("PDA: símbolo de pila desconocido en transición",
             "POST", "pda/simulate",
             {**PDA_VALID,
              "transitions": "q0,a,X->q1,A$\nq1,b,A->q2,",
              "input_string": "ab"}, 400,
             "Pop de símbolo X que no está en stack_alpha")

    run_test("PDA simulate: cadena muy larga (500 chars)",
             "POST", "pda/simulate",
             {**PDA_VALID, "input_string": "a" * 500}, 200,
             "Cadena larga — no debe colgar")

    run_test("PDA: falta campo obligatorio (start_symbol)",
             "POST", "pda/validate",
             {k: v for k, v in PDA_VALID.items() if k != "start_symbol"}, 422,
             "Pydantic debe rechazar por campo faltante")

    # ══════════════════════════════════════════════════════════════════════
    section("TURING — Casos válidos normales")
    # ══════════════════════════════════════════════════════════════════════

    run_test("Turing simulate básico (cinta=11)",
             "POST", "turing/simulate", TURING_VALID, 200)

    run_test("Turing graph básico",
             "POST", "turing/graph", TURING_VALID, 200)

    run_test("Turing: cinta vacía",
             "POST", "turing/simulate",
             {**TURING_VALID, "cinta": ""}, 200,
             "Cinta vacía — debe responder REJECTED, no crashear")

    # ══════════════════════════════════════════════════════════════════════
    section("TURING — Casos límite y entradas inválidas")
    # ══════════════════════════════════════════════════════════════════════

    run_test("Turing: loop infinito controlado por max_steps",
             "POST", "turing/simulate",
             {
                 "states": "q0",
                 "initial": "q0",
                 "accepts": "q1",          # q1 nunca se alcanza
                 "transitions": "q0,1->q0,1,R",  # bucle infinito
                 "cinta": "111",
                 "head_pos": 0,
                 "max_steps": 20,
             }, 200,
             "Debe devolver result=TIMEOUT, no colgarse")

    run_test("Turing: head_pos fuera de la cinta",
             "POST", "turing/simulate",
             {**TURING_VALID, "head_pos": 999}, 200,
             "Cabeza fuera del rango — debe manejar el caso")

    run_test("Turing: head_pos negativo",
             "POST", "turing/simulate",
             {**TURING_VALID, "head_pos": -1}, 200,
             "head_pos negativo")

    run_test("Turing: initial state no existe en states",
             "POST", "turing/simulate",
             {**TURING_VALID, "initial": "qX"}, 400,
             "Estado inicial inexistente")

    run_test("Turing: transiciones vacías",
             "POST", "turing/simulate",
             {**TURING_VALID, "transitions": ""}, 200,
             "Sin transiciones — debe rechazar la cadena")

    run_test("Turing: max_steps = 0",
             "POST", "turing/simulate",
             {**TURING_VALID, "max_steps": 0}, 400,
             "max_steps < 1 no tiene sentido — debe rechazarse con mensaje claro (H-54)")

    run_test("Turing: max_steps muy alto (100000)",
             "POST", "turing/simulate",
             {
                 "states": "q0,q1",
                 "initial": "q0",
                 "accepts": "q1",
                 "transitions": "q0,1->q1,1,R",
                 "cinta": "1",
                 "head_pos": 0,
                 "max_steps": 100000,
             }, 400,
             "max_steps por encima del tope (1000) — debe rechazarse para no generar respuestas enormes (H-52)")

    run_test("Turing: falta campo 'cinta' (requerido)",
             "POST", "turing/simulate",
             {k: v for k, v in TURING_VALID.items() if k != "cinta"}, 422,
             "Pydantic debe rechazar por campo faltante")

    # ══════════════════════════════════════════════════════════════════════
    section("GENERAL — Payloads malformados")
    # ══════════════════════════════════════════════════════════════════════

    run_test("POST con body vacío {}",
             "POST", "pda/validate", {}, 422,
             "Body vacío — Pydantic debe rechazar")

    run_test("POST con body null (None enviado como JSON)",
             "POST", "turing/simulate", None, 422,
             "Body nulo")

    run_test("Campos con tipos incorrectos (int en vez de str)",
             "POST", "pda/validate",
             {**PDA_VALID, "states": 12345}, 422,
             "Pydantic debe rechazar tipo incorrecto")

    run_test("Regex: parámetro 'exp' faltante (GET sin params)",
             "GET", "regex/to-automaton", {}, 422,
             "Parámetro requerido faltante")

    run_test("Endpoint inexistente",
             "GET", "esto/no/existe", {}, 404,
             "Ruta que no existe")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  AUTECH — Suite de casos límite{RESET}")
    print(f"{BOLD}  Base URL: {BASE_URL}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    if not wakeup():
        return

    run_all_tests()

    # ── Resumen ───────────────────────────────────────────────────────────
    total = passed + failed + warned
    print(f"\n\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RESUMEN{RESET}")
    print(f"{'='*60}")
    print(f"  Total de pruebas : {total}")
    print(f"  {GREEN}Pasaron  (PASS){RESET} : {passed}")
    print(f"  {YELLOW}Advertencias (WARN){RESET} : {warned}  ← error esperado, código distinto")
    print(f"  {RED}Fallaron (FAIL){RESET} : {failed}")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}Tu API maneja todos los casos límite correctamente.{RESET}")
    else:
        print(f"\n  {RED}{BOLD}Hay {failed} caso(s) que necesitan atención antes de subir a Play Store.{RESET}")
        print(f"  {RED}Revisa los [FAIL] arriba para ver qué devolvió la API en cada caso.{RESET}")

    # ── Guardar reporte ───────────────────────────────────────────────────
    out = "edge_case_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"total": total, "passed": passed,
                        "warned": warned, "failed": failed},
            "tests": results_log,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Reporte completo guardado en: {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()