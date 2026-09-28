# -*- coding: utf-8 -*-
"""
performance_test.py  –  AUTECH Render API  (version final)
===========================================================
Rutas y schemas tomados del /docs en produccion.
Requiere solo: pip install requests
Uso          : python performance_test.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev

import requests

# ─── Configuración ─────────────────────────────────────────────────────────────
BASE_URL       = "https://api-autech.onrender.com"
TOTAL_REQUESTS = 50
CONCURRENCY    = 10
TIMEOUT        = 45
WAKEUP_WAIT    = 90

# ─── Payloads exactos según /docs ──────────────────────────────────────────────

# AutomatonToRegexRequest
AUTOMATON = {
    "states":      ["q0", "q1", "q2"],
    "transitions": {"q0": {"a": "q1"}, "q1": {"b": "q2"}},
    "initial":     "q0",
    "accepting":   ["q2"],
    "alphabet":    ["a", "b"],
}

# PDARequest  (snake_case, campos string)
PDA_DEF = {
    "states":        "q0,q1,q2",
    "input_alpha":   "a,b",
    "stack_alpha":   "$,A",
    "start_state":   "q0",
    "start_symbol":  "$",
    "accept_states": "q2",
    "transitions":   "q0,a,$->q1,A$\nq1,b,A->q2,",
}

# TuringRequest
TURING = {
    "states":      "q0,q1,q2",
    "initial":     "q0",
    "accepts":     "q2",
    "transitions": "q0,1->q1,1,R\nq1,1->q2,1,R",
    "cinta":       "11",
    "head_pos":    0,
    "max_steps":   50,
}

# ─── Endpoints ─────────────────────────────────────────────────────────────────
ENDPOINTS = [
    # Nombre                           Método  Path                       Payload
    ("Regex -> Automata (a|b)*abb",   "GET",  "regex/to-automaton",      {"exp": "(a|b)*abb"}),
    ("Regex -> Automata a*(b|c)",     "GET",  "regex/to-automaton",      {"exp": "a*(b|c)"}),
    ("Automata -> Regex",             "POST", "regex/automaton-to-regex", AUTOMATON),
    ("Op. Regex: union(a,b)",         "POST", "regex/operation",         {"operation": "union",      "regex1": "a",  "regex2": "b"}),
    ("Op. Regex: kleene(ab)",         "POST", "regex/operation",         {"operation": "kleene",     "regex1": "ab"}),
    ("Op. Regex: complement(a)",      "POST", "regex/operation",         {"operation": "complement", "regex1": "a"}),
    ("Op. Regex: intersection(a,b)",  "POST", "regex/operation",         {"operation": "intersection","regex1": "a", "regex2": "b"}),
    ("PDA Validate",                  "POST", "pda/validate",            PDA_DEF),
    ("PDA Simulate (ab)",             "POST", "pda/simulate",            {**PDA_DEF, "input_string": "ab"}),
    ("PDA -> CFG",                    "POST", "pda/to-cfg",              PDA_DEF),
    ("Turing Graph",                  "POST", "turing/graph",            TURING),
    ("Turing Simulate (cinta=11)",    "POST", "turing/simulate",         TURING),
]


# ─── Wake-up ───────────────────────────────────────────────────────────────────

def wakeup():
    url      = f"{BASE_URL}/"
    deadline = time.monotonic() + WAKEUP_WAIT
    attempt  = 0
    print(f"\n  Despertando la API (max {WAKEUP_WAIT}s)...")
    while time.monotonic() < deadline:
        attempt += 1
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                print(f"  API activa tras {attempt} intento(s).\n")
                return True
            print(f"  Intento {attempt}: HTTP {r.status_code} – reintentando en 5s...")
        except requests.exceptions.ConnectionError:
            print(f"  Intento {attempt}: sin conexion – reintentando en 5s...")
        except requests.exceptions.Timeout:
            print(f"  Intento {attempt}: timeout – reintentando en 5s...")
        time.sleep(5)
    print("  [ERROR] La API no respondio. Verifica el servicio en Render.")
    return False


# ─── Una sola peticion ─────────────────────────────────────────────────────────

def single_request(method, url, payload):
    start = time.monotonic()
    try:
        if method == "GET":
            resp = requests.get(url, params=payload, timeout=TIMEOUT)
        else:
            resp = requests.post(url, json=payload, timeout=TIMEOUT)
        latency = time.monotonic() - start
        ok = (resp.status_code == 200)
        return {"ok": ok, "status": resp.status_code, "latency": latency,
                "error": None if ok else resp.text[:150]}
    except Exception as exc:
        return {"ok": False, "status": "ERR", "latency": time.monotonic() - start,
                "error": str(exc)[:150]}


# ─── Benchmark de un endpoint ──────────────────────────────────────────────────

def bench_endpoint(name, method, path, payload):
    url   = f"{BASE_URL}/{path}"
    probe = single_request(method, url, payload)
    if not probe["ok"]:
        return {"name": name, "skipped": True,
                "probe_status": probe["status"], "probe_error": probe["error"]}

    results    = []
    wall_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(single_request, method, url, payload)
                   for _ in range(TOTAL_REQUESTS)]
        for f in as_completed(futures):
            results.append(f.result())
    wall_time = time.monotonic() - wall_start

    successes  = [r["latency"] for r in results if r["ok"]]
    failures   = [r for r in results if not r["ok"]]
    sorted_lat = sorted(successes)
    p95 = sorted_lat[max(0, int(len(sorted_lat) * 0.95) - 1)] if sorted_lat else 0

    return {
        "name":         name,
        "skipped":      False,
        "total":        TOTAL_REQUESTS,
        "successes":    len(successes),
        "failures":     len(failures),
        "success_rate": len(successes) / TOTAL_REQUESTS * 100,
        "lat_min":      min(successes, default=0),
        "lat_avg":      mean(successes) if successes else 0,
        "lat_p95":      p95,
        "lat_max":      max(successes, default=0),
        "lat_std":      stdev(successes) if len(successes) > 1 else 0,
        "throughput":   len(successes) / wall_time if wall_time > 0 else 0,
        "wall_time":    wall_time,
        "errors":       list({r["error"] for r in failures if r["error"]})[:3],
    }


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 68)
    print("  AUTECH - Pruebas de rendimiento de la API")
    print(f"  Base URL    : {BASE_URL}")
    print(f"  Requests/ep : {TOTAL_REQUESTS}  |  Concurrencia: {CONCURRENCY}")
    print("=" * 68)

    if not wakeup():
        return

    all_results = []
    for name, method, path, payload in ENDPOINTS:
        print(f"[...] {name} ...")
        result = bench_endpoint(name, method, path, payload)
        all_results.append(result)

        if result["skipped"]:
            print(f"    [SKIP] HTTP {result['probe_status']}: {result['probe_error']}")
        else:
            icon = "[OK]" if result["success_rate"] == 100 else "[!!]"
            print(f"    {icon}  {result['successes']}/{result['total']} ok "
                  f"| avg {result['lat_avg']:.3f}s "
                  f"| P95 {result['lat_p95']:.3f}s "
                  f"| {result['throughput']:.1f} req/s")
            for err in result["errors"]:
                print(f"         x {err}")

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 68)
    print("  RESUMEN FINAL")
    print("=" * 68)
    print(f"  {'Endpoint':<40} {'Exito':>6} {'Avg':>7} {'P95':>7} {'Max':>7} {'req/s':>6}")
    print("  " + "-" * 66)
    for r in all_results:
        if r["skipped"]:
            print(f"  {'  ' + r['name'][:38]:<40}  SALTADO (HTTP {r['probe_status']})")
        else:
            flag = "    " if r["success_rate"] == 100 else "[!] "
            print(f"  {flag}{r['name'][:38]:<38} "
                  f"{r['success_rate']:>5.1f}% "
                  f"{r['lat_avg']:>6.3f}s "
                  f"{r['lat_p95']:>6.3f}s "
                  f"{r['lat_max']:>6.3f}s "
                  f"{r['throughput']:>5.1f}")

    out = "resultados_rendimiento.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Resultados guardados en: {out}")
    print("=" * 68)


if __name__ == "__main__":
    main()