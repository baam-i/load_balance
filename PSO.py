# -*- coding: utf-8 -*-
# Requisitos: Python 3.9+, numpy; opcional: scikit-learn para TF-IDF
# Cumple: multiprocessing (asignación explícita por proceso), PSO para cortes,
# sliding windows de 40, montos: 20k..200k, medición de tiempos.

import os, re, csv, time
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

# ---------------------------
# Configuración
# ---------------------------
DEFAULT_WORKERS = 8
WINDOW_SIZE = 40
PLANNING_SIZES = [20000, 40000, 60000, 80000, 100000,
                  120000, 140000, 160000, 180000, 200000]

# ---------------------------
# Limpieza de texto
# ---------------------------
CONTRACTIONS = {
    "can't":"cannot","won't":"will not","n't":" not","'re":" are","'s":" is","'d":" would",
    "'ll":" will","'t":" not","'ve":" have","'m":" am"
}
STOPWORDS = set(""" the and is in at of a to for on it this that""".split())

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticonos
    "\U0001F300-\U0001F5FF"  # símbolos/pictogramas
    "\U0001F680-\U0001F6FF"  # transporte/mapa
    "\U0001F1E0-\U0001F1FF"  # banderas
    "\u2600-\u26FF"          # varios
    "\u2700-\u27BF"          # dingbats
    "]+", flags=re.UNICODE
)
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")

def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = re.sub(re.escape(k), v, text, flags=re.IGNORECASE)
    return text

def clean_text(t: str, stopwords: set = STOPWORDS) -> str:
    if not t:
        return ""
    t = t.lower()
    t = expand_contractions(t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)      # si prefieres conservar palabra, reemplaza '#' por vacío
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r"\d+", " ", t)      # elimina números
    t = NON_ALPHA_RE.sub(" ", t)    # elimina puntuación y símbolos
    t = MULTISPACE_RE.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in stopwords and len(w) > 1]
    return " ".join(tokens)

# ---------------------------
# Sliding windows
# ---------------------------
def make_windows(n_items: int, win_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + win_size, n_items)) for i in range(0, n_items, win_size)]

def window_costs(texts: List[str], windows: List[Tuple[int, int]]) -> np.ndarray:
    # costo aproximado: suma de longitudes de texto por ventana
    costs = []
    for a, b in windows:
        total = sum(len(texts[i]) for i in range(a, b))
        costs.append(total)
    return np.array(costs, dtype=float)

# ---------------------------
# PSO para particionar ventanas contiguas entre n_workers
# Dimensión = n_workers - 1 (puntos de corte entre ventanas)
# ---------------------------
def eval_partition(cuts: np.ndarray, costs: np.ndarray, n_parts: int) -> float:
    n = len(costs)
    # recortar y ordenar cortes a [1, n-1]
    cuts = np.clip(np.round(cuts).astype(int), 1, max(1, n - 1))
    cuts = np.unique(cuts)
    # completar o recortar número de cortes
    while len(cuts) < n_parts - 1:
        all_c = np.r_[0, cuts, n]
        seg_lens = np.diff(all_c)
        j = int(np.argmax(seg_lens))
        new_cut = all_c[j] + seg_lens[j] // 2
        cuts = np.sort(np.r_[cuts, new_cut])
    if len(cuts) > n_parts - 1:
        cuts = cuts[: n_parts - 1]
    # cargas por segmento
    bounds = np.r_[0, cuts, n]
    loads = [costs[bounds[i]:bounds[i+1]].sum() for i in range(n_parts)]
    mx = max(loads) if loads else 0.0
    std = float(np.std(loads)) if loads else 0.0
    return mx + 1e-3 * std

def pso_partition(costs: np.ndarray, n_parts: int, iters: int = 100,
                  particles: int = 40, w: float = 0.729,
                  c1: float = 1.49445, c2: float = 1.49445, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(costs)
    dim = max(1, n_parts - 1)
    # inicializar cortes en dominio continuo [1, n-1]
    X = rng.uniform(1, max(2, n - 1), size=(particles, dim))
    V = rng.normal(0, 1, size=(particles, dim)) * 0.5
    P = X.copy()
    P_cost = np.array([eval_partition(p, costs, n_parts) for p in P])
    g_idx = int(np.argmin(P_cost))
    g = P[g_idx].copy()
    g_cost = P_cost[g_idx]
    for _ in range(iters):
        r1 = rng.random(size=V.shape)
        r2 = rng.random(size=V.shape)
        V = w * V + c1 * r1 * (P - X) + c2 * r2 * (g - X)
        X = X + V
        X = np.clip(X, 1, max(2, n - 1))
        C = np.array([eval_partition(x, costs, n_parts) for x in X])
        improved = C < P_cost
        P[improved] = X[improved]
        P_cost[improved] = C[improved]
        j = int(np.argmin(P_cost))
        if P_cost[j] < g_cost:
            g_cost, g = P_cost[j], P[j].copy()
    cuts = np.clip(np.round(g).astype(int), 1, max(1, n - 1))
    cuts = np.unique(cuts)
    while len(cuts) < n_parts - 1:
        all_c = np.r_[0, cuts, n]
        seg_lens = np.diff(all_c)
        j = int(np.argmax(seg_lens))
        new_cut = all_c[j] + seg_lens[j] // 2
        cuts = np.sort(np.r_[cuts, new_cut])
    if len(cuts) > n_parts - 1:
        cuts = cuts[: n_parts - 1]
    return np.r_[0, cuts, n]  # incluye límites 0 y n

# ---------------------------
# Worker: limpia textos según ventanas asignadas
# ---------------------------
def worker_clean_texts(args):
    worker_id, windows, texts, stopwords = args
    t0 = time.perf_counter()
    out = []
    for a, b in windows:
        for i in range(a, b):
            out.append((i, clean_text(texts[i], stopwords)))
    elapsed = time.perf_counter() - t0
    return worker_id, out, elapsed

# Target top-level para Windows (spawn)
def runner_entry(arg, q):
    wid, winlist, txts, sw = arg
    res = worker_clean_texts((wid, winlist, txts, sw))
    q.put(res)

# ---------------------------
# Vectorización opcional (TF-IDF)
# ---------------------------
def maybe_tfidf(cleaned_texts: List[str], use_tfidf: bool = True):
    if not use_tfidf:
        return None, None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2))
        X = vec.fit_transform(cleaned_texts)
        return X, vec
    except Exception:
        return None, None

# ---------------------------
# E/S y pipeline
# ---------------------------
def load_texts_from_csv(path: str, text_col: str = "text") -> List[str]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row.get(text_col, "") or "")
    return rows

def run_pipeline(csv_path: str,
                 planning_sizes: List[int] = PLANNING_SIZES,
                 n_workers: int = DEFAULT_WORKERS,
                 window_size: int = WINDOW_SIZE,
                 pso_iters: int = 100,
                 pso_particles: int = 40,
                 use_tfidf: bool = True,
                 custom_stopwords: set = None,
                 seed: int = 42):
    texts_all = load_texts_from_csv(csv_path, "text")
    stopwords = custom_stopwords if custom_stopwords is not None else STOPWORDS
    results_by_size: Dict[int, Dict[str, Any]] = {}
    times_by_size: Dict[int, float] = {}

    for N in planning_sizes:
        N_eff = min(N, len(texts_all))
        if N_eff == 0:
            results_by_size[N] = {"cleaned": [], "X": None, "vectorizer": None,
                                  "per_worker_times": [], "cuts": np.array([0,0]), "windows": []}
            times_by_size[N] = 0.0
            continue

        batch = texts_all[:N_eff]

        # ventanas y costos
        windows = make_windows(N_eff, window_size)
        costs = window_costs(batch, windows)

        # cortes PSO entre ventanas para n_workers segmentos contiguos
        cuts = pso_partition(costs, n_parts=n_workers,
                             iters=pso_iters, particles=pso_particles, seed=seed)

        # asignación de ventanas por proceso según cortes (en índices de ventana)
        per_worker_windows = []
        for j in range(n_workers):
            a, b = int(cuts[j]), int(cuts[j+1])
            win_slice = windows[a:b] if a < b else []
            per_worker_windows.append(win_slice)

        t0 = time.perf_counter()

        # lanzar procesos con asignación fija
        args = [(wid, per_worker_windows[wid], batch, stopwords) for wid in range(n_workers)]
        q = mp.Queue()
        procs = []
        for a in args:
            p = mp.Process(target=runner_entry, args=(a, q))
            p.start()
            procs.append(p)

        per_worker_times = []
        cleaned_pairs = []
        for _ in range(n_workers):
            wid, out, elapsed = q.get()
            cleaned_pairs.extend(out)
            per_worker_times.append((wid, elapsed))
        for p in procs:
            p.join()

        # reconstruir orden original
        cleaned_pairs.sort(key=lambda t: t[0])
        cleaned_texts = [t for _, t in cleaned_pairs]

        # vectorización opcional
        X, vec = maybe_tfidf(cleaned_texts, use_tfidf=use_tfidf)

        elapsed_total = time.perf_counter() - t0

        results_by_size[N] = {
            "cleaned": cleaned_texts,
            "X": X,
            "vectorizer": vec,
            "per_worker_times": per_worker_times,
            "cuts": cuts,
            "windows": windows
        }
        times_by_size[N] = elapsed_total

    return results_by_size, times_by_size

if __name__ == "__main__":
    # Requerido en Windows/macOS con spawn; asegura targets importables y evita errores de pickling
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()

    # Ajusta la ruta a tu CSV real (columna 'text' como en el ejemplo compartido)
    csv_path = "Suicide_Detection.csv"

    # Ejecutar pipeline
    results, times = run_pipeline(csv_path,
                                  planning_sizes=PLANNING_SIZES,
                                  n_workers=DEFAULT_WORKERS,
                                  window_size=WINDOW_SIZE,
                                  pso_iters=80,
                                  pso_particles=30,
                                  use_tfidf=True)

    # Ejemplo de reporte en consola
    for N, t in times.items():
        print(f"Monto {N}: {t:.3f} s")
