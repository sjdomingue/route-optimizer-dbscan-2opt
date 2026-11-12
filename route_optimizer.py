import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from sklearn.cluster import DBSCAN


# ------------------------------------------------------------------
# CONFIGURATION: Adjust paths if necessary
# ------------------------------------------------------------------
@dataclass
class Config:
    entrada: str = "data/sample_input.csv"
    salida: str = "data/optimized_route.csv"
    usar_tomtom: bool = False  # Set to True to use the API.
    tomtom_api_key: str = ""   # Paste your TomTom API key here.
    radio_cluster_m: float = 150.0
    min_puntos_cluster: int = 3
    max_clusters_tomtom: int = 70
    max_puntos_tomtom_cluster: int = 12
    # Real start point. Leave both as None to start from the south.
    inicio_latitud: float | None = None
    inicio_longitud: float | None = None
    # "distancia" minimizes meters; "tiempo" minimizes estimated seconds.
    criterio_tomtom: str = "tiempo"


CFG = Config()

# EPSG:9377 is MAGNA-SIRGAS 2018 / Origen Nacional.
TRANSFORMADOR = Transformer.from_crs("EPSG:9377", "EPSG:4326", always_xy=True)


def matriz_haversine(latlon):
    lat = np.radians(latlon[:, 0])
    lon = np.radians(latlon[:, 1])
    dlat = lat[:, None] - lat
    dlon = lon[:, None] - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def greedy(inicio, matriz):
    n = len(matriz)
    visitados = np.zeros(n, dtype=bool)
    ruta = [inicio]
    visitados[inicio] = True
    actual = inicio
    for _ in range(1, n):
        distancias = matriz[actual].copy()
        distancias[visitados] = np.inf
        actual = int(np.argmin(distancias))
        ruta.append(actual)
        visitados[actual] = True
    return ruta


def costo(ruta, matriz):
    r = np.asarray(ruta)
    return float(matriz[r[:-1], r[1:]].sum())


def two_opt(ruta, matriz, max_iter=100):
    ruta = list(ruta)
    if len(ruta) < 4:
        return ruta
    for _ in range(max_iter):
        mejoro = False
        for i in range(1, len(ruta) - 2):
            for j in range(i + 1, len(ruta) - 1):
                a, b, c, d = ruta[i - 1], ruta[i], ruta[j], ruta[j + 1]
                if matriz[a, c] + matriz[b, d] < matriz[a, b] + matriz[c, d] - 1e-9:
                    ruta[i:j + 1] = reversed(ruta[i:j + 1])
                    mejoro = True
        if not mejoro:
            break
    return ruta


def matriz_tomtom(centroides_latlon, api_key, criterio="distancia"):
    """Real distance/time matrix by roads for a small list of points."""
    n = len(centroides_latlon)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = centroides_latlon[i]
            lat2, lon2 = centroides_latlon[j]
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{lat1},{lon1}:{lat2},{lon2}/json"
            try:
                respuesta = requests.get(
                    url,
                    params={"key": api_key, "routeType": "fastest", "travelMode": "car", "traffic": "false"},
                    timeout=15,
                )
                respuesta.raise_for_status()
                resumen = respuesta.json()["routes"][0]["summary"]
                metros = (resumen["travelTimeInSeconds"] if criterio == "tiempo"
                          else resumen["lengthInMeters"])
            except (requests.RequestException, KeyError, IndexError, ValueError):
                metros = matriz_haversine(np.array([[lat1, lon1], [lat2, lon2]]))[0, 1]
                if criterio == "tiempo":
                    metros = metros / (25_000 / 3600)  # Fallback: 25 km/h.
            matriz[i, j] = matriz[j, i] = metros
            time.sleep(0.05)
    return matriz


def crear_clusters(xy, radio, min_puntos):
    etiquetas = DBSCAN(eps=radio, min_samples=min_puntos).fit_predict(xy)
    grupos = []
    for etiqueta in sorted(set(etiquetas)):
        indices = np.where(etiquetas == etiqueta)[0]
        if etiqueta == -1:
            grupos.extend([np.array([i]) for i in indices])
        else:
            grupos.append(indices)
    return grupos


def optimizar(df, cfg):
    xy = df[["xcoord", "ycoord"]].to_numpy(float)
    latlon = df[["latitud", "longitud"]].to_numpy(float)
    grupos = crear_clusters(xy, cfg.radio_cluster_m, cfg.min_puntos_cluster)
    print(f"Clusters to sort: {len(grupos)}")

    centroides_xy = np.array([xy[g].mean(axis=0) for g in grupos])
    lon, lat = TRANSFORMADOR.transform(centroides_xy[:, 0], centroides_xy[:, 1])
    centroides_latlon = np.column_stack([lat, lon])

    if cfg.usar_tomtom and len(grupos) <= cfg.max_clusters_tomtom:
        if not cfg.tomtom_api_key:
            raise ValueError("TomTom activated, but tomtom_api_key is missing.")
        print("Querying TomTom between cluster centroids...")
        matriz_clusters = matriz_tomtom(centroides_latlon, cfg.tomtom_api_key, cfg.criterio_tomtom)
    else:
        if cfg.usar_tomtom:
            print(f"TomTom between clusters skipped: {len(grupos)} exceeds limit {cfg.max_clusters_tomtom}.")
        matriz_clusters = matriz_haversine(centroides_latlon)

    if cfg.inicio_latitud is not None and cfg.inicio_longitud is not None:
        origen = np.array([[cfg.inicio_latitud, cfg.inicio_longitud]])
        inicio = int(np.argmin(matriz_haversine(np.vstack([origen, centroides_latlon]))[0, 1:]))
        punto_anterior = origen[0]
    else:
        inicio = int(np.argmin(centroides_latlon[:, 0]))
        punto_anterior = None
    orden_clusters = two_opt(greedy(inicio, matriz_clusters), matriz_clusters)

    ruta_global = []
    for ci in orden_clusters:
        indices = grupos[ci]
        sub = latlon[indices]
        if len(indices) == 1:
            ruta_local = [0]
        else:
            usar_tomtom_local = cfg.usar_tomtom and len(indices) <= cfg.max_puntos_tomtom_cluster
            if usar_tomtom_local:
                print(f"TomTom inside cluster: {len(indices)} parcels")
                matriz_sub = matriz_tomtom(sub, cfg.tomtom_api_key, cfg.criterio_tomtom)
            else:
                matriz_sub = matriz_haversine(sub)
            if punto_anterior is None:
                inicio_local = int(np.argmin(sub[:, 0]))
            else:
                inicio_local = int(np.argmin(matriz_haversine(np.vstack([punto_anterior, sub]))[0, 1:]))
            ruta_local = two_opt(greedy(inicio_local, matriz_sub), matriz_sub, max_iter=60)
        globales = indices[ruta_local]
        ruta_global.extend(globales.tolist())
        punto_anterior = latlon[globales[-1]]

    return df.iloc[ruta_global].copy(), costo(ruta_global, matriz_haversine(latlon)), len(grupos)


def main():
    if not os.path.isfile(CFG.entrada):
        raise FileNotFoundError(f"CSV not found: {CFG.entrada}")

    # Load data
    df = pd.read_csv(CFG.entrada, sep=";", encoding="utf-8-sig", dtype=str)
    requeridas = {"xcoord", "ycoord"}
    faltantes = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Missing columns: {', '.join(sorted(faltantes))}")

    df["xcoord"] = pd.to_numeric(df["xcoord"], errors="coerce")
    df["ycoord"] = pd.to_numeric(df["ycoord"], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["xcoord", "ycoord"]).copy()
    print(f"Valid rows: {len(df)} out of {antes}")

    lon, lat = TRANSFORMADOR.transform(df["xcoord"].to_numpy(), df["ycoord"].to_numpy())
    df["latitud"] = lat
    df["longitud"] = lon

    # Validate geographic region (Colombia)
    validas = df["latitud"].between(-5, 14) & df["longitud"].between(-82, -66)
    if not validas.all():
        raise ValueError("Coordinates do not fall within the expected region. Check EPSG:9377.")

    resultado, metros, clusters = optimizar(df, CFG)
    resultado["ORDEN_VISITA"] = range(1, len(resultado) + 1)
    resultado.to_csv(CFG.salida, sep=";", encoding="utf-8-sig", index=False)
    print(f"Optimized route: {len(resultado)} points, {clusters} clusters, {metros / 1000:.2f} km in straight line")
    print(f"File created: {CFG.salida}")


if __name__ == "__main__":
    main()