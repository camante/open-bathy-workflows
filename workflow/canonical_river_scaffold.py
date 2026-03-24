from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple


_DEFAULT_MAINSTEM_COMPONENT = "all_edges"


def _parse_aoi(aoi: str) -> Tuple[float, float, float, float]:
    west, east, south, north = [float(x) for x in str(aoi).split('/')[:4]]
    return west, east, south, north


def _format_aoi(bounds: Tuple[float, float, float, float]) -> str:
    return '/'.join(f"{v:.12g}" for v in bounds)


def _bounds_to_dict(bounds: Tuple[float, float, float, float]) -> Dict[str, float]:
    west, east, south, north = bounds
    return {
        'west': float(west),
        'east': float(east),
        'south': float(south),
        'north': float(north),
    }


def expand_aoi(aoi: str, halo_km: float) -> str:
    west, east, south, north = _parse_aoi(aoi)
    if halo_km <= 0:
        return _format_aoi((west, east, south, north))
    halo_deg_lat = halo_km / 111.32
    mid_lat = max(min((south + north) / 2.0, 89.0), -89.0)
    import math

    halo_deg_lon = halo_km / max(111.32 * math.cos(math.radians(mid_lat)), 1.0)
    return _format_aoi((west - halo_deg_lon, east + halo_deg_lon, south - halo_deg_lat, north + halo_deg_lat))


@dataclass(frozen=True)
class RiverAoiDomains:
    export_aoi: str
    solve_aoi: str
    scaffold_aoi: str
    halo_km: float
    trusted_halo_m: float

    @property
    def export_bounds(self) -> Tuple[float, float, float, float]:
        return _parse_aoi(self.export_aoi)

    @property
    def solve_bounds(self) -> Tuple[float, float, float, float]:
        return _parse_aoi(self.solve_aoi)

    @property
    def scaffold_bounds(self) -> Tuple[float, float, float, float]:
        return _parse_aoi(self.scaffold_aoi)

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.update({
            'export_bounds': _bounds_to_dict(self.export_bounds),
            'solve_bounds': _bounds_to_dict(self.solve_bounds),
            'scaffold_bounds': _bounds_to_dict(self.scaffold_bounds),
            'export_role': 'delivered tile/domain for downstream conditioning',
            'solve_role': 'halo-expanded river solve domain used to reduce AOI-edge sensitivity',
            'scaffold_role': 'canonical scaffold domain used for stable river topology and guidance generation',
            'trusted_export_role': 'trusted export interior inset inside the solve domain; only this export interior may influence final river conditioning',
            'domain_contract': 'scaffold/solve domain is halo-expanded; export/trusted interior is constrained to inset channel interior and excludes estuary-transition pixels',
        })
        return payload


def get_river_aoi_domains(aoi: str, halo_km: float, trusted_halo_m: float = 0.0) -> RiverAoiDomains:
    export_aoi = _format_aoi(_parse_aoi(aoi))
    solve_aoi = expand_aoi(export_aoi, halo_km)
    return RiverAoiDomains(
        export_aoi=export_aoi,
        solve_aoi=solve_aoi,
        scaffold_aoi=solve_aoi,
        halo_km=float(halo_km),
        trusted_halo_m=float(trusted_halo_m),
    )


def scaffold_domain_metadata(aoi: str, halo_km: float, trusted_halo_m: float) -> Dict[str, object]:
    return get_river_aoi_domains(aoi, halo_km, trusted_halo_m).as_dict()


def scaffold_identity_payload(*, domains: RiverAoiDomains, hydrography_source: str, tnm_dataset: str, tnm_enable: bool, snap_m: float, da_raster_fingerprint: str | None = None, da_raster_band: int = 1, da_raster_units: str = "km2") -> Dict[str, object]:
    return {
        "scaffold_aoi": domains.scaffold_aoi,
        "solve_aoi": domains.solve_aoi,
        "export_aoi": domains.export_aoi,
        "halo_km": float(domains.halo_km),
        "trusted_halo_m": float(domains.trusted_halo_m),
        "hydrography_source": str(hydrography_source),
        "tnm_dataset": str(tnm_dataset),
        "tnm_enable": bool(tnm_enable),
        "snap_m": float(snap_m),
        "da_raster_fingerprint": da_raster_fingerprint,
        "da_raster_band": int(da_raster_band or 1),
        "da_raster_units": str(da_raster_units or "km2"),
    }


def scaffold_identity_hash(*, domains: RiverAoiDomains, hydrography_source: str, tnm_dataset: str, tnm_enable: bool, snap_m: float, da_raster_fingerprint: str | None = None, da_raster_band: int = 1, da_raster_units: str = "km2") -> str:
    payload = scaffold_identity_payload(
        domains=domains,
        hydrography_source=hydrography_source,
        tnm_dataset=tnm_dataset,
        tnm_enable=tnm_enable,
        snap_m=snap_m,
        da_raster_fingerprint=da_raster_fingerprint,
        da_raster_band=da_raster_band,
        da_raster_units=da_raster_units,
    )
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def scaffold_cache_dir(*, cache_root: str | Path, domains: RiverAoiDomains, hydrography_source: str, tnm_dataset: str, tnm_enable: bool, snap_m: float, da_raster_fingerprint: str | None = None, da_raster_band: int = 1, da_raster_units: str = "km2") -> Path:
    ident = scaffold_identity_hash(
        domains=domains,
        hydrography_source=hydrography_source,
        tnm_dataset=tnm_dataset,
        tnm_enable=tnm_enable,
        snap_m=snap_m,
        da_raster_fingerprint=da_raster_fingerprint,
        da_raster_band=da_raster_band,
        da_raster_units=da_raster_units,
    )
    return Path(cache_root) / "river_scaffold" / ident


def cached_scaffold_paths(*, cache_root: str | Path, domains: RiverAoiDomains, hydrography_source: str, tnm_dataset: str, tnm_enable: bool, snap_m: float, da_raster_fingerprint: str | None = None, da_raster_band: int = 1, da_raster_units: str = "km2") -> Dict[str, str]:
    cache_dir = scaffold_cache_dir(
        cache_root=cache_root,
        domains=domains,
        hydrography_source=hydrography_source,
        tnm_dataset=tnm_dataset,
        tnm_enable=tnm_enable,
        snap_m=snap_m,
        da_raster_fingerprint=da_raster_fingerprint,
        da_raster_band=da_raster_band,
        da_raster_units=da_raster_units,
    )
    return {
        "cache_dir": str(cache_dir),
        "network_gpkg": str(cache_dir / "river_network.gpkg"),
        "network_manifest": str(cache_dir / "river_network_manifest.json"),
        "hydrologic_solve_domain": str(cache_dir / "hydrologic_solve_domain.json"),
        "manifest": str(cache_dir / "river_scaffold_manifest.json"),
    }


def scaffold_cache_hit(*, manifest_path: str | Path, expected_network_gpkg: str | Path | None = None) -> bool:
    manifest = Path(manifest_path)
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if str(payload.get("network_status")) != "success":
        return False
    network_path = Path(expected_network_gpkg) if expected_network_gpkg else Path(str(payload.get("network_gpkg") or ""))
    return network_path.exists()





def _topology_signature(edges) -> str:
    cols = [c for c in ["from_node", "to_node", "component_id", "length_m", "s_m_min", "s_m_max"] if c in edges.columns]
    if not cols:
        return hashlib.sha1(str(len(edges)).encode("utf-8")).hexdigest()[:16]
    payload = []
    for _, row in edges[cols].fillna("nan").iterrows():
        payload.append([str(row[c]) for c in cols])
    return hashlib.sha1(json.dumps(payload, sort_keys=False).encode("utf-8")).hexdigest()[:16]


def _dominant_component_summary(edges) -> Dict[str, object]:
    if len(edges) == 0 or "component_id" not in edges.columns:
        return {
            "component_id": _DEFAULT_MAINSTEM_COMPONENT,
            "length_m": None,
            "selection_basis": "all_edges",
        }
    work = edges.copy()
    length_col = "length_m" if "length_m" in work.columns else None
    if length_col is not None:
        work[length_col] = work[length_col].astype(float).fillna(0.0)
        grouped = work.groupby("component_id", dropna=False)[length_col].sum().sort_values(ascending=False)
        component_id = grouped.index[0]
        total_length = float(grouped.iloc[0])
        return {
            "component_id": str(component_id),
            "length_m": total_length,
            "selection_basis": "max_component_length_m",
        }
    counts = work.groupby("component_id", dropna=False).size().sort_values(ascending=False)
    component_id = counts.index[0]
    return {
        "component_id": str(component_id),
        "length_m": None,
        "selection_basis": "max_component_edge_count",
    }


def _stationing_basis_payload(edges, dominant_component_id: str) -> Dict[str, object]:
    station_fields = [c for c in ["s_m_from", "s_m_to", "s_m_min", "s_m_max"] if c in edges.columns]
    basis = {
        "stationing_fields_present": station_fields,
        "mainstem_component_id": dominant_component_id,
        "stationing_basis": "graph_edges station fields exported from canonical scaffold network",
    }
    if len(edges) == 0 or not station_fields:
        basis["stationing_origin_m"] = None
        basis["stationing_terminus_m"] = None
        return basis
    vals = []
    for col in station_fields:
        try:
            vals.extend([float(v) for v in edges[col].dropna().tolist()])
        except (TypeError, ValueError):
            continue
    if not vals:
        basis["stationing_origin_m"] = None
        basis["stationing_terminus_m"] = None
        return basis
    basis["stationing_origin_m"] = float(min(vals))
    basis["stationing_terminus_m"] = float(max(vals))
    basis["stationing_span_m"] = float(max(vals) - min(vals))
    return basis


def scaffold_product_paths(*, cache_root: str | Path, domains: RiverAoiDomains, hydrography_source: str, tnm_dataset: str, tnm_enable: bool, snap_m: float, da_raster_fingerprint: str | None = None, da_raster_band: int = 1, da_raster_units: str = "km2") -> Dict[str, str]:
    cache = cached_scaffold_paths(
        cache_root=cache_root,
        domains=domains,
        hydrography_source=hydrography_source,
        tnm_dataset=tnm_dataset,
        tnm_enable=tnm_enable,
        snap_m=snap_m,
        da_raster_fingerprint=da_raster_fingerprint,
        da_raster_band=da_raster_band,
        da_raster_units=da_raster_units,
    )
    cache_dir = Path(cache["cache_dir"])
    return {
        **cache,
        "graph_edges_gpkg": str(cache_dir / "scaffold_graph_edges.gpkg"),
        "graph_nodes_gpkg": str(cache_dir / "scaffold_graph_nodes.gpkg"),
        "mainstem_edges_gpkg": str(cache_dir / "scaffold_mainstem_edges.gpkg"),
        "stationing_json": str(cache_dir / "scaffold_stationing_basis.json"),
        "summary_json": str(cache_dir / "scaffold_summary.json"),
    }


def persist_scaffold_products(*, network_gpkg: str | Path, product_paths: Dict[str, str]) -> Dict[str, object]:
    """Persist scaffold-derived downstream products from the canonical network GPKG.

    This exports reusable scaffold products beyond the raw halo-domain network:
    graph edges/nodes, a dominant-component (mainstem-like) edge subset, stationing
    basis metadata, and a topology summary. The goal is to make phase-3 scaffold
    caching produce deterministic downstream artifacts rather than only a cached
    extraction.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("geopandas is required to persist scaffold products") from exc

    net = Path(network_gpkg)
    if not net.exists():
        raise FileNotFoundError(f"network_gpkg not found: {net}")

    edges = gpd.read_file(net, layer="graph_edges")
    nodes = gpd.read_file(net, layer="graph_nodes")

    edges_path = Path(product_paths["graph_edges_gpkg"])
    nodes_path = Path(product_paths["graph_nodes_gpkg"])
    mainstem_path = Path(product_paths["mainstem_edges_gpkg"])
    station_path = Path(product_paths["stationing_json"])
    summary_path = Path(product_paths["summary_json"])
    for out in [edges_path, nodes_path, mainstem_path, station_path, summary_path]:
        out.parent.mkdir(parents=True, exist_ok=True)

    for out in [edges_path, nodes_path, mainstem_path]:
        if out.exists():
            out.unlink()

    edges.to_file(edges_path, layer="graph_edges", driver="GPKG")
    nodes.to_file(nodes_path, layer="graph_nodes", driver="GPKG")

    dominant = _dominant_component_summary(edges)
    component_id = dominant["component_id"]
    if component_id != _DEFAULT_MAINSTEM_COMPONENT and "component_id" in edges.columns:
        mainstem_edges = edges.loc[edges["component_id"].astype(str) == str(component_id)].copy()
    else:
        mainstem_edges = edges.copy()
    mainstem_edges.to_file(mainstem_path, layer="graph_edges", driver="GPKG")

    stationing = _stationing_basis_payload(mainstem_edges if len(mainstem_edges) else edges, str(component_id))
    summary = {
        "graph_edges_count": int(len(edges)),
        "graph_nodes_count": int(len(nodes)),
        "mainstem_component_id": str(component_id),
        "mainstem_selection_basis": dominant["selection_basis"],
        "mainstem_length_m": dominant["length_m"],
        "mainstem_edges_count": int(len(mainstem_edges)),
        "edge_columns": [str(c) for c in edges.columns],
        "node_columns": [str(c) for c in nodes.columns],
        "stationing_fields_present": stationing.get("stationing_fields_present", []),
        "topology_signature": _topology_signature(edges),
    }
    station_path.write_text(json.dumps(stationing, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "graph_edges_gpkg": str(edges_path),
        "graph_nodes_gpkg": str(nodes_path),
        "mainstem_edges_gpkg": str(mainstem_path),
        "mainstem_solve_layer_name": "mainstem_solve_network",
        "stationing_json": str(station_path),
        "summary_json": str(summary_path),
        "graph_edges_count": int(len(edges)),
        "graph_nodes_count": int(len(nodes)),
        "mainstem_component_id": str(component_id),
        "mainstem_edges_count": int(len(mainstem_edges)),
        "topology_signature": summary["topology_signature"],
        "stationing_basis": stationing,
        "summary": summary,
    }




def scaffold_products_complete(product_paths: Dict[str, str]) -> bool:
    required = [
        "graph_edges_gpkg",
        "graph_nodes_gpkg",
        "mainstem_edges_gpkg",
        "stationing_json",
        "summary_json",
    ]
    return all(Path(product_paths[k]).exists() for k in required)

def build_scaffold_manifest(*, domains: RiverAoiDomains, network_gpkg: str | None = None, provenance_lock: str | None = None) -> Dict[str, object]:
    payload = domains.as_dict()
    payload.update({
        'scaffold_product_type': 'canonical_river_scaffold_domain_manifest',
        'network_gpkg': str(network_gpkg) if network_gpkg else None,
        'provenance_lock': str(provenance_lock) if provenance_lock else None,
        'scaffold_artifacts_definition': {
            'network_gpkg': 'stable halo-domain river network product generated on scaffold_aoi when available',
            'network_manifest': 'explicit receipt for export/solve/scaffold AOIs, halo rationale, named mainstem_solve_network layer, and hydrologic solve-domain linkage used by river_network.py',
            'hydrologic_solve_domain': 'explicit hydrologic solve-domain contract including outlet anchors, estuary handoff proxies, major-system selection, and deterministic tie-break rules',
            'provenance_lock': 'hashable provenance lock for the scaffold-domain network build',
            'trusted_export_role': payload['trusted_export_role'],
            'scaffold_products': 'cached downstream products exported from the canonical scaffold network (graph edges/nodes, dominant-component edges, stationing basis, topology summary)',
        },
    })
    return payload


def write_scaffold_manifest(path: str | Path, *, domains: RiverAoiDomains, network_gpkg: str | None = None, provenance_lock: str | None = None, extra: Dict[str, object] | None = None) -> Path:
    out = Path(path)
    payload = build_scaffold_manifest(domains=domains, network_gpkg=network_gpkg, provenance_lock=provenance_lock)
    if extra:
        payload.update(extra)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out



def nested_aoi_relationship(export_aoi: str, candidate_aoi: str) -> Dict[str, object]:
    """Describe whether candidate_aoi contains the export_aoi and the overlap area."""
    ew, ee, es, en = _parse_aoi(export_aoi)
    cw, ce, cs, cn = _parse_aoi(candidate_aoi)
    overlap_w = max(ew, cw)
    overlap_e = min(ee, ce)
    overlap_s = max(es, cs)
    overlap_n = min(en, cn)
    overlap_wd = max(0.0, overlap_e - overlap_w)
    overlap_hd = max(0.0, overlap_n - overlap_s)
    export_area = max(0.0, ee - ew) * max(0.0, en - es)
    overlap_area = overlap_wd * overlap_hd
    contains_export = (cw <= ew) and (ce >= ee) and (cs <= es) and (cn >= en)
    return {
        'contains_export_aoi': bool(contains_export),
        'overlap_fraction_of_export': float(overlap_area / max(export_area, 1e-12)),
        'export_bounds': _bounds_to_dict((ew, ee, es, en)),
        'candidate_bounds': _bounds_to_dict((cw, ce, cs, cn)),
        'overlap_bounds': _bounds_to_dict((overlap_w, overlap_e, overlap_s, overlap_n)) if overlap_area > 0 else None,
    }
