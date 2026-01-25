#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
region_resolver.py - Resolve AOI -> State -> Regression/Curve coefficients.

This module provides:
- Reverse-geocode AOI centroid to US state abbreviation (via Census geocoder).
- Resolve bankfull depth curve coefficients and Q2 regressions from:
  1) NSS Services (best-effort, cached)
  2) Local registry in sdb_config.json (authoritative fallback)

Scientific intent: avoid hard-coded "example" coefficients and make the pipeline
able to use *published* state/regional regressions when available.
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from nss_services import NSSClient, NSSConfig

log = logging.getLogger("river.region")

_CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"

def _http_get_json(url: str, timeout_s: int = 20, user_agent: str = "cudem-river-bathy") -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": user_agent})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def state_from_lonlat(lon: float, lat: float, timeout_s: int = 20) -> Optional[str]:
    """
    Return 2-letter US state postal abbreviation for a lon/lat point.
    Best-effort; returns None if unavailable.
    """
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    url = _CENSUS_URL + "?" + urlencode(params)
    try:
        obj = _http_get_json(url, timeout_s=timeout_s)
        # Navigate Census geocoder response structure
        res = (obj or {}).get("result", {}).get("geographies", {})
        # Prefer States
        states = res.get("States") or res.get("State") or []
        if isinstance(states, list) and states:
            st = states[0].get("STUSAB") or states[0].get("stusab")
            if st:
                return str(st).upper()
        # fallback: other structures
        for v in res.values():
            if isinstance(v, list) and v:
                st = v[0].get("STUSAB") or v[0].get("stusab")
                if st:
                    return str(st).upper()
    except Exception as e:
        log.debug("state_from_lonlat failed: %s", e)
    return None

@dataclass
class Curve:
    c: float
    f: float
    da_units: str
    y_units: str
    source: str
    name: str

def _registry_lookup(cfg: Dict[str, Any], kind: str, key: str) -> Optional[Curve]:
    reg = (cfg.get("river") or {}).get("registry") or {}
    if kind == "bankfull":
        table = reg.get("bankfull_depth_curves") or {}
        entry = table.get(key)
        if not isinstance(entry, dict):
            return None
        return Curve(
            c=float(entry["c"]),
            f=float(entry["f"]),
            da_units=str(entry.get("da_units","km2")),
            y_units=str(entry.get("depth_units","m")),
            source=str(entry.get("source","registry")),
            name=key,
        )
    if kind == "q2":
        table = reg.get("q2_regressions") or {}
        entry = table.get(key)
        if not isinstance(entry, dict):
            return None
        return Curve(
            c=float(entry["c"]),
            f=float(entry["f"]),
            da_units=str(entry.get("da_units","km2")),
            y_units=str(entry.get("q_units","cms")),
            source=str(entry.get("source","registry")),
            name=key,
        )
    return None

def resolve_region_key(cfg: Dict[str, Any], state_abbr: str, kind: str) -> Optional[str]:
    reg = (cfg.get("river") or {}).get("registry") or {}
    rules = reg.get("auto_region_rules") or {}
    st = state_abbr.upper()
    if st in rules and isinstance(rules[st], dict):
        return rules[st].get(kind)
    return None

def resolve_bankfull_curve(cfg: Dict[str, Any], lon: float, lat: float, cache_root: Optional[str] = None) -> Tuple[Optional[Curve], str]:
    st = state_from_lonlat(lon, lat, timeout_s=int(((cfg.get("river") or {}).get("nss_services") or {}).get("timeout_s", 20)))
    if not st:
        return None, "Could not resolve state for AOI centroid"
    key = resolve_region_key(cfg, st, "bankfull") or "default"
    curve = _registry_lookup(cfg, "bankfull", key)
    if curve:
        return curve, f"Registry bankfull curve: {key} (state={st})"
    return None, f"No bankfull curve in registry for state={st} key={key}"

def resolve_q2_regression(cfg: Dict[str, Any], lon: float, lat: float, cache_root: Optional[str] = None) -> Tuple[Optional[Curve], str]:
    river_cfg = cfg.get("river") or {}
    nss_cfg_d = river_cfg.get("nss_services") or {}
    st = state_from_lonlat(lon, lat, timeout_s=int(nss_cfg_d.get("timeout_s", 20)))
    if not st:
        return None, "Could not resolve state for AOI centroid"
    # 1) Try NSS services for Q2 equation detail (best-effort).
    try:
        nss_cfg = NSSConfig(
            enabled=bool(nss_cfg_d.get("enabled", True)),
            base_url=str(nss_cfg_d.get("base_url", "https://streamstats.usgs.gov/nssservices")),
            timeout_s=int(nss_cfg_d.get("timeout_s", 20)),
            cache_dir=str(nss_cfg_d.get("cache_dir") or (os.path.join(cache_root, "nss") if cache_root else "cache/nss")),
            cache_ttl_days=int(nss_cfg_d.get("cache_ttl_days", 30)),
            user_agent=str(nss_cfg_d.get("user_agent", "cudem-river-bathy")),
        )
        client = NSSClient(nss_cfg)
        detail, msg = client.find_peakflow_q2_equation(st)
        if detail:
            # We don't assume field names; store raw detail and let higher-level
            # code decide how to use it (often StreamStats computes Q2 from basin chars).
            return Curve(c=float("nan"), f=float("nan"), da_units="unknown", y_units="cms",
                         source=f"NSSServices (raw detail cached) | {msg}", name=f"{st}_NSS_Q2"), msg
    except Exception as e:
        log.debug("NSS lookup failed: %s", e)

    # 2) Fallback to registry power law (DA -> Q2).
    key = resolve_region_key(cfg, st, "q2") or "default"
    curve = _registry_lookup(cfg, "q2", key)
    if curve:
        return curve, f"Registry Q2 regression: {key} (state={st})"
    return None, f"No Q2 regression available for state={st} (and NSS lookup failed/unavailable)"
