#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nss_services.py - Minimal client for USGS StreamStats NSS Services.

Goal
----
Fetch published regression equation metadata (and when possible, coefficients)
from the NSS Services API, and cache responses locally.

Notes
-----
- NSS Services documentation (REST example): GET /nssservices/regions (Host: streamstats.usgs.gov). 
- StreamStats has announced changes/decommissioning for some services; keep this client resilient by:
  * Making base_url configurable
  * Caching responses
  * Falling back to local registry coefficients when endpoints/fields change
"""


import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

log = logging.getLogger("river.nss")

@dataclass
class NSSConfig:
    enabled: bool = True
    base_url: str = "https://streamstats.usgs.gov/nssservices"
    timeout_s: int = 20
    cache_dir: str = "cache/nss"
    cache_ttl_days: int = 30
    user_agent: str = "cudem-river-bathy (NSSServices client)"

def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]

def _cache_path(cfg: NSSConfig, url: str) -> str:
    _safe_mkdir(cfg.cache_dir)
    return os.path.join(cfg.cache_dir, f"{_cache_key(url)}.json")

def _cache_ok(path: str, ttl_days: int) -> bool:
    if not os.path.exists(path):
        return False
    age_s = time.time() - os.path.getmtime(path)
    return age_s <= (ttl_days * 86400)

def _http_get_json(url: str, timeout_s: int, user_agent: str) -> Any:
    req = Request(url, headers={"Accept": "application/json", "User-Agent": user_agent})
    with urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def cached_get_json(cfg: NSSConfig, url: str) -> Any:
    """
    GET json, with on-disk caching (best-effort).
    """
    path = _cache_path(cfg, url)
    if _cache_ok(path, cfg.cache_ttl_days):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            log.debug("Optional step failed; continuing.", exc_info=True)

    obj = _http_get_json(url, cfg.timeout_s, cfg.user_agent)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        log.debug("Failed to write cache %s", path)

    return obj

class NSSClient:
    def __init__(self, cfg: NSSConfig):
        self.cfg = cfg

    def _url(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        base = self.cfg.base_url.rstrip("/") + "/"
        u = urljoin(base, path.lstrip("/"))
        if params:
            u = u + ("?" + urlencode(params))
        return u

    def get_regions(self) -> List[Dict[str, Any]]:
        """
        Return region list (states/metro areas). Endpoint is documented by example:
        GET /nssservices/regions. 
        """
        url = self._url("/regions")
        obj = cached_get_json(self.cfg, url)
        if isinstance(obj, dict):
            # Some deployments wrap list in 'regions'
            for k in ("regions", "Regions", "data"):
                if k in obj and isinstance(obj[k], list):
                    return obj[k]
        return obj if isinstance(obj, list) else []

    def pick_region_for_state(self, state_abbr: str) -> Optional[Dict[str, Any]]:
        st = state_abbr.upper()
        for r in self.get_regions():
            # common fields seen in various StreamStats APIs
            states = r.get("states") or r.get("States") or r.get("state") or r.get("State")
            if isinstance(states, str):
                if st in [s.strip().upper() for s in states.replace(";", ",").split(",")]:
                    return r
            if isinstance(states, list):
                if st in [str(s).upper() for s in states]:
                    return r
            # sometimes region id == state
            rid = str(r.get("code") or r.get("Code") or r.get("regionID") or r.get("RegionID") or r.get("id") or r.get("ID") or "")
            if rid.upper() == st:
                return r
        return None

    def get_region_statistics(self, region_id: str) -> List[Dict[str, Any]]:
        """
        Try a few patterns used across NSS deployments.
        """
        rid = str(region_id)
        paths = [
            f"/regions/{rid}/statistics",
            f"/regions/{rid}/Statistics",
            f"/regions/{rid}/stats",
        ]
        for p in paths:
            try:
                obj = cached_get_json(self.cfg, self._url(p))
                if isinstance(obj, dict):
                    for k in ("statistics", "Statistics", "data"):
                        if k in obj and isinstance(obj[k], list):
                            return obj[k]
                if isinstance(obj, list):
                    return obj
            except Exception:
                continue
        return []

    def get_stat_detail(self, region_id: str, stat_id: str) -> Dict[str, Any]:
        rid = str(region_id)
        sid = str(stat_id)
        paths = [
            f"/regions/{rid}/statistics/{sid}",
            f"/regions/{rid}/Statistics/{sid}",
            f"/regions/{rid}/stats/{sid}",
        ]
        for p in paths:
            try:
                obj = cached_get_json(self.cfg, self._url(p))
                return obj if isinstance(obj, dict) else {"data": obj}
            except Exception:
                continue
        return {}

    def find_peakflow_q2_equation(self, state_abbr: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Best-effort attempt to locate the 2-year peak flow regression.

        Returns (equation_dict, debug_message).
        """
        if not self.cfg.enabled:
            return None, "NSS disabled"
        reg = self.pick_region_for_state(state_abbr)
        if not reg:
            return None, f"No NSS region found for state={state_abbr}"
        rid = str(reg.get("regionID") or reg.get("RegionID") or reg.get("code") or reg.get("Code") or reg.get("id") or reg.get("ID") or state_abbr)
        stats = self.get_region_statistics(rid)
        # Heuristics: look for Q2, 2-year, P2, PK2
        candidates = []
        for s in stats:
            name = str(s.get("name") or s.get("Name") or s.get("statisticName") or s.get("StatisticName") or "")
            sid = str(s.get("id") or s.get("ID") or s.get("statisticID") or s.get("StatisticID") or "")
            key = (name + " " + sid).lower()
            if any(tok in key for tok in ("2-year", "2 year", "q2", "pk2", "p2", "peak2", "peak flow 2")):
                candidates.append((sid, name))
        if not candidates and stats:
            # last resort: try any statistic with recurrence interval == 2
            for s in stats:
                ri = s.get("recurrenceInterval") or s.get("RecurrenceInterval")
                if str(ri) in ("2", "2.0"):
                    name = str(s.get("name") or s.get("Name") or "")
                    sid = str(s.get("id") or s.get("ID") or "")
                    candidates.append((sid, name))
        if not candidates:
            return None, f"Region={rid}: no Q2-like statistic found (n={len(stats)})"
        sid, name = candidates[0]
        detail = self.get_stat_detail(rid, sid)
        return detail, f"Region={rid} stat={sid} ({name})"
