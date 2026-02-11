from datetime import datetime

def vehicule_prioritisation(user_profile, departure_time, charge_level):
    """
    Evaluate the priority score of a vehicle based on user profile,
    departure time, and battery charge level.

    Parameters:
    user_profile: str - 'Urgences', 'Administration', or 'Autres'
    departure_time: str - 'HH:MM'
    charge_level: float - Battery charge level (0.0 to 1.0)

    Returns:
    priority_score: float
    """

    # Priority based on user profile
    if user_profile == 'Urgences':
        Pu = 7
    elif user_profile == 'Administration':
        Pu = 2
    elif user_profile == 'Autres':
        Pu = 1
    else:
        raise ValueError("Invalid user profile")

    # Current time and departure time as datetime
    now = datetime.now()
    dep_time = datetime.strptime(departure_time, "%H:%M")

    # Attach today's date to departure time
    dep_time = dep_time.replace(
        year=now.year, month=now.month, day=now.day
    )

    # If departure is earlier than now → assume next day
    if dep_time < now:
        dep_time = dep_time.replace(day=now.day + 1)

    # Difference in minutes
    time_difference = (dep_time - now).total_seconds() / 60

    # Priority based on time difference
    if time_difference < 60:
        H = 3
    elif time_difference < 120:
        H = 2
    else:
        H = 1

    # Final priority score
    priority_score = Pu + H + 2 * (1 - charge_level)

    return priority_score

# ============================================================
# EMS + SCENARIOS + COURBES + TABLEAU CONTRAINTES (AUDIT)
# À coller DANS LE MÊME NOTEBOOK que "score de priorisation"
# IMPORTANT: la fonction vehicule_prioritisation(...) doit déjà exister
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# 0) Sécurité: vérifier que la fonction score existe
# -----------------------------
if "vehicule_prioritisation" not in globals():
    raise NameError(
        "La fonction 'vehicule_prioritisation' n'existe pas dans ce notebook.\n"
        "Colle d'abord le code 'score de priorisation' (avec vehicule_prioritisation) au-dessus."
    )

# -----------------------------
# 1) Modèles EV / BESS
# -----------------------------
@dataclass
class EV:
    ev_id: str
    profile: str                # 'Urgences', 'Administration', 'Autres'
    arrival_min: int
    departure_min: int
    E_max_kWh: float
    P_max_kW: float
    soc_ini_pct: float
    soc_target_pct: float

    soc_pct: float = None
    energy_received_kWh: float = 0.0

    def __post_init__(self):
        self.soc_pct = float(self.soc_ini_pct)

    def active(self, t_min: int) -> bool:
        return (self.arrival_min <= t_min < self.departure_min) and (self.soc_pct < self.soc_target_pct - 1e-9)

    def remaining_need_kWh(self) -> float:
        need_pct = max(self.soc_target_pct - self.soc_pct, 0.0)
        return (need_pct / 100.0) * self.E_max_kWh

@dataclass
class BESS:
    E_max_kWh: float = 600.0
    E_min_kWh: float = 120.0
    P_ch_max_kW: float = 120.0
    P_dis_max_kW: float = 240.0
    eff: float = 0.87
    E_kWh: float = 360.0  # énergie initiale (ex: 60% * 600)

    def soc_pct(self) -> float:
        return 100.0 * self.E_kWh / self.E_max_kWh

# -----------------------------
# 2) Scénarios (PV + Load + EV)
# -----------------------------
SCENARIOS: Dict[str, Dict] = {
    # 3 classiques
    "pluvieux": {
        "pv_peak_kW": 25.0,
        "pv_width_min": 240,
        "load_base_kW": 60.0,
        "load_multiplier": 1.00,
        "n_evs": 20,
        "seed_evs": 42,
        "bess_soc0": 0.60,
    },
    "nuageux": {
        "pv_peak_kW": 120.0,
        "pv_width_min": 260,
        "load_base_kW": 60.0,
        "load_multiplier": 1.00,
        "n_evs": 20,
        "seed_evs": 42,
        "bess_soc0": 0.60,
    },
    "ensoleille": {
        "pv_peak_kW": 200.0,
        "pv_width_min": 300,
        "load_base_kW": 60.0,
        "load_multiplier": 1.00,
        "n_evs": 20,
        "seed_evs": 42,
        "bess_soc0": 0.60,
    },

    # enrichissement
    "ete_canicul": {
        "pv_peak_kW": 230.0,
        "pv_width_min": 320,
        "load_base_kW": 70.0,
        "load_multiplier": 1.15,
        "n_evs": 22,
        "seed_evs": 7,
        "bess_soc0": 0.50,
    },
    "hiver_froid": {
        "pv_peak_kW": 80.0,
        "pv_width_min": 220,
        "load_base_kW": 75.0,
        "load_multiplier": 1.20,
        "n_evs": 18,
        "seed_evs": 101,
        "bess_soc0": 0.70,
    },
    "pic_vehicules": {
        "pv_peak_kW": 120.0,
        "pv_width_min": 260,
        "load_base_kW": 60.0,
        "load_multiplier": 1.00,
        "n_evs": 35,
        "seed_evs": 123,
        "bess_soc0": 0.60,
    },
}

# -----------------------------
# 3) Profils PV / Load
# -----------------------------
def pv_profile_kW(t_min: int, pv_peak_kW: float, pv_width_min: float) -> float:
    """PV gaussienne centrée à midi (720 min). pv_width_min contrôle l'étalement."""
    x = (t_min - 720) / (pv_width_min / 2.0)
    return float(pv_peak_kW * np.exp(-0.5 * x * x))

def load_profile_kW(t_min: int, base_kW: float, mult: float) -> float:
    """Charge bâtiment: base + 3 pics (matin/midi/aprem), mult ajuste tout."""
    hour = t_min / 60.0
    morning = 40.0 * np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
    noon = 55.0 * np.exp(-0.5 * ((hour - 13.0) / 2.0) ** 2)
    afternoon = 45.0 * np.exp(-0.5 * ((hour - 17.5) / 1.8) ** 2)
    return float(mult * (base_kW + morning + noon + afternoon))

# -----------------------------
# 4) Génération d'un parc VE (reproductible)
# -----------------------------
def generate_evs(seed: int, n_evs: int) -> List[EV]:
    rng = np.random.default_rng(seed)
    profiles = rng.choice(["Urgences", "Administration", "Autres"], size=n_evs, p=[0.2, 0.3, 0.5])
    evs: List[EV] = []

    for i in range(n_evs):
        arrival = int(rng.integers(7 * 60, 18 * 60))  # 7h..18h
        stay = int(rng.integers(60, 8 * 60))          # 1h..8h
        departure = min(arrival + stay, 23 * 60 + 55)

        E_max = float(rng.choice([50.0, 52.0, 60.0]))
        P_max = float(rng.choice([6.6, 11.0, 22.0]))
        soc_ini = float(rng.integers(10, 70))
        soc_target = float(rng.choice([70.0, 80.0, 90.0, 100.0]))
        soc_target = max(soc_target, soc_ini + 5)
        soc_target = min(soc_target, 100.0)

        evs.append(EV(
            ev_id=f"EV{i+1}",
            profile=str(profiles[i]),
            arrival_min=arrival,
            departure_min=departure,
            E_max_kWh=E_max,
            P_max_kW=P_max,
            soc_ini_pct=soc_ini,
            soc_target_pct=soc_target
        ))
    return evs

# -----------------------------
# 5) Allocation puissance VE (utilise vehicule_prioritisation)
# -----------------------------
def allocate_ev_power(
    evs: List[EV],
    t_min: int,
    dt_h: float,
    nb_bornes: int,
    P_borne_uni_max: float,
    P_borne_tot_max: float,
) -> Dict[str, float]:
    active = [ev for ev in evs if ev.active(t_min)]
    if not active:
        return {}

    scored: List[Tuple[EV, float]] = []
    for ev in active:
        dep_h = ev.departure_min // 60
        dep_m = ev.departure_min % 60
        dep_str = f"{dep_h:02d}:{dep_m:02d}"
        score = float(vehicule_prioritisation(ev.profile, dep_str, ev.soc_pct / 100.0))
        scored.append((ev, score))

    scored.sort(key=lambda x: x[1], reverse=True)  # tri décroissant
    selected = scored[:nb_bornes]                  # max nb bornes

    alloc: Dict[str, float] = {}
    remaining_total = P_borne_tot_max

    for ev, _s in selected:
        if remaining_total <= 1e-9:
            break

        P_cap = min(ev.P_max_kW, P_borne_uni_max)
        need_kWh = ev.remaining_need_kWh()
        P_need = need_kWh / dt_h if dt_h > 0 else 0.0
        P_set = min(P_cap, P_need, remaining_total)

        if P_set > 1e-9:
            alloc[ev.ev_id] = float(P_set)
            remaining_total -= P_set

    return alloc

# -----------------------------
# 6) KPI
# -----------------------------
def jain_index(x: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    if np.all(x <= 1e-12):
        return 1.0
    return float((x.sum() ** 2) / (len(x) * (x**2).sum() + 1e-12))

# -----------------------------
# 7) Simulation journée + AUDIT CONTRAINTES
# -----------------------------
def simulate_day(
    scenario_name: str,
    dt_min: int = 5,
    nb_bornes: int = 6,
    P_borne_uni_max: float = 22.0,
    P_borne_tot_max: float = 132.0,
    P_grid_max: float = 300.0,
    tariff_hp: float = 0.24,
    tariff_hc: float = 0.17,
) -> Dict:
    sc = SCENARIOS[scenario_name]

    # init BESS
    bess = BESS(
        E_max_kWh=600.0,
        E_min_kWh=120.0,
        P_ch_max_kW=120.0,
        P_dis_max_kW=240.0,
        eff=0.87,
        E_kWh=float(sc["bess_soc0"] * 600.0),
    )

    # init EVs
    evs = generate_evs(seed=int(sc["seed_evs"]), n_evs=int(sc["n_evs"]))
    ev_ids = [ev.ev_id for ev in evs]

    # temps
    n_steps = int(24 * 60 / dt_min)
    t_min_arr = np.arange(n_steps) * dt_min
    dt_h = dt_min / 60.0

    # historiques
    pv_hist = np.zeros(n_steps)
    load_hist = np.zeros(n_steps)
    ev_hist = np.zeros(n_steps)
    grid_imp_hist = np.zeros(n_steps)
    grid_exp_hist = np.zeros(n_steps)
    bess_p_hist = np.zeros(n_steps)        # +décharge, -charge
    bess_soc_hist = np.zeros(n_steps)
    sci_inst_hist = np.full(n_steps, np.nan)
    cost_hp_cum = np.zeros(n_steps)
    cost_hc_cum = np.zeros(n_steps)
    soc_ev_hist = np.zeros((n_steps, len(evs)))

    # --- Historiques contraintes (audit) ---
    n_bornes_used_hist = np.zeros(n_steps, dtype=int)
    p_borne_max_hist = np.zeros(n_steps)     # max puissance sur une borne (kW)
    p_ev_target_hist = np.zeros(n_steps)     # avant curtailment réseau (kW)
    grid_over_hist = np.zeros(n_steps)       # dépassement import réseau (kW) si >0

    bess_under_hist = np.zeros(n_steps)      # sous E_min (kWh) si >0
    bess_over_hist = np.zeros(n_steps)       # au-dessus E_max (kWh) si >0
    bess_dis_over_hist = np.zeros(n_steps)   # dépassement P_dis_max (kW) si >0
    bess_ch_over_hist = np.zeros(n_steps)    # dépassement P_ch_max (kW) si >0

    # cumul SCI
    pv_energy_kWh = 0.0
    export_energy_kWh = 0.0

    # cumul coût
    cum_hp = 0.0
    cum_hc = 0.0

    for k in range(n_steps):
        t_min = int(t_min_arr[k])
        hour = t_min / 60.0
        is_hc = (hour >= 22.0) or (hour < 6.0)

        pv = pv_profile_kW(t_min, sc["pv_peak_kW"], sc["pv_width_min"])
        load = load_profile_kW(t_min, sc["load_base_kW"], sc["load_multiplier"])

        pv_hist[k] = pv
        load_hist[k] = load

        # (1) PV -> bâtiment
        pv_to_build = min(pv, load)
        remaining_pv = pv - pv_to_build
        deficit_build = load - pv_to_build  # >=0
        grid_import = deficit_build

        # (2) Allocation VE via score
        alloc = allocate_ev_power(
            evs=evs,
            t_min=t_min,
            dt_h=dt_h,
            nb_bornes=nb_bornes,
            P_borne_uni_max=P_borne_uni_max,
            P_borne_tot_max=P_borne_tot_max,
        )
        P_ev_target = float(sum(alloc.values()))

        # audit EV/bornes
        n_bornes_used_hist[k] = len(alloc)
        p_borne_max_hist[k] = float(max(alloc.values())) if alloc else 0.0
        p_ev_target_hist[k] = P_ev_target

        # PV -> VE
        pv_to_ev = min(remaining_pv, P_ev_target)
        remaining_pv -= pv_to_ev
        ev_need_after_pv = P_ev_target - pv_to_ev

        # (3) BESS -> VE (décharge) si besoin
        P_bess_dis_internal = 0.0  # interne (avant rendement)
        P_bess_to_ev = 0.0         # utile (après rendement)
        if ev_need_after_pv > 1e-9 and bess.E_kWh > bess.E_min_kWh + 1e-9:
            P_req_internal = ev_need_after_pv / bess.eff
            max_by_power = bess.P_dis_max_kW
            max_by_energy = (bess.E_kWh - bess.E_min_kWh) / dt_h if dt_h > 0 else 0.0
            P_bess_dis_internal = min(P_req_internal, max_by_power, max_by_energy)
            P_bess_to_ev = P_bess_dis_internal * bess.eff
            bess.E_kWh -= P_bess_dis_internal * dt_h

        ev_need_after_pv_bess = ev_need_after_pv - P_bess_to_ev

        # (4) réseau -> reste VE + bâtiment
        grid_import += max(ev_need_after_pv_bess, 0.0)

        # contrainte P_grid_max : on réduit en priorité la charge VE (pas le bâtiment)
        if grid_import > P_grid_max + 1e-9:
            excess = grid_import - P_grid_max
            to_reduce = min(excess, P_ev_target)
            P_ev_target -= to_reduce
            grid_import -= to_reduce

            # réduction sur alloc (on coupe les derniers, stratégie simple)
            for ev_id in list(alloc.keys())[::-1]:
                if to_reduce <= 1e-9:
                    break
                cut = min(alloc[ev_id], to_reduce)
                alloc[ev_id] -= cut
                to_reduce -= cut
                if alloc[ev_id] <= 1e-9:
                    del alloc[ev_id]

        # audit dépassement réseau (après curtailment)
        grid_over_hist[k] = max(grid_import - P_grid_max, 0.0)

        P_ev_final = float(sum(alloc.values()))
        ev_hist[k] = P_ev_final

        # (5) charge BESS en HC (22h-6h) avec marge réseau
        P_bess_ch_from_grid = 0.0
        if is_hc and bess.E_kWh < bess.E_max_kWh - 1e-9:
            margin = max(P_grid_max - grid_import, 0.0)
            if margin > 1e-9:
                max_by_power = bess.P_ch_max_kW
                max_by_energy = (bess.E_max_kWh - bess.E_kWh) / (dt_h * bess.eff) if dt_h > 0 else 0.0
                P_bess_ch_from_grid = min(margin, max_by_power, max_by_energy)
                grid_import += P_bess_ch_from_grid
                bess.E_kWh += P_bess_ch_from_grid * dt_h * bess.eff

        # (6) PV restant -> charge BESS puis export
        P_bess_ch_from_pv = 0.0
        grid_export = 0.0
        if remaining_pv > 1e-9:
            if bess.E_kWh < bess.E_max_kWh - 1e-9:
                max_by_power = bess.P_ch_max_kW
                max_by_energy = (bess.E_max_kWh - bess.E_kWh) / (dt_h * bess.eff) if dt_h > 0 else 0.0
                P_bess_ch_from_pv = min(remaining_pv, max_by_power, max_by_energy)
                bess.E_kWh += P_bess_ch_from_pv * dt_h * bess.eff
                remaining_pv -= P_bess_ch_from_pv
            grid_export = max(remaining_pv, 0.0)

        # update EV SOC
        for i_ev, ev in enumerate(evs):
            if ev.ev_id in alloc:
                P_i = alloc[ev.ev_id]
                e_i = P_i * dt_h
                ev.energy_received_kWh += e_i
                ev.soc_pct = min(100.0, ev.soc_pct + 100.0 * e_i / ev.E_max_kWh)
            soc_ev_hist[k, i_ev] = ev.soc_pct

        # historiques réseau et bess
        grid_imp_hist[k] = grid_import
        grid_exp_hist[k] = grid_export
        bess_p_hist[k] = (P_bess_dis_internal) - (P_bess_ch_from_grid + P_bess_ch_from_pv)
        bess_soc_hist[k] = bess.soc_pct()

        # --- audit BESS (énergie + puissance) ---
        bess_under_hist[k] = max(bess.E_min_kWh - bess.E_kWh, 0.0)
        bess_over_hist[k]  = max(bess.E_kWh - bess.E_max_kWh, 0.0)

        bess_dis_over_hist[k] = max(P_bess_dis_internal - bess.P_dis_max_kW, 0.0)
        P_bess_ch_internal = (P_bess_ch_from_grid + P_bess_ch_from_pv)
        bess_ch_over_hist[k] = max(P_bess_ch_internal - bess.P_ch_max_kW, 0.0)

        # SCI instantané
        pv_e = pv * dt_h
        exp_e = grid_export * dt_h
        pv_energy_kWh += pv_e
        export_energy_kWh += exp_e
        if pv_e > 1e-12:
            sci_inst_hist[k] = 100.0 * (pv_e - exp_e) / pv_e

        # coût (import réseau uniquement)
        price = tariff_hc if is_hc else tariff_hp
        cost_step = grid_import * dt_h * price
        if is_hc:
            cum_hc += cost_step
        else:
            cum_hp += cost_step
        cost_hp_cum[k] = cum_hp
        cost_hc_cum[k] = cum_hc

    # KPI finaux
    sci_global = 100.0 * (pv_energy_kWh - export_energy_kWh) / (pv_energy_kWh + 1e-12)
    energies = np.array([ev.energy_received_kWh for ev in evs], dtype=float)
    iej = jain_index(energies)
    peak_imp = float(np.max(grid_imp_hist))
    E_import_total = float(np.sum(grid_imp_hist) * dt_h)
    cost_total = float(cost_hp_cum[-1] + cost_hc_cum[-1])

    return {
        "scenario": scenario_name,
        "t_min": t_min_arr,
        "pv_kW": pv_hist,
        "load_kW": load_hist,
        "ev_kW": ev_hist,
        "grid_import_kW": grid_imp_hist,
        "grid_export_kW": grid_exp_hist,
        "bess_power_kW": bess_p_hist,
        "bess_soc_pct": bess_soc_hist,
        "soc_ev_pct": soc_ev_hist,
        "ev_ids": ev_ids,
        "SCI_inst_pct": sci_inst_hist,
        "SCI_global_pct": float(sci_global),
        "IEJ": float(iej),
        "peak_grid_import_kW": peak_imp,
        "E_import_total_kWh": E_import_total,
        "cost_hp_cum": cost_hp_cum,
        "cost_hc_cum": cost_hc_cum,
        "cost_total": cost_total,

        # --- AUDIT CONTRAINTES ---
        "audit": {
            "nb_bornes": nb_bornes,
            "P_borne_uni_max": P_borne_uni_max,
            "P_borne_tot_max": P_borne_tot_max,
            "P_grid_max": P_grid_max,
            "bess_E_min_kWh": bess.E_min_kWh,
            "bess_E_max_kWh": bess.E_max_kWh,
            "bess_P_ch_max_kW": bess.P_ch_max_kW,
            "bess_P_dis_max_kW": bess.P_dis_max_kW,

            "n_bornes_used_hist": n_bornes_used_hist,
            "p_borne_max_hist": p_borne_max_hist,
            "p_ev_target_hist": p_ev_target_hist,
            "grid_over_hist": grid_over_hist,

            "bess_under_hist": bess_under_hist,
            "bess_over_hist": bess_over_hist,
            "bess_dis_over_hist": bess_dis_over_hist,
            "bess_ch_over_hist": bess_ch_over_hist,
        },
    }

# -----------------------------
# 8) Courbes (par scénario)
# -----------------------------
def plot_scenario(res: Dict):
    t_h = res["t_min"] / 60.0
    sc = res["scenario"]

    # A) Flux de puissance
    plt.figure()
    plt.plot(t_h, res["pv_kW"], label="PV (kW)", linewidth=2)
    plt.plot(t_h, res["load_kW"], label="Bâtiment (kW)", linewidth=2)
    plt.plot(t_h, res["ev_kW"], label="VE (kW)", linewidth=2)
    plt.plot(t_h, res["grid_import_kW"], label="Grid import (kW)", linewidth=2)
    plt.plot(t_h, res["grid_export_kW"], label="Grid export (kW)", linewidth=2)
    plt.plot(t_h, res["bess_power_kW"], label="BESS (+dis / -ch) (kW)", linewidth=2)
    plt.xlabel("Heure (h)")
    plt.ylabel("Puissance (kW)")
    plt.title(f"Flux de puissance — {sc}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # B) SOC BESS
    plt.figure()
    plt.plot(t_h, res["bess_soc_pct"], linewidth=2)
    plt.xlabel("Heure (h)")
    plt.ylabel("SOC BESS (%)")
    plt.title(f"SOC BESS — {sc}")
    plt.grid(True)
    plt.show()

    # C) SOC VE
    plt.figure()
    for i in range(res["soc_ev_pct"].shape[1]):
        plt.plot(t_h, res["soc_ev_pct"][:, i], linewidth=1)
    plt.xlabel("Heure (h)")
    plt.ylabel("SOC VE (%)")
    plt.title(f"SOC des VE — {sc}")
    plt.grid(True)
    plt.show()

    # D) SCI instant + global
    plt.figure()
    plt.plot(t_h, res["SCI_inst_pct"], linewidth=2)
    plt.xlabel("Heure (h)")
    plt.ylabel("SCI instantané (%)")
    plt.title(f"SCI instantané — {sc} | SCI global = {res['SCI_global_pct']:.1f}%")
    plt.grid(True)
    plt.show()

    # E) Coût cumulé
    plt.figure()
    total = res["cost_hp_cum"] + res["cost_hc_cum"]
    plt.plot(t_h, res["cost_hp_cum"], label="Coût cumulé HP (€)", linewidth=2)
    plt.plot(t_h, res["cost_hc_cum"], label="Coût cumulé HC (€)", linewidth=2)
    plt.plot(t_h, total, label="Total (€)", linewidth=2)
    plt.xlabel("Heure (h)")
    plt.ylabel("Coût cumulé (€)")
    plt.title(f"Coût cumulé — {sc} | Total = {res['cost_total']:.2f} €")
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------
# 9) Courbes comparatives (plusieurs scénarios) + print KPI
# -----------------------------
def plot_compare(results: List[Dict]):
    plt.figure()
    for res in results:
        t_h = res["t_min"] / 60.0
        plt.plot(t_h, res["grid_import_kW"], linewidth=2, label=res["scenario"])
    plt.xlabel("Heure (h)")
    plt.ylabel("Import réseau (kW)")
    plt.title("Comparaison — Import réseau")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    for res in results:
        t_h = res["t_min"] / 60.0
        plt.plot(t_h, res["bess_soc_pct"], linewidth=2, label=res["scenario"])
    plt.xlabel("Heure (h)")
    plt.ylabel("SOC BESS (%)")
    plt.title("Comparaison — SOC BESS")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    for res in results:
        t_h = res["t_min"] / 60.0
        plt.plot(t_h, res["SCI_inst_pct"], linewidth=2, label=res["scenario"])
    plt.xlabel("Heure (h)")
    plt.ylabel("SCI instantané (%)")
    plt.title("Comparaison — SCI instantané")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\n===== KPI COMPARATIFS =====")
    for res in results:
        print(
            f"{res['scenario']:>12} | SCI_global={res['SCI_global_pct']:.2f}%"
            f" | IEJ={res['IEJ']:.4f}"
            f" | Pic_import={res['peak_grid_import_kW']:.1f} kW"
            f" | E_import={res['E_import_total_kWh']:.1f} kWh"
            f" | Coût={res['cost_total']:.2f} €"
        )
    print("==========================\n")

# -----------------------------
# 10) Tableau CONTRAINTES (audit)
# -----------------------------
def build_constraints_table(results: List[Dict]) -> pd.DataFrame:
    rows = []

    for res in results:
        sc = res["scenario"]
        a = res["audit"]

        # signaux
        n_b = a["n_bornes_used_hist"]
        p_bmax = a["p_borne_max_hist"]
        p_ev_final = res["ev_kW"]
        p_ev_target = a["p_ev_target_hist"]
        g_imp = res["grid_import_kW"]

        # violations = nombre de pas de temps en violation
        viol_nb_bornes = int(np.sum(n_b > a["nb_bornes"]))
        viol_p_borne = int(np.sum(p_bmax > a["P_borne_uni_max"] + 1e-9))
        viol_p_tot = int(np.sum(p_ev_final > a["P_borne_tot_max"] + 1e-9))
        viol_grid = int(np.sum(g_imp > a["P_grid_max"] + 1e-9))

        viol_bess_under = int(np.sum(a["bess_under_hist"] > 1e-9))
        viol_bess_over = int(np.sum(a["bess_over_hist"] > 1e-9))
        viol_bess_dis = int(np.sum(a["bess_dis_over_hist"] > 1e-9))
        viol_bess_ch = int(np.sum(a["bess_ch_over_hist"] > 1e-9))

        # energie bess (reconstituée via SOC*Emax)
        E_bess_kWh = res["bess_soc_pct"] * a["bess_E_max_kWh"] / 100.0

        row = {
            "scenario": sc,

            # bornes
            "max_bornes_utilisees": int(np.max(n_b)),
            "limite_bornes": a["nb_bornes"],
            "OK_bornes": viol_nb_bornes == 0,
            "viol_bornes_steps": viol_nb_bornes,

            # P borne
            "max_P_une_borne_kW": float(np.max(p_bmax)),
            "limite_P_une_borne_kW": a["P_borne_uni_max"],
            "OK_P_une_borne": viol_p_borne == 0,
            "viol_P_une_borne_steps": viol_p_borne,

            # P total VE
            "max_P_total_VE_kW": float(np.max(p_ev_final)),
            "limite_P_total_VE_kW": a["P_borne_tot_max"],
            "OK_P_total_VE": viol_p_tot == 0,
            "viol_P_total_VE_steps": viol_p_tot,

            # réseau
            "max_import_reseau_kW": float(np.max(g_imp)),
            "limite_import_reseau_kW": a["P_grid_max"],
            "OK_reseau": viol_grid == 0,
            "viol_reseau_steps": viol_grid,

            # BESS énergie
            "min_E_BESS_kWh": float(np.min(E_bess_kWh)),
            "max_E_BESS_kWh": float(np.max(E_bess_kWh)),
            "Emin_kWh": a["bess_E_min_kWh"],
            "Emax_kWh": a["bess_E_max_kWh"],
            "OK_BESS_energie": (viol_bess_under == 0 and viol_bess_over == 0),
            "viol_BESS_energie_steps": viol_bess_under + viol_bess_over,

            # BESS puissance
            "OK_BESS_puissance": (viol_bess_dis == 0 and viol_bess_ch == 0),
            "viol_BESS_puissance_steps": viol_bess_dis + viol_bess_ch,

            # info curtailment
            "P_EV_target_max_kW": float(np.max(p_ev_target)),
            "P_EV_final_max_kW": float(np.max(p_ev_final)),
            "max_depassement_reseau_kW": float(np.max(a["grid_over_hist"])),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    df["ALL_OK"] = (
        df["OK_bornes"] &
        df["OK_P_une_borne"] &
        df["OK_P_total_VE"] &
        df["OK_reseau"] &
        df["OK_BESS_energie"] &
        df["OK_BESS_puissance"]
    )

    df = df.sort_values(by=["ALL_OK", "scenario"], ascending=[False, True]).reset_index(drop=True)
    return df

# -----------------------------
# 11) LANCEMENT
# -----------------------------
scenario_list = ["pluvieux", "nuageux", "ensoleille", "ete_canicul", "hiver_froid", "pic_vehicules"]

all_results = []
for sc in scenario_list:
    res = simulate_day(sc, dt_min=5)
    all_results.append(res)

    print(f"\n--- {sc} ---")
    print(f"SCI global     : {res['SCI_global_pct']:.2f} %")
    print(f"IEJ (Jain)     : {res['IEJ']:.4f}")
    print(f"Pic import grid: {res['peak_grid_import_kW']:.2f} kW")
    print(f"E import total : {res['E_import_total_kWh']:.2f} kWh")
    print(f"Coût total     : {res['cost_total']:.2f} €")

    # Courbes détaillées par scénario
    plot_scenario(res)

# Courbes comparatives multi-scénarios
plot_compare(all_results)

# Tableau audit contraintes
df_constraints = build_constraints_table(all_results)
print("\n===== TABLEAU CONTRAINTES (audit) =====")
print(df_constraints.to_string(index=False))
print("=======================================\n")

# (option) Affichage DataFrame plus lisible dans Jupyter
try:
    display(df_constraints)
except NameError:
    pass