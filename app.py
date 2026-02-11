# app.py
import importlib
import inspect
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler


# ============================================================
# CONFIG UI
# ============================================================
st.set_page_config(
    page_title="SmartGrid / EMS — Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SmartGrid / EMS — Interface")
st.caption("Scénario unique • Comparaison • Audit (1 ligne = 1 contrainte)")

DEFAULT_MODULE = "untitled9"  # <-- change en "exosmartgrid1" si besoin


# ============================================================
# MODULE LOADER
# ============================================================
def load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        st.error(
            f"Impossible d'importer le module **{module_name}**.\n\n"
            f"Assure-toi que `{module_name}.py` est dans le même dossier que `app.py`.\n\n"
            f"Détail: {e}"
        )
        st.stop()


def ensure_state():
    st.session_state.setdefault("single_res", None)          # dict
    st.session_state.setdefault("compare_scenarios", [])     # list[str]
    st.session_state.setdefault("compare_results", [])       # list[dict]
    st.session_state.setdefault("audit_long_single", None)   # DataFrame
    st.session_state.setdefault("audit_long_compare", None)  # DataFrame


ensure_state()


# ============================================================
# HELPERS
# ============================================================
def render_matplotlib_figures_to_streamlit():
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        st.pyplot(fig, clear_figure=False)


def call_simulate_day(m, sc: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Appelle simulate_day en envoyant uniquement les kwargs acceptés par sa signature.
    """
    sig = inspect.signature(m.simulate_day)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in params.items() if k in accepted}
    return m.simulate_day(sc, **filtered)


def show_kpis(res: Dict[str, Any]):
    cols = st.columns(5)

    def metric(col, label, key, fmt=None):
        if key in res and res[key] is not None:
            v = res[key]
            if fmt and isinstance(v, (int, float)):
                col.metric(label, fmt.format(v))
            else:
                col.metric(label, str(v))
        else:
            col.metric(label, "—")

    metric(cols[0], "SCI global (%)", "SCI_global_pct", "{:.2f}")
    metric(cols[1], "IEJ (Jain)", "IEJ", "{:.4f}")
    metric(cols[2], "Pic import (kW)", "peak_grid_import_kW", "{:.2f}")
    metric(cols[3], "E import (kWh)", "E_import_total_kWh", "{:.2f}")
    metric(cols[4], "Coût total (€)", "cost_total", "{:.2f}")


def kpi_dataframe(scenarios: List[str], results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sc, res in zip(scenarios, results):
        rows.append(
            {
                "Scénario": sc,
                "SCI global (%)": res.get("SCI_global_pct"),
                "IEJ (Jain)": res.get("IEJ"),
                "Pic import (kW)": res.get("peak_grid_import_kW"),
                "E import (kWh)": res.get("E_import_total_kWh"),
                "Coût total (€)": res.get("cost_total"),
            }
        )
    return pd.DataFrame(rows)


def audit_to_long(df_audit: pd.DataFrame) -> pd.DataFrame:
    """
    Objectif: 1 ligne = 1 contrainte (par scénario).

    Supporte:
    - Format déjà long (scenario/constraint/value/limit/respect_contrainte/message)
    - Format large -> melt (1 ligne/scénario et colonnes = contraintes)
    """
    if df_audit is None or df_audit.empty:
        return df_audit

    df = df_audit.copy()

    # Normalisation minimale (avant conversion)
    rename_to_std = {
        "Scénario": "scenario",
        "Scenario": "scenario",
        "scenario": "scenario",
        "sc": "scenario",
        "Contrainte": "constraint",
        "contrainte": "constraint",
        "constraint": "constraint",
        "Valeur": "value",
        "valeur": "value",
        "value": "value",
        "Seuil": "limit",
        "seuil": "limit",
        "limit": "limit",
        "OK ?": "respect_contrainte",
        "ok": "respect_contrainte",
        "respect_contrainte": "respect_contrainte",
        "Détail": "message",
        "detail": "message",
        "message": "message",
    }
    df.rename(columns={c: rename_to_std[c] for c in df.columns if c in rename_to_std}, inplace=True)

    # Déjà long ?
    is_long = ("constraint" in df.columns) and ("scenario" in df.columns)
    if is_long:
        # garder un ordre agréable si colonnes existent
        base = [c for c in ["scenario", "constraint", "value", "limit", "respect_contrainte", "message"] if c in df.columns]
        extra = [c for c in df.columns if c not in base]
        return df[base + extra]

    # Sinon, format large -> melt
    scen_col: Optional[str] = None
    for c in ["scenario", "Scénario", "Scenario", "sc"]:
        if c in df.columns:
            scen_col = c
            break
    if scen_col is None:
        df["scenario"] = "global"
        scen_col = "scenario"

    value_vars = [c for c in df.columns if c != scen_col]
    long_df = df.melt(id_vars=[scen_col], value_vars=value_vars, var_name="constraint", value_name="value")
    long_df.rename(columns={scen_col: "scenario"}, inplace=True)
    return long_df


def prettify_audit_table(df_long: pd.DataFrame) -> Styler:
    """
    Rend le tableau audit lisible:
    colonnes FR + tri + coloration OK
    """
    pretty = df_long.copy()

    # Renommer vers FR
    rename_fr = {
        "scenario": "Scénario",
        "constraint": "Contrainte",
        "value": "Valeur",
        "limit": "Seuil",
        "respect_contrainte": "OK ?",
        "message": "Détail",
    }
    pretty.rename(columns={k: v for k, v in rename_fr.items() if k in pretty.columns}, inplace=True)

    # Réordonner
    ordered = ["Scénario", "Contrainte", "Valeur", "Seuil", "OK ?", "Détail"]
    existing = [c for c in ordered if c in pretty.columns]
    remaining = [c for c in pretty.columns if c not in existing]
    pretty = pretty[existing + remaining]

    # Tri
    if "Scénario" in pretty.columns and "Contrainte" in pretty.columns:
        pretty = pretty.sort_values(["Scénario", "Contrainte"], kind="stable")

    styler = pretty.style

    # Coloration OK ?
    if "OK ?" in pretty.columns:
        def ok_style(x):
            if x is True:
                return "background-color:#d4edda;font-weight:700;"
            if x is False:
                return "background-color:#f8d7da;font-weight:700;"
            return ""

        styler = styler.applymap(ok_style, subset=["OK ?"])

    # Format des nombres
    num_cols = [c for c in pretty.columns if pd.api.types.is_numeric_dtype(pretty[c])]
    if num_cols:
        styler = styler.format({c: "{:.3g}" for c in num_cols})

    styler = styler.set_properties(**{"white-space": "pre-wrap"})
    return styler


def audit_metrics(df_long: pd.DataFrame) -> Dict[str, Any]:
    if df_long is None or df_long.empty:
        return {"nb_ok": 0, "nb_total": 0, "rate": 0.0, "has_ok": False}

    has_ok = "respect_contrainte" in df_long.columns
    nb_total = int(len(df_long))
    if not has_ok:
        return {"nb_ok": None, "nb_total": nb_total, "rate": None, "has_ok": False}

    nb_ok = int((df_long["respect_contrainte"] == True).sum())
    rate = (nb_ok / nb_total * 100.0) if nb_total else 0.0
    return {"nb_ok": nb_ok, "nb_total": nb_total, "rate": rate, "has_ok": True}


# ============================================================
# SIDEBAR — MODULE + INPUTS
# ============================================================
st.sidebar.header("Configuration")
with st.sidebar.expander("Module Python", expanded=True):
    module_name = st.sidebar.text_input("Nom du module (sans .py)", value=DEFAULT_MODULE).strip()
m = load_module(module_name)

# Check required functions
if not hasattr(m, "simulate_day"):
    st.error(f"Il manque `simulate_day(...)` dans **{module_name}.py**.")
    st.stop()

HAS_SCENARIOS = hasattr(m, "SCENARIOS") and isinstance(getattr(m, "SCENARIOS"), dict)
HAS_PLOT_SCENARIO = hasattr(m, "plot_scenario")
HAS_PLOT_COMPARE = hasattr(m, "plot_compare")
HAS_AUDIT = hasattr(m, "build_constraints_table")

scenario_list = list(m.SCENARIOS.keys()) if HAS_SCENARIOS else [
    "pluvieux", "nuageux", "ensoleille", "ete_canicul", "hiver_froid", "pic_vehicules"
]

st.sidebar.divider()
st.sidebar.subheader("Inputs simulation")

with st.sidebar.container(border=True):
    scenario = st.sidebar.selectbox("Scénario", scenario_list, index=0)
    dt_min = st.sidebar.select_slider("Pas de temps dt (min)", options=[1, 2, 5, 10, 15], value=5)

with st.sidebar.expander("Paramètres avancés", expanded=True):
    colA, colB = st.sidebar.columns(2)
    nb_bornes = colA.number_input("Nb bornes", min_value=1, max_value=200, value=6, step=1)
    P_borne_uni_max = colB.number_input("P max / borne (kW)", min_value=1.0, max_value=350.0, value=22.0, step=1.0)

    P_borne_tot_default = float(nb_bornes) * float(P_borne_uni_max)
    P_borne_tot_max = st.sidebar.number_input(
        "P totale VE max (kW)",
        min_value=1.0, max_value=5000.0,
        value=float(P_borne_tot_default),
        step=1.0,
        help="Limite globale de puissance VE (toutes bornes confondues).",
    )

    P_grid_max = st.sidebar.number_input("Import réseau max (kW)", min_value=1.0, max_value=10000.0, value=300.0, step=10.0)

    col1, col2 = st.sidebar.columns(2)
    tariff_hp = col1.number_input("Tarif HP (€/kWh)", min_value=0.0, max_value=5.0, value=0.24, step=0.01)
    tariff_hc = col2.number_input("Tarif HC (€/kWh)", min_value=0.0, max_value=5.0, value=0.17, step=0.01)

st.sidebar.divider()
st.sidebar.subheader("Comparaison")
selected = st.sidebar.multiselect("Scénarios à comparer", scenario_list, default=scenario_list[:3])

st.sidebar.divider()
st.sidebar.subheader("Actions")
run_one = st.sidebar.button("▶ Lancer scénario", use_container_width=True)
run_compare = st.sidebar.button("📊 Comparer", use_container_width=True)

st.sidebar.divider()
with st.sidebar.expander("Affichage", expanded=False):
    show_tables = st.checkbox("Afficher tableau KPI", value=True)
    show_debug = st.checkbox("Afficher debug (JSON)", value=False)


# Paramètres à passer à simulate_day (on filtrera selon signature)
SIM_PARAMS = {
    "dt_min": int(dt_min),
    "nb_bornes": int(nb_bornes),
    "P_borne_uni_max": float(P_borne_uni_max),
    "P_borne_tot_max": float(P_borne_tot_max),
    "P_grid_max": float(P_grid_max),
    "tariff_hp": float(tariff_hp),
    "tariff_hc": float(tariff_hc),
}


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🧪 Scénario unique", "📊 Comparaison", "✅ Audit"])


# ============================================================
# TAB 1 — SINGLE
# ============================================================
with tab1:
    st.subheader("Scénario unique")

    if run_one:
        with st.spinner(f"Simulation : {scenario}…"):
            res = call_simulate_day(m, scenario, SIM_PARAMS)
        st.session_state["single_res"] = res
        st.session_state["audit_long_single"] = None  # reset audit cache single

    res = st.session_state["single_res"]
    if res is None:
        st.info("Clique sur **▶ Lancer scénario** dans la sidebar.")
    else:
        show_kpis(res)

        st.divider()
        st.subheader("Courbes")
        plt.close("all")
        if HAS_PLOT_SCENARIO:
            m.plot_scenario(res)
            render_matplotlib_figures_to_streamlit()
        else:
            st.warning("Fonction `plot_scenario(res)` non trouvée dans le module.")

        if show_debug:
            with st.expander("Debug — résultat brut"):
                st.json(res)


# ============================================================
# TAB 2 — COMPARE
# ============================================================
with tab2:
    st.subheader("Comparaison multi-scénarios")

    if run_compare:
        if not selected:
            st.warning("Sélectionne au moins un scénario.")
        else:
            with st.spinner("Simulations + comparaison…"):
                results = [call_simulate_day(m, sc, SIM_PARAMS) for sc in selected]
            st.session_state["compare_scenarios"] = list(selected)
            st.session_state["compare_results"] = results
            st.session_state["audit_long_compare"] = None  # reset audit cache compare

    scenarios_saved = st.session_state["compare_scenarios"]
    results_saved = st.session_state["compare_results"]

    if not scenarios_saved or not results_saved:
        st.info("Choisis des scénarios puis clique sur **📊 Comparer**.")
    else:
        if show_tables:
            st.markdown("### KPI par scénario")
            df_kpi = kpi_dataframe(scenarios_saved, results_saved)
            st.dataframe(
                df_kpi,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "SCI global (%)": st.column_config.NumberColumn(format="%.2f"),
                    "IEJ (Jain)": st.column_config.NumberColumn(format="%.4f"),
                    "Pic import (kW)": st.column_config.NumberColumn(format="%.2f"),
                    "E import (kWh)": st.column_config.NumberColumn(format="%.2f"),
                    "Coût total (€)": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        st.divider()
        st.markdown("### Courbes comparatives")
        plt.close("all")
        if HAS_PLOT_COMPARE:
            m.plot_compare(results_saved)
            render_matplotlib_figures_to_streamlit()
        else:
            st.warning("Fonction `plot_compare(results)` non trouvée dans le module.")

        if show_debug:
            with st.expander("Debug — résultats bruts"):
                st.json(results_saved)


# ============================================================
# TAB 3 — AUDIT (1 LINE PER CONSTRAINT)
# ============================================================
with tab3:
    st.subheader("Audit des contraintes — 1 ligne = 1 contrainte")

    if not HAS_AUDIT:
        st.error("Je ne trouve pas `build_constraints_table(results)` dans ton module.")
        st.stop()

    has_compare = bool(st.session_state["compare_results"])
    has_single = st.session_state["single_res"] is not None

    if not (has_compare or has_single):
        st.info("Lance d’abord une simulation (scénario unique ou comparaison).")
    else:
        colL, colR = st.columns([2, 3], vertical_alignment="center")
        with colL:
            source = st.radio(
                "Source à auditer",
                options=["Dernière comparaison", "Dernier scénario unique"],
                index=0 if has_compare else 1,
                disabled=not (has_compare or has_single),
            )
        with colR:
            if source == "Dernière comparaison":
                st.write("Scénarios:", ", ".join(st.session_state["compare_scenarios"]) or "—")
            else:
                st.write("Scénario:", scenario if has_single else "—")

        if source == "Dernière comparaison" and has_compare:
            results_for_audit = st.session_state["compare_results"]
            cache_key = "audit_long_compare"
        else:
            results_for_audit = [st.session_state["single_res"]]
            cache_key = "audit_long_single"

        # Build / cache audit
        if st.session_state[cache_key] is None:
            with st.spinner("Construction de l’audit…"):
                df_raw = m.build_constraints_table(results_for_audit)
                df_long = audit_to_long(df_raw)
            st.session_state[cache_key] = df_long
        else:
            df_long = st.session_state[cache_key]

        if df_long is None or df_long.empty:
            st.warning("Audit vide : ta fonction d’audit n’a rien renvoyé.")
        else:
            # Filters
            st.divider()
            f1, f2, f3, f4 = st.columns([1, 1, 1.2, 2.8])

            only_failed = f1.toggle("Échecs", value=False)
            only_ok = f2.toggle("OK", value=False)

            scenario_filter = None
            if "scenario" in df_long.columns:
                scenario_filter = f3.selectbox(
                    "Filtrer scénario",
                    options=["(tous)"] + sorted(df_long["scenario"].astype(str).unique().tolist()),
                    index=0,
                )
            search = f4.text_input("Filtrer texte", value="", placeholder="ex: P_grid_max, SOC, borne…")

            df_view = df_long.copy()

            if scenario_filter and scenario_filter != "(tous)" and "scenario" in df_view.columns:
                df_view = df_view[df_view["scenario"].astype(str) == scenario_filter]

            if "respect_contrainte" in df_view.columns:
                if only_failed and not only_ok:
                    df_view = df_view[df_view["respect_contrainte"] == False]
                elif only_ok and not only_failed:
                    df_view = df_view[df_view["respect_contrainte"] == True]

            if search.strip():
                s = search.strip().lower()
                obj_cols = [c for c in df_view.columns if df_view[c].dtype == "object"]
                if obj_cols:
                    mask = False
                    for c in obj_cols:
                        mask = mask | df_view[c].astype(str).str.lower().str.contains(s, na=False)
                    df_view = df_view[mask]

            # Metrics (on calcule sur df_long global)
            met = audit_metrics(df_long)
            if met["has_ok"]:
                m1, m2, m3 = st.columns(3)
                m1.metric("Contraintes respectées", f'{met["nb_ok"]} / {met["nb_total"]}')
                m2.metric("Taux conformité", f'{met["rate"]:.1f} %')
                m3.metric("Échecs", f'{met["nb_total"] - met["nb_ok"]}')
            else:
                st.info("Pas de colonne `respect_contrainte` -> pas de taux OK/KO.")

            st.markdown("### Tableau d’audit (format long)")
            st.dataframe(prettify_audit_table(df_view), use_container_width=True)

            st.divider()
            csv = df_long.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger audit (CSV)",
                data=csv,
                file_name="audit_contraintes_long.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if show_debug:
                with st.expander("Debug — audit long brut"):
                    st.dataframe(df_long, use_container_width=True)
