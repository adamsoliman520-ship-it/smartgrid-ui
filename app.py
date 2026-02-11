# app.py
import importlib
import inspect
from typing import Any, Dict, List

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="SmartGrid / EMS — Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("SmartGrid / EMS — Inputs → Résultats")
st.caption("Simule un scénario, compare plusieurs scénarios, puis audite clairement les contraintes.")


# ============================================================
# MODULE LOADER
# ============================================================
DEFAULT_MODULE = "untitled9"  # <-- change en "exosmartgrid1" si besoin


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


# ============================================================
# SIDEBAR — CONFIG + INPUTS
# ============================================================
st.sidebar.header("Configuration")

with st.sidebar.expander("Module Python", expanded=True):
    module_name = st.text_input("Nom du module (sans .py)", value=DEFAULT_MODULE).strip()
    m = load_module(module_name)

# Verify required functions
required = ["simulate_day"]
missing = [fn for fn in required if not hasattr(m, fn)]
if missing:
    st.error(
        f"Il manque des fonctions dans **{module_name}.py** : {', '.join(missing)}.\n\n"
        f"Minimum requis : `simulate_day(...)`."
    )
    st.stop()

HAS_SCENARIOS = hasattr(m, "SCENARIOS") and isinstance(getattr(m, "SCENARIOS"), dict)
HAS_PLOT_SCENARIO = hasattr(m, "plot_scenario")
HAS_PLOT_COMPARE = hasattr(m, "plot_compare")
HAS_AUDIT = hasattr(m, "build_constraints_table")

# Scenarios list
if HAS_SCENARIOS:
    scenario_list = list(m.SCENARIOS.keys())
else:
    scenario_list = ["pluvieux", "nuageux", "ensoleille", "ete_canicul", "hiver_froid", "pic_vehicules"]

st.sidebar.divider()
st.sidebar.subheader("Inputs simulation")

with st.sidebar.container(border=True):
    scenario = st.selectbox("Scénario", scenario_list, index=0)
    dt_min = st.select_slider("Pas de temps dt (min)", options=[1, 2, 5, 10, 15], value=5)

with st.sidebar.expander("Paramètres avancés", expanded=True):
    colA, colB = st.columns(2)
    nb_bornes = colA.number_input("Nb bornes", min_value=1, max_value=200, value=6, step=1)
    P_borne_uni_max = colB.number_input("P max / borne (kW)", min_value=1.0, max_value=350.0, value=22.0, step=1.0)

    # auto total default (but editable)
    P_borne_tot_default = float(nb_bornes) * float(P_borne_uni_max)
    P_borne_tot_max = st.number_input(
        "P totale VE max (kW)",
        min_value=1.0,
        max_value=5000.0,
        value=float(P_borne_tot_default),
        step=1.0,
        help="Limite globale de puissance VE (toutes bornes confondues).",
    )

    P_grid_max = st.number_input("Import réseau max (kW)", min_value=1.0, max_value=10000.0, value=300.0, step=10.0)

    col1, col2 = st.columns(2)
    tariff_hp = col1.number_input("Tarif HP (€/kWh)", min_value=0.0, max_value=5.0, value=0.24, step=0.01)
    tariff_hc = col2.number_input("Tarif HC (€/kWh)", min_value=0.0, max_value=5.0, value=0.17, step=0.01)

st.sidebar.divider()
st.sidebar.subheader("Actions")

run_one = st.sidebar.button("▶ Lancer le scénario", use_container_width=True)
run_compare = st.sidebar.button("📊 Comparer", use_container_width=True)

selected = st.sidebar.multiselect(
    "Scénarios à comparer",
    scenario_list,
    default=scenario_list[:3],
    help="Sélectionne plusieurs scénarios puis clique sur Comparer.",
)

st.sidebar.divider()
with st.sidebar.expander("Qualité d’affichage", expanded=False):
    show_debug = st.checkbox("Afficher debug (dict JSON)", value=False)
    show_tables = st.checkbox("Afficher tableaux (KPI / audit)", value=True)


# ============================================================
# HELPERS
# ============================================================
def _call_simulate_day(sc: str) -> Dict[str, Any]:
    """
    Appelle simulate_day en envoyant seulement les kwargs acceptés par la signature.
    -> évite le try/except TypeError et rend l'interface plus robuste.
    """
    sig = inspect.signature(m.simulate_day)
    accepted = set(sig.parameters.keys())

    kwargs = {
        "dt_min": int(dt_min),
        "nb_bornes": int(nb_bornes),
        "P_borne_uni_max": float(P_borne_uni_max),
        "P_borne_tot_max": float(P_borne_tot_max),
        "P_grid_max": float(P_grid_max),
        "tariff_hp": float(tariff_hp),
        "tariff_hc": float(tariff_hc),
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    # 1er argument attendu: scenario
    return m.simulate_day(sc, **filtered_kwargs)


def show_kpis(res: Dict[str, Any]):
    cols = st.columns(5)

    def metric(col, label, key, fmt=None):
        if key in res and res[key] is not None:
            val = res[key]
            if fmt and isinstance(val, (int, float)):
                col.metric(label, fmt.format(val))
            else:
                col.metric(label, str(val))
        else:
            col.metric(label, "—")

    metric(cols[0], "SCI global (%)", "SCI_global_pct", "{:.2f}")
    metric(cols[1], "IEJ (Jain)", "IEJ", "{:.4f}")
    metric(cols[2], "Pic import (kW)", "peak_grid_import_kW", "{:.2f}")
    metric(cols[3], "E import (kWh)", "E_import_total_kWh", "{:.2f}")
    metric(cols[4], "Coût total (€)", "cost_total", "{:.2f}")


def render_matplotlib_figures_to_streamlit():
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        st.pyplot(fig, clear_figure=False)


def kpi_dataframe(selected_scenarios: List[str], results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for sc, res in zip(selected_scenarios, results):
        rows.append(
            {
                "Scénario": sc,
                "SCI global (%)": res.get("SCI_global_pct", None),
                "IEJ (Jain)": res.get("IEJ", None),
                "Pic import (kW)": res.get("peak_grid_import_kW", None),
                "E import (kWh)": res.get("E_import_total_kWh", None),
                "Coût total (€)": res.get("cost_total", None),
            }
        )
    return pd.DataFrame(rows)


def prettify_audit_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Rend le tableau audit lisible:
    - renomme si possible
    - met la colonne respect_contrainte en couleur
    - formatte les nombres
    """
    pretty = df.copy()

    # Renommage léger si colonnes connues (sinon on garde)
    rename_map = {
        "respect_contrainte": "OK ?",
        "contrainte": "Contrainte",
        "constraint": "Contrainte",
        "scenario": "Scénario",
        "sc": "Scénario",
        "message": "Détail",
        "detail": "Détail",
        "value": "Valeur",
        "limit": "Seuil",
    }
    pretty.rename(columns={k: v for k, v in rename_map.items() if k in pretty.columns}, inplace=True)

    # Mettre "OK ?" en première colonne si présent
    if "OK ?" in pretty.columns:
        cols = ["OK ?"] + [c for c in pretty.columns if c != "OK ?"]
        pretty = pretty[cols]

    # Styler
    styler = pretty.style

    # Coloration OK ?
    if "OK ?" in pretty.columns:
        def ok_color(x):
            if x is True:
                return "background-color: #d4edda; font-weight: 700;"
            if x is False:
                return "background-color: #f8d7da; font-weight: 700;"
            return ""

        styler = styler.applymap(ok_color, subset=["OK ?"])

    # Format nombres
    num_cols = [c for c in pretty.columns if pd.api.types.is_numeric_dtype(pretty[c])]
    if num_cols:
        styler = styler.format({c: "{:.3g}" for c in num_cols})

    # Petits réglages visuels
    styler = styler.set_properties(**{"white-space": "pre-wrap"})
    return styler


def ensure_session_state():
    st.session_state.setdefault("last_single", None)  # dict result
    st.session_state.setdefault("last_compare_scenarios", [])  # list[str]
    st.session_state.setdefault("last_compare_results", [])  # list[dict]
    st.session_state.setdefault("last_audit_df", None)  # pd.DataFrame


ensure_session_state()


# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🧪 Scénario unique", "📊 Comparaison", "✅ Audit contraintes"])


# ----------------------------
# TAB 1: Single scenario
# ----------------------------
with tab1:
    st.subheader("Simulation — Scénario unique")

    if run_one:
        with st.spinner(f"Simulation en cours : {scenario}…"):
            res = _call_simulate_day(scenario)
        st.session_state["last_single"] = res

    res = st.session_state["last_single"]
    if res is None:
        st.info("Lance un scénario via la sidebar (bouton **▶ Lancer le scénario**).")
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
            with st.expander("Debug — Résultat brut (dict)"):
                st.json(res)


# ----------------------------
# TAB 2: Compare
# ----------------------------
with tab2:
    st.subheader("Comparaison multi-scénarios")

    if run_compare:
        if not selected:
            st.warning("Sélectionne au moins un scénario dans la sidebar.")
        else:
            with st.spinner("Simulations + agrégation…"):
                results = [_call_simulate_day(sc) for sc in selected]
            st.session_state["last_compare_scenarios"] = list(selected)
            st.session_state["last_compare_results"] = results

            # Reset audit cache
            st.session_state["last_audit_df"] = None

    selected_saved = st.session_state["last_compare_scenarios"]
    results_saved = st.session_state["last_compare_results"]

    if not selected_saved or not results_saved:
        st.info("Choisis des scénarios dans la sidebar puis clique sur **📊 Comparer**.")
    else:
        if show_tables:
            st.markdown("### KPI par scénario")
            df_kpi = kpi_dataframe(selected_saved, results_saved)

            # Dataframe lisible (format + largeur)
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
            with st.expander("Debug — Résultats bruts (liste de dict)"):
                st.json(results_saved)


# ----------------------------
# TAB 3: Audit only
# ----------------------------
with tab3:
    st.subheader("Audit des contraintes (tableau lisible)")

    if not HAS_AUDIT:
        st.error(
            "Je ne trouve pas `build_constraints_table(results)` dans ton module.\n\n"
            "➡️ Ajoute-la dans ton fichier (ou adapte le nom dans l’interface)."
        )
    else:
        # Source results: prefer compare; fallback to single
        has_compare = bool(st.session_state["last_compare_results"])
        has_single = st.session_state["last_single"] is not None

        col_left, col_right = st.columns([2, 3], vertical_alignment="center")
        with col_left:
            source = st.radio(
                "Source à auditer",
                options=["Dernière comparaison", "Dernier scénario unique"],
                index=0 if has_compare else 1,
                disabled=not (has_compare or has_single),
                help="L’audit utilise les résultats simulés (pas de re-simulation ici).",
            )

        with col_right:
            if source == "Dernière comparaison":
                st.write("Scénarios:", ", ".join(st.session_state["last_compare_scenarios"]) or "—")
            else:
                st.write("Scénario:", scenario if has_single else "—")

        if not (has_compare or has_single):
            st.info("Lance d’abord une simulation (scénario unique ou comparaison) via la sidebar.")
        else:
            # Build results list
            if source == "Dernière comparaison" and has_compare:
                results_for_audit = st.session_state["last_compare_results"]
            else:
                results_for_audit = [st.session_state["last_single"]]

            # Build / cache audit df
            if st.session_state["last_audit_df"] is None:
                with st.spinner("Construction de l’audit…"):
                    df_audit = m.build_constraints_table(results_for_audit)
                st.session_state["last_audit_df"] = df_audit
            else:
                df_audit = st.session_state["last_audit_df"]

            # Controls
            st.divider()
            top_cols = st.columns([1, 1, 2, 2])
            only_failed = top_cols[0].toggle("Afficher seulement les échecs", value=False)
            only_ok = top_cols[1].toggle("Afficher seulement les OK", value=False)

            # Find OK column name
            ok_col = "respect_contrainte" if "respect_contrainte" in df_audit.columns else ("OK ?" if "OK ?" in df_audit.columns else None)

            filtered = df_audit.copy()
            if ok_col is not None:
                if only_failed and not only_ok:
                    filtered = filtered[filtered[ok_col] == False]
                elif only_ok and not only_failed:
                    filtered = filtered[filtered[ok_col] == True]

            # Summary metrics
            if ok_col is not None:
                nb_ok = int((df_audit[ok_col] == True).sum())
                nb_total = int(len(df_audit))
                rate = (nb_ok / nb_total * 100.0) if nb_total else 0.0
                m1, m2, m3 = st.columns(3)
                m1.metric("Contraintes respectées", f"{nb_ok} / {nb_total}")
                m2.metric("Taux conformité", f"{rate:.1f} %")
                m3.metric("Échecs", f"{nb_total - nb_ok}")

            # Display styled table
            st.markdown("### Tableau d’audit")
            styled = prettify_audit_table(filtered)
            st.dataframe(styled, use_container_width=True)

            # Download CSV
            st.divider()
            csv = df_audit.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger l’audit complet (CSV)",
                data=csv,
                file_name="audit_contraintes.csv",
                mime="text/csv",
                use_container_width=True,
            )
