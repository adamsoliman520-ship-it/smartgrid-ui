# app.py
import importlib
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="SmartGrid / EMS — Interface", layout="wide")
st.title("Interface SmartGrid / EMS — Inputs → Résultats (KPI, courbes, audit)")


# ============================================================
# CHOIX DU MODULE (ton fichier principal)
# - Mets ici le bon nom: exosmartgrid ou exosmartgrid1
# ============================================================
DEFAULT_MODULE = "untitled9"  # <-- change en "exosmartgrid1" si besoin


def load_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        st.error(
            f"Impossible d'importer le module **{module_name}**.\n\n"
            f"Assure-toi que le fichier `{module_name}.py` est dans le même dossier que `app.py`.\n\n"
            f"Détail: {e}"
        )
        st.stop()


# Sidebar: module selection
st.sidebar.header("Configuration")
module_name = st.sidebar.text_input("Nom du module (sans .py)", value=DEFAULT_MODULE).strip()
m = load_module(module_name)


# ============================================================
# VERIF FONCTIONS ATTENDUES
# ============================================================
required = ["simulate_day"]
missing = [fn for fn in required if not hasattr(m, fn)]
if missing:
    st.error(
        f"Il manque des fonctions dans **{module_name}.py** : {', '.join(missing)}.\n\n"
        f"Il faut au minimum `simulate_day(...)`."
    )
    st.stop()

HAS_SCENARIOS = hasattr(m, "SCENARIOS") and isinstance(getattr(m, "SCENARIOS"), dict)
HAS_PLOT_SCENARIO = hasattr(m, "plot_scenario")
HAS_PLOT_COMPARE = hasattr(m, "plot_compare")
HAS_AUDIT = hasattr(m, "build_constraints_table")


# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("Inputs simulation")

# Scenario list
if HAS_SCENARIOS:
    scenario_list = list(m.SCENARIOS.keys())
else:
    scenario_list = ["pluvieux", "nuageux", "ensoleille", "ete_canicul", "hiver_froid", "pic_vehicules"]

scenario = st.sidebar.selectbox("Scénario", scenario_list, index=0)

dt_min = st.sidebar.selectbox("Pas de temps dt (min)", [1, 2, 5, 10, 15], index=2)

# Paramètres (avec valeurs par défaut raisonnables)
nb_bornes = st.sidebar.number_input("Nb bornes", min_value=1, max_value=200, value=6, step=1)
P_borne_uni_max = st.sidebar.number_input("P max / borne (kW)", min_value=1.0, max_value=350.0, value=22.0, step=1.0)
P_borne_tot_max = st.sidebar.number_input("P totale VE max (kW)", min_value=1.0, max_value=5000.0, value=float(nb_bornes) * 22.0, step=1.0)
P_grid_max = st.sidebar.number_input("Import réseau max (kW)", min_value=1.0, max_value=10000.0, value=300.0, step=10.0)

tariff_hp = st.sidebar.number_input("Tarif HP (€/kWh)", min_value=0.0, max_value=5.0, value=0.24, step=0.01)
tariff_hc = st.sidebar.number_input("Tarif HC (€/kWh)", min_value=0.0, max_value=5.0, value=0.17, step=0.01)

run_one = st.sidebar.button("▶ Lancer ce scénario", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Comparaison multi-scénarios")
selected = st.sidebar.multiselect("Scénarios à comparer", scenario_list, default=scenario_list[:3])
run_compare = st.sidebar.button("📊 Comparer", use_container_width=True)


# ============================================================
# HELPERS
# ============================================================
def run_sim(sc: str) -> dict:
    """
    Appelle simulate_day en passant des paramètres si la signature les accepte.
    On tente "riche", puis fallback sur minimal.
    """
    try:
        return m.simulate_day(
            sc,
            dt_min=int(dt_min),
            nb_bornes=int(nb_bornes),
            P_borne_uni_max=float(P_borne_uni_max),
            P_borne_tot_max=float(P_borne_tot_max),
            P_grid_max=float(P_grid_max),
            tariff_hp=float(tariff_hp),
            tariff_hc=float(tariff_hc),
        )
    except TypeError:
        # Si ton simulate_day n'accepte pas tous ces kwargs
        return m.simulate_day(sc, dt_min=int(dt_min))


def show_kpis(res: dict):
    # On affiche les KPI s'ils existent dans le dict
    # (sinon on évite de casser l'UI)
    cols = st.columns(5)

    def metric_if_exists(col, label, key, fmt=None):
        if key in res:
            val = res[key]
            if fmt and isinstance(val, (int, float)):
                col.metric(label, fmt.format(val))
            else:
                col.metric(label, str(val))
        else:
            col.metric(label, "—")

    metric_if_exists(cols[0], "SCI global (%)", "SCI_global_pct", "{:.2f}")
    metric_if_exists(cols[1], "IEJ (Jain)", "IEJ", "{:.4f}")
    metric_if_exists(cols[2], "Pic import (kW)", "peak_grid_import_kW", "{:.2f}")
    metric_if_exists(cols[3], "E import (kWh)", "E_import_total_kWh", "{:.2f}")
    metric_if_exists(cols[4], "Coût total (€)", "cost_total", "{:.2f}")


def render_matplotlib_figures_to_streamlit():
    # Convertit toutes les figures ouvertes en sorties Streamlit
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        st.pyplot(fig, clear_figure=False)


# ============================================================
# PAGE CONTENT
# ============================================================
tab1, tab2 = st.tabs(["Scénario unique", "Comparaison + Audit"])


# ----------------------------
# TAB 1: Single scenario
# ----------------------------
with tab1:
    st.subheader("Simulation — Scénario unique")

    if run_one:
        res = run_sim(scenario)

        show_kpis(res)

        st.divider()
        st.subheader("Courbes")

        plt.close("all")
        if HAS_PLOT_SCENARIO:
            # Ton code existant crée probablement des figures matplotlib
            m.plot_scenario(res)
            render_matplotlib_figures_to_streamlit()
        else:
            st.info("Fonction `plot_scenario(res)` non trouvée dans le module. (KPI affichés quand même.)")

        # Option: afficher un “résultat brut” (utile debug)
        with st.expander("Voir le dictionnaire de résultats (debug)"):
            st.json(res)


# ----------------------------
# TAB 2: Compare + Audit
# ----------------------------
with tab2:
    st.subheader("Comparaison multi-scénarios")

    if run_compare and selected:
        results = [run_sim(sc) for sc in selected]

        # ---- KPIs synthèse
        st.markdown("### KPI par scénario")
        rows = []
        for sc, res in zip(selected, results):
            rows.append(
                {
                    "scenario": sc,
                    "SCI_global_pct": res.get("SCI_global_pct", None),
                    "IEJ": res.get("IEJ", None),
                    "peak_grid_import_kW": res.get("peak_grid_import_kW", None),
                    "E_import_total_kWh": res.get("E_import_total_kWh", None),
                    "cost_total": res.get("cost_total", None),
                }
            )
        df_kpi = pd.DataFrame(rows)
        st.dataframe(df_kpi, use_container_width=True)

        # ---- Plots comparaison
        st.divider()
        st.markdown("### Courbes comparatives")

        plt.close("all")
        if HAS_PLOT_COMPARE:
            m.plot_compare(results)
            render_matplotlib_figures_to_streamlit()
        else:
            st.info("Fonction `plot_compare(results)` non trouvée dans le module. (Tableau KPI affiché quand même.)")

        # ---- Audit contraintes
        st.divider()
        st.markdown("### Audit des contraintes")

        if HAS_AUDIT:
            df_audit = m.build_constraints_table(results)

            # Affichage + style simple si la colonne existe
            if "respect_contrainte" in df_audit.columns:
                styled = df_audit.style.applymap(
                    lambda x: "background-color: #d4edda" if x is True else ("background-color: #f8d7da" if x is False else ""),
                    subset=["respect_contrainte"],
                )
                st.dataframe(styled, use_container_width=True)
                nb_ok = int((df_audit["respect_contrainte"] == True).sum())
                nb_total = int(len(df_audit))
                st.metric("Contraintes respectées", f"{nb_ok} / {nb_total}")
            else:
                st.dataframe(df_audit, use_container_width=True)

            # Download CSV
            csv = df_audit.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger audit (CSV)",
                data=csv,
                file_name="audit_contraintes.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "Je ne trouve pas `build_constraints_table(results)` dans ton module.\n\n"
                "➡️ Ajoute-la dans ton fichier (ou donne-moi le nom exact de la fonction d'audit)."
            )
    else:
        st.info("Choisis des scénarios à gauche puis clique sur **Comparer**.")