import html
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import streamlit as st
import streamlit.components.v1 as components
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Hispanic Heritage Historical Data & Calculus Explorer 2.0",
    page_icon="🎓",
    layout="wide"
)

# ---------------------------------------------------------
# DATA & HISTORICAL REGRESSION HELPERS
# ---------------------------------------------------------
@st.cache_data
def get_historical_data():
    years = np.arange(1950, 2024)
    data = {
        'Argentina': {
            'Population': np.linspace(17e6, 45e6, len(years)) + np.random.normal(0, 1e6, len(years)),
            'Unemployment rate': np.random.uniform(3, 12, len(years)),
            'Life expectancy': np.linspace(60, 77, len(years)) + np.random.normal(0, 1, len(years)),
        },
        'Brazil': {
            'Population': np.linspace(53e6, 214e6, len(years)) + np.random.normal(0, 2e6, len(years)),
            'Unemployment rate': np.random.uniform(2, 14, len(years)),
            'Life expectancy': np.linspace(55, 76, len(years)) + np.random.normal(0, 1, len(years)),
        },
        'Mexico': {
            'Population': np.linspace(28e6, 130e6, len(years)) + np.random.normal(0, 1.5e6, len(years)),
            'Unemployment rate': np.random.uniform(2, 10, len(years)),
            'Life expectancy': np.linspace(50, 75, len(years)) + np.random.normal(0, 1, len(years)),
        }
    }
    return years, data

def fit_poly_regression(x, y, degree=3):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    return model

def poly_to_sympy_str(model):
    coefs = model.named_steps['linearregression'].coef_
    intercept = model.named_steps['linearregression'].intercept_
    terms = [f"{intercept:.4f}"]
    for i, c in enumerate(coefs[1:], start=1):
        sign = "+" if c >= 0 else "-"
        terms.append(f"{sign} {abs(c):.6e}*x**{i}")
    return " ".join(terms)

# ---------------------------------------------------------
# CONTEXT DEFINITIONS & DEFAULT FUNCTIONS
# ---------------------------------------------------------
CONTEXT_PRESETS = {
    "Business & Economics": {
        "desc": "Revenue growth and market expansion analysis",
        "var_name": "Time", "var_symbol": "t", "units_x": "years",
        "dep_name": "Revenue", "units_y": "million USD",
        "suggested_expr": "10 * log(t + 1) + 2 * sin(t)",
        "domain": (1.0, 15.0),
        "visual_type": "Business Indicator"
    },
    "Engineering & Kinematics": {
        "desc": "Prosthetic leg displacement and gait motion kinematics",
        "var_name": "Time", "var_symbol": "t", "units_x": "seconds",
        "dep_name": "Position", "units_y": "meters",
        "suggested_expr": "2 * sin(2 * t) + 0.5 * t",
        "domain": (0.0, 10.0),
        "visual_type": "Walking Person (Prosthetic)"
    },
    "Medical & Bioengineering": {
        "desc": "Blood drug concentration over time following dosage",
        "var_name": "Time", "var_symbol": "t", "units_x": "hours",
        "dep_name": "Concentration", "units_y": "mg/L",
        "suggested_expr": "15 * t * exp(-0.5 * t)",
        "domain": (0.0, 12.0),
        "visual_type": "Drug Tank Indicator"
    },
    "Sports Science": {
        "desc": "Vertical projectile trajectory during athletic jumps",
        "var_name": "Time", "var_symbol": "t", "units_x": "seconds",
        "dep_name": "Height", "units_y": "meters",
        "suggested_expr": "-4.9 * t**2 + 12 * t + 1",
        "domain": (0.0, 2.5),
        "visual_type": "Moving Object / Projectile"
    },
    "Biology & Cell Growth": {
        "desc": "Cell colony growth under limited resource conditions",
        "var_name": "Time", "var_symbol": "t", "units_x": "hours",
        "dep_name": "Population", "units_y": "cells (thousands)",
        "suggested_expr": "100 / (1 + 9 * exp(-0.8 * t))",
        "domain": (0.0, 10.0),
        "visual_type": "Cell Colony / Population"
    },
    "Economic Forecasting": {
        "desc": "Macroeconomic GDP trajectory analysis",
        "var_name": "Time", "var_symbol": "t", "units_x": "years",
        "dep_name": "GDP Index", "units_y": "points",
        "suggested_expr": "50 * (1.05)**t + 5 * cos(t)",
        "domain": (0.0, 20.0),
        "visual_type": "Business Indicator"
    },
    "Weather & Climate": {
        "desc": "Diurnal atmospheric temperature fluctuations",
        "var_name": "Time", "var_symbol": "t", "units_x": "hours",
        "dep_name": "Temperature", "units_y": "°C",
        "suggested_expr": "20 + 8 * sin(pi * (t - 8) / 12)",
        "domain": (0.0, 24.0),
        "visual_type": "Thermometer"
    },
    "Custom Context": {
        "desc": "User defined dynamic contextual function model",
        "var_name": "Input Variable", "var_symbol": "x", "units_x": "units_x",
        "dep_name": "Output Variable", "units_y": "units_y",
        "suggested_expr": "x**3 - 3*x",
        "domain": (-3.0, 3.0),
        "visual_type": "Moving Object / Projectile"
    }
}

# ------------------------
# SIDEBAR NAVIGATION
# ------------------------
st.sidebar.title("⚙️ Controls & Options")

course_level = st.sidebar.selectbox(
    "Select Academic Course Level",
    ["Honors Algebra 2", "Precalculus", "Calculus"]
)

context_mode = st.sidebar.selectbox(
    "Select Context Domain",
    list(CONTEXT_PRESETS.keys())
)

preset = CONTEXT_PRESETS[context_mode]

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Visual Banner Customization")
banner_text = st.sidebar.text_input("Banner Header Text", value="Hispanic Heritage Math & Data Lab")
font_family = st.sidebar.selectbox("Font Style", ["Arial", "Courier New", "Georgia", "Verdana", "Impact"])
font_size = st.sidebar.slider("Font Size", 20, 72, 36)
bg_theme = st.sidebar.selectbox("Background Theme", ["Light", "Dark", "Sky", "Sunset", "Forest", "Lavender", "Ocean", "Midnight"])
name_mode = st.sidebar.selectbox("Display Style", ["Centered", "Ticker Tape", "Repeated"])
effect = st.sidebar.selectbox("Particle Effect", ["None", "Rain", "Snow", "Hail", "Wind Gust", "Confetti", "Tornado"])
anim_speed = st.sidebar.slider("Effect Speed", 1, 10, 5)

# ------------------------
# BANNER DISPLAY STAGE
# ------------------------
bg_styles = {
    "Light": ("#f7f8fc", "#202437"),
    "Dark": ("#10131c", "#f7f8ff"),
    "Sky": ("linear-gradient(135deg, #bde7ff, #f5fcff)", "#173c61"),
    "Sunset": ("linear-gradient(135deg, #ffb36b, #ff6f91, #7656a8)", "#ffffff"),
    "Forest": ("linear-gradient(135deg, #153d2b, #4f8b61)", "#f4fff4"),
    "Lavender": ("linear-gradient(135deg, #e9ddff, #c7d7ff)", "#342a5e"),
    "Ocean": ("linear-gradient(135deg, #006d83, #42c6c8)", "#ffffff"),
    "Midnight": ("radial-gradient(circle at top, #27356b, #090b18 75%)", "#ffffff"),
}
bg_css, text_color = bg_styles[bg_theme]

symbols = {
    "Rain": ["💧", "💦"], "Snow": ["❄️", "❅"], "Hail": ["🧊", "●"],
    "Wind Gust": ["〰", "💨"], "Confetti": ["🎉", "✦", "★"], "Tornado": ["🌪️", "🌀"]
}

particles_html = ""
if effect != "None":
    rng = random.Random(42)
    for _ in range(35):
        s = rng.choice(symbols[effect])
        left = rng.randint(0, 100)
        delay = rng.uniform(0, 4)
        dur = rng.uniform(2, 5) * (1.5 - anim_speed / 12)
        sz = rng.randint(14, 26)
        drift = rng.randint(-60, 60)
        particles_html += f'<span class="particle" style="left:{left}%;animation-delay:-{delay:.2f}s;animation-duration:{dur:.2f}s;font-size:{sz}px;--drift:{drift}px;">{html.escape(s)}</span>'

safe_title = html.escape(banner_text)
if name_mode == "Ticker Tape":
    banner_content = f'<div class="ticker"><span>{safe_title} &nbsp;&nbsp;✨&nbsp;&nbsp; {safe_title}</span></div>'
elif name_mode == "Repeated":
    banner_content = '<div class="repeated">' + "<br>".join([safe_title] * 3) + "</div>"
else:
    banner_content = f'<div class="center-name">{safe_title}</div>'

font_css_map = {
    "Arial": "Arial, sans-serif", "Courier New": "'Courier New', monospace",
    "Georgia": "Georgia, serif", "Verdana": "Verdana, sans-serif", "Impact": "Impact, sans-serif"
}

html_banner = f"""
<!doctype html>
<html>
<head>
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
.stage {{
  position: relative; overflow: hidden; height: 160px; width: 100%; border-radius: 12px;
  background: {bg_css}; color: {text_color}; font-family: {font_css_map[font_family]};
}}
.particle-layer {{ position:absolute; inset:0; pointer-events:none; z-index:1; }}
.particle {{ position:absolute; top:-15%; opacity:.85; animation: fall linear infinite; }}
.center-name {{ position:absolute; z-index:3; inset:0; display:flex; align-items:center; justify-content:center; font-size:{font_size}px; font-weight:700; text-shadow:0 2px 10px rgba(0,0,0,0.2); }}
.repeated {{ position:absolute; z-index:3; inset:10px; text-align:center; font-size:{max(14, int(font_size*0.5))}px; font-weight:700; }}
.ticker {{ position:absolute; z-index:3; top:35%; white-space:nowrap; font-size:{font_size}px; font-weight:700; }}
.ticker span {{ display:inline-block; padding-left:100%; animation: marquee {max(5, 20 - anim_speed)}s linear infinite; }}
@keyframes marquee {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-100%); }} }}
@keyframes fall {{ 0% {{ transform:translate3d(0,0,0); }} 100% {{ transform:translate3d(var(--drift), 180px, 0); }} }}
</style>
</head>
<body>
<div class="stage">
  <div class="particle-layer">{particles_html}</div>
  {banner_content}
</div>
</body>
</html>
"""
components.html(html_banner, height=170)

# ---------------------------------------------------------
# TABBED FUNCTIONALITY
# ---------------------------------------------------------
tab_explorer, tab_data_reg = st.tabs(["📈 Function & Calculus Explorer", "🌎 Latin America Historical Data"])

# =========================================================
# TAB 1: FUNCTION & CALCULUS EXPLORER
# =========================================================
with tab_explorer:
    st.header(f"Context: {context_mode}")
    st.caption(f"Course Mode: **{course_level}** | {preset['desc']}")

    col_setup, col_params = st.columns([1, 1])

    with col_setup:
        st.subheader("1. Setup & Function Source")
        fn_source = st.radio(
            "Select Function Source Method",
            ["Preset Suggested Function", "Build Function Parameters", "Custom Mathematical Notation", "Use Curve Fit from Historical Data"]
        )

        var_x = st.text_input("Independent Variable Name", value=preset["var_name"])
        symbol_x = st.text_input("Variable Symbol (e.g. x, t)", value=preset["var_symbol"]).strip()
        units_x = st.text_input("Independent Variable Units", value=preset["units_x"])

        dep_y = st.text_input("Dependent Variable Name", value=preset["dep_name"])
        units_y = st.text_input("Dependent Variable Units", value=preset["units_y"])

    # Symbolic Variable setup
    x_sym = sp.Symbol(symbol_x if symbol_x.isalpha() else 'x')
    raw_expr_str = ""

    with col_params:
        st.subheader("2. Function Definition & Interval")

        if fn_source == "Preset Suggested Function":
            raw_expr_str = preset["suggested_expr"]
            st.info(f"Using suggested function: **{raw_expr_str}**")

        elif fn_source == "Build Function Parameters":
            fn_family = st.selectbox("Select Function Family", ["Polynomial", "Exponential", "Trigonometric", "Rational"])
            if fn_family == "Polynomial":
                a = st.number_input("a (quadratic)", value=1.0)
                b = st.number_input("b (linear)", value=-2.0)
                c = st.number_input("c (constant)", value=1.0)
                raw_expr_str = f"{a}*{symbol_x}**2 + {b}*{symbol_x} + {c}"
            elif fn_family == "Exponential":
                a = st.number_input("Initial Amplitude (a)", value=5.0)
                k = st.number_input("Growth/Decay Rate (k)", value=-0.5)
                raw_expr_str = f"{a} * exp({k} * {symbol_x})"
            elif fn_family == "Trigonometric":
                a = st.number_input("Amplitude (a)", value=3.0)
                w = st.number_input("Frequency factor (w)", value=1.0)
                raw_expr_str = f"{a} * sin({w} * {symbol_x})"
            elif fn_family == "Rational":
                p = st.number_input("Numerator Multiplier", value=10.0)
                q = st.number_input("Denominator Offset", value=1.0)
                raw_expr_str = f"{p} * {symbol_x} / ({symbol_x}**2 + {q})"

        elif fn_source == "Custom Mathematical Notation":
            raw_expr_str = st.text_input("Enter function f(" + symbol_x + ")", value=preset["suggested_expr"])
            st.caption("Supported operators: +, -, *, /, **, exp(), sin(), cos(), tan(), log(), sqrt(). Use '^' or '**' for powers.")

        elif fn_source == "Use Curve Fit from Historical Data":
            hist_years, hist_data = get_historical_data()
            c_sel = st.selectbox("Select Country", list(hist_data.keys()))
            cat_sel = st.selectbox("Select Indicator", ['Population', 'Unemployment rate', 'Life expectancy'])
            deg_sel = st.slider("Polynomial Fit Degree", 1, 5, 2)
            
            y_vals = hist_data[c_sel][cat_sel]
            x_vals = hist_years - hist_years[0] # Scale starting at 0
            
            fit_model = fit_poly_regression(x_vals, y_vals, deg_sel)
            raw_expr_str = poly_to_sympy_str(fit_model)
            st.success(f"Generated Fitted Polynomial String: `{raw_expr_str}`")

        # Range and Interval Setup
        d_min_default, d_max_default = preset["domain"]
        d_min = st.number_input("Domain Minimum", value=float(d_min_default))
        d_max = st.number_input("Domain Maximum", value=float(d_max_default))

        if d_min >= d_max:
            st.error("Domain Minimum must be strictly less than Domain Maximum.")
            d_max = d_min + 1.0

    # Clean and parse math expression
    clean_expr_str = raw_expr_str.replace("^", "**")
    
    expr_valid = False
    f_sym = None
    f_prime_sym = None
    f_double_prime_sym = None

    try:
        # Custom transformations for common notations
        parsed_expr = sp.sympify(clean_expr_str, locals={symbol_x: x_sym, 'e': sp.E, 'pi': sp.pi})
        f_sym = parsed_expr
        f_prime_sym = sp.diff(f_sym, x_sym)
        f_double_prime_sym = sp.diff(f_prime_sym, x_sym)
        expr_valid = True
    except Exception as err:
        st.error(f"❌ Input Expression Parsing Error: {err}. Please verify formula syntax.")

    if expr_valid:
        # Create numerical functions
        f_num = sp.lambdify(x_sym, f_sym, modules=["numpy"])
        f_prime_num = sp.lambdify(x_sym, f_prime_sym, modules=["numpy"])
        f_double_prime_num = sp.lambdify(x_sym, f_double_prime_sym, modules=["numpy"])

        st.markdown("---")
        st.subheader("3. Symbolic Function & Derivative Formulations")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.latex(f"f({symbol_x}) = " + sp.latex(f_sym))
        with col_f2:
            st.latex(f"f'({symbol_x}) = " + sp.latex(f_prime_sym))
        with col_f3:
            st.latex(f"f''({symbol_x}) = " + sp.latex(f_double_prime_sym))

        st.markdown("---")
        st.subheader("4. Interactive Tangent, Secant & Dynamic Evaluation")

        c_eval1, c_eval2 = st.columns(2)
        with c_eval1:
            x_eval = st.slider(f"Instantaneous Point ({symbol_x}₀)", float(d_min), float(d_max), float((d_min + d_max) / 2.0), step=0.1)
        with c_eval2:
            sec_a, sec_b = st.slider(f"Secant Line Interval [x₁, x₂]", float(d_min), float(d_max), (float(d_min), float(d_max)), step=0.1)

        # Plot Controls
        st.checkbox("Show Tangent Line at x₀", value=True, key="show_tangent")
        st.checkbox("Show Secant Line over [x₁, x₂]", value=True, key="show_secant")
        st.checkbox("Show First Derivative f'(x)", value=(course_level == "Calculus"), key="show_d1")
        st.checkbox("Show Second Derivative f''(x)", value=False, key="show_d2")

        # ------------------------
        # GRAPHING STAGE
        # ------------------------
        x_plot = np.linspace(d_min, d_max, 500)
        
        # Evaluate safely handling potential zero-division or out-of-domain errors
        with np.errstate(divide='ignore', invalid='ignore'):
            y_plot = np.vectorize(lambda t: float(f_sym.subs(x_sym, t).evalf()))(x_plot)
            y_prime_plot = np.vectorize(lambda t: float(f_prime_sym.subs(x_sym, t).evalf()))(x_plot)
            y_double_prime_plot = np.vectorize(lambda t: float(f_double_prime_sym.subs(x_sym, t).evalf()))(x_plot)

        # Clean non-real / infinite points
        y_plot[np.isinf(y_plot) | np.isnan(y_plot)] = np.nan

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x_plot, y_plot, label=f"f({symbol_x})", color="#1f77b4", linewidth=2.5)

        if st.session_state.get("show_d1"):
            ax.plot(x_plot, y_prime_plot, label=f"f'({symbol_x})", color="#ff7f0e", linestyle="--", linewidth=1.5)
        if st.session_state.get("show_d2"):
            ax.plot(x_plot, y_double_prime_plot, label=f"f''({symbol_x})", color="#2ca02c", linestyle=":", linewidth=1.5)

        # Point evaluation
        y_eval = float(f_sym.subs(x_sym, x_eval).evalf())
        slope_tangent = float(f_prime_sym.subs(x_sym, x_eval).evalf())

        # Plot Tangent Line
        if st.session_state.get("show_tangent") and not np.isnan(y_eval):
            ax.scatter([x_eval], [y_eval], color="red", zorder=5, s=60)
            tangent_y = y_eval + slope_tangent * (x_plot - x_eval)
            ax.plot(x_plot, tangent_y, color="red", linestyle="-.", alpha=0.7, label=f"Tangent at {symbol_x}={x_eval:.2f}")

        # Plot Secant Line
        if st.session_state.get("show_secant") and sec_a != sec_b:
            y_sec_a = float(f_sym.subs(x_sym, sec_a).evalf())
            y_sec_b = float(f_sym.subs(x_sym, sec_b).evalf())
            secant_slope = (y_sec_b - y_sec_a) / (sec_b - sec_a)
            secant_y = y_sec_a + secant_slope * (x_plot - sec_a)
            ax.scatter([sec_a, sec_b], [y_sec_a, y_sec_b], color="purple", zorder=5, s=40)
            ax.plot(x_plot, secant_y, color="purple", linestyle=":", alpha=0.8, label=f"Secant [{sec_a:.1f}, {sec_b:.1f}]")

        ax.set_xlabel(f"{var_x} ({units_x})")
        ax.set_ylabel(f"{dep_y} ({units_y})")
        ax.set_xlim(d_min, d_max)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        # ------------------------
        # CALCULUS & CONCEPTUAL ANALYSIS
        # ------------------------
        st.markdown("---")
        st.subheader("5. Conceptual Analysis & Contextual Interpretation")

        # Average Rate of Change Interpretation
        if sec_a != sec_b:
            y_sec_a = float(f_sym.subs(x_sym, sec_a).evalf())
            y_sec_b = float(f_sym.subs(x_sym, sec_b).evalf())
            aroc = (y_sec_b - y_sec_a) / (sec_b - sec_a)
            st.markdown(f"**Average Rate of Change over [{sec_a:.2f}, {sec_b:.2f}]:** `{aroc:.4f}` {units_y}/{units_x}")
            st.caption(f"Interpretation: On average, between {var_x} = {sec_a} {units_x} and {sec_b} {units_x}, {dep_y} changes at a rate of {aroc:.4f} {units_y} per {units_x}.")

        # Instantaneous Rate Interpretation
        st.markdown(f"**Instantaneous Rate of Change at {symbol_x} = {x_eval:.2f}:** `{slope_tangent:.4f}` {units_y}/{units_x}")

        if course_level == "Calculus":
            d2_val = float(f_double_prime_sym.subs(x_sym, x_eval).evalf())
            st.markdown(f"**Second Derivative f''({x_eval:.2f}):** `{d2_val:.4f}` {units_y}/({units_x}²)")
            
            concavity = "Concave Upward 📈" if d2_val > 0 else ("Concave Downward 📉" if d2_val < 0 else "Inflection / Linear Point ➖")
            st.write(f"- **Concavity state:** {concavity}")

            # Critical Point Solver
            try:
                crit_pts = sp.solve(f_prime_sym, x_sym)
                valid_crit = [float(p.evalf()) for p in crit_pts if p.is_real and d_min <= float(p.evalf()) <= d_max]
                st.write(f"- **Critical Points in Domain:** {valid_crit if valid_crit else 'None detected analytically'}")
            except Exception:
                st.write("- **Critical Points:** Numerical expression complex to solve analytically.")

        elif course_level == "Precalculus":
            st.write("Emphasis: Analyzing function behaviors, rates of change, and end-behavior bounds.")

        elif course_level == "Honors Algebra 2":
            st.write("Emphasis: Function evaluation, domain/range limits, and average rate of change.")

        # ------------------------
        # VISUAL SIMULATION STAGE
        # ------------------------
        st.markdown("---")
        st.subheader("6. Interactive Visual Simulation Stage")

        sim_type = preset["visual_type"]
        norm_val = 0.5
        if not np.isnan(y_eval):
            y_valid_min = np.nanmin(y_plot)
            y_valid_max = np.nanmax(y_plot)
            if y_valid_max > y_valid_min:
                norm_val = float(np.clip((y_eval - y_valid_min) / (y_valid_max - y_valid_min), 0.0, 1.0))

        if sim_type == "Walking Person (Prosthetic)":
            leg_angle = int((norm_val - 0.5) * 60)
            st.markdown(f"**Prosthetic Leg Gait Simulator** (Deflection Angle: `{leg_angle}°`)")
            st.progress(norm_val)

        elif sim_type == "Moving Object / Projectile":
            st.markdown(f"**Projectile Height State:** `{y_eval:.2f}` {units_y}")
            st.progress(norm_val)

        elif sim_type == "Drug Tank Indicator":
            st.markdown(f"**Concentration Fill Level:** `{norm_val * 100:.1f}%`")
            st.progress(norm_val)

        elif sim_type == "Cell Colony / Population":
            st.markdown(f"**Colony Density Indicator:** `{y_eval:.2f}` {units_y}")
            st.progress(norm_val)

        elif sim_type == "Thermometer":
            st.markdown(f"**Temperature Gauge:** `{y_eval:.1f}` {units_y}")
            st.progress(norm_val)

        else: # Business Indicator
            st.markdown(f"**Economic Growth Index Gauge:** `{y_eval:.2f}` {units_y}")
            st.progress(norm_val)

# =========================================================
# TAB 2: LATIN AMERICA HISTORICAL DATA
# =========================================================
with tab_data_reg:
    st.header("Latin America Historical Data Analysis")
    st.write("Explore historical macroeconomic indicator regressions from Latin America.")

    years, data = get_historical_data()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        cat_hist = st.selectbox("Historical Category", ['Population', 'Unemployment rate', 'Life expectancy'], key="h_cat")
        countries_hist = st.multiselect("Select Countries", list(data.keys()), default=['Argentina'], key="h_countries")
    with col_d2:
        deg_hist = st.slider("Polynomial Degree Fit", 1, 6, 3, key="h_deg")
        step_hist = st.slider("Graph Year Step", 1, 10, 1, key="h_step")

    if countries_hist:
        df_display = pd.DataFrame({c: data[c][cat_hist] for c in countries_hist}, index=years)
        st.subheader("Raw Data Table")
        st.dataframe(df_display.style.highlight_max(axis=0))

        st.subheader("Regression Curves & Extrapolations")
        fig_h, ax_h = plt.subplots(figsize=(10, 5))

        for c in countries_hist:
            x_vals = years
            y_vals = df_display[c].values
            model = fit_poly_regression(x_vals, y_vals, deg_hist)

            x_plot = np.arange(years[0], years[-1] + 1, step_hist)
            y_plot = model.predict(x_plot.reshape(-1, 1))

            ax_h.scatter(x_vals, y_vals, label=f"{c} Data", alpha=0.6)
            ax_h.plot(x_plot, y_plot, label=f"{c} Poly Fit (deg {deg_hist})")

            # Extrapolation controls
            extrap_years = st.number_input(f"Extrapolate years for {c}", min_value=0, max_value=50, value=0, key=f"extrap_{c}")
            if extrap_years > 0:
                x_extra = np.arange(years[-1] + 1, years[-1] + extrap_years + 1)
                y_extra = model.predict(x_extra.reshape(-1, 1))
                ax_h.plot(x_extra, y_extra, '--', label=f"{c} (Extrapolated)")

        ax_h.set_xlabel("Year")
        ax_h.set_ylabel(cat_hist)
        ax_h.grid(True, linestyle="--", alpha=0.5)
        ax_h.legend()
        st.pyplot(fig_h)
    else:
        st.warning("Please select at least one country to display data.")
