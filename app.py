import html
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Latin America Data & Motion Lab", page_icon="✨", layout="wide")

st.title("Daniel Estrella's Latin America Data Regression & Motion Lab")

# ------------------------
# Data & Helper Functions
# ------------------------
@st.cache_data
def get_data():
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

def analyze_function(x, y, model):
    deriv = np.gradient(model.predict(x.reshape(-1, 1)), x)
    max_idx = np.argmax(y)
    min_idx = np.argmin(y)
    fastest_incr_idx = np.argmax(deriv)
    fastest_decr_idx = np.argmin(deriv)
    return {
        'Local max': (x[max_idx], y[max_idx]),
        'Local min': (x[min_idx], y[min_idx]),
        'Fastest increase': (x[fastest_incr_idx], deriv[fastest_incr_idx]),
        'Fastest decrease': (x[fastest_decr_idx], deriv[fastest_decr_idx]),
        'Domain': (x[0], x[-1]),
        'Range': (min(y), max(y))
    }

def format_analysis(analysis, category, country):
    return (
        f"### Analysis for {category} in {country}\n"
        f"- Local maximum: {analysis['Local max'][1]:.2f} at year {int(analysis['Local max'][0])}\n"
        f"- Local minimum: {analysis['Local min'][1]:.2f} at year {int(analysis['Local min'][0])}\n"
        f"- Fastest increase: {analysis['Fastest increase'][1]:.2f} per year at year {int(analysis['Fastest increase'][0])}\n"
        f"- Fastest decrease: {analysis['Fastest decrease'][1]:.2f} per year at year {int(analysis['Fastest decrease'][0])}\n"
        f"- Domain: {analysis['Domain'][0]} to {analysis['Domain'][1]}\n"
        f"- Range: {analysis['Range'][0]:.2f} to {analysis['Range'][1]:.2f}\n"
    )

# ------------------------
# Sidebar Controls
# ------------------------
years, data = get_data()

with st.sidebar:
    st.header("Visual Customization")
    name = st.text_input("Banner Message", value="Latin America Data Lab", max_chars=100)
    font_family = st.selectbox("Font type", ["Arial", "Courier New", "Georgia", "Times New Roman", "Verdana", "Trebuchet MS", "Impact"])
    font_size = st.slider("Font size", min_value=20, max_value=100, value=36, step=2)
    background = st.selectbox("Background theme", ["Light", "Dark", "Sky", "Sunset", "Forest", "Lavender", "Ocean", "Midnight"])
    name_mode = st.selectbox("Name display mode", ["Centered", "Ticker Tape", "Repeated"])
    effect = st.selectbox("Dynamic action", ["None", "Rain", "Snow", "Hail", "Wind Gust", "Confetti", "Tornado"])
    speed = st.slider("Animation speed", min_value=1, max_value=10, value=5, help="1 is slow; 10 is fast.")

    st.markdown("---")
    st.header("Regression Controls")
    category = st.selectbox("Select Data Category", ['Population', 'Unemployment rate', 'Life expectancy'])
    countries = st.multiselect("Select countries to display", list(data.keys()), default=['Argentina'])
    year_step = st.slider("Graph increments (years)", 1, 10, 1)
    degree = st.slider("Degree of polynomial regression", 3, 6, 3)

# ------------------------
# Motion Banner Stage
# ------------------------
backgrounds = {
    "Light": ("#f7f8fc", "#202437"),
    "Dark": ("#10131c", "#f7f8ff"),
    "Sky": ("linear-gradient(135deg, #bde7ff, #f5fcff)", "#173c61"),
    "Sunset": ("linear-gradient(135deg, #ffb36b, #ff6f91, #7656a8)", "#ffffff"),
    "Forest": ("linear-gradient(135deg, #153d2b, #4f8b61)", "#f4fff4"),
    "Lavender": ("linear-gradient(135deg, #e9ddff, #c7d7ff)", "#342a5e"),
    "Ocean": ("linear-gradient(135deg, #006d83, #42c6c8)", "#ffffff"),
    "Midnight": ("radial-gradient(circle at top, #27356b, #090b18 75%)", "#ffffff"),
}
bg, text_color = backgrounds[background]

symbols = {
    "Rain": ["💧", "💦"],
    "Snow": ["❄️", "❅", "✻"],
    "Hail": ["🧊", "●"],
    "Wind Gust": ["〰", "💨", "﹏"],
    "Confetti": ["🎉", "✦", "●", "▲", "■", "✿"],
    "Tornado": ["🌪️", "〰", "·"],
}

effect_items = []
if effect != "None":
    rng = random.Random(23)
    count = 54 if effect in ("Rain", "Snow", "Hail", "Confetti") else 28
    for _ in range(count):
        symbol = rng.choice(symbols[effect])
        left = rng.randint(0, 100)
        delay = rng.uniform(0, 5)
        duration = rng.uniform(1.5, 5.5) * (1.7 - speed / 12)
        size = rng.randint(12, 27) if effect != "Tornado" else rng.randint(15, 30)
        drift = rng.randint(-90, 90)
        
        animations = {
            "Rain": "fall", "Hail": "fall",
            "Snow": "snowfall", "Wind Gust": "gust",
            "Confetti": "confetti"
        }
        animation = animations.get(effect, "spin")
        
        effect_items.append(
            f'<span class="particle {animation}" style="left:{left}%;'
            f'animation-delay:-{delay:.2f}s;animation-duration:{duration:.2f}s;'
            f'font-size:{size}px;--drift:{drift}px;--spin:{rng.randint(180, 720)}deg;">'
            f'{html.escape(symbol)}</span>'
        )

safe_name = html.escape(name.strip() or "Latin America Data Lab")
if name_mode == "Ticker Tape":
    name_content = f'<div class="ticker"><span>{safe_name}&nbsp;&nbsp;&nbsp; ✨ &nbsp;&nbsp;&nbsp;{safe_name}&nbsp;&nbsp;&nbsp; ✨ &nbsp;&nbsp;&nbsp;</span></div>'
elif name_mode == "Repeated":
    name_content = '<div class="repeated">' + "<br>".join([safe_name] * 3) + "</div>"
else:
    name_content = f'<div class="center-name">{safe_name}</div>'

font_css = {
    "Arial": "Arial, sans-serif",
    "Courier New": "'Courier New', monospace",
    "Georgia": "Georgia, serif",
    "Times New Roman": "'Times New Roman', serif",
    "Verdana": "Verdana, sans-serif",
    "Trebuchet MS": "'Trebuchet MS', sans-serif",
    "Impact": "Impact, sans-serif",
}[font_family]

html_doc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
.stage {{
  position: relative; isolation: isolate; overflow: hidden;
  height: 220px; width: 100%; border-radius: 12px;
  background: {bg}; color: {text_color};
  font-family: {font_css}; box-shadow: inset 0 0 0 1px rgba(120,120,150,.18);
}}
.particle-layer {{ position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:1; }}
.particle {{ position:absolute; top:-12%; opacity:.8; user-select:none; will-change:transform; }}
.center-name {{
  position:absolute; z-index:3; left:5%; right:5%; top:50%;
  transform:translateY(-50%); text-align:center; overflow-wrap:anywhere;
  font-size:{font_size}px; font-weight:700; line-height:1.18;
  text-shadow: 0 2px 16px rgba(0,0,0,.13);
}}
.repeated {{
  position:absolute; z-index:3; inset:7% 4%; text-align:center;
  font-size:{max(14, int(font_size * .6))}px; font-weight:700;
  line-height:1.4; overflow-wrap:anywhere;
}}
.ticker {{ position:absolute; z-index:3; top:35%; white-space:nowrap; font-size:{font_size}px; font-weight:700; }}
.ticker span {{ display:inline-block; padding-left:100%; animation: marquee {max(7, 24 - speed * 1.5)}s linear infinite; }}
@keyframes marquee {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-100%); }} }}
@keyframes fall {{ from {{ transform:translate3d(0,-10vh,0); }} to {{ transform:translate3d(var(--drift),115vh,0); }} }}
@keyframes snowfall {{ 0% {{ transform:translate3d(0,-10vh,0) rotate(0); }} 50% {{ transform:translate3d(calc(var(--drift) * -1),50vh,0) rotate(180deg); }} 100% {{ transform:translate3d(var(--drift),115vh,0) rotate(360deg); }} }}
@keyframes gust {{ 0% {{ transform:translate3d(-20vw,10vh,0) rotate(0); opacity:0; }} 15% {{ opacity:.85; }} 85% {{ opacity:.85; }} 100% {{ transform:translate3d(120vw,-8vh,0) rotate(20deg); opacity:0; }} }}
@keyframes confetti {{ 0% {{ transform:translate3d(0,-10vh,0) rotate(0); }} 100% {{ transform:translate3d(var(--drift),115vh,0) rotate(var(--spin)); }} }}
@keyframes spin {{ 0% {{ transform:translate3d(0,-12vh,0) rotate(0) scale(.8); }} 50% {{ transform:translate3d(var(--drift),50vh,0) rotate(360deg) scale(1.3); }} 100% {{ transform:translate3d(0,115vh,0) rotate(720deg) scale(.8); }} }}
</style>
</head>
<body>
<div class="stage">
  <div class="particle-layer">{''.join(effect_items)}</div>
  {name_content}
</div>
</body>
</html>
"""

components.html(html_doc, height=230, scrolling=False)

# ------------------------
# Data Table & Regression Analysis
# ------------------------
if not countries:
    st.warning("Please select at least one country from the sidebar to view analysis.")
else:
    st.subheader("Raw Data Table (Editable)")
    df_display = pd.DataFrame({country: data[country][category] for country in countries}, index=years)
    edited_df = st.data_editor(df_display, num_rows="dynamic")

    # Regression Graph
    st.subheader("Regression Graph")
    fig, ax = plt.subplots(figsize=(10, 5))

    extrap_values = {}
    for country in countries:
        x = years
        y = edited_df[country].values
        model = fit_poly_regression(x, y, degree)
        
        x_plot = np.arange(years[0], years[-1] + 1, year_step)
        y_plot = model.predict(x_plot.reshape(-1, 1))
        
        extrap_years = st.number_input(f"Extrapolate years for {country}", min_value=0, max_value=50, value=0, key=f"extrap_{country}")
        if extrap_years > 0:
            x_extra = np.arange(years[-1] + 1, years[-1] + extrap_years + 1)
            y_extra = model.predict(x_extra.reshape(-1, 1))
            ax.plot(x_extra, y_extra,