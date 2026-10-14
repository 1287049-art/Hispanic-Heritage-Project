import html
import math
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from streamlit_autorefresh import st_autorefresh
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer 2.0", page_icon="📈", layout="wide")
st.title("Hispanic Heritage Historical Data Explorer 2.0")
st.caption("Explore Hispanic/Latino history, contributions, and historical population data through graphs, functions, and calculus.")

# Hispanic Heritage content: historical milestones, Census data, and STEM contributions.
heritage_tabs = st.tabs([
    "Hispanic Heritage history",
    "Population data + calculus",
    "Hispanic/Latino STEM pioneers",
])

with heritage_tabs[0]:
    st.subheader("Why Hispanic Heritage Month begins on September 15")
    st.write(
        "National Hispanic Heritage Month runs from September 15 through October 15. "
        "It recognizes the histories, cultures, and contributions of Hispanic Americans. "
        "The start date connects to independence celebrations in several Latin American countries."
    )
    timeline = pd.DataFrame([
        {"Year/date": "September 15", "Milestone": "Costa Rica, El Salvador, Guatemala, Honduras, and Nicaragua celebrate independence day."},
        {"Year/date": "September 16", "Milestone": "Mexico celebrates its Independence Day."},
        {"Year/date": "1968", "Milestone": "Congress established Hispanic Heritage Week; President Lyndon B. Johnson issued the first proclamation."},
        {"Year/date": "1988", "Milestone": "Congress expanded the observance into a month, September 15–October 15."},
    ])
    st.dataframe(timeline, hide_index=True, use_container_width=True)
    st.info(
        "Hispanic and Latino are broad terms that include many different national origins, "
        "cultures, histories, and identities—not one single culture."
    )
    st.markdown(
        "**Sources:** [Library of Congress: National Hispanic Heritage Month](https://guides.loc.gov/latinx-studies/nhhm) · "
        "[Law Library of Congress: history and legislation](https://wwws.loc.gov/law/help/commemorative-observations/hispanic-heritage.php)"
    )

with heritage_tabs[1]:
    st.subheader("How the U.S. Hispanic population changed over time")
    st.write(
        "Use the historical Census figures below to explore change over time. "
        "The data are decennial snapshots, not annual measurements. Census definitions and methods can change, "
        "so comparisons should be interpreted with that limitation in mind."
    )
    heritage_data = pd.DataFrame({
        "Census year": [1980, 1990, 2000, 2010, 2020],
        "Hispanic population (millions)": [14.6, 22.4, 35.3, 50.5, 62.1],
        "Share of U.S. population (%)": [6.4, 9.0, 12.5, 16.3, 18.7],
    })
    measure = st.radio(
        "Choose what to graph",
        ["Hispanic population (millions)", "Share of U.S. population (%)"],
        horizontal=True,
        key="heritage_measure",
    )
    fit_degree_heritage = st.slider(
        "Polynomial model degree",
        min_value=1,
        max_value=3,
        value=2,
        key="heritage_degree",
        help="A higher-degree curve can fit these five points more closely, but may behave poorly outside the observed years.",
    )
    y_values = heritage_data[measure].to_numpy(dtype=float)
    years = heritage_data["Census year"].to_numpy(dtype=float)
    t_values = years - 1980.0
    heritage_coeffs = np.polyfit(t_values, y_values, fit_degree_heritage)
    heritage_poly = np.poly1d(heritage_coeffs)
    t_grid = np.linspace(0, 40, 400)
    year_grid = t_grid + 1980
    pop_fig = go.Figure()
    pop_fig.add_trace(go.Scatter(
        x=years, y=y_values, mode="markers+lines", name="Census data",
        marker=dict(size=10), line=dict(width=3),
    ))
    pop_fig.add_trace(go.Scatter(
        x=year_grid, y=heritage_poly(t_grid), mode="lines",
        name=f"Degree {fit_degree_heritage} polynomial model",
        line=dict(dash="dash", width=2),
    ))
    pop_fig.update_layout(
        template="plotly_white", height=420,
        xaxis_title="Census year", yaxis_title=measure,
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=15, r=15, t=35, b=15),
    )
    st.plotly_chart(pop_fig, use_container_width=True)
    st.dataframe(heritage_data, hide_index=True, use_container_width=True)

    first_value = float(y_values[0])
    last_value = float(y_values[-1])
    overall_change = last_value - first_value
    percent_change = 100 * overall_change / first_value
    average_decade_change = overall_change / 4
    m1, m2, m3 = st.columns(3)
    m1.metric("Change, 1980–2020", f"+{overall_change:.1f}" + (" million" if "millions" in measure else " percentage points"))
    m2.metric("Relative change", f"{percent_change:.1f}%")
    m3.metric("Average per decade", f"{average_decade_change:.2f}" + (" million" if "millions" in measure else " percentage points"))

    selected_year = st.slider(
        "Inspect the fitted model at a year",
        min_value=1980,
        max_value=2020,
        value=2000,
        step=1,
        key="heritage_selected_year",
    )
    selected_t = float(selected_year - 1980)
    fitted_value = float(heritage_poly(selected_t))
    derivative_poly = np.polyder(heritage_poly)
    model_rate = float(derivative_poly(selected_t))
    label_unit = "million people" if "millions" in measure else "percent of the U.S. population"
    rate_unit = "million people per year" if "millions" in measure else "percentage points per year"
    r1, r2 = st.columns(2)
    r1.metric(f"Model value in {selected_year}", f"{fitted_value:.2f}")
    r2.metric("Instantaneous model rate", f"{model_rate:.3f}")
    st.write(
        f"At {selected_year}, this fitted curve gives about {fitted_value:.2f} {label_unit}. "
        f"Its derivative is about {model_rate:.3f} {rate_unit}. This is the slope of the selected mathematical model, "
        "not a directly measured annual Census change."
    )
    st.caption(
        "Data source: U.S. Census Bureau historical Hispanic-origin counts and shares, rounded for classroom use. "
        "The fitted polynomial is a classroom model, not an official Census forecast. Avoid extrapolating it beyond 1980–2020."
    )
    st.markdown(
        "**Reference:** [U.S. Census Bureau working paper with historical Hispanic population tables](https://www2.census.gov/library/working-papers/2025/adrm/ces/CES-WP-25-50.pdf)"
    )

with heritage_tabs[2]:
    st.subheader("Hispanic/Latino contributions in science and engineering")
    st.write(
        "Hispanic and Latino history includes contributions across science, engineering, environmental research, "
        "medicine, public service, literature, music, sports, and many other fields. Here are two examples connected to STEM."
    )
    st.markdown("### Ellen Ochoa — engineering and space exploration")
    st.write(
        "Ellen Ochoa is an engineer and former NASA astronaut. In 1993, she became the first Hispanic woman to travel to space. "
        "Her engineering research included optical systems for information processing, and she later led NASA's Johnson Space Center."
    )
    st.markdown("[NASA biography: Ellen Ochoa](https://www.nasa.gov/former-astronaut-ellen-ochoa/)")
    st.divider()
    st.markdown("### Mario J. Molina — chemistry and the atmosphere")
    st.write(
        "Mario Molina, a chemist born in Mexico City, shared the 1995 Nobel Prize in Chemistry for research on atmospheric chemistry, "
        "including the formation and decomposition of ozone. His work helped show how certain human-made chemicals damage the ozone layer."
    )
    st.markdown("[Nobel Prize biography: Mario J. Molina](https://www.nobelprize.org/prizes/chemistry/1995/molina/facts/)")
    st.info(
        "Explore the mathematics connection: scientific work uses functions to model changing quantities, derivatives to describe rates, "
        "and data analysis to test whether a model agrees with observations."
    )
    st.markdown(
        "**Explore more:** [Library of Congress Hispanic heritage resources](https://guides.loc.gov/latinx-studies/nhhm)"
    )

st.divider()
st.header("Interactive Function & Calculus Lab")
st.write(
    "Use the general math explorer below to connect functions, derivatives, and rates of change "
    "to other applications. The Hispanic population data above is the project’s historical-data feature."
)

x = sp.Symbol("x", real=True)
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCALS = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "sqrt": sp.sqrt,
          "log": sp.log, "ln": sp.log, "exp": sp.exp, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]
CONTEXTS = ["Business", "Engineering", "Medical", "Sports science", "Biology", "Bioengineering", "Economic forecasting", "Weather", "Custom"]

# function, independent variable, x unit, dependent variable, y unit, description, suggested visual
SUGGESTIONS = {
 "Business": ("-2*(x-5)^2+80", "Items sold", "items", "Revenue", "dollars", "Revenue rises toward an optimum, then falls in this simplified model.", "Business / economic indicator"),
 "Engineering": ("-4.9*x^2+24*x+1.5", "Time", "seconds", "Height", "meters", "A simplified projectile-height model; interpret only while the object is in flight.", "Moving object or projectile"),
 "Medical": ("8*exp(-0.35*x)", "Time", "hours", "Concentration", "mg/L", "An illustrative concentration-decay model, not medical advice.", "Tank / concentration indicator"),
 "Sports science": ("-4.9*x^2+5*x+0.8", "Time", "seconds", "Jump height", "meters", "A simplified vertical-jump model.", "Moving object or projectile"),
 "Biology": ("100*x/(x+4)", "Time", "days", "Population index", "index units", "A simple saturating population index, not a fitted dataset.", "Population / cell colony"),
 "Bioengineering": ("35*sin(2*pi*x/2)", "Time", "seconds", "Joint angle", "degrees", "A simplified periodic joint-angle pattern, not a clinical gait model.", "Walking person with prosthetic leg"),
 "Economic forecasting": ("100*exp(0.04*x)", "Time", "years", "Economic index", "index points", "An illustrative growth trend, not an official forecast.", "Business / economic indicator"),
 "Weather": ("18+7*sin(2*pi*(x-6)/24)", "Time of day", "hours", "Temperature", "°C", "A simplified repeating daily temperature pattern.", "Thermometer / weather indicator"),
 "Custom": ("x^2-4*x+3", "Input", "units", "Output", "units", "Describe your own situation and variables.", "Business / economic indicator"),
}

def parse_function(source):
    if not source or not source.strip():
        raise ValueError("Enter a function, for example x^2 - 4*x + 3.")
    if len(source) > 250 or "__" in source or re.search(r"[{}\[\];]", source):
        raise ValueError("Use a function under 250 characters with ordinary math notation only.")
    try:
        expr = parse_expr(source, local_dict=LOCALS, transformations=TRANSFORMS, evaluate=True)
    except Exception as exc:
        raise ValueError("Couldn't read that function. Try x^2, 1/(x-2), exp(x), log(x), or sin(x).") from exc
    if not isinstance(expr, sp.Expr) or (expr.free_symbols - {x}):
        raise ValueError("Use x as the independent variable; unknown symbols are not supported.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The expression contains an undefined or infinite constant.")
    return sp.simplify(expr)

def val(expr, point):
    try:
        z = complex(sp.N(expr.subs(x, float(point))))
        return float(z.real) if abs(z.imag) < 1e-7 and math.isfinite(z.real) else None
    except Exception:
        return None

def roots_in(expr, lo, hi, samples=1000):
    roots = []
    try:
        answer = sp.solveset(sp.Eq(expr, 0), x, domain=sp.Interval(lo, hi))
        if isinstance(answer, sp.FiniteSet):
            for root in answer:
                q = float(sp.N(root))
                if lo-1e-7 <= q <= hi+1e-7 and not any(abs(q-r)<1e-4 for r in roots): roots.append(q)
    except Exception:
        pass
    try:
        fn = sp.lambdify(x, expr, modules=["numpy"])
        grid = np.linspace(lo, hi, samples)
        with np.errstate(all="ignore"):
            yy = np.asarray(fn(grid), dtype=float)
            if yy.ndim == 0: yy = np.full_like(grid, float(yy))
        for i in range(len(grid)-1):
            aa, bb, ya, yb = grid[i], grid[i+1], yy[i], yy[i+1]
            if not (np.isfinite(ya) and np.isfinite(yb)): continue
            if abs(ya) < 1e-7: root = float(aa)
            elif ya*yb < 0:
                left, right, fl = float(aa), float(bb), float(ya)
                for _ in range(45):
                    mid = (left+right)/2; fm = val(expr, mid)
                    if fm is None: break
                    if fl*fm <= 0: right = mid
                    else: left, fl = mid, fm
                root = (left+right)/2
            else: continue
            if lo-1e-7 <= root <= hi+1e-7 and not any(abs(root-r)<1e-4 for r in roots): roots.append(root)
    except Exception:
        pass
    return sorted(roots)

def curve(expr, xx):
    try:
        fn = sp.lambdify(x, expr, modules=["numpy"])
        with np.errstate(all="ignore"):
            yy = np.asarray(fn(xx), dtype=float)
            if yy.ndim == 0: yy = np.full_like(xx, float(yy))
        yy[~np.isfinite(yy)] = np.nan
        finite = yy[np.isfinite(yy)]
        if finite.size: yy[np.abs(yy) > max(1e4, np.percentile(np.abs(finite), 95)*30)] = np.nan
        return yy
    except Exception:
        return np.full_like(xx, np.nan)

def fmt(number): return "undefined" if number is None else f"{number:.5g}"

def extremum_test(point):
    eps = max(1e-5, abs(point)*1e-5)
    left, right = val(fp, point-eps), val(fp, point+eps)
    if left is None or right is None: return "inconclusive near a domain boundary"
    if left > 1e-7 and right < -1e-7: return "local maximum (f′ changes + to −)"
    if left < -1e-7 and right > 1e-7: return "local minimum (f′ changes − to +)"
    return "not classified as an extremum (no sign change detected)"

with st.sidebar:
    st.header("1 · Course and context")
    course = st.selectbox("Course mode", COURSES, index=2)
    context = st.selectbox("Context", CONTEXTS)
    default_fn, default_x, default_xu, default_y, default_yu, default_desc, default_visual = SUGGESTIONS[context]
    st.caption(default_desc)
    source = st.radio("Function source", ["Suggested function", "Build with parameters", "Enter my own function", "Fit curve to uploaded CSV data"])
    st.header("2 · Model details")
    x_name = st.text_input("Independent variable", default_x)
    x_unit = st.text_input("Independent variable unit", default_xu)
    y_name = st.text_input("Dependent variable", default_y)
    y_unit = st.text_input("Dependent variable unit", default_yu)
    description = st.text_area("Context description", default_desc, height=80)
    ca, cb = st.columns(2)
    with ca: lo = st.number_input("Interval start", value=0.0, step=0.5)
    with cb: hi = st.number_input("Interval end", value=10.0, step=0.5)
    function_text = default_fn
    uploaded_df = None
    fit_points = None
    if source == "Fit curve to uploaded CSV data":
        csv_file = st.file_uploader("Upload CSV with numeric x and y columns", type=["csv"])
        fit_degree = st.selectbox("Polynomial fit degree", [1, 2, 3, 4, 5], index=2)
        if csv_file is not None:
            try:
                uploaded_df = pd.read_csv(csv_file)
                numeric_cols = list(uploaded_df.select_dtypes(include="number").columns)
                if len(numeric_cols) < 2:
                    st.error("Your CSV needs at least two numeric columns: one for x and one for y.")
                    uploaded_df = None
                else:
                    x_col = st.selectbox("Column for independent variable x", numeric_cols, index=0)
                    y_options = [col for col in numeric_cols if col != x_col]
                    y_col = st.selectbox("Column for dependent variable y", y_options, index=0)
                    clean = uploaded_df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(clean) <= fit_degree:
                        st.error(f"A degree-{fit_degree} fit needs at least {fit_degree + 1} valid data rows.")
                    else:
                        coeffs = np.polyfit(clean[x_col].to_numpy(float), clean[y_col].to_numpy(float), fit_degree)
                        poly = np.poly1d(coeffs)
                        terms = []
                        for i, coefficient in enumerate(coeffs):
                            power = fit_degree - i
                            if abs(coefficient) < 1e-12:
                                continue
                            if power == 0: term = f"({coefficient:.12g})"
                            elif power == 1: term = f"({coefficient:.12g})*x"
                            else: term = f"({coefficient:.12g})*x^{power}"
                            terms.append(term)
                        function_text = "+".join(terms) if terms else "0"
                        fit_points = (clean[x_col].to_numpy(float), clean[y_col].to_numpy(float), x_col, y_col, poly)
                        st.success(f"Fitted a degree-{fit_degree} polynomial to {len(clean)} rows.")
                        st.code("f(x) = " + function_text)
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
    elif source == "Build with parameters":
        family = st.selectbox("Function family", ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
        if family == "Polynomial":
            aa=st.number_input("Quadratic a",value=1.0,step=0.5); bb=st.number_input("Linear b",value=-4.0,step=0.5); cc=st.number_input("Constant c",value=3.0,step=0.5)
            function_text=f"({aa})*x^2+({bb})*x+({cc})"
        elif family == "Rational":
            aa=st.number_input("Numerator scale",value=10.0,step=1.0); bb=st.number_input("Denominator shift",value=2.0,step=0.5); cc=st.number_input("Vertical offset",value=0.0,step=0.5)
            function_text=f"({aa})/(x-({bb}))+({cc})"
        elif family == "Exponential":
            aa=st.number_input("Initial scale",value=2.0,step=0.5); bb=st.number_input("Growth/decay rate",value=0.2,step=0.05); cc=st.number_input("Vertical offset",value=0.0,step=0.5)
            function_text=f"({aa})*exp(({bb})*x)+({cc})"
        elif family == "Logarithmic":
            aa=st.number_input("Log scale",value=2.0,step=0.5); bb=st.number_input("Horizontal shift",value=1.0,step=0.5); cc=st.number_input("Vertical offset",value=0.0,step=0.5)
            function_text=f"({aa})*log(x-({bb}))+({cc})"
        else:
            aa=st.number_input("Amplitude",value=5.0,step=0.5); bb=st.number_input("Angular frequency",value=1.0,step=0.25); cc=st.number_input("Vertical shift",value=0.0,step=0.5)
            function_text=f"({aa})*sin(({bb})*x)+({cc})"
    elif source == "Enter my own function":
        function_text=st.text_input("f(x) =",default_fn,help="Examples: x^2-4*x+3, 1/(x-2), exp(x), log(x), sin(x), cos(x), tan(x).")
    st.header("3 · Visual simulation")
    visuals=["Walking person with prosthetic leg","Moving object or projectile","Tank / concentration indicator","Population / cell colony","Thermometer / weather indicator","Business / economic indicator"]
    visual=st.selectbox("Contextual visual",visuals,index=visuals.index(default_visual))
    speed=st.slider("Playback speed",1,10,5)
    play=st.toggle("Play animation",value=False)
    reset=st.button("Reset animation")

if hi <= lo:
    st.error("Interval end must be greater than interval start."); st.stop()
if source == "Fit curve to uploaded CSV data" and fit_points is None:
    st.warning("Upload a CSV with at least two numeric columns and enough valid rows to fit a curve."); st.stop()
try: f=parse_function(function_text)
except ValueError as exc:
    st.error(f"Function input error: {exc}")
    st.info("Correct the expression. Supported examples: x^2, 1/(x-2), exp(x), log(x), sin(x), cos(x), tan(x).")
    st.stop()
fp, fpp = sp.simplify(sp.diff(f,x)), sp.simplify(sp.diff(f,x,2))
if "motion_x" not in st.session_state: st.session_state.motion_x=float(lo)
if reset: st.session_state.motion_x=float(lo)
if play:
    st_autorefresh(interval=1100-speed*90,key="animation_refresh")
    st.session_state.motion_x += (hi-lo)/(22-speed)
    if st.session_state.motion_x > hi: st.session_state.motion_x=float(lo)
else: st.session_state.motion_x=min(max(st.session_state.motion_x,lo),hi)
if play or reset:
    st.session_state["current_input"] = float(st.session_state.motion_x)

st.subheader("Function and derivatives")
c1,c2,c3=st.columns(3)
c1.markdown("**Function**"); c1.latex("f(x)="+sp.latex(f))
c2.markdown("**First derivative**"); c2.latex("f'(x)="+sp.latex(fp) if course!="Honors Algebra 2" else r"\text{Optional extension}")
c3.markdown("**Second derivative**"); c3.latex("f''(x)="+sp.latex(fpp) if course=="Calculus" else r"\text{Calculus preview}")
if course=="Honors Algebra 2": st.caption("Focus: function values, graph shape, domain, and average rate of change.")
elif course=="Precalculus": st.caption("Focus: function families, transformations, domain, and rates of change; derivatives are a preview.")
else: st.caption("Calculus mode includes derivative tests, extrema, monotonicity, concavity, and inflection analysis.")

_,d1,d2=st.columns(3)
current=st.slider("Current input x",float(lo),float(hi),float(min(max(st.session_state.motion_x,lo),hi)),key="current_input")
show_fp=d1.checkbox("Show f′(x)",value=True); show_fpp=d2.checkbox("Show f″(x)",value=(course=="Calculus"))
aoc1,aoc2=st.columns(2)
a=aoc1.slider("Average-rate interval start",float(lo),float(hi),float(lo),key="avg_start")
b=aoc2.slider("Average-rate interval end",float(lo),float(hi),float(hi),key="avg_end")
xs=np.linspace(lo,hi,1000); fig=go.Figure()
fig.add_trace(go.Scatter(x=xs,y=curve(f,xs),name="f(x)",mode="lines",line=dict(width=3)))
if fit_points is not None:
    px, py, x_col_name, y_col_name, _poly = fit_points
    fig.add_trace(go.Scatter(x=px, y=py, name="Uploaded data", mode="markers", marker=dict(size=8, color="#7b2cbf")))
if show_fp: fig.add_trace(go.Scatter(x=xs,y=curve(fp,xs),name="f′(x)",mode="lines",line=dict(dash="dash")))
if show_fpp: fig.add_trace(go.Scatter(x=xs,y=curve(fpp,xs),name="f″(x)",mode="lines",line=dict(dash="dot")))
fy,slope=val(f,current),val(fp,current)
if fy is not None:
    fig.add_trace(go.Scatter(x=[current],y=[fy],name="Current point",mode="markers",marker=dict(size=12)))
    if slope is not None:
        tx=np.array([lo,hi]); fig.add_trace(go.Scatter(x=tx,y=fy+slope*(tx-current),name="Tangent line",mode="lines",line=dict(color="red",dash="dash")))
if b>a:
    ya,yb=val(f,a),val(f,b)
    if ya is not None and yb is not None:
        avg=(yb-ya)/(b-a); sx=np.array([lo,hi])
        fig.add_trace(go.Scatter(x=sx,y=ya+avg*(sx-a),name="Secant line",mode="lines",line=dict(color="green",dash="dash")))
        fig.add_trace(go.Scatter(x=[a,b],y=[ya,yb],name="Interval endpoints",mode="markers"))
fig.update_layout(template="plotly_white",height=520,margin=dict(l=15,r=15,t=35,b=15),xaxis_title=f"{x_name} ({x_unit})",yaxis_title=f"{y_name} ({y_unit})",legend=dict(orientation="h",y=1.12))
st.plotly_chart(fig,use_container_width=True)

st.subheader("Rates of change")
r1,r2,r3=st.columns(3)
r1.metric("f(x)",fmt(fy)); r2.metric("Instantaneous rate f′(x)",fmt(slope),help=f"Units: {y_unit} per {x_unit}."); r3.metric("Second derivative f″(x)",fmt(val(fpp,current)),help=f"Units: {y_unit} per {x_unit} squared.")
if b>a:
    ya,yb=val(f,a),val(f,b)
    if ya is not None and yb is not None: st.write(f"**Average rate of change:** {fmt((yb-ya)/(b-a))} {y_unit} per {x_unit}.")
    else: st.warning("Average rate is undefined because an endpoint is outside the real-valued domain.")
else: st.info("Choose different interval endpoints to calculate average rate of change.")

st.subheader("Interpretation in context")
if fy is None: st.warning(f"The function is undefined at {x_name} = {current:g}; move the point to a valid input.")
else:
    st.write(f"At **{x_name} = {current:.4g} {x_unit}**, the model predicts **{y_name} = {fy:.4g} {y_unit}**.")
    if slope is not None:
        trend="increasing" if slope>1e-7 else "decreasing" if slope < -1e-7 else "locally flat"
        st.write(f"The first derivative is {slope:.4g} {y_unit} per {x_unit}; the modeled quantity is **{trend}** at this input.")
    curv=val(fpp,current)
    if curv is not None:
        shape="concave up" if curv>1e-7 else "concave down" if curv < -1e-7 else "has approximately zero second derivative"
        st.write(f"The second derivative is {curv:.4g} {y_unit} per {x_unit}², so the curve is **{shape}** locally.")
if context in ("Engineering","Sports science"): st.caption("For position-versus-time, f′ represents velocity and f″ represents acceleration. For height-versus-time, they represent vertical velocity and acceleration.")
elif context in ("Business","Economic forecasting"): st.caption("The first derivative describes marginal change; the second derivative indicates whether the rate of growth or decline is itself changing.")
elif context=="Medical": st.caption("This is an educational concentration model only, not medical advice or a dosing recommendation.")
elif context=="Bioengineering": st.caption("For the joint-angle example, derivatives represent angular rate and change in angular rate; this is not a clinical gait assessment.")
else: st.caption(description)

st.subheader("Calculus analysis")
if course=="Calculus":
    critical=roots_in(fp,lo,hi); crit_values=[(q,val(f,q)) for q in critical if val(f,q) is not None]
    st.markdown("**Critical points and derivative-test classification**")
    if crit_values:
        for q,yq in crit_values: st.write(f"- x ≈ {q:.6g}, f(x) ≈ {yq:.6g}: {extremum_test(q)}.")
    else: st.write("No stationary critical points detected in this interval. Points where f′ is undefined may also be critical when f is defined.")
    trends=[]; bounds=[lo]+critical+[hi]
    for u,v in zip(bounds[:-1],bounds[1:]):
        if v-u<1e-7: continue
        d=val(fp,(u+v)/2)
        if d is not None: trends.append(f"{'increasing' if d>1e-7 else 'decreasing' if d < -1e-7 else 'approximately constant'} on ({u:.4g}, {v:.4g})")
    st.write("**Intervals:** "+("; ".join(trends) if trends else "Could not reliably determine intervals."))
    st.markdown("**Absolute extrema on the selected closed interval**")
    candidates=[(lo,val(f,lo)),(hi,val(f,hi))]+crit_values; candidates=[p for p in candidates if p[1] is not None]
    if candidates:
        mn,mx=min(candidates,key=lambda p:p[1]),max(candidates,key=lambda p:p[1])
        st.write(f"- Minimum candidate: f({mn[0]:.5g}) ≈ {mn[1]:.5g}.")
        st.write(f"- Maximum candidate: f({mx[0]:.5g}) ≈ {mx[1]:.5g}.")
        st.caption("Numerical candidates only. Discontinuities or excluded points can mean absolute extrema do not exist.")
    else: st.write("No valid extrema candidates found.")
    st.markdown("**Concavity and inflection points**")
    inflect=roots_in(fpp,lo,hi); confirmed=[]
    for q in inflect:
        eps=max(1e-4,(hi-lo)*1e-5); yl=val(fpp,max(lo,q-eps)); yr=val(fpp,min(hi,q+eps)); yq=val(f,q)
        if yl is not None and yr is not None and yq is not None and yl*yr < -1e-10: confirmed.append((q,yq,yl,yr))
    if confirmed:
        for q,yq,yl,yr in confirmed:
            before="concave up" if yl>0 else "concave down"; after="concave up" if yr>0 else "concave down"
            st.write(f"- ({q:.6g}, {yq:.6g}); concavity changes from {before} to {after}.")
    else: st.write("No inflection points with a confirmed concavity change were detected.")
    concavity=[]
    for u,v in zip([lo]+inflect,inflect+[hi]):
        if v-u<1e-7: continue
        q=val(fpp,(u+v)/2)
        if q is not None: concavity.append(f"{'concave up' if q>0 else 'concave down' if q<0 else 'near-linear'} on ({u:.4g}, {v:.4g})")
    st.write("**Concavity intervals:** "+("; ".join(concavity) if concavity else "Not determined reliably."))
else: st.info("Choose Calculus mode for critical-point tests, absolute-extrema candidates, monotonic intervals, and inflection analysis.")

st.markdown("**Domain restrictions and discontinuities**")
denom=sp.denom(sp.together(f)); notes=[]
if denom!=1:
    poles=roots_in(denom,lo,hi)
    if poles: notes.append("Denominator may be zero near x ≈ "+", ".join(f"{q:.5g}" for q in poles)+".")
for node in sp.preorder_traversal(f):
    if node.func==sp.log: notes.append("Logarithms require their argument to be greater than zero."); break
    if node.func==sp.sqrt: notes.append("Square roots require their radicand to be nonnegative for real outputs."); break
if notes:
    for note in notes: st.write("- "+note)
else: st.write("No common denominator, logarithm, or square-root restriction was detected in this interval. This numerical check is not a proof of the full domain.")

st.subheader("Context simulation")
st.caption("The visual level is normalized to finite function values in the selected interval. It is an illustrative scale, not a physical measurement unless the model supports that interpretation.")
ys=curve(f,xs); finite=ys[np.isfinite(ys)]
if fy is not None and finite.size:
    ymin,ymax=float(np.min(finite)),float(np.max(finite)); level=.5 if abs(ymax-ymin)<1e-10 else min(1,max(0,(fy-ymin)/(ymax-ymin))); pct=int(round(level*100))
else: pct=0
emojis={"Walking person with prosthetic leg":"🚶 🦿","Moving object or projectile":"🚀","Tank / concentration indicator":"🧪","Population / cell colony":"🦠","Thermometer / weather indicator":"🌡️","Business / economic indicator":"📊"}
left,right=st.columns([1,2])
with left: st.markdown(f"<div style='font-size:4rem;text-align:center'>{emojis[visual]}</div><div style='text-align:center;font-weight:600'>{html.escape(visual)}</div>",unsafe_allow_html=True)
with right:
    st.write(f"**Simulation input:** {current:.4g} {x_unit} · **Output:** {fmt(fy)} {y_unit}"); st.progress(pct,text=f"Normalized model level: {pct}%")
    descriptions={"Walking person with prosthetic leg":"The indicator follows the selected function as a conceptual gait-cycle cue; it is not a biomechanical or clinical simulation.","Moving object or projectile":"The indicator tracks the modeled output. If f is height versus time, interpret the value as modeled height.","Tank / concentration indicator":"The fill level is normalized output, not a real volume, dose, or treatment recommendation.","Population / cell colony":"The indicator reflects normalized model output, not a measured cell count.","Thermometer / weather indicator":"Read the actual modeled temperature and units above; the bar itself is normalized.","Business / economic indicator":"The indicator reflects normalized output; use the graph and derivatives to understand growth and changing rate."}
    st.write(descriptions[visual])
st.caption("Symbolic derivatives are exact when SymPy can express them. Detected critical points and inflection points may be numerical approximations; zeros are not automatically classified as extrema or inflection points.")
