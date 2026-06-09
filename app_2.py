import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import expon, kstest, norm, triang, weibull_min


st.set_page_config(
    page_title="Queue Data Playground",
    page_icon=":bar_chart:",
    layout="wide",
)


PRIMARY = "#2563eb"
SECONDARY = "#f97316"
REQUIRED_COLUMNS = ["interarrival_time", "service_time"]


def format_number(value, decimals=2, prefix="", suffix=""):
    if isinstance(value, (int, np.integer)):
        return f"{prefix}{value:,}{suffix}"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def result_table(rows):
    st.table(pd.DataFrame(rows, columns=["Item", "Value"]))


def parse_numeric_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    cleaned_numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.fillna(cleaned_numeric)


def load_queue_data(uploaded_file):
    file_name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        last_error = None
        for encoding in ["utf-8-sig", "cp1252", "latin1"]:
            try:
                return pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding=encoding)
            except UnicodeDecodeError as error:
                last_error = error
            except pd.errors.ParserError as error:
                last_error = error
        raise ValueError(f"Could not read CSV file. Last parser error: {last_error}")

    if file_name.endswith(".xlsx"):
        return pd.read_excel(BytesIO(raw), engine="openpyxl")

    raise ValueError("Please upload a .csv or .xlsx file.")


def validate_queue_data(df):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}"

    clean_df = df[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        clean_df[column] = parse_numeric_series(clean_df[column])

    if clean_df[REQUIRED_COLUMNS].isna().any().any():
        return None, "Both required columns must contain numeric values only."
    if (clean_df[REQUIRED_COLUMNS] < 0).any().any():
        return None, "Time values must be non-negative."
    if clean_df.empty:
        return None, "The uploaded file does not contain any rows."

    return clean_df, None


def interpret_p_value(p_value, alpha=0.05):
    if p_value < alpha:
        return (
            f"At alpha = {alpha:.2f}, this small p-value suggests the selected data does not "
            "fit the hypothesized distribution well."
        )

    return (
        f"At alpha = {alpha:.2f}, there is not enough evidence to reject the hypothesized "
        "distribution for this selected data."
    )


def run_continuous_gof(values, distribution):
    if len(values) < 5:
        raise ValueError("At least 5 observations are recommended for a Kolmogorov-Smirnov test.")
    if values.nunique() < 2:
        raise ValueError("The selected variable needs at least two different numeric values.")

    data = values.astype(float).to_numpy()

    if distribution == "Normal":
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        if sigma <= 0:
            raise ValueError("Normal distribution requires positive standard deviation.")
        statistic, p_value = kstest(data, "norm", args=(mu, sigma))
        parameters = [["mu", format_number(mu)], ["sigma", format_number(sigma)]]
        raw_parameters = {"mu": mu, "sigma": sigma}
    elif distribution == "Exponential":
        scale = float(np.mean(data))
        if scale <= 0:
            raise ValueError("Exponential distribution requires positive mean.")
        statistic, p_value = kstest(data, "expon", args=(0, scale))
        parameters = [["loc", 0], ["scale", format_number(scale)]]
        raw_parameters = {"loc": 0.0, "scale": scale}
    elif distribution == "Triangular":
        c, loc, scale = triang.fit(data)
        if scale <= 0:
            raise ValueError("Triangular fit produced a non-positive scale.")
        statistic, p_value = kstest(data, "triang", args=(c, loc, scale))
        parameters = [
            ["c", format_number(c, 4)],
            ["loc", format_number(loc)],
            ["scale", format_number(scale)],
        ]
        raw_parameters = {"c": c, "loc": loc, "scale": scale}
    else:
        shape, loc, scale = weibull_min.fit(data, floc=0)
        if shape <= 0 or scale <= 0:
            raise ValueError("Weibull fit produced non-positive shape or scale.")
        statistic, p_value = kstest(data, "weibull_min", args=(shape, loc, scale))
        parameters = [
            ["shape", format_number(shape, 4)],
            ["loc", format_number(loc)],
            ["scale", format_number(scale)],
        ]
        raw_parameters = {"shape": shape, "loc": loc, "scale": scale}

    return {
        "test": "Kolmogorov-Smirnov goodness-of-fit",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "parameters": parameters,
        "raw_parameters": raw_parameters,
        "sample_size": len(data),
    }


def make_gof_histogram(values, variable_label, distribution=None, result=None):
    clean_values = pd.Series(values).dropna()
    fig = go.Figure()
    bin_count = min(max(clean_values.nunique(), 8), 30)
    _, edges = np.histogram(clean_values, bins=bin_count)
    bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0

    fig.add_trace(
        go.Histogram(
            x=clean_values,
            name="Observed data",
            marker_color=PRIMARY,
            opacity=0.78,
            nbinsx=bin_count,
            hovertemplate=f"{variable_label}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
        )
    )

    if result is not None:
        params = result["raw_parameters"]
        sample_size = result["sample_size"]
        x = np.linspace(float(clean_values.min()), float(clean_values.max()), 400)
        if distribution == "Normal":
            y = norm.pdf(x, params["mu"], params["sigma"])
        elif distribution == "Exponential":
            y = expon.pdf(x, loc=params["loc"], scale=params["scale"])
        elif distribution == "Triangular":
            y = triang.pdf(x, params["c"], loc=params["loc"], scale=params["scale"])
        else:
            y = weibull_min.pdf(
                x,
                params["shape"],
                loc=params["loc"],
                scale=params["scale"],
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y * sample_size * bin_width,
                name=f"Fitted {distribution}",
                mode="lines",
                line=dict(color=SECONDARY, width=3),
                hovertemplate=f"{variable_label}: %{{x:.2f}}<br>Expected count: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Observed data for {variable_label}",
        xaxis_title=f"{variable_label} (minutes)",
        yaxis_title="Count",
        bargap=0.08,
        margin=dict(l=20, r=20, t=70, b=20),
        height=430,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def calculate_mm_c_queue(lambda_rate, mu_rate, servers):
    if lambda_rate <= 0:
        raise ValueError("Arrival rate must be greater than 0.")
    if mu_rate <= 0:
        raise ValueError("Service rate must be greater than 0.")
    if servers < 1:
        raise ValueError("Number of servers must be at least 1.")

    utilization = lambda_rate / (servers * mu_rate)
    if utilization >= 1:
        return {
            "stable": False,
            "utilization": utilization,
        }

    traffic_intensity = lambda_rate / mu_rate
    base_sum = sum(
        (traffic_intensity**n) / math.factorial(n)
        for n in range(servers)
    )
    tail_term = (
        traffic_intensity**servers
        / (math.factorial(servers) * (1 - utilization))
    )
    p0 = 1 / (base_sum + tail_term)
    average_number_waiting = (
        p0
        * (traffic_intensity**servers)
        * utilization
        / (math.factorial(servers) * ((1 - utilization) ** 2))
    )
    average_waiting_time_queue = average_number_waiting / lambda_rate
    average_time_system = average_waiting_time_queue + (1 / mu_rate)
    average_number_system = lambda_rate * average_time_system

    return {
        "stable": True,
        "utilization": utilization,
        "p0": p0,
        "average_number_waiting": average_number_waiting,
        "average_number_system": average_number_system,
        "average_waiting_time_queue": average_waiting_time_queue,
        "average_time_system": average_time_system,
    }


def format_queue_time(hours):
    return f"{hours:.4f} hours ({hours * 60:.2f} minutes)"


def render_goodness_of_fit_tab():
    st.subheader("Goodness-of-Fit for Queue Data")
    st.write(
        "Upload a CSV or Excel file with `interarrival_time` and `service_time`, then test "
        "which continuous distribution fits each time variable."
    )

    uploaded_file = st.file_uploader(
        "Upload queue time data",
        type=["csv", "xlsx"],
        help="Required columns: interarrival_time and service_time.",
    )

    if uploaded_file is None:
        st.write("Expected columns: `interarrival_time`, `service_time`.")
        return

    try:
        raw_df = load_queue_data(uploaded_file)
    except Exception as error:
        st.error(f"Could not load file: {error}")
        return

    queue_df, validation_error = validate_queue_data(raw_df)
    if validation_error:
        st.warning(validation_error)
        st.dataframe(raw_df.head(8), hide_index=True, width="stretch")
        return

    control_col, preview_col = st.columns([0.9, 1.5], gap="large")
    with control_col:
        variable_label = st.selectbox("Choose variable", REQUIRED_COLUMNS)
        distribution = st.selectbox(
            "Hypothesized distribution",
            ["Normal", "Exponential", "Triangular", "Weibull"],
        )
        compute = st.button("Compute", type="primary")
        st.caption(f"Valid rows loaded: {len(queue_df)}. Time unit: minutes.")

    values = queue_df[variable_label].dropna()

    with preview_col:
        histogram_slot = st.empty()
        histogram_slot.plotly_chart(make_gof_histogram(values, variable_label), width="stretch")

    if not compute:
        st.write("Click **Compute** to fit parameters and run the goodness-of-fit test.")
        return

    try:
        result = run_continuous_gof(values, distribution)
    except ValueError as error:
        st.warning(str(error))
        return

    histogram_slot.plotly_chart(
        make_gof_histogram(values, variable_label, distribution, result),
        width="stretch",
    )

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Test statistic", f"{result['statistic']:.4f}")
    metric_col_2.metric("p-value", f"{result['p_value']:.4f}")
    metric_col_3.metric("Sample size", result["sample_size"])

    st.write(interpret_p_value(result["p_value"]))
    st.caption("The p-value is approximate because parameters are estimated from the selected data.")

    st.write("Estimated parameters")
    result_table(result["parameters"])


def render_queueing_theory_tab():
    st.subheader("Queueing Theory")
    st.write(
        "Calculate steady-state M/M/c queue metrics using the arrival rate, service rate per server, "
        "and number of parallel servers."
    )

    control_col, result_col = st.columns([0.9, 1.5], gap="large")
    with control_col:
        lambda_rate = st.number_input(
            "Arrival rate, lambda (customers per hour)",
            min_value=0.01,
            value=30.0,
            step=1.0,
            format="%.2f",
        )
        mu_rate = st.number_input(
            "Service rate, mu (customers served per server per hour)",
            min_value=0.01,
            value=20.0,
            step=1.0,
            format="%.2f",
        )
        servers = st.number_input(
            "Number of servers, c",
            min_value=1,
            value=2,
            step=1,
        )

        result = calculate_mm_c_queue(lambda_rate, mu_rate, int(servers))
        st.metric("Utilization", f"{result['utilization']:.2%}")

        if result["stable"]:
            st.write(
                "The system is stable because utilization is below 100%, so steady-state queue "
                "metrics can be calculated."
            )
        else:
            st.warning(
                "The system is unstable because utilization is at least 100%. In steady state, "
                "the queue would grow over time, so waiting-time and queue-length metrics are not computed."
            )

    with result_col:
        if result["stable"]:
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric(
                "Average Number Waiting (Lq)",
                f"{result['average_number_waiting']:.2f}",
            )
            metric_col_2.metric(
                "Average Number in System (L)",
                f"{result['average_number_system']:.2f}",
            )

            metric_col_3, metric_col_4 = st.columns(2)
            metric_col_3.metric(
                "Average Waiting Time in Queue (Wq)",
                format_queue_time(result["average_waiting_time_queue"]),
            )
            metric_col_4.metric(
                "Average Time in System (W)",
                format_queue_time(result["average_time_system"]),
            )

            st.write("Formula reference")
            result_table(
                [
                    ["Model", "M/M/c"],
                    ["Utilization", "rho = lambda / (c * mu)"],
                    ["Probability system is empty", "P0 = [sum((lambda/mu)^n / n!) + tail term]^-1"],
                    ["Average Number Waiting", "Lq = P0(lambda/mu)^c rho / (c!(1-rho)^2)"],
                    ["Average Waiting Time in Queue", "Wq = Lq / lambda"],
                    ["Average Time in System", "W = Wq + 1 / mu"],
                    ["Average Number in System", "L = lambda * W"],
                ]
            )
            st.caption(f"Probability the system is empty, P0 = {result['p0']:.4f}")
        else:
            st.write("Formula reference")
            result_table(
                [
                    ["Model", "M/M/c"],
                    ["Utilization", "rho = lambda / (c * mu)"],
                    ["Stability condition", "rho < 1"],
                ]
            )


st.title("Queue Data Playground")
st.caption("Upload queue time data, fit distributions, and calculate M/M/c queue metrics.")

gof_tab, queue_tab = st.tabs(["Goodness-of-Fit", "Queueing Theory"])

with gof_tab:
    render_goodness_of_fit_tab()

with queue_tab:
    render_queueing_theory_tab()
