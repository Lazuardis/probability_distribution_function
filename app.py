import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binom, expon, norm, poisson, triang, weibull_min


st.set_page_config(
    page_title="Sandbox: Probability Distribution",
    page_icon=":bar_chart:",
    layout="wide",
)


PRIMARY = "#2563eb"
SECONDARY = "#f97316"
FILL = "rgba(37, 99, 235, 0.18)"


def format_number(value, decimals=2, prefix="", suffix=""):
    if isinstance(value, (int, np.integer)):
        return f"{prefix}{value:,}{suffix}"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def stats_table(rows):
    st.dataframe(
        pd.DataFrame(rows, columns=["Metric", "Value"]),
        hide_index=True,
        width="stretch",
    )


def make_discrete_chart(x, pmf, cdf, title, x_label, y_label, show_cdf):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=pmf,
            name=y_label,
            marker_color=PRIMARY,
            hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
        )
    )

    if show_cdf:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cdf,
                name="Cumulative probability",
                mode="lines+markers",
                yaxis="y2",
                line=dict(color=SECONDARY, width=3),
                hovertemplate=f"{x_label}: %{{x}}<br>CDF: %{{y:.4f}}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="Cumulative probability",
                overlaying="y",
                side="right",
                range=[0, 1.02],
                tickformat=".0%",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        bargap=0.18,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        height=470,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def make_continuous_chart(x, pdf, cdf, title, x_label, show_cdf, samples=None):
    fig = go.Figure()

    if samples is not None:
        fig.add_trace(
            go.Histogram(
                x=samples,
                histnorm="probability density",
                name="Simulated histogram",
                marker_color="rgba(148, 163, 184, 0.35)",
                opacity=0.65,
                nbinsx=30,
                hovertemplate=f"{x_label}: %{{x:.2f}}<br>Density: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            name="Probability density",
            mode="lines",
            fill="tozeroy",
            fillcolor=FILL,
            line=dict(color=PRIMARY, width=3),
            hovertemplate=f"{x_label}: %{{x:.2f}}<br>PDF: %{{y:.4f}}<extra></extra>",
        )
    )

    if show_cdf:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cdf,
                name="Cumulative probability",
                mode="lines",
                yaxis="y2",
                line=dict(color=SECONDARY, width=3, dash="dot"),
                hovertemplate=f"{x_label}: %{{x:.2f}}<br>CDF: %{{y:.4f}}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="Cumulative probability",
                overlaying="y",
                side="right",
                range=[0, 1.02],
                tickformat=".0%",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Probability density",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
        height=470,
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def render_binomial():
    st.subheader("Binomial PMF")
    st.write("Scenario: estimate how many sales calls convert in one campaign batch.")
    st.write(
        "Use this when every call has the same conversion chance and the team wants to understand "
        "how likely different conversion counts are before staffing or quota decisions."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        n = st.slider("Number of calls", min_value=1, max_value=200, value=40, step=1)
        p = st.slider(
            "Probability of conversion",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
            format="%.2f",
        )
        show_cdf = st.toggle("Show CDF", value=False, key="binomial_cdf")

        mean = n * p
        variance = n * p * (1 - p)
        stats_table(
            [
                ["Formula", "P(X = k) = C(n, k) p^k (1-p)^(n-k)"],
                ["Expected conversions", format_number(mean)],
                ["Variance", format_number(variance)],
                ["Standard deviation", format_number(math.sqrt(variance))],
            ]
        )

    x = np.arange(0, n + 1)
    pmf = binom.pmf(x, n, p)
    cdf = binom.cdf(x, n, p)

    with chart_col:
        st.plotly_chart(
            make_discrete_chart(
                x,
                pmf,
                cdf,
                "Distribution of converted customers per batch",
                "Converted customers",
                "Probability mass",
                show_cdf,
            ),
            width="stretch",
        )


def render_poisson():
    st.subheader("Poisson PMF")
    st.write("Scenario: estimate how many support tickets arrive during a typical operating hour.")
    st.write(
        "Use this for demand planning when events happen independently over time, such as tickets, "
        "walk-ins, calls, or orders arriving at an average hourly rate."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        lam = st.slider(
            "Average tickets per hour",
            min_value=0.1,
            max_value=40.0,
            value=7.5,
            step=0.1,
        )
        show_cdf = st.toggle("Show CDF", value=False, key="poisson_cdf")

        stats_table(
            [
                ["Formula", "P(X = k) = e^-lambda lambda^k / k!"],
                ["Expected tickets", format_number(lam)],
                ["Variance", format_number(lam)],
                ["Standard deviation", format_number(math.sqrt(lam))],
            ]
        )

    upper = max(12, int(poisson.ppf(0.999, lam)) + 2)
    x = np.arange(0, upper + 1)
    pmf = poisson.pmf(x, lam)
    cdf = poisson.cdf(x, lam)

    with chart_col:
        st.plotly_chart(
            make_discrete_chart(
                x,
                pmf,
                cdf,
                "Distribution of support tickets per hour",
                "Tickets per hour",
                "Probability mass",
                show_cdf,
            ),
            width="stretch",
        )


def render_normal():
    st.subheader("Normal/Gaussian PDF")
    st.write("Scenario: model monthly revenue around a forecast with recurring business volatility.")
    st.write(
        "Use this when revenue tends to fluctuate around a central forecast because many small, "
        "independent business factors push results above or below plan."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        mu = st.slider(
            "Expected monthly revenue",
            min_value=10_000,
            max_value=500_000,
            value=120_000,
            step=5_000,
            format="$%d",
        )
        sigma = st.slider(
            "Revenue volatility",
            min_value=1_000,
            max_value=150_000,
            value=25_000,
            step=1_000,
            format="$%d",
        )
        show_cdf = st.toggle("Show CDF", value=False, key="normal_cdf")
        show_hist = st.toggle(
            "Show simulated histogram", value=True, key="normal_hist"
        )

        stats_table(
            [
                ["Formula", "f(x) = (1 / sigma sqrt(2pi)) e^(-(x-mu)^2 / 2sigma^2)"],
                ["Expected revenue", format_number(mu, 0, "$")],
                ["Variance", format_number(sigma**2, 0, "$")],
                ["Standard deviation", format_number(sigma, 0, "$")],
            ]
        )

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 450)
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)
    samples = np.random.default_rng(42).normal(mu, sigma, 350) if show_hist else None

    with chart_col:
        st.plotly_chart(
            make_continuous_chart(
                x,
                pdf,
                cdf,
                "Distribution of monthly revenue",
                "Monthly revenue",
                show_cdf,
                samples,
            ),
            width="stretch",
        )


def render_exponential():
    st.subheader("Exponential PDF")
    st.write("Scenario: model the waiting time between online customer purchases.")
    st.write(
        "Use this when the business cares about the time gap until the next purchase, signup, or "
        "transaction after one has just occurred."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        lam = st.slider(
            "Average purchase arrival rate per hour",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1,
        )
        show_cdf = st.toggle("Show CDF", value=False, key="exponential_cdf")
        show_hist = st.toggle(
            "Show simulated histogram", value=True, key="exponential_hist"
        )

        scale = 1 / lam
        stats_table(
            [
                ["Formula", "f(x) = lambda e^(-lambda x), x >= 0"],
                ["Expected time between purchases", f"{scale:.2f} hours"],
                ["Variance", f"{scale**2:.2f} hours^2"],
                ["Standard deviation", f"{scale:.2f} hours"],
            ]
        )

    x = np.linspace(0, expon.ppf(0.995, scale=scale), 450)
    pdf = expon.pdf(x, scale=scale)
    cdf = expon.cdf(x, scale=scale)
    samples = np.random.default_rng(42).exponential(scale, 350) if show_hist else None

    with chart_col:
        st.plotly_chart(
            make_continuous_chart(
                x,
                pdf,
                cdf,
                "Distribution of time between online purchases",
                "Time between purchases in hours",
                show_cdf,
                samples,
            ),
            width="stretch",
        )


def render_triangular():
    st.subheader("Triangular PDF")
    st.write("Scenario: model project delivery time from optimistic, most likely, and pessimistic estimates.")
    st.write(
        "Use this for early planning when the team has expert estimates but not enough historical "
        "data for a richer delivery-time model."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        optimistic = st.slider("Optimistic days", 1, 60, 12, 1)
        pessimistic = st.slider(
            "Pessimistic days",
            min_value=optimistic + 2,
            max_value=180,
            value=max(optimistic + 20, 45),
            step=1,
        )
        most_likely = st.slider(
            "Most likely days",
            min_value=optimistic + 1,
            max_value=pessimistic - 1,
            value=min(max(optimistic + 10, 25), pessimistic - 1),
            step=1,
        )
        show_cdf = st.toggle("Show CDF", value=False, key="triangular_cdf")
        show_hist = st.toggle(
            "Show simulated histogram", value=True, key="triangular_hist"
        )

        loc = optimistic
        scale = pessimistic - optimistic
        c = (most_likely - optimistic) / scale
        mean = (optimistic + most_likely + pessimistic) / 3
        variance = (
            optimistic**2
            + most_likely**2
            + pessimistic**2
            - optimistic * most_likely
            - optimistic * pessimistic
            - most_likely * pessimistic
        ) / 18
        stats_table(
            [
                ["Formula", "Piecewise triangular density over [a, b] with mode c"],
                ["Expected delivery time", f"{mean:.2f} days"],
                ["Variance", f"{variance:.2f} days^2"],
                ["Standard deviation", f"{math.sqrt(variance):.2f} days"],
            ]
        )

    x = np.linspace(optimistic, pessimistic, 450)
    pdf = triang.pdf(x, c, loc=loc, scale=scale)
    cdf = triang.cdf(x, c, loc=loc, scale=scale)
    samples = (
        np.random.default_rng(42).triangular(
            optimistic, most_likely, pessimistic, 350
        )
        if show_hist
        else None
    )

    with chart_col:
        st.plotly_chart(
            make_continuous_chart(
                x,
                pdf,
                cdf,
                "Distribution of project delivery time",
                "Delivery time in days",
                show_cdf,
                samples,
            ),
            width="stretch",
        )


def render_weibull():
    st.subheader("Weibull PDF")
    st.write("Scenario: model time until equipment or system component failure.")
    st.write(
        "Use this for reliability and maintenance planning, especially when failure risk changes "
        "as equipment ages or usage accumulates."
    )

    control_col, chart_col = st.columns([0.92, 1.58], gap="large")
    with control_col:
        shape = st.slider("Shape", min_value=0.2, max_value=6.0, value=1.8, step=0.1)
        scale = st.slider(
            "Scale / characteristic life",
            min_value=1.0,
            max_value=240.0,
            value=72.0,
            step=1.0,
        )
        show_cdf = st.toggle("Show CDF", value=False, key="weibull_cdf")
        show_hist = st.toggle(
            "Show simulated histogram", value=True, key="weibull_hist"
        )

        mean = scale * math.gamma(1 + 1 / shape)
        variance = scale**2 * (
            math.gamma(1 + 2 / shape) - math.gamma(1 + 1 / shape) ** 2
        )
        stats_table(
            [
                ["Formula", "f(x) = (k/lambda)(x/lambda)^(k-1)e^(-(x/lambda)^k)"],
                ["Expected time until failure", f"{mean:.2f} hours"],
                ["Variance", f"{variance:.2f} hours^2"],
                ["Standard deviation", f"{math.sqrt(variance):.2f} hours"],
            ]
        )

    x_max = weibull_min.ppf(0.995, shape, scale=scale)
    x = np.linspace(0, x_max, 450)
    pdf = weibull_min.pdf(x, shape, scale=scale)
    cdf = weibull_min.cdf(x, shape, scale=scale)
    samples = scale * np.random.default_rng(42).weibull(shape, 350) if show_hist else None

    with chart_col:
        st.plotly_chart(
            make_continuous_chart(
                x,
                pdf,
                cdf,
                "Distribution of equipment or system life",
                "Time until failure in hours",
                show_cdf,
                samples,
            ),
            width="stretch",
        )


st.title("Sandbox: Probability Distribution")
st.caption(
    "Adjust business assumptions and watch the probability mass or density curve update automatically."
)

discrete_tab, continuous_tab = st.tabs(["Discrete Distributions", "Continuous Distributions"])

with discrete_tab:
    selected = st.selectbox(
        "Choose a discrete distribution",
        ["Binomial: sales-call conversions", "Poisson: support tickets"],
    )
    if selected.startswith("Binomial"):
        render_binomial()
    else:
        render_poisson()

with continuous_tab:
    selected = st.selectbox(
        "Choose a continuous distribution",
        [
            "Normal/Gaussian: monthly revenue",
            "Exponential: purchase timing",
            "Triangular: project delivery",
            "Weibull: component failure",
        ],
    )
    if selected.startswith("Normal"):
        render_normal()
    elif selected.startswith("Exponential"):
        render_exponential()
    elif selected.startswith("Triangular"):
        render_triangular()
    else:
        render_weibull()
