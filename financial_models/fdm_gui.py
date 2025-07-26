import asyncio
import base64
import io
import pandas as pd
import logging
import random
from datetime import datetime, timedelta

import httpx
import matplotlib.pyplot as plt
import numpy as np
from nicegui import page, ui

from pyDOE import lhs

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fdm_gui")

# === STATE ===
latest_vector = {"S_grid": [], "prices": [], "final_price": 0.0}
bs_price_label = None
# === VALID METHODS BY OPTION STYLE ===
european_methods = [
    "explicit",
    "implicit",
    "crank",
    "compact",
]
american_methods = ["psor"]
# === PAGE CONFIG ===
page.title = "FDM Calculator"
ui.dark_mode().enable()
test_results_df = pd.DataFrame()
latest_csv_filename = ""

# === MAIN UI ===
with ui.row().classes("w-full justify-center"):
    with ui.column().classes(
        "max-w-6xl w-full bg-gray-900 text-white p-4 gap-2 items-center"
    ):
        ui.markdown("## 🎓 Finite Difference Method Option Pricing Tool").classes(
            "text-2xl mb-4"
        )
        ui.markdown("### Method & Market Parameters").classes("text-xl")

        with ui.row().classes("gap-4"):
            # Method dropdown, defaulting to European methods
            method = ui.select(
                european_methods, label="Finite Difference Method", value="explicit"
            ).classes("w-56 border border-gray-700")

            # Option type dropdown
            option_type = ui.select(
                ["European", "American"], label="Option Style", value="European"
            ).classes("w-56 border border-gray-700")

            # Callback: change method options based on selected option type
            def update_method_options():
                if option_type.value == "American":
                    method.options = american_methods
                    method.value = american_methods[0]
                else:
                    method.options = european_methods
                    method.value = european_methods[0]

            # Trigger callback initially and on update
            option_type.on("update:model-value", lambda e: update_method_options())
            update_method_options()  # Call once at init

            vol_source = ui.select(
                ["User-defined", "Implied"],
                label="Volatility Source",
                value="User-defined",
            ).classes("w-56 border border-gray-700")
            grid_scheme = ui.select(
                ["uniform", "adaptive"], label="Grid Scheme", value="uniform"
            ).classes("w-56 border border-gray-700")

        with ui.row().classes("gap-4"):
            N = ui.number("N (Grid steps)", value=10).classes(
                "w-56 border border-gray-700"
            )
            M = ui.number("M (Time steps)", value=10).classes(
                "w-56 border border-gray-700"
            )
            Smax = ui.number("Smax", value=100).classes("w-56 border border-gray-700")
            K = ui.number("K (Strike)", value=50).classes("w-56 border border-gray-700")
        ## Market parameters
        with ui.row().classes("gap-4"):
            r = ui.number("r (Interest rate)", value=0.05).classes(
                "w-56 border border-gray-700"
            )
            sigma = ui.number("σ (Volatility)", value=0.2).classes(
                "w-56 border border-gray-700"
            )
            omega = ui.number("ω (Relaxation)", value=1.2).classes(
                "w-56 border border-gray-700"
            )
            S0 = ui.number("S₀ (Spot Price)", value=50.0).classes(
                "w-56 border border-gray-700"
            )
        # Optional parameters for specific methods
        with ui.row().classes("gap-4"):
            datetime_start = (
                ui.input("Start Datetime")
                .props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"')
                .classes("w-56 border border-gray-700")
            )
            datetime_end = (
                ui.input("End Datetime")
                .props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"')
                .classes("w-56 border border-gray-700")
            )
            is_call = ui.toggle({True: "Call", False: "Put"}, value=True).classes(
                "w-56 border border-gray-700"
            )
            cfl_toggle = ui.toggle({"off": "off", "on": "on"}, value="off").classes(
                "w-56 border border-gray-700"
            )
        # Optional parameters for specific methods
        with ui.row().classes("gap-4"):
            tol = ui.number("Tolerance", value=1e-6).classes(
                "w-56 border border-gray-700"
            )
            beta = ui.number("β (Fractional time)", value=0.8).classes(
                "w-56 border border-gray-700"
            )
            dx = ui.number("dx (Compact dx)", value=1.0).classes(
                "w-56 border border-gray-700"
            )
            max_iter = (
                ui.number("Max Iter", value=10000)
                .props("step=100")
                .classes("w-56 border border-gray-700")
            )

            def update_method_param_fields():
                def mark_inactive(field):
                    field.enabled = False
                    field.classes(
                        remove="w-56 border-gray-700", add="w-56 border-red-500"
                    )

                def mark_active(field):
                    field.enabled = True
                    field.classes(
                        remove="w-56 border-red-500", add="w-56 border-gray-700"
                    )

                # PSOR fields
                if method.value == "psor":
                    mark_active(omega)
                    mark_active(tol)
                    mark_active(max_iter)

                    mark_inactive(beta)
                    mark_inactive(dx)
                elif method.value == "fractional":
                    mark_active(beta)

                    mark_inactive(omega)
                    mark_inactive(tol)
                    mark_inactive(max_iter)
                    mark_inactive(dx)
                elif method.value == "compact":
                    mark_active(dx)
                    mark_active(max_iter)

                    mark_inactive(omega)
                    mark_inactive(tol)
                    mark_inactive(beta)
                else:
                    # Default: all optional inactive
                    mark_inactive(omega)
                    mark_inactive(tol)
                    mark_inactive(max_iter)
                    mark_inactive(beta)
                    mark_inactive(dx)

            option_type.on(
                "update:model-value",
                lambda e: update_method_options()
                or update_method_param_fields()
                or reset_outputs(),
            )
            method.on(
                "update:model-value",
                lambda e: update_method_param_fields() or reset_outputs(),
            )

            update_method_options()
            update_method_param_fields()

            def reset_outputs():
                final_price.text = "Final Price:"
                comparison_label.text = "Comparison Result:"
                if bs_price_label:
                    bs_price_label.text = ""

        with ui.row().classes("gap-4"):
            final_price = ui.label("Final Price:").classes(
                "text-green-400 text-xl border border-green-400 p-2 rounded w-full"
            )
        with ui.row().classes("gap-4"):
            comparison_label = ui.label("Comparison Result:").classes(
                "text-yellow-400 text-xl border border-yellow-400 p-2 rounded w-full"
            )

        with ui.row().classes("gap-4 mt-4"):
            ui.button(
                "Run FDM Solver",
                on_click=lambda: asyncio.create_task(compute_fdm()),
                color="blue",
            )
            compare_button = ui.button(
                "Compare with Binomial",
                on_click=lambda: asyncio.create_task(compare_with_binomial()),
                color="orange",
            )
            compare_button.bind_visibility_from(
                option_type, "value", lambda v: v == "American"
            )

            ui.button(
                "Show 2D Plot",
                on_click=lambda: asyncio.create_task(show_vector_plot("2d")),
                color="green",
            )

            ui.button(
                "Show 3D Line",
                on_click=lambda: asyncio.create_task(show_vector_plot("3d")),
                color="cyan",
            )
            ui.button(
                "Show Surface",
                on_click=lambda: asyncio.create_task(show_surface_plot()),
                color="purple",
            )
            ui.button(
                "Generate Random Test Cases",
                on_click=lambda: asyncio.create_task(generate_random_test_cases()),
                color="red",
            )

# === TABLE POPUP ===
with ui.dialog() as popup_table, ui.card().classes("bg-gray-900 p-4 w-[600px]"):
    ui.label("📋 Output Table").classes("text-white text-lg mb-2")
    with ui.element("div").classes("h-[300px] overflow-y-auto"):
        output_table = ui.table(
            columns=[
                {"name": "index", "label": "Index", "field": "index"},
                {"name": "value", "label": "Value", "field": "value"},
            ],
            rows=[],
        ).classes("w-full text-white")
    ui.button("Close", on_click=popup_table.close).classes("mt-4")



# === 2D/3D POPUP ===
with ui.dialog() as popup3d, ui.card().classes("bg-gray-900 p-4 w-[1000px] h-[700px]"):
    plot3d_image = ui.image().classes("w-full h-[600px] object-contain")
    ui.button("Close", on_click=popup3d.close).classes("mt-4")


with ui.dialog() as popup_test_table, ui.card().classes(
    "bg-gray-900 p-4 w-full max-w-[95vw] max-h-[95vh] overflow-hidden"
):
    ui.label("🧪 Random Test Case Results").classes("text-white text-2xl mb-4")

    with ui.row().classes("w-full items-center justify-between"):
        method_filter = ui.select(
            options=["All"] + european_methods + american_methods,
            value="All",
            label="Filter by Method"
        ).classes("w-64 text-black")

        sort_delta = ui.toggle({"Δ ↑": "Δ ↑", "Δ ↓": "Δ ↓"}, value="Δ ↑")
        sort_delta.on("update:model-value", lambda e: update_filtered_table())
        method_filter.on("update:model-value", lambda e: update_filtered_table())

    with ui.element("div").classes("overflow-x-auto overflow-y-auto w-full"):
        test_case_table = ui.table(columns=[], rows=[]).classes("min-w-full text-white text-sm")

    with ui.row().classes("justify-end w-full mt-4 gap-4"):
        ui.button("Download CSV", on_click=lambda: ui.download(latest_csv_filename)).classes("bg-blue-700")
        ui.button("Close", on_click=popup_test_table.close).classes("bg-gray-700")


async def compute_fdm():
    """
    Compute the option price using the selected FDM method and parameters.
    This function handles both European and American options, and compares results with Black-Scholes or Binomial methods.
    """
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)

        async with httpx.AsyncClient() as client:
            # Run FDM solver
            resp_fdm = await client.post(
                f"http://localhost:8000/fdm/{method.value}", json=params
            )
            resp_fdm.raise_for_status()
            data = resp_fdm.json()
            final = float(data.get("final_price", 0.0))
            print(f"The final is: {final:}")
            # Clamp absurdly high or negative values (optional safeguard)
            if abs(final) > 1e4:
                final_price.text = f"Final Price: {final:.4e}"  # scientific
            else:
                final_price.text = f"Final Price: {final:.4f}"  # fixed-point
            if option_type.value == "European":
                # Also run Black-Scholes for comparison
                bs_params = {
                    "S": S0.value,
                    "K": K.value,
                    "T": T_calc,
                    "r": r.value,
                    "sigma": sigma.value,
                    "is_call": is_call.value,
                }

                resp_bs = await client.post(
                    "http://localhost:8000/fdm/black_scholes", json=bs_params
                )
                resp_bs.raise_for_status()
                bs_data = resp_bs.json()
                bs_price = bs_data.get("price", 0.0)
                # Optional safeguard: clamp absurd values (e.g., explosion due to instability)

                diff = abs(final - bs_price)

                def smart_format(value: float) -> str:
                    if not isinstance(value, float):
                        return "NaN"
                    if abs(value) > 1e6 or abs(value) < 1e-3:
                        return f"{value:.2e}"
                    return f"{value:.4f}"

                comparison_label.text = (
                    f"{method.value.capitalize()} vs Black-Scholes:\n"
                    f"FDM: {smart_format(final)} | BS: {smart_format(bs_price)} | Δ: {smart_format(diff)}"
                )

            else:
                # Compare PSOR (American) vs Implicit (European)
                european_params = params.copy()
                european_params["option_style"] = "European"
                european_params["method"] = "implicit"

                resp_implicit = await client.post(
                    "http://localhost:8000/fdm/implicit", json=european_params
                )
                resp_implicit.raise_for_status()
                implicit_result = resp_implicit.json().get("final_price", 0.0)

                diff = abs(final - implicit_result)
                comparison_label.text = f"PSOR vs Implicit:\nPSOR: {final:.4f} | Implicit: {implicit_result:.4f} | Δ: {diff:.4f}"

    except Exception as e:
        logger.exception("FDM Error")
        ui.notify(f"❌ FDM Error: {str(e)}", type="negative")


def safe_notify(message: str, type: str = "info", timeout: int = 5000) -> None:
    """
    Safely notify the user with a message.
    Uses NiceGUI's task system to ensure UI updates are handled correctly.
    """
    with ui.tasks():
        ui.notify(message, type=type, timeout=timeout)


async def compare_with_binomial() -> None:
    """
    Compare PSOR (American) method with Binomial (European) method.
    This function computes the option price using both methods and displays the results.
    """
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)

        async with httpx.AsyncClient() as client:
            resp_psor = await client.post("http://localhost:8000/fdm/psor", json=params)
            resp_psor.raise_for_status()
            result_psor = resp_psor.json().get("final_price", 0.0)

            european_params = params.copy()
            european_params["option_style"] = "European"
            resp_binom = await client.post(
                "http://localhost:8000/fdm/binomial", json=european_params
            )
            result_binom = resp_binom.json().get("final_price", 0.0)

        diff = abs(result_psor - result_binom)
        final_price.text = f"Final Price (PSOR): {result_psor:.4f}"
        comparison_label.text = f"📊 PSOR (American): {result_psor:.4f} vs Binomial (European): {result_binom:.4f} | Δ = {diff:.4f}"

    except Exception as e:
        logger.exception("Binomial Comparison Error")
        safe_notify(f"❌ Comparison with Binomial failed: {str(e)}", type="negative")


# === BLACK-SCHOLES ===
async def compute_bs() -> None:
    """
    Compute the Black-Scholes price using the provided parameters.
    This function retrieves the parameters from the UI and sends them to the server for calculation.
    """
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = {
            "S": S0.value,
            "K": K.value,
            "T": T_calc,
            "r": r.value,
            "sigma": sigma.value,
            "is_call": is_call.value,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://localhost:8000/fdm/black_scholes", json=params
            )
            resp.raise_for_status()
            data = resp.json()
            bs_price_label.text = f"Black-Scholes Price: {data['price']:.4f}"
    except Exception as e:
        logger.exception("BS Error")
        ui.notify(f"❌ Black-Scholes Error: {str(e)}", type="negative")


# === SHOW VECTOR (2D/3D LINE) ===
async def show_vector_plot(plot_type: str) -> None:
    """
    Show a 2D or 3D vector plot of the option prices against asset prices.
    This function retrieves the computed option prices and asset prices from the server
    and generates the plot using Matplotlib.
    """
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:8000/fdm/{method.value}_vector", json=params
            )
            resp.raise_for_status()
            data = resp.json()

        S_grid = data["S_grid"]
        prices = data["prices"]
        latest_vector["S_grid"] = S_grid
        latest_vector["prices"] = prices
        output_table.rows = [
            {"index": i, "value": prices[i]} for i in range(len(prices))
        ]

        fig = plt.figure()
        if plot_type == "2d":
            ax = fig.add_subplot(111)
            ax.plot(S_grid, prices)
            ax.set_title("Option Price vs Asset Price")
            ax.set_xlabel("S")
            ax.set_ylabel("Price")
            ax.grid(True)
        elif plot_type == "3d":
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(S_grid, [0] * len(S_grid), prices)
            ax.set_xlabel("S")
            ax.set_ylabel("Time")
            ax.set_zlabel("Price")
            ax.set_title("Option Price at Maturity")
            ax.view_init(elev=30, azim=-135)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plot3d_image.source = (
            f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
        )
        popup3d.open()
        plt.close(fig)
    except Exception as e:
        logger.exception("Vector Plot Error")
        ui.notify(f"❌ Plot Error: {str(e)}", type="negative")


# === SHOW SURFACE PLOT ===
async def show_surface_plot() -> None:
    """
    Show a 3D surface plot of the option prices against asset prices and time.
    This function retrieves the computed option prices and asset prices from the server
    and generates the surface plot using Matplotlib.
    """
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)
        params = params
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://localhost:8000/fdm/{method.value}_surface", json=params
            )
            resp.raise_for_status()
            data = resp.json()
        S = np.array(data["S_grid"])
        t = np.array(data["t_grid"])
        V = np.array(data["price_surface"])
        S_mesh, t_mesh = np.meshgrid(S, t)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            S_mesh,
            t_mesh,
            V,
            cmap="coolwarm",
            edgecolor="k",
            linewidth=0.3,
            antialiased=True,
        )
        ax.set_xlabel("S")
        ax.set_ylabel("T")
        ax.set_zlabel("Option Price")
        ax.set_title("Option Price Surface")
        ax.view_init(elev=30, azim=-135)
        fig.colorbar(surf, shrink=0.5, aspect=12, label="Price")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plot3d_image.source = (
            f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
        )
        popup3d.open()
        plt.close(fig)
    except Exception as e:
        logger.exception("Surface Plot Error")
        ui.notify(f"❌ Surface plot error: {str(e)}", type="negative")



# === MATURITY ===
def compute_maturity(start_str, end_str) -> float:
    """
    Compute the maturity in years based on the provided start and end dates.
    If the dates are invalid, defaults to 1 year from now.
    """
    try:
        start = datetime.strptime(start_str.strip(), "%Y-%m-%d")
    except:
        start = datetime.now()
    try:
        end = datetime.strptime(end_str.strip(), "%Y-%m-%d")
    except:
        end = start + timedelta(days=365)
    return max((end - start).total_seconds() / (365 * 24 * 60 * 60), 1e-6)




def get_method_explanation(method: str) -> str:
    explanations = {
        "explicit": (
            "**Explicit Method**:\n"
            "- Fast, conditionally stable.\n"
            "- CFL condition applied: (σ²·S²·Δt) / Δx² < 1.\n"
            "- Small Δt and fine spatial grid needed.\n"
        ),
        "implicit": (
            "**Implicit Method**:\n"
            "- Unconditionally stable.\n"
            "- Suitable for stiff PDEs and longer maturity.\n"
            "- Requires solving tridiagonal systems.\n"
        ),
        "crank": (
            "**Crank-Nicolson Method**:\n"
            "- Combines explicit and implicit (trapezoidal rule).\n"
            "- 2nd-order accurate in time and space.\n"
            "- May require Rannacher smoothing for early exercise.\n"
        ),
        "compact": (
            "**Compact Scheme**:\n"
            "- Higher-order accuracy with fewer grid points.\n"
            "- Adds spatial derivative correction terms.\n"
            "- Good for smooth payoff and dense grids.\n"
        ),
        "fractional": (
            "**Time-Fractional Method**:\n"
            "- Captures memory effects in markets.\n"
            "- Uses fractional Caputo derivatives.\n"
            "- Computationally intensive.\n"
        ),
        "exponential": (
            "**Exponential Integral Scheme**:\n"
            "- Effective for exponential discounting.\n"
            "- May use exponential transformation of grid.\n"
            "- Handles long maturities well.\n"
        ),
        "psor": (
            "**PSOR Method (American Options)**:\n"
            "- Handles early exercise constraint.\n"
            "- Uses projected iterative solvers.\n"
            "- Requires careful tuning of ω (relaxation).\n"
        ),
        "All": (
            "**All Methods Summary**:\n"
            "- Explicit: Simple, fast, CFL-sensitive.\n"
            "- Implicit: Robust, solves linear systems.\n"
            "- Crank: Balanced, accurate, slightly costlier.\n"
            "- Compact: High-order, efficient on dense grids.\n"
            "- PSOR: For American options, handles early exercise.\n"
            "- Fractional: Nonlocal memory effects.\n"
            "- Exponential: Good for discount-sensitive long options.\n"
        )
    }
    return explanations.get(method, "No explanation available.")



def update_filtered_table():
    if test_results_df.empty:
        return

    df = test_results_df.copy()

    # ✅ Filter by Method
    if method_filter.value != "All":
        df = df[df["Method"] == method_filter.value]

    # ✅ Sort by Δ
    ascending = sort_delta.value == "asc" or sort_delta.value == "Δ ↑"
    df = df.sort_values("Δ", ascending=ascending)

    # ✅ Update table
    test_case_table.columns = [{"name": c, "label": c, "field": c} for c in df.columns]
    test_case_table.rows = df.to_dict("records")



def sort_results(results: list[dict], order: str) -> list[dict]:
    reverse = order == "Δ ↓"
    return sorted(
        results,
        key=lambda x: float(x["Δ"]) if x["Δ"] is not None else float("inf"),
        reverse=reverse
    )


# === REGIME-BASED MONEINESS + LHS SAMPLING ===
def generate_regime_parameters(sample):
    moneyness_levels = [0.8, 0.9, 1.0, 1.1, 1.2]
    m = random.choice(moneyness_levels)
    sigma = 0.1 + sample[0] * 0.5
    r = 0.01 + sample[1] * 0.09
    T = 0.1 + sample[2] * 1.9
    K = random.uniform(100, 500)
    S0 = K * m
    Smax = 2 * S0
    return S0, K, sigma, r, T, Smax

# === TEST CASE GENERATOR ===
async def generate_random_test_cases():
    global test_results_df, latest_csv_filename
    european_methods = ["explicit", "implicit", "crank", "compact"]
    american_methods = ["psor"]
    all_methods = european_methods + american_methods
    test_results = []
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_csv_filename = f"fdm_test_results_{now}.csv"
    lhs_samples = lhs(3, samples=100)

    for method in all_methods:
        is_american = method in american_methods
        for is_call_val in [True, False]:
            count = 0
            sample_idx = 0
            while count < 100 and sample_idx < len(lhs_samples):
                S0, K, sigma, r, T, Smax = generate_regime_parameters(lhs_samples[sample_idx])
                sample_idx += 1

                p = {
                    "option_style": "American" if is_american else "European",
                    "vol_source": "User-defined",
                    "grid_scheme": "uniform",
                    "S0": S0,
                    "K": K,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                    "is_call": is_call_val,
                    "Smax": Smax,
                    "cfl": False,
                }

                if method == "explicit":
                    N = random.randint(30, 80)
                    M = random.randint(40, 150)
                    dx = Smax / N
                    dt = T / M
                    cfl = (sigma ** 2 * S0 ** 2 * dt) / dx ** 2
                    if cfl > 1:
                        continue
                    p["N"] = N
                    p["M"] = M

                elif method in ["implicit", "crank"]:
                    p["N"] = random.randint(50, 120)
                    p["M"] = random.randint(60, 250)

                elif method == "compact":
                    p["N"] = 100
                    p["M"] = 100
                    p["dx"] = round(random.uniform(0.5, 1.5), 2)
                    p["maxIter"] = random.randint(3000, 8000)

                elif method == "psor":
                    p.update({
                        "omega": round(random.uniform(1.1, 1.8), 2),
                        "maxIter": random.randint(5000, 15000),
                        "tol": random.choice([1e-5, 1e-6]),
                        "is_american": True,
                    })

                    for key in ["option_style", "vol_source", "grid_scheme", "cfl"]:
                        p.pop(key, None)
                try:
                    async with httpx.AsyncClient() as client:
                        fdm_resp = await client.post(f"http://localhost:8000/fdm/{method}", json=p)
                        fdm_resp.raise_for_status()
                        fdm_price = float(fdm_resp.json().get("final_price", 0.0))

                        if not is_american:
                            ref_payload = {
                                "S": S0, "K": K, "T": T, "r": r,
                                "sigma": sigma, "is_call": is_call_val
                            }
                            ref_resp = await client.post("http://localhost:8000/fdm/black_scholes", json=ref_payload)
                            ref_resp.raise_for_status()
                            ref_price = float(ref_resp.json().get("price", 0.0))

                        elif method == "psor":
                            binomial_payload = {
                            "N": random.randint(80, 200),
                            "T": float(T),
                            "K": float(K),
                            "r": float(r),
                            "sigma": float(sigma),
                            "is_call": bool(is_call_val),
                            "is_american": False,
                            "S0": float(S0)
                            }
                            ref_resp = await client.post("http://localhost:8000/fdm/binomial", json=binomial_payload)
                            ref_resp.raise_for_status()
                            ref_price = float(ref_resp.json().get("final_price", 0.0))


                        else:
                            ref_price = None  # ✅ MODIFIED: Skip reference for PSOR

                        delta = abs(fdm_price - ref_price) if ref_price is not None else None  # ✅ MODIFIED: Guarded delta

                        test_results.append({
                            "Test": len(test_results) + 1,
                            "Method": method,
                            "Style": "American" if is_american else "European",
                            "Call": is_call_val,
                            "S0": round(S0, 2),
                            "K": round(K, 2),
                            "T": round(T, 3),
                            "σ": round(sigma, 3),
                            "r": round(r, 4),
                            "FDM": round(fdm_price, 4),
                            "Reference": round(ref_price, 4) if ref_price is not None else None,
                            "Δ": round(delta, 6) if delta is not None else None,
                        })
                        count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Skipped ({method}): {e}")
                    continue

    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(latest_csv_filename, index=False)
    update_filtered_table()
    popup_test_table.open()
    ui.notify(f"✅ Generated {len(test_results)} test cases", type="positive")

# === PARAM BUILD ===
def build_params(T_calc):
    p = {
        "N": N.value,
        "M": M.value,
        "Smax": Smax.value,
        "T": T_calc,
        "K": K.value,
        "S0": S0.value,
        "r": r.value,
        "sigma": sigma.value,
        "is_call": is_call.value,
        "option_style": option_type.value,
        "vol_source": vol_source.value,
        "grid_scheme": grid_scheme.value,
        "cfl": cfl_toggle.value == "on",
    }
    if method.value == "psor":
        p.update(
            {
                "omega": omega.value,
                "maxIter": max_iter.value,
                "tol": tol.value,
                "is_american": True,
            }
        )
    elif method.value == "fractional":
        p.update({"beta": beta.value})
    elif method.value == "compact":
        p.update({"dx": dx.value, "maxIter": max_iter.value})
    return p
