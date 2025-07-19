import asyncio
import base64
import io
import logging
from datetime import datetime, timedelta

import httpx
import matplotlib.pyplot as plt
import numpy as np
from nicegui import page, ui

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
    "fractional",
    "exponential",
    "compact",
]
american_methods = ["psor"]
# === PAGE CONFIG ===
page.title = "FDM Calculator"
ui.dark_mode().enable()

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
            ).classes("w-56")

            # Option type dropdown
            option_type = ui.select(
                ["European", "American"], label="Option Style", value="European"
            ).classes("w-56")

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
            ).classes("w-56")
            grid_scheme = ui.select(
                ["uniform", "adaptive"], label="Grid Scheme", value="uniform"
            ).classes("w-56")

        with ui.row().classes("gap-4"):
            N = ui.number("N (Grid steps)", value=10).props("step=1").classes("w-56")
            M = ui.number("M (Time steps)", value=10).props("step=1").classes("w-56")
            Smax = ui.number("Smax", value=100).classes("w-56")
            K = ui.number("K (Strike)", value=50).classes("w-56")
        ## Market parameters
        with ui.row().classes("gap-4"):
            r = ui.number("r (Interest rate)", value=0.05).classes("w-56")
            sigma = ui.number("σ (Volatility)", value=0.2).classes("w-56")
            omega = ui.number("ω (Relaxation)", value=1.2).classes("w-56")
            S0 = ui.number("S₀ (Spot Price)", value=50.0).classes("w-56")
        # Optional parameters for specific methods
        with ui.row().classes("gap-4"):
            datetime_start = (
                ui.input("Start Datetime")
                .props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"')
                .classes("w-56")
            )
            datetime_end = (
                ui.input("End Datetime")
                .props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"')
                .classes("w-56")
            )
            is_call = ui.toggle({True: "Call", False: "Put"}, value=True).classes(
                "w-56"
            )
            cfl_toggle = ui.toggle({"off": "off", "on": "on"}, value="off").classes(
                "w-56"
            )
        # Optional parameters for specific methods
        with ui.row().classes("gap-4"):
            tol = ui.number("Tolerance", value=1e-6).classes("w-56")
            beta = ui.number("β (Fractional time)", value=0.8).classes("w-56")
            dx = ui.number("dx (Compact dx)", value=1.0).classes("w-56")
            max_iter = (
                ui.number("Max Iter", value=10000).props("step=100").classes("w-56")
            )
        # === BLACK-SCHOLES PRICE ===
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

            def show_early_exercise_info() -> None:
                """Show early exercise info for American options."""
                ui.notify(
                    "📘 American options allow early exercise before expiry.\n"
                    "This creates a free boundary problem handled using methods like PSOR.",
                    type="info",
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
            ui.button("Show Table", on_click=lambda: popup_table.open()).props(
                "outline"
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
            final = data.get("final_price", 0.0)
            final_price.text = f"Final Price: {final:.4f}"

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

                diff = abs(final - bs_price)
                comparison_label.text = f"{method.value.capitalize()} vs Black-Scholes:\nFDM: {final:.4f} | BS: {bs_price:.4f} | Δ: {diff:.4f}"
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
