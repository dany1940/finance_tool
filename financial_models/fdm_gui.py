from nicegui import ui, page
import httpx
import asyncio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import logging
import numpy as np

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fdm_gui")

# === STATE ===
latest_vector = {"S_grid": [], "prices": [], "final_price": 0.0}
bs_price_label = None

# === PAGE CONFIG ===
page.title = "FDM Calculator"
ui.dark_mode().enable()

# === MAIN UI ===
with ui.row().classes("w-full justify-center"):
    with ui.column().classes("max-w-6xl w-full bg-gray-900 text-white p-4 gap-2 items-center"):
        ui.markdown("## 🎓 Finite Difference Method Option Pricing Tool").classes("text-2xl mb-4")
        ui.markdown("### Method & Market Parameters").classes("text-xl")

        with ui.row().classes("gap-4"):
            method = ui.select(['explicit', 'implicit', 'crank', 'american', 'fractional', 'exponential', 'compact'],
                               label='Finite Difference Method', value='explicit').classes('w-56')
            option_type = ui.select(['European', 'American'], label='Option Style', value='European').classes('w-56')
            vol_source = ui.select(['User-defined', 'Implied'], label='Volatility Source', value='User-defined').classes('w-56')
            grid_scheme = ui.select(['uniform', 'adaptive'], label="Grid Scheme", value='uniform').classes("w-56")

        with ui.row().classes("gap-4"):
            N = ui.number('N (Grid steps)', value=10).props('step=1').classes('w-56')
            M = ui.number('M (Time steps)', value=10).props('step=1').classes('w-56')
            Smax = ui.number('Smax', value=100).classes('w-56')
            K = ui.number('K (Strike)', value=50).classes('w-56')

        with ui.row().classes("gap-4"):
            r = ui.number('r (Interest rate)', value=0.05).classes('w-56')
            sigma = ui.number('σ (Volatility)', value=0.2).classes('w-56')
            omega = ui.number('ω (Relaxation)', value=1.2).classes('w-56')
            S0 = ui.number('S₀ (Spot Price)', value=50.0).classes('w-56')

        with ui.row().classes("gap-4"):
            datetime_start = ui.input('Start Datetime').props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"').classes('w-56')
            datetime_end = ui.input('End Datetime').props('placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM:SS"').classes('w-56')
            is_call = ui.toggle({True: 'Call', False: 'Put'}, value=True).classes('w-56')
            cfl_toggle = ui.toggle({'off': 'off', 'on': 'on'}, value='off').classes('w-56')

        with ui.row().classes("gap-4"):
            tol = ui.number('Tolerance', value=1e-6).classes('w-56')
            beta = ui.number('β (Fractional time)', value=0.8).classes('w-56')
            dx = ui.number('dx (Compact dx)', value=1.0).classes('w-56')
            max_iter = ui.number('Max Iter', value=10000).props('step=100').classes('w-56')

        with ui.row().classes("gap-4"):
            final_price = ui.label("Final Price:").classes("text-green-400 text-xl border border-green-400 p-2 rounded w-full")
        with ui.row().classes("gap-4"):
            bs_price_label = ui.label("Black-Scholes Price:").classes("text-yellow-400 text-xl border border-yellow-400 p-2 rounded w-full")

        with ui.row().classes("gap-4 mt-4"):
            ui.button("Run FDM Solver", on_click=lambda: asyncio.create_task(compute_fdm()), color="blue")
            ui.button("Run Black-Scholes", on_click=lambda: asyncio.create_task(compute_bs()), color="orange")
            ui.button("Show 2D Plot", on_click=lambda: asyncio.create_task(show_vector_plot('2d')), color="green")
            ui.button("Show 3D Line", on_click=lambda: asyncio.create_task(show_vector_plot('3d')), color="cyan")
            ui.button("Show Surface", on_click=lambda: asyncio.create_task(show_surface_plot()), color="purple")
            ui.button("Show Table", on_click=lambda: popup_table.open()).props("outline")

# === TABLE POPUP ===
with ui.dialog() as popup_table, ui.card().classes("bg-gray-900 p-4 w-[600px]"):
    ui.label("📋 Output Table").classes("text-white text-lg mb-2")
    with ui.element("div").classes("h-[300px] overflow-y-auto"):
        output_table = ui.table(columns=[
            {'name': 'index', 'label': 'Index', 'field': 'index'},
            {'name': 'value', 'label': 'Value', 'field': 'value'}
        ], rows=[]).classes("w-full text-white")
    ui.button("Close", on_click=popup_table.close).classes("mt-4")

# === 2D/3D POPUP ===
with ui.dialog() as popup3d, ui.card().classes("bg-gray-900 p-4 w-[1000px] h-[700px]"):
    plot3d_image = ui.image().classes("w-full h-[600px] object-contain")
    ui.button("Close", on_click=popup3d.close).classes("mt-4")

# === COMPUTE FDM ===
async def compute_fdm():
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"http://localhost:8000/fdm/{method.value}", json=params)
            resp.raise_for_status()
            data = resp.json()
            final = data.get("final_price", 0.0)
            final_price.text = f"Final Price: {final:.4f}"
    except Exception as e:
        logger.exception("FDM Error")
        ui.notify(f"❌ FDM Error: {str(e)}", type="negative")

# === BLACK-SCHOLES ===
async def compute_bs():
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = {
            "S": S0.value, "K": K.value, "T": T_calc,
            "r": r.value, "sigma": sigma.value, "is_call": is_call.value,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://localhost:8000/fdm/black_scholes", json=params)
            resp.raise_for_status()
            data = resp.json()
            bs_price_label.text = f"Black-Scholes Price: {data['price']:.4f}"
    except Exception as e:
        logger.exception("BS Error")
        ui.notify(f"❌ Black-Scholes Error: {str(e)}", type="negative")

# === SHOW VECTOR (2D/3D LINE) ===
async def show_vector_plot(plot_type: str):
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"http://localhost:8000/fdm/{method.value}_vector", json=params)
            resp.raise_for_status()
            data = resp.json()

        S_grid = data["S_grid"]
        prices = data["prices"]
        latest_vector["S_grid"] = S_grid
        latest_vector["prices"] = prices
        output_table.rows = [{"index": i, "value": prices[i]} for i in range(len(prices))]

        fig = plt.figure()
        if plot_type == '2d':
            ax = fig.add_subplot(111)
            ax.plot(S_grid, prices)
            ax.set_title('Option Price vs Asset Price')
            ax.set_xlabel('S')
            ax.set_ylabel('Price')
            ax.grid(True)
        elif plot_type == '3d':
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(S_grid, [0] * len(S_grid), prices)
            ax.set_xlabel("S")
            ax.set_ylabel("Time")
            ax.set_zlabel("Price")
            ax.set_title("Option Price at Maturity")
            ax.view_init(elev=30, azim=-135)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plot3d_image.source = f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'
        popup3d.open()
        plt.close(fig)
    except Exception as e:
        logger.exception("Vector Plot Error")
        ui.notify(f"❌ Plot Error: {str(e)}", type="negative")

# === SHOW SURFACE PLOT ===
async def show_surface_plot():
    try:
        T_calc = compute_maturity(datetime_start.value, datetime_end.value)
        params = build_params(T_calc)
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"http://localhost:8000/fdm/{method.value}_surface", json=params)
            resp.raise_for_status()
            data = resp.json()

        S = np.array(data["S_grid"])
        t = np.array(data["t_grid"])
        V = np.array(data["price_surface"])
        S_mesh, t_mesh = np.meshgrid(S, t)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_mesh, t_mesh, V, cmap='coolwarm', edgecolor='k', linewidth=0.3, antialiased=True)
        ax.set_xlabel("S")
        ax.set_ylabel("T")
        ax.set_zlabel("Option Price")
        ax.set_title("Option Price Surface")
        ax.view_init(elev=30, azim=-135)
        fig.colorbar(surf, shrink=0.5, aspect=12, label="Price")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plot3d_image.source = f'data:image/png;base64,{base64.b64encode(buf.read()).decode()}'
        popup3d.open()
        plt.close(fig)
    except Exception as e:
        logger.exception("Surface Plot Error")
        ui.notify(f"❌ Surface plot error: {str(e)}", type="negative")

# === MATURITY ===
def compute_maturity(start_str, end_str):
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
        "cfl": cfl_toggle.value == 'on',
    }
    if method.value == "american":
        p.update({"omega": omega.value, "maxIter": max_iter.value, "tol": tol.value})
    elif method.value == "fractional":
        p.update({"beta": beta.value})
    elif method.value == "compact":
        p.update({"dx": dx.value, "maxIter": max_iter.value})
    return p
