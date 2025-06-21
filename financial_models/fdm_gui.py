from nicegui import ui, page
import json
import httpx
import asyncio
from datetime import datetime, timedelta
import plotly.graph_objs as go
import logging

# === LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fdm_gui")

# === STATE ===
latest_result = []

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
            option_type = ui.select(['European', 'American'],
                                    label='Option Style', value='European').classes('w-56')
            vol_source = ui.select(['User-defined', 'Implied'],
                                   label='Volatility Source', value='User-defined').classes('w-56')
            grid_scheme = ui.select(['uniform', 'adaptive'],
                                    label="Grid Scheme", value='uniform').classes("w-56")

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

        with ui.row().classes("gap-4 mt-4"):
            ui.button("Run FDM Solver", on_click=lambda: asyncio.create_task(compute_fdm()), color="blue")
            ui.button("Show 2D Plot", on_click=lambda: popup2d.open()).props("outline")
            ui.button("Show 3D Plot", on_click=lambda: popup3d.open()).props("outline")
            ui.button("Show Table", on_click=lambda: popup_table.open()).props("outline")

# === TABLE POPUP ===
with ui.dialog() as popup_table, ui.card().classes("bg-gray-900"):
    ui.label("📋 Output Table").classes("text-white text-lg")
    output_table = ui.table(columns=[
        {'name': 'index', 'label': 'Index', 'field': 'index'},
        {'name': 'value', 'label': 'Value', 'field': 'value'}
    ], rows=[]).classes('w-full max-h-96 text-white')
    ui.button("Close", on_click=popup_table.close)

# === 2D/3D POPUPS ===
with ui.dialog() as popup2d, ui.card().classes("bg-gray-900"):
    popup2d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup2d.close)

with ui.dialog() as popup3d, ui.card().classes("bg-gray-900"):
    popup3d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup3d.close)

# === FDM COMPUTATION ===
async def compute_fdm():
    try:
        logger.info("🚀 Starting FDM computation")
        start = datetime.now()
        end = start + timedelta(days=365)
        if datetime_start.value:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    start = datetime.strptime(datetime_start.value.strip(), fmt)
                    break
                except ValueError:
                    continue
        if datetime_end.value:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    end = datetime.strptime(datetime_end.value.strip(), fmt)
                    break
                except ValueError:
                    continue
        T_calc = (end - start).total_seconds() / (365 * 24 * 60 * 60)
        T_calc = max(T_calc, 1e-6)
        logger.info(f"📐 Computed T: {T_calc:.6f}")
    except Exception as e:
        logger.exception("⚠️ Unexpected datetime error — fallback to T = 1")
        T_calc = 1.0

    params = {
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
        "cfl": cfl_toggle.value == 'on'
    }

    if method.value == 'american':
        params.update({"omega": omega.value, "maxIter": max_iter.value, "tol": tol.value})
    elif method.value == 'fractional':
        params.update({"beta": beta.value})
    elif method.value == 'compact':
        params.update({"dx": dx.value, "maxIter": max_iter.value})

    logger.info(f"📤 Sending to /fdm/{method.value} with:")
    logger.info(json.dumps(params, indent=2))

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"http://localhost:8000/fdm/{method.value}", json=params)
            resp.raise_for_status()
            response_json = resp.json()
            logger.info(f"📥 Response: {json.dumps(response_json, indent=2)}")
            result = response_json.get("result", [])
            final = response_json.get("final_price", 0.0)
            logger.info(f"✅ Received {len(result)} results")
    except Exception as e:
        ui.notify(f"❌ Error: {str(e)}", type="negative")
        logger.exception("API call error")
        return

    latest_result.clear()
    latest_result.extend(result)
    output_table.rows = latest_result

    try:
        final_price.text = f"Final Price: {final:.4f}" if result else "No result."
        logger.info(f"🎯 Final price: {final:.4f}")
    except Exception:
        final_price.text = "No final value returned."
        logger.warning("Could not extract final price")

    fig2d = go.Figure(data=[go.Scatter(
        x=[x["index"] for x in latest_result],
        y=[x["value"] for x in latest_result],
        mode="lines"
    )])
    popup2d_plot.figure = fig2d

    fig3d = go.Figure(data=[go.Scatter3d(
        x=[x["index"] for x in latest_result],
        y=[K.value] * len(latest_result),
        z=[x["value"] for x in latest_result],
        mode="lines"
    )])
    popup3d_plot.figure = fig3d
