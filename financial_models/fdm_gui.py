from nicegui import ui, page
import json
import httpx
import asyncio
import plotly.graph_objs as go
from datetime import datetime

# === GLOBAL STATE ===
state = {}
latest_result = []

# === Page Configuration ===
page.title = "FDM Calculator"
ui.dark_mode().enable()

# === Centered & Wider Layout Container ===
with ui.row().classes("w-full justify-center"):
    with ui.column().classes("max-w-6xl w-full bg-gray-900 text-white p-4 gap-2 items-center"):

        ui.markdown("## 🎓 Finite Difference Method Option Pricing Tool").classes("text-white text-2xl mb-4")
        ui.markdown("### Method & Market Parameters").classes("text-white text-xl")

        # Row 1 - All dropdowns
        with ui.row().classes("gap-4"):
            method = ui.select(['explicit', 'implicit', 'crank', 'american', 'fractional', 'exponential', 'compact'],
                               label='Finite Difference Method', value='explicit').classes('w-56')
            option_type = ui.select(['European', 'American', 'Barrier', 'Asian', 'Digital'],
                                    label='Option Style', value='European').classes('w-56')
            vol_source = ui.select(['User-defined', 'Historical', 'Implied'],
                                   label='Volatility Source', value='User-defined').classes('w-56')
            grid_scheme = ui.select(['uniform', 'adaptive'],
                                    label="Grid Scheme", value='uniform').classes("w-56")

        # Row 2
        with ui.row().classes("gap-4"):
            N = ui.number('N (Grid steps)', value=10).props('step=1').classes('w-56')
            M = ui.number('M (Time steps)', value=10).props('step=1').classes('w-56')
            Smax = ui.number('Smax', value=100).classes('w-56')
            K = ui.number('K (Strike)', value=50).classes('w-56')

        # Row 3
        with ui.row().classes("gap-4"):
            r = ui.number('r (Interest rate)', value=0.05).classes('w-56')
            sigma = ui.number('σ (Volatility)', value=0.2).classes('w-56')
            omega = ui.number('ω (Relaxation)', value=1.2).classes('w-56')
            max_iter = ui.number('Max Iter', value=10000).props('step=100').classes('w-56')

        # Row 4 - Date + Toggles
        with ui.row().classes("gap-4"):
            datetime_start = ui.input('Start Datetime').props('placeholder="YYYYMMDD HHMMSS"').classes('w-56')
            datetime_end = ui.input('End Datetime').props('placeholder="YYYYMMDD HHMMSS"').classes('w-56')
            is_call = ui.toggle({True: 'Call', False: 'Put'}, value=True).classes('w-56')
            cfl_toggle = ui.toggle({'off': 'off', 'on': 'on'}, value='off').classes('w-56')

        # Row 5 - Advanced
        with ui.row().classes("gap-4"):
            tol = ui.number('Tolerance', value=1e-6).classes('w-56')
            beta = ui.number('β (Fractional time)', value=0.8).classes('w-56')
            dx = ui.number('dx (Compact dx)', value=1.0).classes('w-56')
            vector_input = ui.textarea('V (for compact)', value='[1, 2, 3, 4, 5, 6]').classes('w-56').props('rows=1')

        # Row 6 - Output
        with ui.row().classes("gap-4"):
            final_price = ui.label("Final Price:").classes("text-green-400 text-xl border border-green-400 p-2 rounded w-full")

        # Row 7 - Buttons
        with ui.row().classes("gap-4 mt-4"):
            ui.button("Run FDM Solver", on_click=lambda: asyncio.create_task(compute_fdm()), color="blue")
            ui.button("Show 2D Plot", on_click=lambda: popup2d.open()).props("outline")
            ui.button("Show 3D Plot", on_click=lambda: popup3d.open()).props("outline")
            ui.button("Show Table", on_click=lambda: popup_table.open()).props("outline")

# === Table Popup ===
with ui.dialog() as popup_table, ui.card().classes("bg-gray-900"):
    ui.label("📋 Output Table").classes("text-white text-lg")
    output_table = ui.table(
        columns=[
            {'name': 'Index', 'label': 'Index', 'field': 'Index'},
            {'name': 'Value', 'label': 'Value', 'field': 'Value'}
        ],
        rows=[]
    ).classes('w-full max-h-96 text-white')
    ui.button("Close", on_click=popup_table.close)

# === Plot Popups ===
with ui.dialog() as popup2d, ui.card().classes("bg-gray-900"):
    popup2d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup2d.close)

with ui.dialog() as popup3d, ui.card().classes("bg-gray-900"):
    popup3d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup3d.close)

# === Compute FDM Backend ===
async def compute_fdm():
    try:
        start = datetime.strptime(datetime_start.value, "%Y%m%d %H%M%S")
        end = datetime.strptime(datetime_end.value, "%Y%m%d %H%M%S")
        T_calc = (end - start).total_seconds() / (365 * 24 * 60 * 60)
        if T_calc <= 0:
            ui.notify("⚠️ End datetime must be after start.")
            return
    except Exception as e:
        ui.notify(f"⚠️ Invalid datetime: {e}")
        return

    params = {
        "N": N.value, "M": M.value, "Smax": Smax.value, "T": T_calc,
        "K": K.value, "r": r.value, "sigma": sigma.value,
        "is_call": is_call.value,
        "option_style": option_type.value,
        "vol_source": vol_source.value,
        "grid_scheme": grid_scheme.value,
        "cfl": cfl_toggle.value == 'on',
        "datetime_start": datetime_start.value,
        "datetime_end": datetime_end.value
    }

    if method.value == 'american':
        params.update({"omega": omega.value, "maxIter": max_iter.value, "tol": tol.value})
    elif method.value == 'fractional':
        params.update({"beta": beta.value})
    elif method.value == 'compact':
        try:
            V = json.loads(vector_input.value)
            assert isinstance(V, list)
            params.update({"V": V, "dx": dx.value})
        except Exception:
            ui.notify("⚠️ Invalid vector V format.")
            return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f'http://localhost:8000/fdm/{method.value}', json=params)
            resp.raise_for_status()
            result = resp.json().get("result", [])
    except Exception as e:
        ui.notify(f"❌ Error: {e}")
        return

    latest_result.clear()
    latest_result.extend([{"Index": i, "Value": v} for i, v in enumerate(result)])
    output_table.rows = latest_result
    final_price.text = f"Final Price: {result[-1]:.4f}" if result else "No result."

    fig2d = go.Figure(data=[go.Scatter(
        x=[x["Index"] for x in latest_result],
        y=[x["Value"] for x in latest_result],
        mode="lines"
    )])
    popup2d_plot.figure = fig2d

    fig3d = go.Figure(data=[go.Scatter3d(
        x=[x["Index"] for x in latest_result],
        y=[K.value] * len(latest_result),
        z=[x["Value"] for x in latest_result],
        mode="lines"
    )])
    popup3d_plot.figure = fig3d
