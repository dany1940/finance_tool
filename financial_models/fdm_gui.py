from nicegui import ui, page
import os
import json
import httpx
import asyncio
import polars as pl
import plotly.graph_objs as go

# === GLOBAL STATE ===
state = {}
latest_result = []

# === Setup download directory ===
DOWNLOAD_PATH = 'downloads'
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# === Page Configuration ===
page.title = "FDM Calculator"
ui.dark_mode().enable()  # Force dark theme

# === Main Container ===
with ui.column().classes("w-full bg-gray-900 text-white p-4"):

    ui.markdown("## 🎓 Finite Difference Method Option Pricing Tool").classes("text-white text-2xl mb-4")

    with ui.row().classes("w-full justify-between"):

        # === Method & Market Parameters ===
        with ui.column().classes("w-2/3"):
            ui.markdown("### Method & Market Parameters").classes("text-white text-xl")
            method = ui.select(
                ['explicit', 'implicit', 'crank', 'american', 'fractional', 'exponential', 'compact'],
                label='Finite Difference Method',
                value='explicit'
            ).classes('w-80 text-white')

            with ui.row().classes("gap-4 flex-wrap"):
                N = ui.number('N (Grid steps)', value=10).props('step=1')
                M = ui.number('M (Time steps)', value=10).props('step=1')
                Smax = ui.number('Smax', value=100)
                T = ui.number('T (Maturity)', value=1.0)

            with ui.row().classes("gap-4 flex-wrap"):
                K = ui.number('K (Strike)', value=50)
                r = ui.number('r (Interest rate)', value=0.05)
                sigma = ui.number('σ (Volatility)', value=0.2)
                is_call = ui.toggle({True: 'Call', False: 'Put'}, value=True).props('inline')
                option_type = ui.select(['European', 'American'], label='Option Style', value='European').classes('w-64')

        # === Output Table on Right Side ===
        with ui.column().classes("w-1/3 items-end"):
            ui.markdown("### Output Table").classes("text-white text-lg")
            output_table = ui.table(
                columns=[
                    {'name': 'Index', 'label': 'Index', 'field': 'Index'},
                    {'name': 'Value', 'label': 'Value', 'field': 'Value'}
                ],
                rows=[]
            ).classes('w-full max-h-96 text-white')

    # === Advanced Parameters ===
    with ui.expansion("Advanced Parameters", value=True).classes("w-2/3 mt-6"):
        with ui.row().classes("gap-4 flex-wrap"):
            omega = ui.number('ω (Relaxation)', value=1.2)
            max_iter = ui.number('Max Iter', value=10000).props('step=100')
            tol = ui.number('Tolerance', value=1e-6)

        with ui.row().classes("gap-4 flex-wrap"):
            beta = ui.number('β (Fractional time)', value=0.8)
            dx = ui.number('dx (Compact dx)', value=1.0)

        vector_input = ui.textarea('V (for compact)', value='[1, 2, 3, 4, 5, 6]').classes('w-full')

        with ui.row().classes("gap-4 flex-wrap"):
            ticker = ui.input('Stock Ticker (optional)').classes('w-64')
            date = ui.date('Start Date').classes('w-64')

        # Final Price
        final_price = ui.label("").classes("text-green-400 text-xl")

        # Buttons aligned
        with ui.row().classes("gap-4 mt-4"):
            ui.button("Run FDM Solver", on_click=lambda: asyncio.create_task(compute_fdm()), color="blue")
            ui.button("Show 2D Plot", on_click=lambda: popup2d.open()).props("outline")
            ui.button("Show 3D Plot", on_click=lambda: popup3d.open()).props("outline")

# === Plot Modals ===
with ui.dialog() as popup2d, ui.card().classes("bg-gray-900"):
    popup2d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup2d.close)

with ui.dialog() as popup3d, ui.card().classes("bg-gray-900"):
    popup3d_plot = ui.plotly(go.Figure()).classes('w-160 h-96')
    ui.button("Close", on_click=popup3d.close)

# === Helpers ===
def save_to_csv():
    df = pl.DataFrame(latest_result)
    path = os.path.join(DOWNLOAD_PATH, 'fdm_result.csv')
    df.write_csv(path)
    return path

def save_to_excel():
    df = pl.DataFrame(latest_result)
    path = os.path.join(DOWNLOAD_PATH, 'fdm_result.xlsx')
    df.write_excel(path)
    return path

def save_plot(fig, filename):
    fig.write_image(filename)
    return filename

# === Backend Compute ===
async def compute_fdm():
    params = {
        "N": N.value, "M": M.value, "Smax": Smax.value, "T": T.value,
        "K": K.value, "r": r.value, "sigma": sigma.value,
        "is_call": is_call.value, "option_style": option_type.value
    }

    if method.value == 'american':
        params.update({"omega": omega.value, "maxIter": max_iter.value, "tol": tol.value})
    elif method.value == 'fractional':
        params.update({"beta": beta.value})
    elif method.value == 'compact':
        try:
            V = json.loads(vector_input.value)
            assert isinstance(V, list)
            params = {"V": V, "dx": dx.value}
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

    fig3d = go.Figure(data=[go.Scatter3d(
        x=[x["Index"] for x in latest_result],
        y=[K.value] * len(latest_result),
        z=[x["Value"] for x in latest_result],
        mode="lines"
    )])

    popup2d_plot.figure = fig2d
    popup3d_plot.figure = fig3d
