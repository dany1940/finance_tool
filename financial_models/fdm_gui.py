# fdm_gui.py

from nicegui import ui, page
import json, os, httpx, asyncio
import polars as pl
import plotly.graph_objs as go

# === GLOBAL STATE ===
state = {}
latest_result = []

# === Setup for download path ===
DOWNLOAD_PATH = 'downloads'
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# === Set page title (fixes AppConfig.title crash) ===
page.title = "FDM Calculator"

# === GUI Layout ===
ui.markdown("# 📊 Advanced FDM Financial Tool")

method = ui.select(
    ['explicit', 'implicit', 'crank', 'american', 'fractional', 'exponential', 'compact'],
    label='Finite Difference Method',
    value='explicit'
).classes('w-80')

# === Input Fields ===
with ui.row():
    N = ui.number('N (Grid steps)', value=10).props('step=1')
    M = ui.number('M (Time steps)', value=10).props('step=1')
    Smax = ui.number('Smax', value=100)
    T = ui.number('T (Maturity)', value=1.0)

with ui.row():
    K = ui.number('K (Strike)', value=50)
    r = ui.number('r (Interest rate)', value=0.05)
    sigma = ui.number('σ (Volatility)', value=0.2)
    is_call = ui.checkbox('Call Option', value=True)

with ui.row():
    omega = ui.number('ω (Relaxation)', value=1.2)
    max_iter = ui.number('Max Iter', value=10000).props('step=100')
    tol = ui.number('Tolerance', value=1e-6)

with ui.row():
    beta = ui.number('β (Fractional time)', value=0.8)
    dx = ui.number('dx (Compact dx)', value=1.0)

vector_input = ui.textarea('V (for compact)', value='[1, 2, 3, 4, 5, 6]').classes('w-full')

with ui.row():
    ticker = ui.input('Stock Ticker (optional)').classes('w-64')
    date = ui.date('Start Date').classes('w-64')

# === Plot Area ===
fig2d = go.Figure()
fig3d = go.Figure()
plot2d = ui.plotly(fig2d).classes('w-full h-96')
plot3d = ui.plotly(fig3d).classes('w-full h-96')

# === Table Output ===
output_table = ui.table(columns=[
    {'name': 'Index', 'label': 'Index', 'field': 'Index'},
    {'name': 'Value', 'label': 'Value', 'field': 'Value'}
], rows=[]).classes('w-full')

# === Helper Functions ===
def save_to_csv():
    df = pl.DataFrame(latest_result)
    path = os.path.join(DOWNLOAD_PATH, 'fdm_result.csv')
    df.write_csv(path)
    return '/downloads/fdm_result.csv'

def save_to_excel():
    df = pl.DataFrame(latest_result)
    path = os.path.join(DOWNLOAD_PATH, 'fdm_result.xlsx')
    df.write_excel(path)
    return '/downloads/fdm_result.xlsx'

# === Async Computation ===
async def compute_fdm():
    params = {
        "N": N.value,
        "M": M.value,
        "Smax": Smax.value,
        "T": T.value,
        "K": K.value,
        "r": r.value,
        "sigma": sigma.value,
        "is_call": is_call.value,
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

    plot2d.figure = go.Figure(data=[go.Scatter(
        x=[x["Index"] for x in latest_result],
        y=[x["Value"] for x in latest_result],
        mode="lines"
    )])
    plot3d.figure = go.Figure(data=[go.Scatter3d(
        x=[x["Index"] for x in latest_result],
        y=[K.value] * len(latest_result),
        z=[x["Value"] for x in latest_result],
        mode="lines"
    )])

    state['download_link_csv'].target = save_to_csv()
    state['download_link_excel'].target = save_to_excel()
    state['download_link_csv'].visible = True
    state['download_link_excel'].visible = True

# === Buttons & Download Links ===
with ui.row():
    ui.button("Run FDM Solver", on_click=lambda: asyncio.create_task(compute_fdm()), color="primary")

with ui.row():
    state['download_link_csv'] = ui.link("⬇️ Download CSV", target="", new_tab=True).classes("text-blue-600").bind_visibility_from(False)
    state['download_link_excel'] = ui.link("⬇️ Download Excel", target="", new_tab=True).classes("text-green-600").bind_visibility_from(False)
