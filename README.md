
# 📌 High-Performance Finite Difference Method (FDM) Option Pricing System

🚀 **Built with Python, C++, Pybind11, FastAPI, and NiceGUI — Optimized for Financial Modeling and Real-Time Visualization** 🚀

---

## 📌 Architecture Overview

The system integrates:

✅ **C++ Backends** – High-performance numerical solvers using finite difference methods
✅ **Pybind11** – Bridges C++ methods into Python
✅ **FastAPI** – Exposes backend functionality as clean, typed RESTful APIs
✅ **NiceGUI + Matplotlib** – Interactive GUI with dynamic charts and result tables
✅ **Poetry** – Dependency management and environment setup
✅ **Makefile** – Build automation for C++ shared object (`.so`) generation

---

## 📌 FDM Features Supported

- ✅ Explicit Scheme
- ✅ Implicit Scheme
- ✅ Crank-Nicolson Scheme
- ✅ American Option Pricing (via PSOR)
- ✅ Exponential Integral
- ✅ Fractional Time Derivatives
- ✅ Compact Schemes (Advanced)
- ✅ Surface Generation for 3D Visualization

---

## 📌 Folder Structure Overview

financial_tool/ ├── financial_models/ │ ├── fdm_gui.py # GUI frontend (NiceGUI + Matplotlib) │ ├── routers/ # FastAPI routers │ ├── src/ # C++ source files (.cpp) │ ├── include/ # C++ headers │ ├── build/ # Compiled object files │ ├── financial_models_wrapper.cpp/.so # Pybind11 integration │ ├── main.py # FastAPI entrypoint │ ├── Makefile # Build C++ FDM solvers │ └── tests/ # Pytest test suite


---

## 📌 Getting Started

### 🧰 1. Install Dependencies

#### ✅ Linux/macOS

```bash
sudo apt install python3.11 python3.11-venv g++ make cmake

curl -sSL https://install.python-poetry.org | python3.11 -

cd financial_tool
poetry shell
poetry install

make
✅ Windows 

# Step 1: Install Python 3.11+ from https://www.python.org/downloads/windows/
# Make sure to check "Add Python to PATH" during installation

# Step 2: Install Poetry (user-level, no admin needed)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Close and reopen PowerShell

# Step 3: Verify Poetry installed
poetry --version

# Step 4: Create project folder and initialize
mkdir financial_tool
cd financial_tool
poetry new financial_models
cd financial_models

# Step 5: Activate Poetry environment
poetry shell

# Step 6: Add dependencies
poetry add fastapi uvicorn nicegui matplotlib numpy

# Step 7: If repo cloned, install dependencies
poetry install

# Step 8: Compile C++ solvers (run Makefile if supported)
make
▶️ 2. Run the Application

cd financial_tool/financial_models
uvicorn app:app --reload --host 0.0.0.0 --port 8000
Open in your browser:
🧪 Swagger API: http://localhost:8000/docs
🖥️ FDM GUI Frontend: http://localhost:8000/

✅ 3. Run Tests

cd financial_tool/financial_models
poetry run pytest tests/
📌 Example Endpoints

POST /fdm/explicit – Compute price vector using explicit scheme
POST /fdm/explicit_surface – Return full price surface for 3D visualization
Similar endpoints for implicit, crank, american, etc.


Developed by Florin Dumitrascu
King’s College London | Quantitative Financial Modelling
