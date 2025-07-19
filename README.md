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

```
financial_tool/
├── financial_models/
│   ├── fdm_gui.py                 # GUI frontend (NiceGUI + Matplotlib)
│   ├── routers/                   # FastAPI routers
│   ├── src/                       # C++ source files (.cpp)
│   ├── include/                   # C++ headers
│   ├── build/                     # Compiled object files
│   ├── financial_models_wrapper.cpp/.so # Pybind11 integration
│   ├── main.py                    # FastAPI entrypoint
│   ├── Makefile                   # Build C++ FDM solvers
│   └── tests/                     # Pytest test suite
```

---

## 📌 Getting Started

### 🧰 1. Install Dependencies

#### ✅ Linux/macOS

```bash
# Prerequisites
sudo apt install python3.11 python3.11-venv g++ make cmake

# Install Poetry
curl -sSL https://install.python-poetry.org | python3.11 -

# Clone repo and initialize
cd financial_tool
poetry shell
poetry install

# Compile C++ solvers
make
```

#### ✅ Windows (via WSL recommended)

```bash
# Use WSL with Ubuntu
sudo apt update
sudo apt install python3.11 python3.11-venv g++ make cmake

# Install Poetry
curl -sSL https://install.python-poetry.org | python3.11 -

# Clone and build
cd financial_tool
poetry shell
poetry install
make
```

---

### ▶️ 2. Run the Application

```bash
cd financial_tool/financial_models

# Start FastAPI + GUI
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then open:
🧪 **Swagger API:** http://localhost:8000/docs
🖥️ **FDM GUI Frontend:** http://localhost:8000

---

### ✅ 3. Run Tests

```bash
cd financial_tool/financial_models
poetry run pytest tests/
```

---

## 📌 Example Endpoints

- `POST /fdm/explicit` – Compute price vector using explicit scheme
- `POST /fdm/explicit_surface` – Return full price surface for 3D visualization
- Similar endpoints available for `implicit`, `crank`, `american`, etc.

---

## 📌 Authors

Developed by **Florin Dumitrascu**
King’s College London | Quantitative Financial Modelling
