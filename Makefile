# Compiler and Flags
CXX = clang++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup
PYTHON = poetry run python
# Ensure pybind11 is correctly included
# Ensure pybind11 is correctly included
PYTHON = poetry run python  # if using poetry for virtualenv

# Directories
FINANCIAL_MODELS_DIR = financial_models
QUIC_SERVER_DIR = quic_server
FINANCE_TOOL_DIR = finance_tool
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

# Targets for building financial models and quic server
all: build_financial_models build_quic_server install_financial_tools

# Build Financial Models
build_financial_models:
	$(MAKE) -C $(FINANCIAL_MODELS_DIR)

# Build Quic Server
build_quic_server:
	$(MAKE) -C $(QUIC_SERVER_DIR)

# Install Financial Tools using Poetry (ensure the proper Python environment is set up)
install_financial_tools:
	poetry install --directory $(FINANCE_TOOL_DIR) --no-root

# Run tests
test: test_financial_models

test_financial_models:
	$(PYTHON) -m pytest tests/test_financial_models.py



# Clean all builds
clean:
	$(MAKE) -C $(FINANCIAL_MODELS_DIR) clean
	$(MAKE) -C $(QUIC_SERVER_DIR) clean
	rm -rf $(FINANCE_TOOL_DIR)/.venv $(BUILD_DIR)/*

# Rule to compile .cpp to .o (for financial models or other components)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Rule to link object files into a shared library (financial models)
$(OUTPUT): $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(OUTPUT) $^ $(PYTHON_LIB)
