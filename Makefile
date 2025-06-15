# Compiler and Flags
CXX = clang++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup
PYTHON = poetry run python
# Ensure pybind11 is correctly included
# Ensure pybind11 is correctly included
PYTHON = poetry run python  # if using poetry for virtualenv

# Directories
FINANCIAL_MODELS_DIR = financial_models
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

# Targets for building financial models and quic server
all: build_financial_models

# Build Financial Models
build_financial_models:
	$(MAKE) -C $(FINANCIAL_MODELS_DIR)

# Run tests
test: test_financial_models

test_financial_models:
	$(PYTHON) -m pytest tests/test_financial_models.py



# Clean all builds
clean:
	$(MAKE) -C $(FINANCIAL_MODELS_DIR) clean

# Rule to compile .cpp to .o (for financial models or other components)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Rule to link object files into a shared library (financial models)
$(OUTPUT): $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(OUTPUT) $^ $(PYTHON_LIB)
