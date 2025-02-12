# Compiler
CXX = clang++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup -Iinclude

# Python & pybind11 Paths
PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
PYTHON_LIB := $(shell python3-config --ldflags)

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
OUTPUT = financial_models_wrapper.so

# Source Files
SRCS = $(SRC_DIR)/black_scholes.cpp $(SRC_DIR)/monte_carlo.cpp $(SRC_DIR)/finite_difference.cpp $(SRC_DIR)/utils.cpp financial_models_wrapper.cpp
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Build Shared Library
all: $(OUTPUT)

# Compile .cpp to .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(PYTHON_INCLUDE) -c $< -o $@

# Link .o files to create the shared library
$(OUTPUT): $(OBJS)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(PYTHON_INCLUDE) -o $(OUTPUT) $^ $(PYTHON_LIB)

# Clean Build
clean:
	rm -f $(BUILD_DIR)/*.o $(OUTPUT)
