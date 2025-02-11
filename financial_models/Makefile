# Compiler
CXX = clang++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -undefined dynamic_lookup

# Python & pybind11 Paths
PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
PYTHON_LIB := $(shell python3-config --ldflags)

# Source and Output Files
SRC_DIR = src
INCLUDE_DIR = include
OUTPUT = financial_models$(shell python3-config --extension-suffix)

# Source Files
SRCS = $(SRC_DIR)/black_scholes.cpp $(SRC_DIR)/monte_carlo.cpp $(SRC_DIR)/finite_difference.cpp $(SRC_DIR)/utils.cpp financial_models_wrapper.cpp
OBJS = $(SRCS:.cpp=.o)

# Build Shared Library
all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -I$(INCLUDE_DIR) $(PYTHON_INCLUDE) -o $(OUTPUT) $^ $(PYTHON_LIB)

# Clean Build
clean:
	rm -f $(OUTPUT) $(SRC_DIR)/*.o
