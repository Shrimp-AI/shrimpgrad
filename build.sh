
# Step 1: Compile Zig code
echo "Compiling zshrimpgrad Zig..."
cd backend
zig build
cd ..

# Step 2: Move the compiled library to the Python package
echo "Moving compiled zshrimpgrad to Python package..."
cp backend/zig-out/lib/*.dylib shrimpgrad/lib/

# Step 3: Build the Python package
echo "Building shrimpgrad Python package..."

# python3 setup.py sdist bdist_wheel
python -m build
python -m pip install .
cd ..

echo "Build complete."