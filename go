#!/usr/bin/env bash

set -e

SRC_DIR="src"
BUILD_DIR="build"

CPP_SRC_DIR="$SRC_DIR/cpp"
CPP_BUILD_DIR="$BUILD_DIR/cpp"

CXX="g++"
CXXFLAGS="-std=c++17 -Wall"


# -----------------------------
# HELP / HINT (Interactive)
# -----------------------------
help() {
cat <<HEREDOC
Usage: ./go <command> [options]

Commands:
=== 0. 🛠 PREREQUISITES          ===
=== 1. 💻 LOCAL DEVELOPMENT      ===
=== 2. 🧪 TESTING AND ANALYSIS   ===
=== 3. 🚀 EXPORT & DEPLOY      ===

Enter a number to see details:
HEREDOC

read -rn 1 option
echo ""; echo ""

case ${option} in
  0)
    echo "=== 🛠 PREREQUISITES ==="
    echo "⚙️ install_tools                     -- * Install the following tools: node (>=24), npm, yarn"
    ;;

  1)
    echo "=== 💻 LOCAL DEVELOPMENT    ==="
    echo "▶️  dev               -- Start development build"
    echo "📦  build             -- Build production App"
    echo "🐞  debug             -- Build Debug App"
    ;;

  2)
    echo "=== 🔎 TESTING AND ANALYSIS ==="
    echo "🪄 lint                -- Run analyze & lint code"
    echo "🧪 test                -- Run tests"
    ;;

  3)
    echo "=== 🚀 EXPORT AND DEPLOY ==="
    echo "🗑️  clean             -- Delete build file"
    echo "🎨  format            -- Format code"
    ;;
  *)
    echo "Section $option does not exist"
    ;;
esac
}

# ---------------------------------------------------------------------------------------
# 0)                  === 🛠 PREREQUISITES ===
# ---------------------------------------------------------------------------------------

install_tools(){
  echo "📦 Installing development tools..."
  brew install llvm clang-format

 # Xcode CLI tools (ignore error if already installed)
  xcode-select --install 2>/dev/null || true

  LLVM_PATH="/opt/homebrew/opt/llvm/bin"
  SHELL_RC="$HOME/.zshrc"

  if ! grep -q "$LLVM_PATH" "$SHELL_RC"; then
      echo ""
      echo "🔧 Adding LLVM to PATH in $SHELL_RC"
      echo "export PATH=\"$LLVM_PATH:\$PATH\"" >> "$SHELL_RC"
      echo "✅ PATH updated. Restart terminal or run:"
      echo "source $SHELL_RC"
  else
      echo "✅ LLVM already in PATH"
  fi

  echo ""
   echo "🔎 Verifying installation..."
   clang-tidy --version || echo "clang-tidy not found"
   clang-format --version || echo "clang-format not found"
}

# ---------------------------------------------------------------------------------------
#  1)                === 💻 LOCAL DEVELOPMENT ===
# ---------------------------------------------------------------------------------------

build() {
    echo "⏳ Building project..."
    mkdir -p "$BUILD_DIR"
    $CXX $CXXFLAGS $CPP_SRC_DIR/*.cpp -o $CPP_BUILD_DIR
}

debug() {
    echo "🐞 Building debug version..."
    $CXX $CXXFLAGS -g "$CPP_BUILD_DIR/*.cpp" -o $CPP_BUILD_DIR
    gdb ./$CPP_BUILD_DIR
}

dev() {
    gcc --version
    g++ --version

    build

    echo "▶️ Running program..."
    ./$CPP_BUILD_DIR
}

# ---------------------------------------------------------------------------------------
#  3)                === 🔎 TESTING AND ANALYSIS ===
# ---------------------------------------------------------------------------------------

lint() {
    echo "🪄 Running Lint"

    files=$(find "$CPP_SRC_DIR" -name "*.cpp")
    if [ -z "$files" ]; then
        echo "No C++ files found"
        exit 1
    fi

    echo "🧹 Running clang-tidy..."
    for file in $files; do
        echo "Checking $file"
        clang-tidy "$file" -checks='clang-analyzer-*,modernize-*,performance-*' -- $CXXFLAGS
    done
}

test() {
    echo "🧪 Running tests..."

    files=$(find test -name "*.cpp" 2>/dev/null)
    if [ -z "$files" ]; then
        echo "⚠️ No test files found in ./test"
        return
    fi

    mkdir -p "$CPP_BUILD_DIR"

    TEST_BIN="$CPP_BUILD_DIR/test_app"

    $CXX $CXXFLAGS $files -o "$TEST_BIN"

    "$TEST_BIN"

    echo "✅ Tests completed"
}

format() {
    echo "🎨 Formatting C++ code..."

    files=$(find "$CPP_SRC_DIR" -name "*.cpp" -o -name "*.h")
    if [ -z "$files" ]; then
        echo "No C++ files found"
        exit 1
    fi

    for file in $files; do
        clang-format -i "$file"
        echo "Formatted $file"
    done
}

# ---------------------------------------------------------------------------------------
#  4)                "=== 4. 🚀 EXPORT & COMPRESS ==="
# ---------------------------------------------------------------------------------------

clean() {
    echo "🗑️ Delete build..."
    rm -rf $BUILD_DIR/*
}


# -----------------------------
# Main
# -----------------------------
subcommand=$1
case $subcommand in
"" | "-h" | "--help")
  help
  ;;
*)
  shift
  "${subcommand}" "$@"
  ;;
esac