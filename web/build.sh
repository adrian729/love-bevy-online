#!/usr/bin/env bash
# Build the browser version into dist/. Mirrors what the Dockerfile does;
# run it locally to test the exact artifact that gets deployed:
#
#   web/build.sh
#   python3 -m http.server -d dist 8080   # then open http://localhost:8080
#
# Needs: `rustup target add wasm32-unknown-unknown` (one-time).
set -euo pipefail
cd "$(dirname "$0")/.."

# No default features (drops native-only dynamic linking); bevy/webgpu
# replaces the default WebGL2 web backend — the compute sim needs it.
cargo build --release --target wasm32-unknown-unknown \
  --no-default-features --features bevy/webgpu

# wasm-bindgen-cli must match the wasm-bindgen *crate* version in Cargo.lock
# exactly, or the generated JS glue is rejected at load time.
WB_VERSION=$(grep -A1 'name = "wasm-bindgen"' Cargo.lock | grep version | head -1 | cut -d'"' -f2)
if ! wasm-bindgen --version 2>/dev/null | grep -q "$WB_VERSION"; then
  cargo install wasm-bindgen-cli --version "$WB_VERSION" --locked
fi

rm -rf dist
wasm-bindgen --target web --no-typescript --out-dir dist --out-name boids \
  target/wasm32-unknown-unknown/release/boids.wasm
cp web/index.html dist/

# Precompressed copies for nginx's gzip_static (compress once at build time
# instead of per request; the .wasm is by far the heaviest file).
gzip -9 -k -f dist/boids_bg.wasm dist/boids.js

ls -lh dist/
echo "dist/ ready — test with: python3 -m http.server -d dist 8080"
