# Browser build, served as a static site. For Coolify: add the repo as an
# application with the "Dockerfile" build pack; the container listens on 80.
# The build compiles Bevy for wasm from scratch — expect 10–20 minutes on a
# typical VPS the first time.

FROM rust:1.93 AS builder
RUN rustup target add wasm32-unknown-unknown
WORKDIR /app
COPY . .
# wasm-bindgen-cli must match the wasm-bindgen crate version in Cargo.lock.
RUN cargo install wasm-bindgen-cli --locked --version \
      "$(grep -A1 'name = "wasm-bindgen"' Cargo.lock | grep version | head -1 | cut -d'"' -f2)"
# Same steps as web/build.sh (kept inline so the image needs no bash script).
RUN cargo build --release --target wasm32-unknown-unknown \
      --no-default-features --features bevy/webgpu \
 && wasm-bindgen --target web --no-typescript --out-dir dist --out-name boids \
      target/wasm32-unknown-unknown/release/boids.wasm \
 && cp web/index.html dist/ \
 && gzip -9 -k -f dist/boids_bg.wasm dist/boids.js

FROM nginx:alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
