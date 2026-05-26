# Hermes Runtime Manager

Runtime Manager is the HTTP/SSE integration surface used by the Go
apiserver. It runs inside the Hermes runtime pod and starts short-lived
Hermes worker subprocesses for agent runs.

The frontend must not call Runtime Manager directly. Cloud/apiserver remains
the only business API entry point and is responsible for authentication,
conversation persistence, model selection, event projection, and audit.

## Runtime Image

Use `Dockerfile.runtime-manager` for the lean Runtime Manager image. It keeps
Hermes core, Runtime Manager, Python dependencies, `kubectl`, and common
network/debug tools, while dropping dashboard/TUI/Playwright/Node layers.

The clean runtime image is expected to be much smaller than the full Hermes
image. Local validation on arm64 produced about `605MB` after the final-stage
ownership optimization, compared with about `3.43GB` for the earlier full
runtime-manager image.

## Build Command

For local or CI builds in China, pass Debian mirror build args explicitly.
Without these args, builds can fail during `apt-get install` with transient
`502 Bad Gateway`, token, or timeout errors from `deb.debian.org` /
Docker Hub.

```bash
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.runtime-manager \
  --build-arg DEBIAN_MIRROR=http://mirrors.aliyun.com/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security \
  -t apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com/apecloud/hermes-agent:runtime-manager-v1 \
  --push \
  .
```

For a single-arch local smoke build:

```bash
docker buildx build \
  --platform linux/arm64 \
  -f Dockerfile.runtime-manager \
  --build-arg DEBIAN_MIRROR=http://mirrors.aliyun.com/debian \
  --build-arg DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security \
  -t hermes-agent-runtime-manager-review:local \
  --load \
  .
```

## Smoke Checks

After a local `--load` build:

```bash
docker image ls hermes-agent-runtime-manager-review:local

docker run --rm --entrypoint /bin/sh hermes-agent-runtime-manager-review:local -lc '
  whoami
  python3 -V
  /opt/hermes/.venv/bin/python - <<PY
import runtime_manager.app, runtime_manager.worker_main
import run_agent, hermes_state
print("imports-ok")
PY
  kubectl version --client=true
'

docker run -d --rm \
  --name hermes-runtime-manager-smoke \
  -e RUNTIME_MANAGER_API_KEY=review-key \
  -p 18765:8765 \
  hermes-agent-runtime-manager-review:local

curl -sS -H 'Authorization: Bearer review-key' \
  http://127.0.0.1:18765/health

docker stop hermes-runtime-manager-smoke
```

Expected health response:

```json
{"status":"ok"}
```

## Notes And Caveats

- Keep LLM provider/model/baseURL/API key out of the Helm chart and container
  environment. Apiserver must resolve the effective Cloud LLM config for the
  current conversation and pass it in each `POST /agent/runs` request.
- `uv` is currently kept in the final image. It costs about `48MB`, but allows
  Hermes lazy dependency installation for optional backends/tools. Remove it
  only if Runtime Manager production mode explicitly forbids runtime dependency
  installs.
- Prefer `COPY --from=builder --chown=hermes:hermes /opt/hermes /opt/hermes`
  in the final stage if changing the Dockerfile. A later recursive
  `chown -R /opt/hermes` can add a large metadata layer; local review observed
  about `68.7MB`.
- The clean image `ENTRYPOINT` is `hermes-runtime-manager`. If the Helm chart
  also sets `args: ["hermes-runtime-manager"]`, that argument is redundant and
  should be kept consistent with the final image entrypoint design.
- Keep the container non-root at runtime. The runtime manager writes user
  homes under `RUNTIME_MANAGER_USERS_ROOT`, usually `/opt/data/users`, backed
  by the chart PVC.
- Runtime Manager should only expose technical run APIs to apiserver. Do not
  expose Hermes profile, `HERMES_HOME`, Runtime Manager API key, or raw worker
  internals to the frontend.
