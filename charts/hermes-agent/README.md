# Hermes Agent Helm Chart

This chart deploys Hermes Agent as an in-cluster runtime manager. It is intended for a Cloud or product backend to call internally; browsers should not connect directly to Hermes.

## What It Installs

- `Deployment` running `hermes-runtime-manager`, a thin HTTP/SSE service that starts short-lived Hermes worker subprocesses.
- `Service` exposing the runtime manager inside the cluster.
- `Secret` for the runtime manager bearer key.
- Optional `PersistentVolumeClaim` mounted as `HERMES_HOME`.
- `ServiceAccount` bound to the predefined `apecloud-cluster-admin` ClusterRole.

## Minimal Install

```bash
helm install hermes-agent ./charts/hermes-agent \
  --set image.registry=apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com \
  --set image.repository=apecloud/hermes-agent \
  --set image.tag=runtime-manager-v1
```

The chart generates a random runtime manager API key when `runtimeManager.apiKey` is empty. For production, set a strong generated key through your secret manager.

LLM provider/model/baseURL/API-key configuration is intentionally not configured in this chart. The product backend should resolve the current user's selected model configuration for each conversation and pass it to Runtime Manager when starting a run, using `model` plus `llm_config` or the equivalent per-run fields.

Hermes profile configuration is also intentionally not configured through Helm values. Each user workspace should be initialized by the product backend or Runtime Manager under that user's `HERMES_HOME`.

## Runtime Manager Configuration

The chart values only configure the Runtime Manager process and Kubernetes deployment surface:

- `runtimeManager.host`, `runtimeManager.port`
- `runtimeManager.usersRoot`
- `runtimeManager.apiKey`
- `runtimeManager.logLevel`
- `runtimeManager.stopGraceSeconds`
- `runtimeManager.maxActiveRuns`
- `runtimeManager.maxActiveRunsPerUser`
- `runtimeManager.maxEventsPerRun`
- `runtimeManager.completedRunTtlSeconds`
- `runtimeManager.allowUnauthenticated`
- `runtimeManager.pythonExecutable`
- `runtimeManager.defaultEnabledToolsets`

The runtime manager API key is stored in the chart Secret. If `secret.create=false` or `secret.name` points to a pre-created Secret, the key name must match `secret.key` (default `runtime-manager-api-key`).

`runtimeManager.defaultEnabledToolsets` defaults to `terminal,file` so the lean runtime image does not initialize optional browser, TTS, image, or messaging tool dependencies during KubeBlocks diagnosis. The apiserver may override `enabled_toolsets` per run; set this value to `all` only for development images that include every optional dependency.

## Kubernetes Access

The chart always binds the runtime `ServiceAccount` to the predefined `apecloud-cluster-admin` ClusterRole and enables service account token mounting by default. Kubernetes diagnostics should use this in-cluster ServiceAccount instead of chart-level kubeconfig Secret configuration.

## Profile Storage

The shared Hermes data root defaults to `/opt/data` and is backed by a PVC when `persistence.enabled=true`. The runtime manager resolves per-user Hermes homes under `runtimeManager.usersRoot` (default `/opt/data/users`) and starts each run in a worker subprocess with a fixed `HERMES_HOME`.

For KubeBlocks Cloud integration, the apiserver should still own tenant authorization, business persistence, and expose only the `/ai-agent/...` business API. Hermes user homes, runtime manager API keys, and raw runtime endpoints should remain internal implementation details.
