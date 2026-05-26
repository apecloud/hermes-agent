# Hermes Agent Helm Chart

This chart deploys Hermes Agent as an in-cluster runtime manager. It is intended for a Cloud or product backend to call internally; browsers should not connect directly to Hermes.

## What It Installs

- `Deployment` running `hermes-runtime-manager`, a thin HTTP/SSE service that starts short-lived Hermes worker subprocesses.
- `Service` exposing the runtime manager inside the cluster.
- `Secret` for the runtime manager bearer key and optional model API key.
- Optional `PersistentVolumeClaim` mounted as `HERMES_HOME`.
- Optional kubeconfig Secret mount for Kubernetes diagnostics.
- Optional ServiceAccount/RBAC wiring.

## Minimal Install

```bash
helm install hermes-agent ./charts/hermes-agent \
  --set image.repository=apecloud/hermes-agent \
  --set image.tag=latest \
  --set model.provider=custom \
  --set model.name=qwen-plus \
  --set model.baseURL=https://example.com/v1 \
  --set model.apiKey=replace-me
```

The chart generates a random runtime manager API key when `runtimeManager.apiKey` is empty and no existing Secret is supplied. For production, provide `runtimeManager.existingSecret` or set a strong generated key through your secret manager.

## Existing Secrets

```yaml
runtimeManager:
  existingSecret: hermes-agent-api
  existingSecretKey: runtime-manager-api-key

model:
  existingSecret: hermes-agent-model
  existingSecretKey: api-key
```

## Kubeconfig

For Kubernetes diagnostics, either enable ServiceAccount/RBAC or mount a kubeconfig Secret:

```yaml
kubeconfig:
  enabled: true
  existingSecret: hermes-kubeconfig
  key: config
```

The chart sets `KUBECONFIG=/opt/data/.kube/config`.

## Profile Storage

The shared Hermes data root defaults to `/opt/data` and is backed by a PVC when `persistence.enabled=true`. The runtime manager resolves per-user Hermes homes under `runtimeManager.usersRoot` (default `/opt/data/users`) and starts each run in a worker subprocess with a fixed `HERMES_HOME`.

For KubeBlocks Cloud integration, the apiserver should still own tenant authorization, business persistence, and expose only the `/ai-agent/...` business API. Hermes user homes, runtime manager API keys, and raw runtime endpoints should remain internal implementation details.
