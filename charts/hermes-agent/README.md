# Hermes Agent Helm Chart

This chart deploys Hermes Agent as an in-cluster runtime manager. It is intended for a Cloud or product backend to call internally; browsers should not connect directly to Hermes.

## What It Installs

- `Deployment` running `hermes-runtime-manager`, a thin HTTP/SSE service that starts short-lived Hermes worker subprocesses.
- `Service` exposing the runtime manager inside the cluster.
- `Secret` for the runtime manager bearer key.
- Optional `PersistentVolumeClaim` mounted as `HERMES_HOME`.
- Optional kubeconfig Secret mount for Kubernetes diagnostics.
- `ServiceAccount` bound to the existing `apecloud-cluster-admin` ClusterRole.

## Minimal Install

```bash
helm install hermes-agent ./charts/hermes-agent \
  --set image.repository=apecloud/hermes-agent \
  --set image.tag=latest
```

The chart generates a random runtime manager API key when `runtimeManager.apiKey` is empty and no existing Secret is supplied. For production, provide `runtimeManager.existingSecret` or set a strong generated key through your secret manager.

LLM provider/model/baseURL/API-key configuration is intentionally not configured in this chart. The product backend should resolve the current user's selected model configuration for each conversation and pass it to Runtime Manager when starting a run, using `model` plus `llm_config` or the equivalent per-run fields.

## Existing Secrets

```yaml
runtimeManager:
  existingSecret: hermes-agent-api
  existingSecretKey: runtime-manager-api-key
```

## Kubernetes Access

The chart always binds the runtime `ServiceAccount` to the existing `apecloud-cluster-admin` ClusterRole and enables service account token mounting by default. If an environment still requires an explicit kubeconfig, mount it as a Secret:

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
