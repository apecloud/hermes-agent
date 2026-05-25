# Hermes Agent Helm Chart

This chart deploys Hermes Agent as an in-cluster runtime API server. It is intended for a Cloud or product backend to call internally; browsers should not connect directly to Hermes.

## What It Installs

- `Deployment` running `hermes gateway run` with the structured API server enabled.
- `Service` exposing the API server inside the cluster.
- `Secret` for the API server bearer key and optional model API key.
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

The chart generates a random API server key when `apiServer.apiKey` is empty and no existing Secret is supplied. For production, provide `apiServer.existingSecret` or set a strong generated key through your secret manager.

## Existing Secrets

```yaml
apiServer:
  existingSecret: hermes-agent-api
  existingSecretKey: api-server-key

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

`HERMES_HOME` defaults to `/opt/data` and is backed by a PVC when `persistence.enabled=true`. Hermes profiles, sessions, skills, memory, and workspace state live under this directory.

For KubeBlocks Cloud integration, the apiserver should still own tenant authorization and expose only the `/ai-agent/...` business API. Hermes profile names, API keys, and raw runtime endpoints should remain internal implementation details.
