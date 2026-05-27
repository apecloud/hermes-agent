---
name: kubeblocks-k8s-diagnosis
description: Use when diagnosing KubeBlocks, Kubernetes, or database operation issues in scoped business clusters.
version: 0.3.0
author: KubeBlocks AI Ops Team
license: Apache-2.0
metadata:
  hermes:
    tags:
      - kubeblocks
      - kubernetes
      - database
      - diagnosis
      - operations
    related_skills: []
---

# KubeBlocks K8s Diagnosis

Use this skill when the user asks about abnormal KubeBlocks, Kubernetes, or database operations behavior in a Cloud-provided business environment.

## Boundaries

- Stay within the injected organization, cluster, namespace, and object contexts.
- If the request is unrelated to KubeBlocks, Kubernetes, or database operations diagnosis, refuse briefly and do not use tools.
- If required context is missing, ask for the missing cluster, namespace, Pod, component, instance, backup, log, event, or performance scope.
- Do not expose Secret values, kubeconfig contents, model/API keys, credentials, approval tokens, or hidden runtime details.
- Do not run mutation/destructive commands unless Runtime Manager requests approval and the user approves through Cloud.

## When To Use

- Pod is not Ready, Pending, CrashLoopBackOff, restarted, or unavailable.
- KubeBlocks cluster/component/instance status is abnormal.
- Recent events or logs show errors or suspicious patterns.
- Backup or restore is abnormal.
- Database service connectivity, endpoint, capacity, performance, or storage behavior is abnormal.
- User asks for a cluster health explanation in a scoped business environment.

## Diagnostic Workflow

1. Establish target and context.
   - Identify the org, cluster, namespace, target object, and user goal from the injected contexts and user message.
   - If multiple contexts are present, keep findings separated by context.

2. Discover relevant resources if needed.
   - Use Kubernetes discovery only when resource kinds are unclear.
   - Useful patterns include `kubectl api-resources | grep -i kubeblocks` and namespaced resource listing.

3. Inspect Kubernetes status.
   - Pod overview: `kubectl -n <namespace> get pods -o wide`
   - Target Pod detail: `kubectl -n <namespace> describe pod <pod>`
   - Events: `kubectl -n <namespace> get events --sort-by=.lastTimestamp`
   - Container logs: `kubectl -n <namespace> logs <pod> -c <container> --tail=200 --since=1h`
   - Services/endpoints if readiness or connectivity is involved.
   - PVCs/nodes/resource pressure if scheduling, storage, or capacity is involved.
   - `kubectl debug` is higher risk and should only be used when Runtime Manager/Hermes asks for dangerous-command approval and the user approves through Cloud.

4. Inspect KubeBlocks resources.
   - Check KubeBlocks Cluster, Component, Instance, Backup, or related CR status when available.
   - Use `kubectl -n <namespace> get <resource> <name> -o yaml` or `describe` with bounded output when relevant.
   - Use `kbcli` if it is installed and useful in the runtime environment.

5. Correlate observations.
   - Do not conclude from one field when status, events, logs, and KubeBlocks resource state disagree.
   - Distinguish Kubernetes phase from readiness and service availability.
   - Treat missing logs/events/status fields as uncertainty, not success.

6. Produce a grounded answer.
   - Summarize observations.
   - State likely cause only when supported.
   - Mark confidence high/medium/low.
   - Recommend the next targeted diagnostic check or approved operation.

## Common Checks

- Pod not ready: phase, readiness, conditions, container states, restart count, owner references, node, recent events, recent container logs.
- Pending: scheduling events, node/resource pressure, PVC binding, image pull, taints/tolerations.
- CrashLoop or restarts: current/previous logs, last state, exit code, restart count, OOMKilled, liveness/readiness probe events.
- Component or instance abnormal: KubeBlocks status, related Pods, events, PVCs, Services, endpoints, and logs.
- Backup abnormal: backup CR/job status, events, controller logs, storage/credential symptoms, database-side errors.
- Connectivity issue: Service, Endpoints/EndpointSlices, Pod readiness, network policy if visible, listener/port state if available.

## Output Checklist

- Diagnosis summary.
- Observations grouped by context and source.
- Likely cause or explicit uncertainty.
- Confidence and reason.
- Recommended next step.
