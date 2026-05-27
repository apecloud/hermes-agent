---
name: kubeblocks-k8s-diagnosis
description: Diagnose business Kubernetes environments and KubeBlocks clusters with kubectl, kbcli, and common network/debug tools.
---

# KubeBlocks Kubernetes Diagnosis

Use this skill when the user asks about a KubeBlocks cluster, database workload,
Kubernetes Pod/Service/StatefulSet/Deployment, backup/restore, network path,
storage, scheduling, readiness/liveness, or operational incident.

## Scope

Cloud passes the current message context in the system prompt. Treat that
context as authoritative for this run. It may include multiple clusters or
namespaces. Do not silently switch to a different cluster.

For each scoped cluster:

1. Confirm the active Kubernetes context.
2. Inspect the target namespace and resources.
3. Prefer read-only commands first.
4. Collect enough observations before giving a conclusion.

## Useful Commands

Start with a narrow inventory:

```bash
kubectl config current-context
kubectl get ns
kubectl get pods -A -o wide
kubectl get events -A --sort-by=.lastTimestamp
```

For a specific namespace or workload:

```bash
kubectl get pods -n <namespace> -o wide
kubectl describe pod -n <namespace> <pod>
kubectl logs -n <namespace> <pod> --previous --tail=200
kubectl get statefulset,deploy,svc,pvc -n <namespace>
kubectl describe pvc -n <namespace> <pvc>
```

For KubeBlocks resources:

```bash
kubectl get clusters.apps.kubeblocks.io -A
kubectl describe cluster.apps.kubeblocks.io -n <namespace> <cluster>
kubectl get opsrequests.apps.kubeblocks.io -n <namespace>
kubectl get backups.dataprotection.kubeblocks.io -n <namespace>
```

If `kbcli` is installed, use it for KubeBlocks-oriented summaries, but do not
depend on it as the only source of evidence.

## Diagnosis Pattern

Classify findings into:

- `symptom`: what the user or system observes.
- `evidence`: exact Kubernetes/KubeBlocks observations supporting the diagnosis.
- `likely cause`: the most plausible explanation.
- `uncertainty`: what is still unknown or not observable.
- `next action`: safe follow-up checks or remediation suggestions.

Prefer concise final answers. Avoid dumping long command output unless it is
necessary to explain the conclusion.
