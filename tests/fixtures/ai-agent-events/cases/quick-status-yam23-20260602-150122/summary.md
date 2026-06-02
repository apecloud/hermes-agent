# quick-status-yam23-20260602-150122

- runtimeRunId: `run_8e9c3d84f55b4ef9b0aca92339c9b62b`
- conversationId: `conv_canonical_20260602_150122`
- status: `completed`
- events: 405
- event ids: 1..405
- full capture from event id 1: True

## Event Types

- `agent.step`: 4
- `message.delta`: 124
- `reasoning.available`: 261
- `run.completed`: 1
- `run.queued`: 1
- `run.running`: 1
- `status.message`: 4
- `tool.completed`: 3
- `tool.generating`: 3
- `tool.started`: 3

## Prompt

快速检查 yam23 集群当前是否健康，只需要做必要的只读检查（Cluster、Pod、Component、PVC 或事件），最多执行 3 到 5 组命令，给出简短结论。
