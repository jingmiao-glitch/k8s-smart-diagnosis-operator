"""
K8s 集群查询工具模块（11 个纯函数）

从 k8s-query-mcp.py 搬移而来，去掉 FastMCP 装饰器，保留 Annotated 类型注解
以便 FastMCP.add_tool() 自动读取参数 schema。

K8s 客户端初始化：优先集群内 incluster_config（Pod 内 ServiceAccount），
降级到本地 kubeconfig（开发环境）。
"""

from typing import Annotated

from pydantic import Field
from kubernetes import client, config

try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()


def _get_core_v1() -> client.CoreV1Api:
    return client.CoreV1Api()


def _get_apps_v1() -> client.AppsV1Api:
    return client.AppsV1Api()


def _parse_cpu_millicores(cpu_str: str) -> float:
    cpu_str = cpu_str.strip()
    if cpu_str.endswith("n"):
        return float(cpu_str[:-1]) / 1_000_000
    elif cpu_str.endswith("u"):
        return float(cpu_str[:-1]) / 1_000
    elif cpu_str.endswith("m"):
        return float(cpu_str[:-1])
    else:
        return float(cpu_str) * 1000


def _parse_memory_mebibytes(mem_str: str) -> float:
    mem_str = mem_str.strip()
    if mem_str.endswith("Ki"):
        return float(mem_str[:-2]) / 1024
    elif mem_str.endswith("Mi"):
        return float(mem_str[:-2])
    elif mem_str.endswith("Gi"):
        return float(mem_str[:-2]) * 1024
    elif mem_str.endswith("Ti"):
        return float(mem_str[:-2]) * 1024 * 1024
    elif mem_str.endswith("k"):
        return float(mem_str[:-1]) / 1000
    elif mem_str.endswith("M"):
        return float(mem_str[:-1])
    elif mem_str.endswith("G"):
        return float(mem_str[:-1]) * 1000
    elif mem_str.endswith("T"):
        return float(mem_str[:-1]) * 1000 * 1000
    else:
        return float(mem_str) / (1024 * 1024)


def get_nodes(
    wide: Annotated[bool, Field(description="是否显示详细信息（内网IP、OS镜像、内核版本、容器运行时）")] = False,
) -> str:
    """获取集群中所有节点的状态信息

    参数：
      wide  是否显示详细信息（默认 False）

    基础列：
      - 节点名称 / 就绪状态 / 节点角色 / kubelet 版本 / 可分配 CPU / 可分配内存

    wide=True 时增加：
      - 内网 IP / OS 镜像 / 内核版本 / 容器运行时
    """
    v1 = _get_core_v1()
    nodes = v1.list_node()

    lines = []
    for node in nodes.items:
        name = node.metadata.name
        ready = "Unknown"
        for condition in node.status.conditions:
            if condition.type == "Ready":
                ready = "Ready" if condition.status == "True" else "NotReady"
                break

        roles = []
        if node.metadata.labels:
            for key in node.metadata.labels:
                if key.startswith("node-role.kubernetes.io/"):
                    roles.append(key.split("/")[-1])
        role = ",".join(roles) if roles else "<none>"

        kubelet_version = (
            node.status.node_info.kubelet_version
            if node.status.node_info
            else "N/A"
        )
        cpu = node.status.allocatable.get("cpu", "N/A")
        memory = node.status.allocatable.get("memory", "N/A")

        if wide:
            internal_ip = "N/A"
            for addr in node.status.addresses or []:
                if addr.type == "InternalIP":
                    internal_ip = addr.address
                    break
            os_image = (
                node.status.node_info.os_image if node.status.node_info else "N/A"
            )
            kernel = (
                node.status.node_info.kernel_version
                if node.status.node_info
                else "N/A"
            )
            runtime = (
                node.status.node_info.container_runtime_version
                if node.status.node_info
                else "N/A"
            )
            lines.append(
                f"{name:<24} {ready:<10} {role:<16} {kubelet_version:<12} "
                f"{cpu:<6} {memory:<10} {internal_ip:<16} {os_image:<24} "
                f"{kernel:<20} {runtime}"
            )
        else:
            lines.append(
                f"{name:<32} {ready:<10} {role:<20} {kubelet_version:<12} {cpu:<6} {memory}"
            )

    if wide:
        header = (
            f"{'NAME':<24} {'STATUS':<10} {'ROLES':<16} {'VERSION':<12} "
            f"{'CPU':<6} {'MEMORY':<10} {'INTERNAL-IP':<16} {'OS-IMAGE':<24} "
            f"{'KERNEL':<20} {'RUNTIME'}"
        )
    else:
        header = (
            f"{'NAME':<32} {'STATUS':<10} {'ROLES':<20} "
            f"{'VERSION':<12} {'CPU':<6} {'MEMORY'}"
        )
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else "未找到任何节点。"


def get_pods(
    namespace: Annotated[str, Field(description="目标命名空间，例如 default、kube-system 等")] = "default",
    wide: Annotated[bool, Field(description="是否显示 Pod IP 和所在节点（-o wide 效果）")] = False,
) -> str:
    """获取指定命名空间下的所有 Pod 及其状态

    参数：
      namespace  命名空间（默认：default）
      wide       是否显示 Pod IP 和所在节点（默认 False）
    """
    v1 = _get_core_v1()
    pods = v1.list_namespaced_pod(namespace)

    lines = []
    for pod in pods.items:
        name = pod.metadata.name
        phase = pod.status.phase if pod.status.phase else "Unknown"

        ready_count = 0
        total_containers = 0
        restarts = 0
        if pod.status.container_statuses:
            total_containers = len(pod.status.container_statuses)
            for cs in pod.status.container_statuses:
                if cs.ready:
                    ready_count += 1
                restarts += cs.restart_count

        ready_str = f"{ready_count}/{total_containers}" if total_containers > 0 else "N/A"
        created = pod.metadata.creation_timestamp.strftime("%m-%d %H:%M") if pod.metadata.creation_timestamp else "N/A"

        if wide:
            pod_ip = pod.status.pod_ip or "N/A"
            node_name = pod.spec.node_name or "N/A"
            lines.append(
                f"{name:<48} {phase:<12} {ready_str:<8} {restarts:<6} "
                f"{pod_ip:<16} {node_name:<20} {created}"
            )
        else:
            lines.append(
                f"{name:<48} {phase:<12} {ready_str:<8} {restarts:<6} {created}"
            )

    if wide:
        header = (
            f"{'NAME':<48} {'PHASE':<12} {'READY':<8} {'RESTARTS':<6} "
            f"{'POD-IP':<16} {'NODE':<20} {'CREATED'}"
        )
    else:
        header = (
            f"{'NAME':<48} {'PHASE':<12} {'READY':<8} {'RESTARTS':<6} {'CREATED'}"
        )
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else f"命名空间 [{namespace}] 下未找到任何 Pod。"


def describe_pod(
    name: Annotated[str, Field(description="Pod 名称，必填。例如 nginx-pod-xxx、coredns-bbdc5fdf6-xxxxx")],
    namespace: Annotated[str, Field(description="Pod 所在的命名空间，默认 default")] = "default",
) -> str:
    """获取单个 Pod 的详细描述信息

    参数：
      name       Pod 名称（必填）
      namespace  命名空间（默认：default）
    """
    v1 = _get_core_v1()
    pod = v1.read_namespaced_pod(name, namespace)

    lines = []
    lines.append(f"名称：         {pod.metadata.name}")
    lines.append(f"命名空间：     {pod.metadata.namespace}")
    lines.append(f"UID：          {pod.metadata.uid}")
    lines.append(f"节点：         {pod.spec.node_name or 'N/A'}")
    if pod.metadata.creation_timestamp:
        lines.append(f"创建时间：     {pod.metadata.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    labels = pod.metadata.labels or {}
    if labels:
        label_str = ", ".join(f"{k}={v}" for k, v in labels.items())
        lines.append(f"标签：         {label_str}")

    lines.append("")
    lines.append(f"状态阶段：     {pod.status.phase or 'Unknown'}")
    lines.append(f"Pod IP：       {pod.status.pod_ip or 'N/A'}")
    lines.append(f"主机 IP：      {pod.status.host_ip or 'N/A'}")

    if pod.status.container_statuses:
        lines.append("")
        lines.append("容器状态：")
        for i, cs in enumerate(pod.status.container_statuses):
            if cs.state.running:
                state = "Running"
            elif cs.state.waiting:
                state = f"Waiting({cs.state.waiting.reason})"
            elif cs.state.terminated:
                state = f"Terminated({cs.state.terminated.reason})"
            else:
                state = "unknown"
            lines.append(
                f"  [{i + 1}] {cs.name:<20} "
                f"{'Ready' if cs.ready else 'NotReady':<10} "
                f"重启 {cs.restart_count:<3} 次  状态：{state}"
            )
    else:
        lines.append("")
        lines.append("无容器状态信息。")

    if pod.spec.containers:
        lines.append("")
        lines.append("容器镜像：")
        for c in pod.spec.containers:
            lines.append(f"  - {c.name:<20} {c.image}")

    return "\n".join(lines)


def get_events(namespace: Annotated[str, Field(description="目标命名空间，例如 default、kube-system 等")] = "default") -> str:
    """获取指定命名空间的事件列表

    参数：
      namespace  命名空间（默认：default）
    """
    v1 = _get_core_v1()
    events = v1.list_namespaced_event(namespace)

    lines = []
    for ev in events.items:
        ev_type = ev.type or "Normal"
        reason = ev.reason or "N/A"
        kind = ev.involved_object.kind if ev.involved_object else "N/A"
        obj_name = ev.involved_object.name if ev.involved_object else "N/A"
        message = (ev.message or "")[:60]
        count = ev.count or 1
        last_ts = ev.last_timestamp or ev.event_time or "N/A"
        if hasattr(last_ts, "strftime"):
            last_ts = last_ts.strftime("%m-%d %H:%M:%S")
        lines.append(
            f"{ev_type:<10} {reason:<22} {kind:<12} "
            f"{obj_name:<32} {count:<6} {last_ts:<20} {message}"
        )

    header = (
        f"{'TYPE':<10} {'REASON':<22} {'KIND':<12} "
        f"{'NAME':<32} {'COUNT':<6} {'LAST-TS':<20} {'MESSAGE'}"
    )
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else f"命名空间 [{namespace}] 下暂无事件。"


def get_namespaces() -> str:
    """获取集群中所有命名空间的列表"""
    v1 = _get_core_v1()
    nss = v1.list_namespace()

    lines = []
    for ns in nss.items:
        name = ns.metadata.name
        phase = ns.status.phase if ns.status.phase else "Active"
        created = (
            ns.metadata.creation_timestamp.strftime("%Y-%m-%d %H:%M")
            if ns.metadata.creation_timestamp
            else "N/A"
        )
        lines.append(f"{name:<40} {phase:<15} {created}")

    header = f"{'NAME':<40} {'STATUS':<15} {'CREATED'}"
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else "未找到任何命名空间。"


def describe_node(
    name: Annotated[str, Field(description="节点名称，必填。例如 ai-agent、worker-1 等")],
) -> str:
    """获取指定节点的详细信息

    参数：
      name  节点名称（必填）
    """
    v1 = _get_core_v1()
    node = v1.read_node(name)

    lines = []
    lines.append(f"名称：             {node.metadata.name}")
    lines.append(f"UID：              {node.metadata.uid}")

    for addr in node.status.addresses or []:
        if addr.type == "Hostname":
            lines.append(f"主机名：           {addr.address}")
        elif addr.type == "InternalIP":
            lines.append(f"内网 IP：          {addr.address}")
        elif addr.type == "ExternalIP":
            lines.append(f"外网 IP：          {addr.address}")

    if node.metadata.creation_timestamp:
        lines.append(f"创建时间：         {node.metadata.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("")
    lines.append("--- 系统信息 ---")
    ni = node.status.node_info
    if ni:
        lines.append(f"OS 镜像：          {ni.os_image or 'N/A'}")
        lines.append(f"操作系统：         {ni.operating_system or 'N/A'}")
        lines.append(f"系统架构：         {ni.architecture or 'N/A'}")
        lines.append(f"内核版本：         {ni.kernel_version or 'N/A'}")
        lines.append(f"容器运行时：       {ni.container_runtime_version or 'N/A'}")
        lines.append(f"kubelet 版本：     {ni.kubelet_version or 'N/A'}")
        lines.append(f"kube-proxy 版本：  {ni.kube_proxy_version or 'N/A'}")

    lines.append("")
    lines.append("--- 资源容量 ---")
    cap = node.status.capacity or {}
    alloc = node.status.allocatable or {}
    for key in ["cpu", "memory", "pods", "ephemeral-storage"]:
        cap_val = cap.get(key, "N/A")
        alloc_val = alloc.get(key, "N/A")
        lines.append(f"  {key:<20}  容量：{str(cap_val):<12}  可分配：{alloc_val}")

    lines.append("")
    lines.append("--- 节点状态（Conditions） ---")
    for cond in node.status.conditions or []:
        lines.append(f"  {cond.type:<20} {cond.status:<10} 原因：{cond.reason or 'N/A'}  消息：{cond.message or 'N/A'}")

    lines.append("")
    lines.append("--- Taints（污点） ---")
    if node.spec.taints:
        for t in node.spec.taints:
            effect = t.effect or "N/A"
            value = t.value or ""
            line = f"  {t.key}={value}  Effect: {effect}"
            if t.time_added:
                line += f"  TimeAdded: {t.time_added}"
            lines.append(line)
    else:
        lines.append("  (无)")

    labels = node.metadata.labels or {}
    if labels:
        lines.append("")
        lines.append("--- 关键标签 ---")
        shown = 0
        for k, v in sorted(labels.items()):
            if k.startswith("node-role.kubernetes.io/") or k.startswith("kubernetes.io/") or k.startswith("kubeadm.alpha"):
                lines.append(f"  {k}={v}")
                shown += 1
        if shown == 0:
            for k, v in sorted(labels.items()):
                lines.append(f"  {k}={v}")

    annotations = node.metadata.annotations or {}
    if annotations:
        lines.append("")
        lines.append("--- 关键注解 ---")
        shown = 0
        for k, v in sorted(annotations.items()):
            if any(kw in k for kw in ["kubeadm", "projectcalico", "cni", "flannel"]):
                lines.append(f"  {k}")
                shown += 1
        if shown == 0:
            for k in sorted(annotations.keys())[:10]:
                lines.append(f"  {k}")

    return "\n".join(lines)


def get_workloads(
    kind: Annotated[str, Field(description="控制器类型：deployment / statefulset / daemonset / all")],
    namespace: Annotated[str, Field(description="目标命名空间，例如 default、kube-system 等")] = "default",
    wide: Annotated[bool, Field(description="是否显示镜像和创建时间")] = False,
) -> str:
    """查询指定命名空间下的控制器（Deployment / StatefulSet / DaemonSet）

    参数：
      kind      控制器类型：deployment / statefulset / daemonset / all
      namespace 命名空间（默认：default）
      wide      是否显示镜像和创建时间（默认 False）
    """
    apps = _get_apps_v1()

    kind = kind.lower().strip()
    valid_kinds = {"deployment", "statefulset", "daemonset", "all"}
    if kind not in valid_kinds:
        return f"不支持的控制器类型。支持：deployment、statefulset、daemonset、all"

    all_items = []

    if kind in ("deployment", "all"):
        for d in apps.list_namespaced_deployment(namespace).items:
            images = [c.image for c in d.spec.template.spec.containers]
            all_items.append({
                "type": "Deployment",
                "name": d.metadata.name,
                "desired": d.spec.replicas or 0,
                "current": d.status.replicas or 0,
                "ready": d.status.ready_replicas or 0,
                "available": d.status.available_replicas or 0,
                "image": images[0] if images else "N/A",
                "created": d.metadata.creation_timestamp,
            })

    if kind in ("statefulset", "all"):
        for s in apps.list_namespaced_stateful_set(namespace).items:
            images = [c.image for c in s.spec.template.spec.containers]
            all_items.append({
                "type": "StatefulSet",
                "name": s.metadata.name,
                "desired": s.spec.replicas or 0,
                "current": s.status.replicas or 0,
                "ready": s.status.ready_replicas or 0,
                "available": s.status.available_replicas or 0,
                "image": images[0] if images else "N/A",
                "created": s.metadata.creation_timestamp,
            })

    if kind in ("daemonset", "all"):
        for d in apps.list_namespaced_daemon_set(namespace).items:
            images = [c.image for c in d.spec.template.spec.containers]
            all_items.append({
                "type": "DaemonSet",
                "name": d.metadata.name,
                "desired": d.status.desired_number_scheduled or 0,
                "current": d.status.current_number_scheduled or 0,
                "ready": d.status.number_ready or 0,
                "available": d.status.number_available or 0,
                "image": images[0] if images else "N/A",
                "created": d.metadata.creation_timestamp,
            })

    lines = []
    for item in all_items:
        if wide:
            created = item["created"].strftime("%m-%d %H:%M") if item["created"] else "N/A"
            lines.append(
                f"{item['type']:<12} {item['name']:<36} "
                f"{item['desired']:<5} {item['current']:<5} {item['ready']:<5} {item['available']:<5} "
                f"{item['image']:<50} {created}"
            )
        else:
            lines.append(
                f"{item['type']:<12} {item['name']:<40} "
                f"{item['desired']:<5} {item['current']:<5} {item['ready']:<5} {item['available']:<5}"
            )

    if not lines:
        return f"命名空间 [{namespace}] 下未找到 {kind} 类型的控制器。"

    if wide:
        header = (
            f"{'KIND':<12} {'NAME':<36} {'DES':<5} {'CUR':<5} {'RDY':<5} "
            f"{'AVAIL':<5} {'IMAGE':<50} {'CREATED'}"
        )
    else:
        header = (
            f"{'KIND':<12} {'NAME':<40} {'DES':<5} {'CUR':<5} {'RDY':<5} {'AVAIL':<5}"
        )
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines)


def describe_controller(
    kind: Annotated[str, Field(description="控制器类型：deployment / statefulset / daemonset")],
    name: Annotated[str, Field(description="控制器名称，必填。例如 nginx、coredns 等")],
    namespace: Annotated[str, Field(description="控制器所在的命名空间，默认 default")] = "default",
) -> str:
    """获取指定控制器的详细信息

    参数：
      kind      控制器类型：deployment / statefulset / daemonset（必填）
      name      控制器名称（必填）
      namespace 命名空间（默认：default）
    """
    apps = _get_apps_v1()

    kind = kind.lower().strip()
    if kind not in ("deployment", "statefulset", "daemonset"):
        return "不支持的控制器类型。支持：deployment、statefulset、daemonset"

    lines = []
    lines.append(f"Kind：            {kind.capitalize()}")
    lines.append(f"名称：            {name}")
    lines.append(f"命名空间：        {namespace}")

    try:
        if kind == "deployment":
            obj = apps.read_namespaced_deployment(name, namespace)
            lines.append(f"策略：            {obj.spec.strategy.type if obj.spec.strategy else 'RollingUpdate'}")
            if obj.spec.strategy and obj.spec.strategy.rolling_update:
                ru = obj.spec.strategy.rolling_update
                lines.append(f"滚动更新：        最大不可用={ru.max_unavailable or '25%'}  最大超卖={ru.max_surge or '25%'}")
            lines.append("")
            lines.append(f"期望副本：        {obj.spec.replicas or 0}")
            lines.append(f"当前副本：        {obj.status.replicas or 0}")
            lines.append(f"就绪副本：        {obj.status.ready_replicas or 0}")
            lines.append(f"可用副本：        {obj.status.available_replicas or 0}")
            lines.append(f"更新副本：        {obj.status.updated_replicas or 0}")
        elif kind == "statefulset":
            obj = apps.read_namespaced_stateful_set(name, namespace)
            lines.append(f"服务名：          {obj.spec.service_name or 'N/A'}")
            lines.append(f"更新策略：        {obj.spec.update_strategy.type if obj.spec.update_strategy else 'RollingUpdate'}")
            lines.append("")
            lines.append(f"期望副本：        {obj.spec.replicas or 0}")
            lines.append(f"当前副本：        {obj.status.replicas or 0}")
            lines.append(f"就绪副本：        {obj.status.ready_replicas or 0}")
            lines.append(f"当前版本副本：    {obj.status.current_replicas or 0}")
            lines.append(f"更新版本副本：    {obj.status.updated_replicas or 0}")
        elif kind == "daemonset":
            obj = apps.read_namespaced_daemon_set(name, namespace)
            lines.append(f"更新策略：        {obj.spec.update_strategy.type if obj.spec.update_strategy else 'RollingUpdate'}")
            lines.append("")
            lines.append(f"期望调度数：      {obj.status.desired_number_scheduled or 0}")
            lines.append(f"当前运行数：      {obj.status.current_number_scheduled or 0}")
            lines.append(f"就绪数：          {obj.status.number_ready or 0}")
            lines.append(f"可用数：          {obj.status.number_available or 0}")
            lines.append(f"更新数：          {obj.status.updated_number_scheduled or 0}")
    except client.exceptions.ApiException as e:
        return f"获取控制器详情失败：{e.reason or 'API 错误'}（{e.status}）"

    container = obj.spec.template.spec.containers[0] if obj.spec.template.spec.containers else None

    if obj.metadata.labels:
        lines.append("")
        lines.append("标签：")
        for k, v in obj.metadata.labels.items():
            lines.append(f"  {k}={v}")

    if obj.spec.selector and obj.spec.selector.match_labels:
        lines.append("")
        lines.append("Pod 选择器：")
        for k, v in obj.spec.selector.match_labels.items():
            lines.append(f"  {k}={v}")

    lines.append("")
    if container:
        lines.append("--- 容器模板 ---")
        lines.append(f"  名称：          {container.name}")
        lines.append(f"  镜像：          {container.image}")
        if container.ports:
            lines.append(f"  端口：")
            for p in container.ports:
                lines.append(f"    {p.name or '<unnamed>'}:{p.container_port}/{p.protocol or 'TCP'}")
        if container.resources:
            req = container.resources.requests or {}
            lim = container.resources.limits or {}
            lines.append(f"  资源限制：")
            for key in ["cpu", "memory"]:
                if key in lim:
                    lines.append(f"    {key}: 请求={req.get(key, 'N/A')}  限制={lim.get(key, 'N/A')}")
        if container.env:
            lines.append(f"  环境变量（前 10 个）：")
            for env in container.env[:10]:
                val = env.value or (env.value_from and "from:secret/configmap" or "N/A")
                lines.append(f"    {env.name}={val}")
        if container.volume_mounts:
            lines.append(f"  挂载卷：")
            for vm in container.volume_mounts:
                lines.append(f"    {vm.name} -> {vm.mount_path}  (只读={str(vm.read_only or False)})")

    if obj.spec.template.spec.volumes:
        lines.append("")
        lines.append("--- 卷（Volumes） ---")
        for v in obj.spec.template.spec.volumes:
            info_parts = [v.name]
            if v.config_map:
                info_parts.append(f"ConfigMap({v.config_map.name})")
            elif v.secret:
                info_parts.append(f"Secret({v.secret.secret_name})")
            elif v.persistent_volume_claim:
                info_parts.append(f"PVC({v.persistent_volume_claim.claim_name})")
            elif v.host_path:
                info_parts.append(f"HostPath({v.host_path.path})")
            elif v.empty_dir:
                info_parts.append("EmptyDir")
            lines.append(f"  {' | '.join(info_parts)}")

    if obj.metadata.creation_timestamp:
        lines.append("")
        lines.append(f"创建时间：        {obj.metadata.creation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


def logs(
    pod_name: Annotated[str, Field(description="目标 Pod 名称，必填。例如 nginx-xxx、coredns-bbdc5fdf6-xxxxx")],
    namespace: Annotated[str, Field(description="Pod 所在的命名空间，默认 default")] = "default",
    container: Annotated[str, Field(description="容器名称（可选）。对于多容器 Pod，可指定查看哪个容器的日志")] = "",
    tail_lines: Annotated[int, Field(description="返回日志的末尾行数，填 -1 表示返回全部日志")] = 100,
    previous: Annotated[bool, Field(description="是否查看上一轮已退出容器的日志（类似 kubectl logs -p）")] = False,
) -> str:
    """获取指定 Pod 的容器日志

    参数：
      pod_name    Pod 名称（必填）
      namespace   命名空间（默认：default）
      container   容器名称（可选，多容器 Pod 必填）
      tail_lines  返回日志的末尾行数（默认：100，-1 表示全部）
      previous    是否查看崩溃退出的上一轮日志（默认 False）
    """
    v1 = _get_core_v1()

    resolved_container = container
    if not resolved_container:
        pod = v1.read_namespaced_pod(pod_name, namespace)
        if pod.spec.containers:
            resolved_container = pod.spec.containers[0].name

    log_kwargs = {
        "name": pod_name,
        "namespace": namespace,
    }
    if tail_lines and tail_lines > 0:
        log_kwargs["tail_lines"] = tail_lines
    if resolved_container:
        log_kwargs["container"] = resolved_container
    if previous:
        log_kwargs["previous"] = True

    try:
        log_data = v1.read_namespaced_pod_log(**log_kwargs)
    except client.exceptions.ApiException as e:
        return f"获取日志失败：{e.reason or 'API 错误'}（{e.status}）"

    if not log_data or log_data.strip() == "":
        return f"Pod [{pod_name}] 在当前尚无日志输出。"

    header_info = f"===== Pod: {pod_name} | 容器: {resolved_container} | 命名空间: {namespace}"
    if tail_lines and tail_lines > 0:
        header_info += f" | 显示末 {tail_lines} 行"
    if previous:
        header_info += " | 上一轮日志"

    return f"{header_info} =====\n\n{log_data.rstrip()}"


def top_node() -> str:
    """查看集群所有节点的 CPU / 内存使用情况

    依赖集群已安装 metrics-server，返回每个节点的：
      - CPU 使用量（cores）和使用率
      - 内存使用量（Mi）和使用率
    """
    v1 = _get_core_v1()

    try:
        metrics = client.CustomObjectsApi().list_cluster_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            plural="nodes",
        )
    except client.exceptions.ApiException as e:
        return f"获取节点指标失败，请确认 metrics-server 已安装：{e.reason or 'API 错误'}（{e.status}）"

    nodes_meta = {}
    for n in v1.list_node().items:
        cpu_cap = n.status.capacity.get("cpu", "0")
        mem_cap = n.status.capacity.get("memory", "0")
        nodes_meta[n.metadata.name] = {"cpu": cpu_cap, "memory": mem_cap}

    lines = []
    for item in metrics.get("items", []):
        name = item["metadata"]["name"]
        usage = item["usage"]
        cpu_usage = usage.get("cpu", "0")
        mem_usage = usage.get("memory", "0")

        cpu_m = _parse_cpu_millicores(cpu_usage)
        mem_mi = _parse_memory_mebibytes(mem_usage)

        meta = nodes_meta.get(name, {"cpu": "0", "memory": "0"})
        cpu_total_m = _parse_cpu_millicores(meta["cpu"])
        mem_total_mi = _parse_memory_mebibytes(meta["memory"])

        cpu_pct = f"{cpu_m / cpu_total_m * 100:.1f}%" if cpu_total_m > 0 else "N/A"
        mem_pct = f"{mem_mi / mem_total_mi * 100:.1f}%" if mem_total_mi > 0 else "N/A"

        lines.append(
            f"{name:<24} {cpu_m:>6}m ({cpu_pct:<6}) "
            f"{mem_mi:>8}Mi ({mem_pct:<6})"
        )

    header = f"{'NODE':<24} {'CPU(cores)':<18} {'MEMORY(bytes)':<20}"
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else "未获取到节点指标数据，请确认 metrics-server 正常运行。"


def top_pod(
    namespace: Annotated[str, Field(description="目标命名空间，例如 default、kube-system 等")] = "default",
) -> str:
    """查看指定命名空间下所有 Pod 的 CPU / 内存使用量

    依赖集群已安装 metrics-server。

    参数：
      namespace  命名空间（默认：default）
    """
    try:
        metrics = client.CustomObjectsApi().list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
    except client.exceptions.ApiException as e:
        return f"获取 Pod 指标失败，请确认 metrics-server 已安装且命名空间正确：{e.reason or 'API 错误'}（{e.status}）"

    lines = []
    for item in metrics.get("items", []):
        pod_name = item["metadata"]["name"]
        containers = item.get("containers", [])
        for c in containers:
            usage = c.get("usage", {})
            cpu_usage = usage.get("cpu", "0")
            mem_usage = usage.get("memory", "0")
            cpu_m = _parse_cpu_millicores(cpu_usage)
            mem_mi = _parse_memory_mebibytes(mem_usage)
            lines.append(
                f"{pod_name:<48} {c['name']:<20} "
                f"{cpu_m:>6}m  {mem_mi:>8}Mi"
            )

    header = f"{'POD':<48} {'CONTAINER':<20} {'CPU':>6}  {'MEMORY':>8}"
    sep = "-" * len(header)
    return f"{header}\n{sep}\n" + "\n".join(lines) if lines else f"命名空间 [{namespace}] 下未找到 Pod 指标数据。"
