"""
Prometheus 查询工具模块（6 个工具函数）

基于 prometheus-api-client==0.7.2 SDK，通过 HTTP API 查询 Prometheus Server。
Prometheus 地址通过环境变量 PROMETHEUS_URL 配置，默认集群内 DNS 地址。

6 个工具：
  1. prom_query_instant   — PromQL 即时查询
  2. prom_query_range     — PromQL 范围查询（时序数据）
  3. node_cpu_usage       — 节点 CPU 使用率趋势
  4. node_memory_usage    — 节点内存使用率趋势
  5. pod_cpu_usage        — Pod CPU 使用率趋势
  6. pod_memory_usage     — Pod 内存使用率趋势
"""

import os
from datetime import datetime, timedelta
from typing import Annotated

from pydantic import Field
from prometheus_api_client import PrometheusConnect

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "")

_prom = None


def _resolve_prometheus_url() -> str:
    """
    自动探测 Prometheus 地址：
    1. 环境变量 PROMETHEUS_URL（最高优先级）
    2. 集群内 DNS（Pod 内 CoreDNS 可用）
    3. 通过 K8s API 获取 Service ClusterIP（宿主机开发环境）
    """
    if PROMETHEUS_URL:
        return PROMETHEUS_URL

    dns_url = "http://kube-prometheus-stack-prometheus.monitoring:9090"
    try:
        import socket
        host = "kube-prometheus-stack-prometheus.monitoring"
        socket.getaddrinfo(host, 9090)
        return dns_url
    except Exception:
        pass

    try:
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        v1 = client.CoreV1Api()
        svc = v1.read_namespaced_service("kube-prometheus-stack-prometheus", "monitoring")
        cluster_ip = svc.spec.cluster_ip
        port = svc.spec.ports[0].port
        return f"http://{cluster_ip}:{port}"
    except Exception:
        return dns_url


def _get_prom() -> PrometheusConnect:
    global _prom
    if _prom is None:
        url = _resolve_prometheus_url()
        _prom = PrometheusConnect(url=url, disable_ssl=True)
    return _prom


def prom_query_instant(
    query: Annotated[str, Field(description="PromQL 查询语句，例如 up、node_memory_MemAvailable_bytes")],
) -> str:
    """PromQL 即时查询

    参数：
      query  PromQL 查询语句（必填）

    返回 JSON 格式的查询结果
    """
    import json
    try:
        prom = _get_prom()
        result = prom.custom_query(query=query)
        if not result:
            return f"查询 [{query}] 无数据返回"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"PromQL 即时查询失败：{e}"


def prom_query_range(
    query: Annotated[str, Field(description="PromQL 查询语句，例如 rate(node_cpu_seconds_total[5m])")],
    start_minutes_ago: Annotated[int, Field(description="起始时间距今多少分钟，默认 60")] = 60,
    step_seconds: Annotated[int, Field(description="采样步长（秒），默认 60")] = 60,
) -> str:
    """PromQL 范围查询（获取时序数据）

    参数：
      query             PromQL 查询语句（必填）
      start_minutes_ago 起始时间距今多少分钟（默认 60）
      step_seconds      采样步长（秒，默认 60）
    """
    import json
    try:
        prom = _get_prom()
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=start_minutes_ago)
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=f"{step_seconds}s",
        )
        if not result:
            return f"范围查询 [{query}] 无数据返回"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"PromQL 范围查询失败：{e}"


def node_cpu_usage(
    node_name: Annotated[str, Field(description="节点名称，例如 ai-agent、worker-1")],
    minutes: Annotated[int, Field(description="查询时间范围（分钟），默认 15")] = 15,
) -> str:
    """查询节点 CPU 使用率趋势

    PromQL: 100 - avg(rate(node_cpu_seconds_total{mode="idle",instance="<node>"}[5m]) off (job)) * 100

    参数：
      node_name  节点名称（必填）
      minutes    查询时间范围（分钟，默认 15）
    """
    import json
    query_str = (
        f"(1 - avg(rate(node_cpu_seconds_total{{"
        f'mode="idle",instance="{node_name}"'
        f"}}[{minutes}m]))) * 100"
    )
    try:
        prom = _get_prom()
        result = prom.custom_query(query=query_str)
        if not result:
            return f"节点 [{node_name}] 无 CPU 使用率数据"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"查询节点 CPU 使用率失败：{e}"


def node_memory_usage(
    node_name: Annotated[str, Field(description="节点名称，例如 ai-agent、worker-1")],
) -> str:
    """查询节点内存使用率

    PromQL: (1 - node_memory_MemAvailable_bytes{instance="<node>"} / node_memory_MemTotal_bytes{instance="<node>"}) * 100

    参数：
      node_name  节点名称（必填）
    """
    import json
    query_str = (
        f"(1 - node_memory_MemAvailable_bytes{{instance=\"{node_name}\"}}"
        f" / node_memory_MemTotal_bytes{{instance=\"{node_name}\"}}) * 100"
    )
    try:
        prom = _get_prom()
        result = prom.custom_query(query=query_str)
        if not result:
            return f"节点 [{node_name}] 无内存使用率数据"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"查询节点内存使用率失败：{e}"


def pod_cpu_usage(
    pod_name: Annotated[str, Field(description="Pod 名称，例如 nginx-deploy-7d4f8b-x9k2m")],
    namespace: Annotated[str, Field(description="Pod 所在的命名空间，默认 default")] = "default",
    minutes: Annotated[int, Field(description="查询时间范围（分钟），默认 15")] = 15,
) -> str:
    """查询 Pod CPU 使用率趋势

    PromQL: rate(container_cpu_usage_seconds_total{namespace="<ns>",pod="<pod>"}[5m])

    参数：
      pod_name   Pod 名称（必填）
      namespace  命名空间（默认 default）
      minutes    查询时间范围（分钟，默认 15）
    """
    import json
    timerange = f"{minutes}m" if minutes >= 5 else "5m"
    query_str = (
        f"sum(rate(container_cpu_usage_seconds_total{{"
        f'namespace="{namespace}",pod="{pod_name}"'
        f"}}[{timerange}])) by (pod)"
    )
    try:
        prom = _get_prom()
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        result = prom.custom_query_range(
            query=query_str,
            start_time=start_time,
            end_time=end_time,
            step="60s",
        )
        if not result:
            return f"Pod [{namespace}/{pod_name}] 无 CPU 使用率数据"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"查询 Pod CPU 使用率失败：{e}"


def pod_memory_usage(
    pod_name: Annotated[str, Field(description="Pod 名称，例如 nginx-deploy-7d4f8b-x9k2m")],
    namespace: Annotated[str, Field(description="Pod 所在的命名空间，默认 default")] = "default",
) -> str:
    """查询 Pod 内存使用量

    PromQL: container_memory_working_set_bytes{namespace="<ns>",pod="<pod>"}

    参数：
      pod_name   Pod 名称（必填）
      namespace  命名空间（默认 default）
    """
    import json
    query_str = (
        f"container_memory_working_set_bytes{{"
        f'namespace="{namespace}",pod="{pod_name}"'
        f"}}"
    )
    try:
        prom = _get_prom()
        result = prom.custom_query(query=query_str)
        if not result:
            return f"Pod [{namespace}/{pod_name}] 无内存使用量数据"
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        return f"查询 Pod 内存使用量失败：{e}"
