#!/usr/bin/env python3
"""
macOS 资源监控脚本 — 极致详细使用指南
======================================

一、脚本功能速览
-----------------
- 系统概览：展示总 CPU、每核负载、1/5/15 分钟平均负载，以及内存、交换区占用和高负载进程。
- 速率差分：按自定义窗口统计网络、磁盘的上下行速率，并用 ASCII sparkline 展示趋势。
- ClashX 深度集成：
  * 实时读取各策略分组当前节点与历史延迟。
  * 主动对热门节点发起 delay 测速，标记在线状态。
  * 聚合节点流量和会话数，高亮流量最大的节点。
  * 罗列活跃连接（进程 → 目标 → 节点 → 上下行速率）。
- 其他能力：进程过滤、彩色条形图、JSON 输出、快照日志、差分阈值高亮等。

二、依赖与准备
---------------
1. Python 3.8 以上版本（默认使用 `/usr/bin/python3`）。
2. 安装 psutil：`pip install psutil`。
3. ClashX 中打开 external-controller（默认 `http://127.0.0.1:9090`），如设有 secret 需记住密钥。

三、快速启动命令
-----------------

  /usr/bin/python3 Notes/python3/mac_system_monitor.py \\
    --interval 5 \\
    --diff-interval 2 \\
    --clash-controller http://127.0.0.1:9090 \\
    --clash-node-limit 6 \\
    --clash-test-url http://www.gstatic.com/generate_204 \\
    --clash-test-timeout 5000 \\
    --clash-connection-limit 8

  注意：续行符 `\` 后必须紧跟换行且不得包含空格。若不便，可改为单行：

  /usr/bin/python3 Notes/python3/mac_system_monitor.py --interval 5 --diff-interval 2 --clash-controller http://127.0.0.1:9090 --clash-node-limit 6 --clash-test-url http://www.gstatic.com/generate_204 --clash-test-timeout 5000 --clash-connection-limit 8

四、核心参数详解
-----------------
- `--interval <秒>`             刷新间隔，默认 5。
- `--diff-interval <秒>`        速率差分窗口；设为 0 可关闭趋势图。
- `--sparkline-length <数>`     趋势条样本长度（默认 20）。
- `--metrics cpu,mem,...`       自定义输出模块，如 `cpu,network,proxy`。
- `--once`                      只输出一次快照。
- `--no-color`                  关闭彩色输出（适合日志环境）。
- `--json` / `--log-file`       启用 JSON 输出或追加日志。
- `--clash-controller URL`      ClashX 控制端地址，默认 `http://127.0.0.1:9090`。
- `--clash-secret KEY`          控制端访问密钥（如配置了 secret）。
- `--clash-node-limit N`        展示的节点数量上限，默认 6。
- `--clash-test-url URL`        主动测速目标 URL（默认 gstatic）。
- `--clash-test-timeout MS`     单次 delay 测速超时，单位毫秒。
- `--clash-connection-limit N`  活跃连接列表长度，默认 8。
- `--process-filter/-p KEY`     进程过滤关键字；可多次指定或用逗号分隔。

五、输出结构说明
-----------------
1. **系统概览**：总览指标、每核条形图、前五高负载进程。
2. **网络与磁盘**：
   - 网络速率达 5 MB/s 或 20 MB/s 会使用黄色/红色高亮。
   - 磁盘仅显示占用率最高的前 4 个挂载点，其余汇总为“已隐藏”。
3. **ClashX 模块**：
   - 展示 ClashX 进程统计、策略分组当前节点与延迟。
   - 主动测速列表会注明节点在线/离线和延迟值。
   - 节点流量统计会对流量最高者加亮显示。
4. **活跃连接**：
   - 记录当前连接的进程、目标、节点、累计流量、上下行速率。
   - 第一条高亮，便于快速锁定带宽大户。
5. **趋势行**：
   - 形式 `趋势: ↑: [._:-]  ↓: [.=-*]`，越靠右表示速率越高。

六、典型使用场景
-----------------
- 快速健康检查：`--once --metrics cpu,network,proxy`。
- 日志记录：配合 `--json` 或 `--log-file monitor.log` 定时运行。
- Clash 调优：
  * 主动测速可辅助挑选节点。
  * 流量统计定位异常高流量节点。
  * 活跃连接帮助追踪对应进程。
- 无 Clash 场景：直接忽略 `proxy` 模块或通过 `--metrics` 排除。

七、常见问题与排障
-------------------
- `psutil` 缺失：按提示执行 `pip install psutil`。
- Clash 控制端连接报错：检查 external-controller 是否启用、端口/secret 是否匹配。
- sparkline 显示异常：确保终端使用 UTF-8，必要时 `export LC_ALL=en_US.UTF-8`。
- 命令复制失败：确认续行符 `\` 后无空格，或改用单行命令。

更多参数与高级用法，请执行 `--help` 查看官方描述。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

try:
    import psutil
except ModuleNotFoundError as exc:  # pragma: no cover
    print("Missing dependency: psutil. Install with `pip install psutil`.", file=sys.stderr)
    raise SystemExit(1) from exc

COLOR_CODES = {
    "reset": "\033[0m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}

METRIC_ALIASES = {
    "cpu": "cpu",
    "processor": "cpu",
    "mem": "memory",
    "memory": "memory",
    "swap": "swap",
    "net": "network",
    "network": "network",
    "disk": "disk",
    "disks": "disk",
    "process": "process",
    "processes": "process",
    "proc": "process",
    "proxy": "proxy",
    "clash": "proxy",
    "clashx": "proxy",
}

DEFAULT_METRICS = ["cpu", "memory", "swap", "network", "disk", "process", "proxy"]
SPARK_CHARS = "._:-=+*#%@"
BYTES_PER_MB = 1024 * 1024
DEFAULT_TEST_URL = "http://www.gstatic.com/generate_204"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="macOS 资源监控，支持定制指标、速率差分与彩色输出。",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="刷新间隔（秒，默认 5）。")
    parser.add_argument("--top", type=int, default=5, help="列出 CPU 占用最高的进程数量（默认 5）。")
    parser.add_argument(
        "--top-memory",
        type=int,
        default=0,
        metavar="N",
        help="额外列出占用内存最高的 N 个进程。",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="逗号分隔的模块列表（cpu,mem,swap,net,disk,process,proxy），默认显示全部。",
    )
    parser.add_argument(
        "--process-filter",
        "-p",
        dest="process_filters",
        action="append",
        metavar="关键字",
        help="仅显示名称或用户名包含关键字的进程，可多次输入或用逗号分隔。",
    )
    parser.add_argument("--log-file", type=str, help="可选：追加写入 JSON 快照的文件路径。")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出（默认为文本仪表盘）。")
    parser.add_argument("--once", action="store_true", help="仅采集一次后退出。")
    parser.add_argument("--no-color", action="store_true", help="禁用彩色输出。")
    parser.add_argument("--bar-width", type=int, default=24, help="条形图宽度（默认 24）。")
    parser.add_argument("--static", action="store_true", help="禁用清屏刷新，逐条输出。")
    parser.add_argument(
        "--diff-interval",
        type=float,
        default=0.0,
        help="计算网络与磁盘速率的窗口（秒）。0 表示禁用速率差分。",
    )
    parser.add_argument(
        "--sparkline-length",
        type=int,
        default=20,
        help="速率趋势图宽度（字符数，默认 20）。",
    )
    parser.add_argument(
        "--clash-controller",
        type=str,
        default="http://127.0.0.1:9090",
        help="ClashX 外部控制端地址（默认 http://127.0.0.1:9090）。",
    )
    parser.add_argument(
        "--clash-secret",
        type=str,
        default=None,
        help="ClashX 控制端的访问密钥（如配置了 external-controller secret）。",
    )
    parser.add_argument(
        "--clash-node-limit",
        type=int,
        default=6,
        help="展示节点延迟的数量上限（默认 6）。",
    )
    parser.add_argument(
        "--clash-test-url",
        type=str,
        default=DEFAULT_TEST_URL,
        help="ClashX 节点测速使用的 URL。",
    )
    parser.add_argument(
        "--clash-test-timeout",
        type=int,
        default=5000,
        help="节点测速超时（毫秒，默认 5000）。",
    )
    parser.add_argument(
        "--clash-connection-limit",
        type=int,
        default=8,
        help="显示活跃连接的数量上限（默认 8）。",
    )
    args = parser.parse_args()
    try:
        args.metrics = normalize_metrics(args.metrics)
    except ValueError as exc:
        parser.error(str(exc))
    args.process_filters = normalize_filters(args.process_filters)
    args.clash_controller = normalize_controller_url(args.clash_controller)
    return args


def normalize_metrics(raw: Optional[str]) -> List[str]:
    if raw is None:
        return DEFAULT_METRICS.copy()
    unique: List[str] = []
    for chunk in raw.split(","):
        key = chunk.strip().lower()
        if not key:
            continue
        alias = METRIC_ALIASES.get(key)
        if not alias:
            raise ValueError(f"未知指标: {chunk.strip()}")
        if alias not in unique:
            unique.append(alias)
    return unique or DEFAULT_METRICS.copy()


def normalize_filters(filters: Optional[List[str]]) -> List[str]:
    if not filters:
        return []
    result: List[str] = []
    for item in filters:
        for chunk in item.split(","):
            clean = chunk.strip()
            if clean:
                result.append(clean)
    return result


def normalize_controller_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    parsed = urlparse.urlparse(raw)
    if not parsed.scheme:
        raw = "http://" + raw
    return raw.rstrip("/")


def format_bytes(num: float) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if abs(num) < step:
            return f"{num:6.1f}{unit}"
        num /= step
    return f"{num:6.1f}EB"


def make_sparkline(values: Deque[float], width: int) -> str:
    if not values:
        return ""
    width = max(4, width)
    sample = list(values)[-width:]
    if not sample:
        return ""

    # Remove leading zeros to give more contrast; keep at least one sample
    while len(sample) > 1 and sample[0] <= 0 and sample[1] <= 0:
        sample.pop(0)

    v_min = min(sample)
    v_max = max(sample)
    if v_max <= 0:
        return "".join(SPARK_CHARS[0] for _ in sample)

    # Collapse negative values to zero for readability
    sample = [max(0.0, val) for val in sample]
    v_min = 0.0
    span = max(v_max - v_min, 1e-9)
    levels = len(SPARK_CHARS) - 1
    chars = []
    for val in sample:
        normalized = (val - v_min) / span
        idx = int(round(normalized * levels))
        chars.append(SPARK_CHARS[max(0, min(levels, idx))])
    return "".join(chars).rstrip("_")


def collect_process_stats(
    limit_cpu: int,
    limit_mem: int,
    filters: List[str],
) -> Dict[str, List[Dict[str, object]]]:
    filters_lower = [item.lower() for item in filters]
    entries: List[Dict[str, object]] = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        try:
            with proc.oneshot():
                name = proc.info.get("name") or "unknown"
                user = proc.info.get("username") or "-"
                haystack = f"{name} {user}".lower()
                if filters_lower and not any(key in haystack for key in filters_lower):
                    continue
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss
                uptime = max(0.0, time.time() - proc.create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        entries.append(
            {
                "pid": proc.info.get("pid"),
                "name": name,
                "user": user,
                "cpu_percent": round(cpu, 2),
                "memory_rss": mem,
                "uptime_sec": int(uptime),
            }
        )

    return {
        "cpu": sorted(entries, key=lambda item: item["cpu_percent"], reverse=True)[: max(0, limit_cpu)]
        if limit_cpu > 0
        else [],
        "memory": sorted(entries, key=lambda item: item["memory_rss"], reverse=True)[: max(0, limit_mem)]
        if limit_mem > 0
        else [],
    }


def collect_clash_status(
    controller: Optional[str],
    secret: Optional[str],
    node_limit: int,
    test_url: str,
    test_timeout: int,
    connection_limit: int,
) -> Dict[str, object]:
    processes: List[Dict[str, object]] = []
    for proc in psutil.process_iter(["pid", "name", "username"]):
        name = (proc.info.get("name") or "").lower()
        if "clash" not in name:
            continue
        try:
            with proc.oneshot():
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss
                uptime = max(0.0, time.time() - proc.create_time())
                status = proc.status()
                try:
                    conn_count = len(proc.net_connections(kind="inet"))
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    conn_count = None
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        processes.append(
            {
                "pid": proc.info.get("pid"),
                "name": proc.info.get("name") or "unknown",
                "user": proc.info.get("username") or "-",
                "cpu_percent": round(cpu, 2),
                "memory_rss": mem,
                "uptime_sec": int(uptime),
                "status": status,
                "connections": conn_count,
            }
        )
    processes.sort(key=lambda item: item["cpu_percent"], reverse=True)

    api_info = (
        fetch_clash_proxy_info(
            controller,
            secret,
            node_limit,
            connection_limit,
            test_url,
            test_timeout,
        )
        if controller
        else None
    )
    return {
        "running": bool(processes),
        "processes": processes,
        "api": api_info,
    }


def fetch_clash_proxy_info(
    controller: Optional[str],
    secret: Optional[str],
    node_limit: int,
    connection_limit: int,
    test_url: str,
    test_timeout: int,
    timeout: float = 1.5,
) -> Dict[str, object]:
    info = {
        "controller": controller,
        "reachable": False,
        "error": None,
        "groups": [],
        "top_nodes": [],
        "tested_nodes": [],
        "connections": {},
    }
    if not controller:
        info["error"] = "未指定控制端地址"
        return info
    url = controller.rstrip("/") + "/proxies"
    headers = {"Accept": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    req = urlrequest.Request(url, headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf8"))
    except urlerror.URLError as exc:
        info["error"] = str(getattr(exc, "reason", exc))
        return info
    except Exception as exc:  # pragma: no cover - defensive
        info["error"] = str(exc)
        return info

    proxies = data.get("proxies", {})
    group_entries = []
    node_entries = []
    selector_types = {"Selector", "URLTest", "Fallback", "LoadBalance"}
    node_types = {
        "VMess",
        "VLESS",
        "Trojan",
        "Shadowsocks",
        "WireGuard",
        "SSR",
        "Snell",
        "SOCKS5",
        "HTTP",
        "Hysteria",
        "Hysteria2",
        "Tuic",
        "ShadowTLS",
        "Reality",
        "SSH",
        "Direct",
        "Reject",
    }
    for name, entry in proxies.items():
        entry_type = entry.get("type", "")
        history = entry.get("history") or []
        last_delay = None
        last_time = None
        if history:
            last = history[-1]
            last_delay = last.get("delay")
            last_time = last.get("time")
        if entry_type in selector_types:
            group_entries.append(
                {
                    "name": name,
                    "type": entry_type,
                    "current": entry.get("now"),
                    "candidates": entry.get("all") or [],
                    "last_delay": last_delay,
                    "last_time": last_time,
                    "alive": entry.get("alive"),
                }
            )
        elif entry_type in node_types:
            node_entries.append(
                {
                    "name": name,
                    "type": entry_type,
                    "last_delay": last_delay,
                    "last_time": last_time,
                    "alive": entry.get("alive"),
                }
            )

    tested_nodes = perform_delay_tests(
        controller=controller,
        secret=secret,
        nodes=node_entries[: max(0, node_limit * 2)],
        test_url=test_url,
        test_timeout=test_timeout,
    )
    if tested_nodes:
        tested_nodes.sort(
            key=lambda item: (0 if (item.get("delay", -1) >= 0) else 1, item.get("delay", float("inf")))
        )

    def delay_sort_key(item: Dict[str, object]) -> Tuple[int, float]:
        delay = item.get("last_delay")
        if delay is None or delay < 0:
            return (1, float("inf"))
        return (0, delay)

    node_entries.sort(key=delay_sort_key)
    connections_info = fetch_clash_connections(controller, secret, connection_limit)

    info.update(
        {
            "reachable": True,
            "groups": group_entries,
            "top_nodes": node_entries[: max(0, node_limit)],
            "tested_nodes": tested_nodes[: max(0, node_limit)] if tested_nodes else [],
            "connections": connections_info,
        }
    )
    return info


def perform_delay_tests(
    controller: Optional[str],
    secret: Optional[str],
    nodes: List[Dict[str, object]],
    test_url: str,
    test_timeout: int,
    per_request_timeout: float = 3.0,
) -> List[Dict[str, object]]:
    if not controller or not nodes:
        return []
    headers = {"Accept": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    results: List[Dict[str, object]] = []
    for node in nodes:
        name = node.get("name")
        if not name:
            continue
        node_type = (node.get("type") or "").lower()
        if node_type == "reject":
            continue
        encoded_name = urlparse.quote(name, safe="")
        url = (
            controller.rstrip("/")
            + f"/proxies/{encoded_name}/delay?timeout={int(max(0, test_timeout))}"
            + f"&url={urlparse.quote(test_url, safe='')}"
        )
        req = urlrequest.Request(url, headers=headers)
        delay_value = -1
        error_msg = None
        try:
            with urlrequest.urlopen(req, timeout=per_request_timeout) as resp:
                payload = json.loads(resp.read().decode("utf8"))
            delay_value = payload.get("delay", -1)
        except Exception as exc:  # pragma: no cover - network dependent
            error_msg = str(exc)
        results.append(
            {
                "name": name,
                "type": node.get("type"),
                "alive": node.get("alive"),
                "delay": delay_value if delay_value is not None else -1,
                "error": error_msg,
            }
        )
    return results


def fetch_clash_connections(
    controller: Optional[str],
    secret: Optional[str],
    limit: int,
    timeout: float = 2.0,
) -> Dict[str, object]:
    if not controller or limit <= 0:
        return {}
    url = controller.rstrip("/") + "/connections"
    headers = {"Accept": "application/json"}
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    req = urlrequest.Request(url, headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf8"))
    except Exception:  # pragma: no cover
        return {}

    connections = data.get("connections") or []
    now = _dt.datetime.now(_dt.timezone.utc)
    entries: List[Dict[str, object]] = []
    agg: Dict[str, Dict[str, object]] = {}

    for conn in connections:
        upload = int(conn.get("upload", 0))
        download = int(conn.get("download", 0))
        total_bytes = upload + download
        metadata = conn.get("metadata") or {}
        host = metadata.get("host") or metadata.get("remoteDestination") or metadata.get("destinationIP")
        process = metadata.get("process") or metadata.get("processPath") or "-"
        chains = conn.get("chains") or []
        final_proxy = chains[-1] if chains else "Unknown"
        start_time = conn.get("start")
        duration_sec = 0.0
        if start_time:
            try:
                start_dt = _dt.datetime.fromisoformat(start_time)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=_dt.timezone.utc)
                duration_sec = max((now - start_dt).total_seconds(), 1e-3)
            except Exception:
                duration_sec = max(conn.get("duration", 0) or 0, 1e-3)
        else:
            duration_sec = max(conn.get("duration", 0) or 0, 1e-3)

        upload_rate = upload / duration_sec
        download_rate = download / duration_sec

        entry = {
            "id": conn.get("id"),
            "host": host,
            "process": process,
            "proxy": final_proxy,
            "chains": chains,
            "upload": upload,
            "download": download,
            "total": total_bytes,
            "upload_rate_mb_s": upload_rate / BYTES_PER_MB,
            "download_rate_mb_s": download_rate / BYTES_PER_MB,
            "rule": conn.get("rule"),
            "rule_payload": conn.get("rulePayload"),
            "start": start_time,
        }
        entries.append(entry)

        agg_entry = agg.setdefault(
            final_proxy,
            {"proxy": final_proxy, "connections": 0, "upload": 0, "download": 0},
        )
        agg_entry["connections"] += 1
        agg_entry["upload"] += upload
        agg_entry["download"] += download

    entries.sort(key=lambda item: item["total"], reverse=True)
    by_proxy = sorted(
        agg.values(),
        key=lambda item: item["upload"] + item["download"],
        reverse=True,
    )

    return {
        "connections": entries[: max(0, limit)],
        "by_proxy": by_proxy,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def collect_snapshot(
    limit_cpu: int,
    limit_mem: int,
    filters: List[str],
    include_processes: bool,
    include_proxy: bool,
    controller: Optional[str],
    secret: Optional[str],
    node_limit: int,
    test_url: str,
    test_timeout: int,
    connection_limit: int,
) -> Dict[str, object]:
    net_total = psutil.net_io_counters(pernic=False)
    net_pernic_raw = psutil.net_io_counters(pernic=True)
    net_pernic = {
        name: {
            "bytes_sent": counters.bytes_sent,
            "bytes_recv": counters.bytes_recv,
            "packets_sent": counters.packets_sent,
            "packets_recv": counters.packets_recv,
        }
        for name, counters in net_pernic_raw.items()
    }

    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    cpu_total = sum(cpu_per_core) / max(1, len(cpu_per_core))
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)

    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    seen_mounts = set()
    disks: List[Dict[str, object]] = []
    for part in psutil.disk_partitions(all=False):
        mount = part.mountpoint
        if mount in seen_mounts or not mount:
            continue
        seen_mounts.add(mount)
        try:
            usage = psutil.disk_usage(mount)
        except PermissionError:
            continue
        disks.append(
            {
                "mount": mount,
                "fstype": part.fstype or "-",
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
            }
        )

    disk_io_total = psutil.disk_io_counters(perdisk=False)
    disk_io_per = psutil.disk_io_counters(perdisk=True)
    disk_io = {
        name: {
            "read_bytes": counters.read_bytes,
            "write_bytes": counters.write_bytes,
            "read_count": counters.read_count,
            "write_count": counters.write_count,
        }
        for name, counters in disk_io_per.items()
    }

    snapshot = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "cpu": {
            "percent_total": round(cpu_total, 2),
            "percent_per_core": [round(val, 2) for val in cpu_per_core],
            "load_average": {"1m": load_avg[0], "5m": load_avg[1], "15m": load_avg[2]},
            "logical": psutil.cpu_count(logical=True),
            "physical": psutil.cpu_count(logical=False),
        },
        "memory": {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        },
        "swap": {
            "total": sm.total,
            "used": sm.used,
            "free": sm.free,
            "percent": sm.percent,
        },
        "network": {
            "bytes_sent": net_total.bytes_sent,
            "bytes_recv": net_total.bytes_recv,
            "packets_sent": net_total.packets_sent,
            "packets_recv": net_total.packets_recv,
            "pernic": net_pernic,
        },
        "disk_io_total": {
            "read_bytes": disk_io_total.read_bytes,
            "write_bytes": disk_io_total.write_bytes,
            "read_count": disk_io_total.read_count,
            "write_count": disk_io_total.write_count,
        },
        "disk_io": disk_io,
        "disks": disks,
        "processes": collect_process_stats(limit_cpu, limit_mem, filters)
        if include_processes
        else {"cpu": [], "memory": []},
        "process_filters": filters if include_processes else [],
        "process_limits": {"cpu": limit_cpu, "memory": limit_mem},
        "clash": collect_clash_status(
            controller,
            secret,
            node_limit,
            test_url,
            test_timeout,
            connection_limit,
        )
        if include_proxy
        else None,
    }
    return snapshot


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def colorize(text: str, color: str, enabled: bool) -> str:
    if not enabled or color not in COLOR_CODES:
        return text
    return f"{COLOR_CODES[color]}{text}{COLOR_CODES['reset']}"


def classify_percent(value: float) -> str:
    if value >= 85.0:
        return "red"
    if value >= 60.0:
        return "yellow"
    return "green"


def render_bar(percent: float, width: int, color_enabled: bool) -> str:
    width = max(10, min(width, 60))
    filled = int(round(percent / 100.0 * width))
    filled = max(0, min(width, filled))
    bar = "#" * filled + "." * (width - filled)
    pct_text = f"{percent:5.1f}%"
    pct_colored = colorize(pct_text, classify_percent(percent), color_enabled)
    return f"[{bar}] {pct_colored}"


def pick_base(history: Deque[Tuple[float, object]], now: float, interval: float) -> Optional[Tuple[float, object]]:
    if not history:
        return None
    target = now - interval
    base = history[0]
    for item in history:
        if item[0] <= target:
            base = item
        else:
            break
    return base


def compute_diff_info(
    snapshot: Dict[str, object],
    history_net: Deque[Tuple[float, Dict[str, object], Dict[str, Dict[str, float]]]],
    history_disk: Deque[Tuple[float, Dict[str, float], Dict[str, Dict[str, float]]]],
    rate_history: Dict[str, Deque[float]],
    interval: float,
    sparkline_len: int,
) -> Optional[Dict[str, object]]:
    now = time.time()
    diff_result: Dict[str, object] = {}

    net_total = snapshot["network"]
    history_net.append((now, net_total, net_total["pernic"]))
    while history_net and now - history_net[0][0] > max(interval * 2, interval + 10):
        history_net.popleft()
    base_net = pick_base(history_net, now, interval)

    disk_total = snapshot["disk_io_total"]
    history_disk.append((now, disk_total, snapshot["disk_io"]))
    while history_disk and now - history_disk[0][0] > max(interval * 2, interval + 10):
        history_disk.popleft()
    base_disk = pick_base(history_disk, now, interval)

    if base_net and now - base_net[0] >= 0.5:
        elapsed = now - base_net[0]
        prev_total = base_net[1]
        sent_bps = max(0.0, (net_total["bytes_sent"] - prev_total["bytes_sent"]) / elapsed)
        recv_bps = max(0.0, (net_total["bytes_recv"] - prev_total["bytes_recv"]) / elapsed)
        rate_history["net_sent"].append(sent_bps / BYTES_PER_MB)
        rate_history["net_recv"].append(recv_bps / BYTES_PER_MB)
        pernic_rates = []
        prev_pernic = base_net[2]
        for name, current in net_total["pernic"].items():
            prev = prev_pernic.get(name)
            if not prev:
                continue
            sent = max(0.0, (current["bytes_sent"] - prev["bytes_sent"]) / elapsed)
            recv = max(0.0, (current["bytes_recv"] - prev["bytes_recv"]) / elapsed)
            if sent == 0 and recv == 0:
                continue
            pernic_rates.append(
                {
                    "name": name,
                    "sent_mb_s": sent / BYTES_PER_MB,
                    "recv_mb_s": recv / BYTES_PER_MB,
                }
            )
        pernic_rates.sort(key=lambda item: item["sent_mb_s"] + item["recv_mb_s"], reverse=True)
        diff_result["network"] = {
            "total": {
                "sent_bps": sent_bps,
                "recv_bps": recv_bps,
                "sent_mb_s": sent_bps / BYTES_PER_MB,
                "recv_mb_s": recv_bps / BYTES_PER_MB,
                "elapsed": elapsed,
            },
            "pernic_top": pernic_rates[:3],
            "spark": {
                "sent": make_sparkline(rate_history["net_sent"], sparkline_len),
                "recv": make_sparkline(rate_history["net_recv"], sparkline_len),
            },
        }

    if base_disk and now - base_disk[0] >= 0.5:
        elapsed = now - base_disk[0]
        prev_total = base_disk[1]
        read_bps = max(0.0, (disk_total["read_bytes"] - prev_total["read_bytes"]) / elapsed)
        write_bps = max(0.0, (disk_total["write_bytes"] - prev_total["write_bytes"]) / elapsed)
        read_iops = max(0.0, (disk_total["read_count"] - prev_total["read_count"]) / elapsed)
        write_iops = max(0.0, (disk_total["write_count"] - prev_total["write_count"]) / elapsed)
        rate_history["disk_read"].append(read_bps / BYTES_PER_MB)
        rate_history["disk_write"].append(write_bps / BYTES_PER_MB)
        prev_perdisk = base_disk[2]
        perdisk_rates = []
        for name, current in snapshot["disk_io"].items():
            prev = prev_perdisk.get(name)
            if not prev:
                continue
            r_bps = max(0.0, (current["read_bytes"] - prev["read_bytes"]) / elapsed)
            w_bps = max(0.0, (current["write_bytes"] - prev["write_bytes"]) / elapsed)
            r_iops = max(0.0, (current["read_count"] - prev["read_count"]) / elapsed)
            w_iops = max(0.0, (current["write_count"] - prev["write_count"]) / elapsed)
            if r_bps == 0 and w_bps == 0:
                continue
            perdisk_rates.append(
                {
                    "name": name,
                    "read_mb_s": r_bps / BYTES_PER_MB,
                    "write_mb_s": w_bps / BYTES_PER_MB,
                    "read_iops": r_iops,
                    "write_iops": w_iops,
                }
            )
        perdisk_rates.sort(key=lambda item: item["read_mb_s"] + item["write_mb_s"], reverse=True)
        diff_result["disk"] = {
            "total": {
                "read_mb_s": read_bps / BYTES_PER_MB,
                "write_mb_s": write_bps / BYTES_PER_MB,
                "read_iops": read_iops,
                "write_iops": write_iops,
            },
            "perdisk_top": perdisk_rates[:3],
            "spark": {
                "read": make_sparkline(rate_history["disk_read"], sparkline_len),
                "write": make_sparkline(rate_history["disk_write"], sparkline_len),
            },
        }

    return diff_result or None


def render_text(
    snapshot: Dict[str, object],
    *,
    metrics: List[str],
    bar_width: int,
    use_color: bool,
    diff_info: Optional[Dict[str, object]] = None,
) -> str:
    metrics_set = set(metrics)
    cpu = snapshot["cpu"]
    memory = snapshot["memory"]
    swap = snapshot["swap"]
    net = snapshot["network"]
    disks = snapshot["disks"]
    processes = snapshot.get("processes", {"cpu": [], "memory": []})
    process_filters = snapshot.get("process_filters", [])
    process_limits = snapshot.get("process_limits", {"cpu": 0, "memory": 0})
    clash_status = snapshot.get("clash")

    header = f"时间: {snapshot['timestamp']}  系统: {snapshot['platform']}"
    lines: List[str] = [header]

    RATE_WARN = 5.0
    RATE_ALERT = 20.0
    DELAY_WARN = 150
    DELAY_ALERT = 250

    def fmt_delay(delay: Optional[float]) -> str:
        if delay is None or delay < 0:
            return "--"
        text = f"{delay} ms"
        if use_color:
            if delay >= DELAY_ALERT:
                return colorize(text, "red", True)
            if delay >= DELAY_WARN:
                return colorize(text, "yellow", True)
        return text

    def fmt_rate(rate: float) -> str:
        text = f"{rate:6.2f} MB/s"
        if use_color:
            if rate >= RATE_ALERT:
                return colorize(text, "red", True)
            if rate >= RATE_WARN:
                return colorize(text, "yellow", True)
        return text

    def highlight_line(text: str, top: bool) -> str:
        if use_color and top:
            return colorize(text, "yellow", True)
        return text

    system_tags = {"cpu", "memory", "swap", "process"}
    if metrics_set & system_tags:
        lines.append("—— 系统概览 ——")
        if "cpu" in metrics_set:
            cpu_heading = (
                f"CPU 总体: {render_bar(cpu['percent_total'], bar_width, use_color)}  "
                f"平均负载(1/5/15): {cpu['load_average']['1m']:.2f} / "
                f"{cpu['load_average']['5m']:.2f} / {cpu['load_average']['15m']:.2f}  "
                f"核心: {cpu['physical']} 物理 / {cpu['logical']} 逻辑"
            )
            lines.append(cpu_heading)
            lines.append("每核负载:")
            row: List[str] = []
            for idx, val in enumerate(cpu["percent_per_core"]):
                label = f"核心{idx:02d}: {render_bar(val, max(12, bar_width // 2), use_color)}"
                row.append(label)
                if len(row) == 2:
                    lines.append("  ".join(row))
                    row = []
            if row:
                lines.append("  ".join(row))
        if "memory" in metrics_set:
            lines.append(
                f"内存使用: {render_bar(memory['percent'], bar_width, use_color)}  "
                f"已用 {format_bytes(memory['used'])} / 总计 {format_bytes(memory['total'])}  "
                f"可用 {format_bytes(memory['available'])}"
            )
        if "swap" in metrics_set:
            lines.append(
                f"交换区:   {render_bar(swap['percent'], bar_width, use_color)}  "
                f"已用 {format_bytes(swap['used'])} / 总计 {format_bytes(swap['total'])}"
            )
        if "process" in metrics_set:
            filters_text = ", ".join(process_filters)
            if filters_text:
                lines.append(f"进程过滤关键字: {filters_text}")
            cpu_limit = process_limits.get("cpu", 0)
            mem_limit = process_limits.get("memory", 0)
            if cpu_limit > 0:
                lines.append("高占用进程（按 CPU 排序）:")
                if processes["cpu"]:
                    for proc in processes["cpu"]:
                        cpu_pct = colorize(
                            f"{proc['cpu_percent']:5.1f}%",
                            classify_percent(proc["cpu_percent"]),
                            use_color,
                        )
                        lines.append(
                            f"  PID {proc['pid']:>6}  CPU {cpu_pct}  内存 {format_bytes(proc['memory_rss'])}  "
                            f"运行 {proc['uptime_sec']:>6}s  {proc['name']} ({proc['user']})"
                        )
                else:
                    lines.append("  无匹配的进程。")
            if mem_limit > 0:
                if cpu_limit > 0:
                    lines.append("")
                lines.append("高占用进程（按内存排序）:")
                if processes["memory"]:
                    for proc in processes["memory"]:
                        cpu_pct = colorize(
                            f"{proc['cpu_percent']:5.1f}%",
                            classify_percent(proc["cpu_percent"]),
                            use_color,
                        )
                        lines.append(
                            f"  PID {proc['pid']:>6}  内存 {format_bytes(proc['memory_rss'])}  CPU {cpu_pct}  "
                            f"运行 {proc['uptime_sec']:>6}s  {proc['name']} ({proc['user']})"
                        )
                else:
                    lines.append("  无匹配的进程。")

    net_disk_tags = {"network", "disk"}
    if metrics_set & net_disk_tags:
        lines.append("")
        lines.append("—— 网络与磁盘 ——")
        if "network" in metrics_set:
            lines.append(
                f"网络累计: ↑{format_bytes(net['bytes_sent'])} ↓{format_bytes(net['bytes_recv'])}  "
                f"数据包 ↑{net['packets_sent']} ↓{net['packets_recv']}"
            )
            if diff_info and diff_info.get("network"):
                net_diff = diff_info["network"]
                rate = net_diff["total"]
                lines.append(
                    f"网络速率: ↑{fmt_rate(rate['sent_mb_s'])} ↓{fmt_rate(rate['recv_mb_s'])} "
                    f"(窗口 {rate['elapsed']:.1f}s)"
                )
                spark = net_diff.get("spark", {})
                if spark.get("sent") or spark.get("recv"):
                    parts = []
                    if spark.get("sent"):
                        parts.append(f"↑: [{spark['sent']}]")
                    if spark.get("recv"):
                        parts.append(f"↓: [{spark['recv']}]")
                    lines.append("趋势: " + "  ".join(parts))
                for item in net_diff.get("pernic_top", []):
                    lines.append(
                        f"  {item['name']:<12} ↑{fmt_rate(item['sent_mb_s'])} ↓{fmt_rate(item['recv_mb_s'])}"
                    )
        if "disk" in metrics_set:
            lines.append("磁盘占用:")
            if disks:
                top_disks = sorted(disks, key=lambda d: d["percent"], reverse=True)[:4]
                for disk in top_disks:
                    percent_text = colorize(
                        f"{disk['percent']:5.1f}%",
                        classify_percent(disk["percent"]),
                        use_color,
                    )
                    lines.append(
                        f"  {disk['mount']:<30} {disk['fstype']:<6} "
                        f"{percent_text}  {format_bytes(disk['used'])}/{format_bytes(disk['total'])}"
                    )
                other_count = max(0, len(disks) - len(top_disks))
                if other_count > 0:
                    lines.append(f"  … 其他挂载点 {other_count} 个（已隐藏）")
            else:
                lines.append("  暂无磁盘数据")
            if diff_info and diff_info.get("disk"):
                disk_diff = diff_info["disk"]
                total = disk_diff["total"]
                lines.append(
                    f"磁盘速率: 读 {fmt_rate(total['read_mb_s'])} 写 {fmt_rate(total['write_mb_s'])} "
                    f"IOPS R {total['read_iops']:6.1f} / W {total['write_iops']:6.1f}"
                )
                spark = disk_diff.get("spark", {})
                if spark.get("read") or spark.get("write"):
                    parts = []
                    if spark.get("read"):
                        parts.append(f"读: [{spark['read']}]")
                    if spark.get("write"):
                        parts.append(f"写: [{spark['write']}]")
                    lines.append("趋势: " + "  ".join(parts))
                for item in disk_diff.get("perdisk_top", []):
                    lines.append(
                        f"  {item['name']:<12} 读 {fmt_rate(item['read_mb_s'])} 写 {fmt_rate(item['write_mb_s'])} "
                        f"IOPS R {item['read_iops']:5.1f} / W {item['write_iops']:5.1f}"
                    )

    if "proxy" in metrics_set:
        lines.append("")
        lines.append("—— ClashX ——")
        if not clash_status:
            lines.append("  无 ClashX 进程或控制端信息。")
        else:
            processes_proxy = clash_status.get("processes") or []
            if clash_status.get("running"):
                total_cpu = sum(proc["cpu_percent"] for proc in processes_proxy)
                total_mem = sum(proc["memory_rss"] for proc in processes_proxy)
                lines.append(
                    f"  进程数: {len(processes_proxy)}  汇总 CPU {total_cpu:5.1f}%  内存 {format_bytes(total_mem)}"
                )
                for proc in processes_proxy[:5]:
                    cpu_pct = colorize(
                        f"{proc['cpu_percent']:5.1f}%",
                        classify_percent(proc["cpu_percent"]),
                        use_color,
                    )
                    conn_text = (
                        f"{proc['connections']} 连接"
                        if proc["connections"] is not None
                        else "连接数未知"
                    )
                    lines.append(
                        f"  PID {proc['pid']:>6} {proc['status']:<10} CPU {cpu_pct} 内存 {format_bytes(proc['memory_rss'])} "
                        f"运行 {proc['uptime_sec']:>6}s {conn_text} {proc['name']} ({proc['user']})"
                    )
            else:
                lines.append("  ClashX 进程未运行。")

            api_info = clash_status.get("api")
            if api_info:
                controller = api_info.get("controller") or ""
                if api_info.get("reachable"):
                    lines.append(f"  控制端: {controller} (已连接)")
                    groups = api_info.get("groups") or []
                    if groups:
                        lines.append("  分组当前节点:")
                        for group in groups[:6]:
                            delay_text = fmt_delay(group.get("last_delay"))
                            lines.append(
                                f"    {group['name']:<16} → {group.get('current') or '未选择':<20} 延迟 {delay_text}"
                            )
                    tested_nodes = api_info.get("tested_nodes") or []
                    if tested_nodes:
                        lines.append("  主动测速结果:")
                        for node in tested_nodes:
                            delay_text = fmt_delay(node.get("delay"))
                            state = "在线" if node.get("alive") else "离线"
                            lines.append(
                                f"    {node['name']:<20} {delay_text}  {state}"
                            )
                    connections_info = api_info.get("connections") or {}
                    by_proxy = connections_info.get("by_proxy") or []
                    if by_proxy:
                        lines.append("  节点流量统计:")
                        for idx, item in enumerate(by_proxy[:6]):
                            line = (
                                f"    {item['proxy']:<16} 连接 {item['connections']:>3}  "
                                f"↑{format_bytes(item['upload'])} ↓{format_bytes(item['download'])}"
                            )
                            lines.append(highlight_line(line, idx == 0))
                    top_connections = connections_info.get("connections") or []
                    if top_connections:
                        lines.append("")
                        lines.append("—— 活跃连接 ——")
                        for idx, conn in enumerate(top_connections):
                            line = (
                                f"  {conn['process'][:20] if conn['process'] else '-':<20} → "
                                f"{(conn['host'] or '-')[:28]:<28} {conn['proxy'] or '-':<18} "
                                f"↑{format_bytes(conn['upload'])} ↓{format_bytes(conn['download'])} "
                                f"速率 ↑{fmt_rate(conn['upload_rate_mb_s'])} ↓{fmt_rate(conn['download_rate_mb_s'])}"
                            )
                            lines.append(highlight_line(line, idx == 0))
                else:
                    err = api_info.get("error") or "未知错误"
                    lines.append(f"  控制端: {controller} (连接失败: {err})")

    return "\n".join(lines)


def append_log(path: str, snapshot: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False))
        handle.write("\n")


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def main() -> None:
    if platform.system() != "Darwin":  # pragma: no cover
        print("Warning: this script is tuned for macOS (Darwin).", file=sys.stderr)

    args = parse_args()
    metrics_list = args.metrics
    metrics_set = set(metrics_list)
    include_processes = "process" in metrics_set
    include_proxy = "proxy" in metrics_set
    diff_interval = max(0.0, args.diff_interval)

    history_net: Deque = deque()
    history_disk: Deque = deque()
    rate_history = {
        "net_sent": deque(maxlen=args.sparkline_length),
        "net_recv": deque(maxlen=args.sparkline_length),
        "disk_read": deque(maxlen=args.sparkline_length),
        "disk_write": deque(maxlen=args.sparkline_length),
    }

    psutil.cpu_percent(interval=None)
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    try:
        cycle = 0
        while True:
            snapshot = collect_snapshot(
                limit_cpu=args.top if include_processes else 0,
                limit_mem=args.top_memory if include_processes else 0,
                filters=args.process_filters if include_processes else [],
                include_processes=include_processes,
                include_proxy=include_proxy,
                controller=args.clash_controller,
                secret=args.clash_secret,
                node_limit=args.clash_node_limit,
                test_url=args.clash_test_url,
                test_timeout=args.clash_test_timeout,
                connection_limit=args.clash_connection_limit,
            )
            if args.json:
                output = json.dumps(snapshot, ensure_ascii=False)
            else:
                use_color = supports_color() and not args.no_color
                width = args.bar_width
                if width == 24:
                    try:
                        term_width = shutil.get_terminal_size(fallback=(120, 20)).columns
                        width = min(max(18, term_width // 5), 40)
                    except OSError:
                        width = 24
                diff_info = None
                if diff_interval > 0:
                    diff_info = compute_diff_info(
                        snapshot,
                        history_net,
                        history_disk,
                        rate_history,
                        diff_interval,
                        args.sparkline_length,
                    )
                if diff_interval > 0 and diff_info is None:
                    time.sleep(max(0.5, diff_interval))
                    continue
                output = render_text(
                    snapshot,
                    metrics=metrics_list,
                    bar_width=width,
                    use_color=use_color,
                    diff_info=diff_info,
                )

            should_clear = (
                not args.once
                and not args.static
                and sys.stdout.isatty()
                and not args.json
            )
            if should_clear:
                clear_screen()
                print(output, end="\n")
                sys.stdout.flush()
            else:
                if cycle > 0 and not args.once:
                    print("-" * 80)
                print(output)

            if args.log_file:
                append_log(args.log_file, snapshot)
            if args.once:
                break
            time.sleep(max(0.5, args.interval))
            cycle += 1
    except KeyboardInterrupt:
        print("\nInterrupted, exiting.", file=sys.stderr)


if __name__ == "__main__":
    main()
