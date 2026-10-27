"""开发模式启动器：热更新（UI 就地刷新 + 非 UI 自动重启）。

用法:
    python dev.py

说明:
    - 仅用于本地开发，不要用于打包产物
    - UI 变更（app/widgets/**、app/styles.qss）：进程内就地刷新，保留传感器连接
    - 其它源码变更：子进程优雅退出（退出码 100），由本脚本整进程重启
    - 其它非零退出码视为崩溃，稍后再拉起
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "main.py"

# 与 main.py 中约定一致
RELOAD_EXIT_CODE = 100
ENV_FLAG = "VITAI_DEV_RELOAD"


def _start_app() -> subprocess.Popen:
    env = os.environ.copy()
    env[ENV_FLAG] = "1"

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    return subprocess.Popen(
        [sys.executable, str(MAIN)],
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
    )


def _stop_app(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def main() -> int:
    print("[dev] 热更新已开启：UI 就地刷新 / 非 UI 自动重启。Ctrl+C 退出。")
    print(f"[dev] 工作目录: {ROOT}")

    proc = _start_app()
    print(f"[dev] 已启动 PID={proc.pid}")

    try:
        while True:
            code = proc.wait()
            if code == RELOAD_EXIT_CODE:
                print("[dev] 检测到非 UI 源码变更，正在重启…")
                proc = _start_app()
                print(f"[dev] 已重启 PID={proc.pid}")
                continue

            if code == 0:
                print("[dev] 应用正常退出。")
                return 0

            print(f"[dev] 进程异常退出 code={code}，3 秒后重启…")
            time.sleep(3)
            proc = _start_app()
            print(f"[dev] 已重启 PID={proc.pid}")
    except KeyboardInterrupt:
        print("\n[dev] 正在退出…")
        _stop_app(proc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
