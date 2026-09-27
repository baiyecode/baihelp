#!/usr/bin/env bash
# ============================================================================
# 端到端验收脚本 —— ch01/pure-chat（spec §10 / task-10 Step 1）
#
# 前提假设（本脚本不负责满足，违反时结果无意义）：
#   1. 服务已在运行：请在仓库根目录先执行 `uv run uvicorn app.main:app`
#      （默认监听 127.0.0.1:8000）。本脚本【不会】自行启动服务。
#      目标地址可用环境变量 BASE_URL 覆盖（默认 http://127.0.0.1:8000）。
#   2. 项目根目录 `.env` 已配置真实凭据（LLM_API_KEY，参考 .env.example）：
#      服务端启动 fail-fast 依赖它，对话/抽取质量依赖真实模型。
#
# 三条验收：
#   ① 流式对话：POST /api/chat/stream（curl -N）——
#      至少一行 `data: {"choices"` 开头的 OpenAI 兼容 delta 事件，
#      且流的最后一个非空事件是 `data: [DONE]`。
#   ② 上下文记忆：同一 session 两轮对话——
#      第一轮告知订单号（输出仅留痕，不判分）；第二轮要求原样复述订单号。
#      模型必须从会话历史里复述数字，这就是上下文记忆验收。
#      判据必须先把所有 delta content 合并成完整回复再匹配——模型的
#      tokenizer 可能把订单号从中间切成多个 delta（如 "…：SO" + "20260927001"），
#      直接 grep 原始流会因 token 边界切分而漏检。
#   ③ 结构化抽取：POST /api/extract——
#      响应必须是可解析 JSON 且含 `complaint_type` 字段。
#
# 结果：逐条打印 PASS/FAIL + 汇总；三条全过 exit 0，任一失败 exit 1；
#       前置健康检查失败（服务未运行/不可达）exit 2 并给出启动提示。
# ============================================================================
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
ORDER_NO="SO20260927001"        # 验收②第一轮告知、第二轮追问的订单号
ORDER_NO_EXTRACT="SO20260927002" # 验收③抽取样例中的订单号
# 会话 id 每次运行随机化：服务端会话是进程内存态、跨脚本运行存活，
# 固定 id 会让重跑时叠加历史（模型回复"再次确认"），破坏测试隔离。
RUN_ID="$(date +%s)$RANDOM"

# uv run 需要 pyproject 上下文：统一切到仓库根目录（本脚本位于 scripts/ 下），
# 使脚本可从任意工作目录调用。
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

pass_count=0
fail_count=0

# 请求体经 UTF-8 临时文件传递：Windows 下内联中文作为 curl.exe 命令行参数
# 会按本地代码页（cp936）编码，服务端按 UTF-8 解析必然失败（400 Bad Request）；
# bash 内建 printf 直写文件保留脚本内的 UTF-8 字节，curl --data-binary @file 原样发送。
BODY_FILE="$(mktemp "${TMPDIR:-/tmp}/acc-body.XXXXXX.json")"
trap 'rm -f "$BODY_FILE"' EXIT

report() { # report <名称> <1=通过|0=失败>
  if [[ "$2" == "1" ]]; then
    pass_count=$((pass_count + 1))
    printf '  [PASS] %s\n' "$1"
  else
    fail_count=$((fail_count + 1))
    printf '  [FAIL] %s\n' "$1"
  fi
}

curl_post() { # curl_post <path> <json-body>（-N 流式；失败不中断，由判据定 PASS/FAIL）
  printf '%s' "$2" > "$BODY_FILE"
  curl -N -s --connect-timeout 5 --max-time 120 \
    -X POST "$BASE_URL$1" \
    -H "Content-Type: application/json" \
    --data-binary @"$BODY_FILE" || true
}

# 前置健康检查（不计入三条验收）：服务不可达时直接给出可操作提示。
echo "== 前置检查：GET $BASE_URL/api/healthz =="
if ! healthz="$(curl -s --connect-timeout 3 --max-time 10 "$BASE_URL/api/healthz")"; then
  echo "服务不可达：请先在仓库根目录运行 \`uv run uvicorn app.main:app\`，再执行本脚本。" >&2
  exit 2
fi
echo "  healthz: $healthz"

echo
echo "== 验收① 流式对话：POST /api/chat/stream（curl -N，delta 逐行可见）=="
stream1="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-1-$RUN_ID\",\"message\":\"你好，我想咨询一下退货\"}")"
printf '%s\n' "$stream1" | sed 's/^/  | /'

delta_ok=0
grep -q '^data: {"choices"' <<<"$stream1" && delta_ok=1
last_event="$(printf '%s\n' "$stream1" | sed -e 's/\r$//' -e '/^[[:space:]]*$/d' | tail -n 1)"
done_ok=0
[[ "$last_event" == "data: [DONE]" ]] && done_ok=1
report "① 流式：含 delta 事件行" "$delta_ok"
report "① 流式：以 data: [DONE] 收尾（末事件=$last_event）" "$done_ok"

echo
echo "== 验收② 上下文记忆：同 session 两轮，第二轮需复述第一轮订单号 =="
echo "  第一轮（留痕，不判分）：告知订单号 $ORDER_NO"
round1="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-2-$RUN_ID\",\"message\":\"我订单号是 ${ORDER_NO}，鞋子开胶了想退货\"}")"
printf '%s\n' "$round1" | sed 's/^/  | /'

echo "  第二轮：要求原样复述 $ORDER_NO"
round2="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-2-$RUN_ID\",\"message\":\"请一字不差地原样告诉我，我刚才说的订单号是多少？\"}")"
printf '%s\n' "$round2" | sed 's/^/  | /'

# 模型必须从会话历史里复述第一轮给出的订单号——上下文记忆验收。
# 先合并所有 delta content（与验收③同为 Python 判据：token 边界可能把订单号
# 切进相邻两个 delta，grep 原始流会漏检；Python 解析同时免疫 JSON 转义差异）。
round2_text="$(printf '%s\n' "$round2" | uv run python -c '
import sys, json
parts = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: {"):
        continue
    try:
        parts.append(json.loads(line[6:])["choices"][0]["delta"].get("content", ""))
    except Exception:
        pass
sys.stdout.write("".join(parts))
')"
ctx_ok=0
grep -qF "$ORDER_NO" <<<"$round2_text" && ctx_ok=1
report "② 上下文：第二轮回复中出现 $ORDER_NO" "$ctx_ok"

echo
echo "== 验收③ 结构化抽取：POST /api/extract =="
extract_body="{\"text\":\"订单${ORDER_NO_EXTRACT}，收到的杯子碎了，我要退款，钱退回到我原支付方式就行\"}"
printf '%s' "$extract_body" > "$BODY_FILE"
resp="$(curl -s --connect-timeout 5 --max-time 120 \
  -X POST "$BASE_URL/api/extract" \
  -H "Content-Type: application/json" \
  --data-binary @"$BODY_FILE")" || true
echo "  | $resp"

# JSON 可解析 + complaint_type 字段存在。显式按 UTF-8 解码字节流，
# 避免 Windows 控制台 locale（如 cp936）误解码中文导致误判。
extract_ok=0
if printf '%s' "$resp" | uv run python -c \
  'import json,sys; d=json.loads(sys.stdin.buffer.read().decode("utf-8")); sys.exit(0 if isinstance(d,dict) and "complaint_type" in d else 1)' \
  2>/dev/null; then
  extract_ok=1
fi
report "③ 抽取：JSON 可解析且含 complaint_type" "$extract_ok"

echo
echo "== 验收汇总 =="
echo "  ① 流式对话（delta 逐行 + [DONE] 收尾）: $([[ $delta_ok == 1 && $done_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ② 上下文记忆（第二轮复述订单号）:       $([[ $ctx_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ③ 结构化抽取（JSON + complaint_type）:  $([[ $extract_ok == 1 ]] && echo PASS || echo FAIL)"

if [[ "$fail_count" -ne 0 ]]; then
  echo "结论: FAIL（$pass_count/4 项判据通过）"
  exit 1
fi
echo "结论: PASS（3/3）"
exit 0
