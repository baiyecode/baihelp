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
#      第一轮告知订单号（输出仅留痕，不判分）；第二轮问"刚才的订单号"，
#      原始流中必须再次出现该订单号。模型必须从会话历史里复述数字，
#      这就是上下文记忆验收。服务端 delta 为紧凑 JSON 且 ensure_ascii=False，
#      ASCII 订单号在线上不会被转义，故直接 grep 原始流即等价于检查 delta 内容。
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

# uv run 需要 pyproject 上下文：统一切到仓库根目录（本脚本位于 scripts/ 下），
# 使脚本可从任意工作目录调用。
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

pass_count=0
fail_count=0

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
  curl -N -s --connect-timeout 5 --max-time 120 \
    -X POST "$BASE_URL$1" \
    -H "Content-Type: application/json" \
    -d "$2" || true
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
  '{"session_id":"acc-1","message":"你好，我想咨询一下退货"}')"
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
  "{\"session_id\":\"acc-2\",\"message\":\"我订单号是 ${ORDER_NO}，鞋子开胶了想退货\"}")"
printf '%s\n' "$round1" | sed 's/^/  | /'

echo "  第二轮：追问 $ORDER_NO 是多少"
round2="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-2\",\"message\":\"我刚才说的订单号是多少？\"}")"
printf '%s\n' "$round2" | sed 's/^/  | /'

# 模型必须从会话历史里复述第一轮给出的订单号——上下文记忆验收。
# delta 是紧凑 JSON 且 ensure_ascii=False，ASCII 数字串不会被转义，
# 因此对原始流 grep 等价于检查 delta content 里的订单号。
ctx_ok=0
grep -qF "$ORDER_NO" <<<"$round2" && ctx_ok=1
report "② 上下文：第二轮流中出现 $ORDER_NO" "$ctx_ok"

echo
echo "== 验收③ 结构化抽取：POST /api/extract =="
extract_body="{\"text\":\"订单${ORDER_NO_EXTRACT}，收到的杯子碎了，我要退款，钱退回到我原支付方式就行\"}"
resp="$(curl -s --connect-timeout 5 --max-time 120 \
  -X POST "$BASE_URL/api/extract" \
  -H "Content-Type: application/json" \
  -d "$extract_body")" || true
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
  echo "结论: FAIL（$pass_count/3 项判据通过）"
  exit 1
fi
echo "结论: PASS（3/3）"
exit 0
