#!/usr/bin/env bash
# ============================================================================
# 端到端验收脚本 —— ch01/pure-chat + ch02/function-calling + ch03/rag-knowledge-base
# （spec §10 / task-10 Step 1 立稿；ch02 task-13 扩展验收④⑤⑥；ch03 task-11 反转⑥、新增⑦）
#
# 前提假设（本脚本不负责满足，违反时结果无意义）：
#   1. 服务已在运行：请在仓库根目录先执行 `uv run uvicorn app.main:app`
#      （默认监听 127.0.0.1:8000）。本脚本【不会】自行启动服务。
#      目标地址可用环境变量 BASE_URL 覆盖（默认 http://127.0.0.1:8000）。
#   2. 项目根目录 `.env` 已配置真实凭据（LLM_API_KEY，参考 .env.example）：
#      服务端启动 fail-fast 依赖它，对话/抽取质量依赖真实模型。
#
# 七条验收：
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
#   ④ 工具调用（ch02）：POST「订单 1001 的物流到哪了」——
#      流中至少一行 `data: {"tool"` 开头的工具事件帧（task-8 落地的帧格式：
#      {"tool":{"name":...,"status":...}}），且末事件仍为 `data: [DONE]`；
#      完整流与合并 delta 回答均照打留痕，回答措辞不判分。
#   ⑤ FAQ 命中（ch02）：POST「退货政策是什么」——
#      工具事件帧存在（query_faq 触发），且合并 delta 回答含「七天」判据词
#      （task-5 种子 FAQ：该条 answer 含「七天」）。
#   ⑥ 向量召回演示（ch03）：POST「邮费是多少」——
#      ch02 时是漏召回演示（种子 question 列刻意不含「邮费/运费」，LIKE 落空）；
#      ch03 向量检索上线后该问句语义命中退货政策.md 的「运费说明」块，判据反转为
#      双条件：工具帧名含 query_faq，且合并回复含「包邮」或「99」（任一即可，
#      双判据降 flaky）。
#   ⑦ 挖矿自检（ch03）：uv run python -m app.knowledge.mine_qa --self-test——
#      零外部依赖（sqlite 内存 + 内联假会话 + 罐头 LLM）的挖矿全管线离线自测，
#      退出码 0 判 PASS。不依赖服务，放在前置健康检查之前执行。
#
# 结果：逐条打印 PASS/FAIL + 汇总；七条验收全部通过 exit 0，任一失败 exit 1；
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

# 以下两个判据辅助函数供验收④⑤⑥使用，逻辑与验收②内联的 delta 合并同源：
# 都是读 SSE 流（stdin）、用 Python 按 UTF-8 解析、写 stdout，免疫
# Windows 控制台 locale（cp936）误解码与 JSON 转义差异。

merge_delta_text() { # 合并所有 OpenAI 兼容 delta content 为完整回复文本
  uv run python -c '
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
'
}

tool_names() { # 依次提取所有工具事件帧的 name 字段，空格分隔（无工具帧输出空串）
  uv run python -c '
import sys, json
names = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: {"):
        continue
    try:
        tool = json.loads(line[6:]).get("tool")
    except Exception:
        continue
    if isinstance(tool, dict) and tool.get("name"):
        names.append(tool["name"])
sys.stdout.write(" ".join(names))
'
}

# 验收⑦（ch03 挖矿自检）不依赖服务：--self-test 零外部依赖，放在前置健康检查
# 之前执行——服务未起时也能单独拿到挖矿自检结果（①-⑥ 仍以服务可达为前提）。
echo
echo "== 验收⑦ 挖矿自检：uv run python -m app.knowledge.mine_qa --self-test =="
self7_ok=0
if uv run python -m app.knowledge.mine_qa --self-test 2>&1 | sed 's/^/  | /'; then
  self7_ok=1
fi
report "⑦ 挖矿自检：--self-test 退出码 0（挖矿全管线离线跑通）" "$self7_ok"

# 前置健康检查（不计入七条验收）：服务不可达时直接给出可操作提示。
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
echo "== 验收④ 工具调用：POST「订单 1001 的物流到哪了」（ch02 Function Calling）=="
stream4="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-4-$RUN_ID\",\"message\":\"订单 1001 的物流到哪了\"}")"
printf '%s\n' "$stream4" | sed 's/^/  | /'

# 工具事件帧判据：帧格式固定为 data: {"tool":{"name":...,"status":...}}（task-8），
# 紧凑 JSON 无空格，直接前缀匹配即可判定「至少调用了一次工具」。
tool4_ok=0
grep -q '^data: {"tool"' <<<"$stream4" && tool4_ok=1
last4="$(printf '%s\n' "$stream4" | sed -e 's/\r$//' -e '/^[[:space:]]*$/d' | tail -n 1)"
done4_ok=0
[[ "$last4" == "data: [DONE]" ]] && done4_ok=1
report "④ 工具：含工具事件帧（data: {\"tool\"...）" "$tool4_ok"
report "④ 工具：以 data: [DONE] 收尾（末事件=$last4）" "$done4_ok"

# 合并 delta 回答照打留痕：物流轨迹由模型自由转述，只验走了工具，不判措辞。
stream4_text="$(printf '%s\n' "$stream4" | merge_delta_text)"
echo "  合并回复：$stream4_text"

echo
echo "== 验收⑤ FAQ 命中：POST「退货政策是什么」（query_faq → 种子 answer 含「七天」）=="
stream5="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-5-$RUN_ID\",\"message\":\"退货政策是什么\"}")"
printf '%s\n' "$stream5" | sed 's/^/  | /'

tool5_ok=0
grep -q '^data: {"tool"' <<<"$stream5" && tool5_ok=1
report "⑤ FAQ：含工具事件帧（query_faq 触发）" "$tool5_ok"

# 「七天」判据词在种子 answer 里，必须合并全部 delta 后再匹配（同验收②的理由）。
stream5_text="$(printf '%s\n' "$stream5" | merge_delta_text)"
echo "  合并回复：$stream5_text"
seven_ok=0
grep -qF "七天" <<<"$stream5_text" && seven_ok=1
report "⑤ FAQ：合并回复含「七天」" "$seven_ok"

echo
echo "== 验收⑥ 向量召回演示：POST「邮费是多少」（ch03 向量检索）=="
echo "  ch02 时代本条是漏召回演示（LIKE 落空）；ch03 向量检索上线后该问句语义"
echo "  命中退货政策.md 的「运费说明」块。判据双条件：走了 query_faq 工具，且"
echo "  合并回复含「包邮」或「99」（任一即可，双判据降 flaky）。"
stream6="$(curl_post /api/chat/stream \
  "{\"session_id\":\"acc-6-$RUN_ID\",\"message\":\"邮费是多少\"}")"
printf '%s\n' "$stream6" | sed 's/^/  | /'

faq6_names="$(printf '%s\n' "$stream6" | tool_names)"
echo "  工具帧名：${faq6_names:-（无）}"
faq6_ok=0
grep -qF "query_faq" <<<"$faq6_names" && faq6_ok=1
report "⑥ 向量召回：工具帧名含 query_faq" "$faq6_ok"

# 「包邮」/「99」判据词在「运费说明」块 answer 里，必须合并全部 delta 后再匹配
# （与验收②⑤同理：token 边界可能把判据词切进相邻两个 delta）。
stream6_text="$(printf '%s\n' "$stream6" | merge_delta_text)"
echo "  合并回复：$stream6_text"
ship6_ok=0
grep -qE "包邮|99" <<<"$stream6_text" && ship6_ok=1
report "⑥ 向量召回：合并回复含「包邮」或「99」" "$ship6_ok"

echo
echo "== 验收汇总 =="
echo "  ① 流式对话（delta 逐行 + [DONE] 收尾）: $([[ $delta_ok == 1 && $done_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ② 上下文记忆（第二轮复述订单号）:       $([[ $ctx_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ③ 结构化抽取（JSON + complaint_type）:  $([[ $extract_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ④ 工具调用（工具帧 + [DONE] 收尾）:     $([[ $tool4_ok == 1 && $done4_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ⑤ FAQ 命中（工具帧 + 回答含「七天」）:  $([[ $tool5_ok == 1 && $seven_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ⑥ 向量召回演示（工具帧 + 回复含「包邮」/「99」）: $([[ $faq6_ok == 1 && $ship6_ok == 1 ]] && echo PASS || echo FAIL)"
echo "  ⑦ 挖矿自检（mine_qa --self-test 退出码 0）:      $([[ $self7_ok == 1 ]] && echo PASS || echo FAIL)"

if [[ "$fail_count" -ne 0 ]]; then
  echo "结论: FAIL（$pass_count/$((pass_count + fail_count)) 项判据通过）"
  exit 1
fi
# PASS 分支同样按动态判据计数输出(与 FAIL 分支口径一致),七条验收语义用文字保留
echo "结论: PASS(七条验收全过,$pass_count 项判据全部通过)"
exit 0
