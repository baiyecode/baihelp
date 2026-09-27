
id        order_no  complaint_type  expected_solution  error
------------------------------------------------------------
reg-01    ok        ok              ok               
reg-02    ok        ok              ok               
reg-03    ok        ok              ok               
reg-04    ok        ok              ok               
reg-05    ok        ok              ok               
reg-06    ok        ok              ok               
reg-07    ok        ok              ok               
reg-08    ok        ok              ok               
reg-09    ok        ok              ok               
noord-01  ok        ok              ok               
noord-02  ok        ok              X                
noord-03  ok        ok              ok               
noord-04  ok        ok              X                
mixed-01  ok        ok              ok               
mixed-02  ok        ok              ok               
mixed-03  ok        ok              X                
typo-01   ok        ok              ok               
typo-02   ok        ok              ok               
typo-03   ok        ok              X                
vague-01  ok        ok              ok               
vague-02  ok        ok              ok               
vague-03  ok        ok              ok               

准确率（22 条）: order_no 22/22 (100.0%) | complaint_type 22/22 (100.0%) | expected_solution 18/22 (81.8%)
门禁: order_no >= 90% -> 达标 | complaint_type == 100% -> 达标 | expected_solution >= 80% -> 达标
门禁结论: PASS

失败详情:
  - noord-02:
      expected: order_no=None, complaint_type='其他', expected_solution='售后页面一直报错，希望人工帮忙处理售后申请'
      got:      order_no=None, complaint_type='其他', expected_solution='帮我想想办法'
  - noord-04:
      expected: order_no=None, complaint_type='其他', expected_solution='客服承诺的优惠券补偿未到账，要求给个说法'
      got:      order_no=None, complaint_type='其他', expected_solution='给我个说法'
  - mixed-03:
      expected: order_no='SO20260913003', complaint_type='物流问题', expected_solution='快递显示签收但实际未收到货，要求查清包裹去向'
      got:      order_no='SO20260913003', complaint_type='物流问题', expected_solution='查清楚快递件去哪了'
  - typo-03:
      expected: order_no='SO20260917003', complaint_type='发票售后', expected_solution='之前忘记开票，希望尽快补开发票，抬头写个人'
      got:      order_no='SO20260917003', complaint_type='发票售后', expected_solution='麻烦尽快补开一下，抬头写‘个人’就行'

---

# Ch02 工具选型评估报告(live)

- 日期:2026-09-28 · 模型:deepseek-v4-flash-0731(阿里云百炼兼容端点)· runner:`evals/run_tool_eval.py`
- 前置说明:首轮 live 跑分(16/19)及三次「复测」实际打在一个**未被杀掉的旧服务进程**上(PID 29836,`pkill -f` 在 Git Bash 对 Windows python 进程静默失效)——修复轮 1-3 后的首次真实验证见下「终验」;首轮数据仅作过程留痕,不作结论依据。

## 终验(修复轮 3 代码真正生效后)

- **整体准确率:18/19(94.7%)** — 门禁 ≥90% 达标
- **none 类误调:0/2** — 达标
- **「邮费」漏召回样例:选中 query_faq** — 达标(工具选对了,LIKE 查空属预期漏召回)
- **门禁结论:PASS**

唯一失败:`logistics-04`「包裹什么时候能到?订单 4004」→ 预测 query_order(期望 query_logistics)。属「时效问题归口」的边界抖动,门禁容忍 1 错;query_order 描述已明示不含预计送达,后续可继续观察。

## 过程记录(工具描述三轮迭代,标注集零改动)

| 轮 | 触发 | 修复 | 提交 |
|----|------|------|------|
| 1 | 验收④:模型拒查裸编号 1001,索要「完整单号」 | id 参数描述放宽「原样传入无需校验格式」 | `70a8b77` |
| 2 | eval 3 失败:两条投诉用例不触发 create_ticket / 物流时效被 query_order 抢答 | 强化 create_ticket 触发条件、时效问题归口 query_logistics | `904f353` |
| 3 | 验收④抖动:模型仍复述长格式示例反问 | **删除「ORD-20260927-001」长格式示例**(反面教材),单向指令「不要质疑格式」 | `c280d52` |

教训:参数示例中的格式样例会被模型当成校验规则反问用户;「示例多样性」在工具描述里是负资产,单一裸示例 + 明确「原样传入」才稳。

## 验收脚本(同日终验)

`bash scripts/acceptance.sh` → **6/6 PASS**(含④物流工具帧、⑤退货政策命中、⑥邮费漏召回演示)。
