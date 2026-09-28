
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

---

# Ch03 检索与挖矿评估报告(live)

- 日期:2026-09-29 · 嵌入:BAAI/bge-m3(SiliconFlow)· 生成模型:deepseek-v4-flash-0731(阿里云百炼)· 向量库:milvus-lite 3.2.1 本地库
- 知识库状态:knowledge_chunks 36 行全 done(语料 20 + 挖矿 16),staging kept 16 / discarded 12
- 前置说明:首轮建库后 live 验收揪出挖矿 Prompt 过滤缺口(mock 工具演示对话与敷衍回答被当知识入库,挤占检索 top 位)→ Prompt 补个案查询/敷衍回答两条排除规则并清库重挖;另有 MySQL 清行后 Milvus 孤儿向量占位问题 → 重建向量侧(见 dev-notes 阶段 13)。本报告为修复后的终验。

## 检索评估(`evals/run_retrieval_eval.py`)

| id | expect | query | 判分 | 命中详情 |
|---|---|---|---|---|
| faq-postage | hit | 邮费是多少 | ok | rank=2 score=0.703 含全部关键词 |
| faq-shipping-rephrase | hit | 快递费怎么算 | ok | rank=1 score=0.700 含全部关键词 |
| nohit-boss | no_hit | 你们老板是谁 | ok | 零命中 |
| nohit-weather | no_hit | 今天天气怎么样 | ok | 零命中(T10 遗留风险项,实测未误召) |
| policy-return-how | hit | 怎么退货 | ok | rank=2 score=0.730 含全部关键词 |
| policy-refund-eta | hit | 退款多久能到账 | ok | rank=1 score=0.838 含全部关键词 |
| policy-return-process | hit | 退货流程是什么 | ok | rank=1 score=0.807 含全部关键词 |
| faq-coupon-expired | hit | 优惠券过期了还能用吗 | ok | rank=1 score=0.790 含全部关键词 |
| faq-invoice | hit | 怎么开发票 | ok | rank=1 score=0.691 含全部关键词 |
| faq-payment | hit | 支持什么付款方式 | ok | rank=1 score=0.752 含全部关键词 |
| faq-shipping-eta | hit | 下单后多久发货 | ok | rank=1 score=0.797 含全部关键词 |
| faq-exchange | hit | 尺码不合适怎么换货 | ok | rank=1 score=0.764 含全部关键词 |
| manual-warranty | hit | 保修期是多久 | ok | rank=1 score=0.744 含全部关键词 |

hit 命中达标(11 条): **11/11** | no_hit 零命中(2 条): **2/2** | **门禁结论: PASS**

## 挖矿评估(`evals/run_mine_eval.py`)

- expect_qa 满足(10 个挖矿样例): **11/11**(含变说法样例「钱什么时候回来」→ 退款时效)
- 噪音行 0 抽取(4 条): **4/4**(纯寒暄 / 转人工 / **敷衍话术 / mock 订单个案**——后两类为 Prompt 修复新增反例)
- **门禁结论: PASS**

## ch02 工具选型回归(`evals/run_tool_eval.py`)

- 整体准确率 19 条: **18/19(94.7%)**;none 类误调 0/2;「邮费」样例(faq-postage-miss)选中 query_faq ✅
- 三门禁(整体 ≥90% / none 误调 0 / 邮费样例必选 query_faq)全过 → **PASS**
- 唯一失败 logistics-04(时效问题被 query_order 抢答),与 ch02 终验完全相同,门禁容忍内

## 验收脚本(同日终验)

`bash scripts/acceptance.sh` → **7/7 验收 PASS、11/11 判据全过(exit 0)**:⑥「邮费是多少」向量召回判「包邮/99」命中,完整回复为「普通地区实付满 99 元包邮;不满 99 元运费 8 元;偏远地区 12 元」;⑤「退货政策」回答含「七天」;⑦ 挖矿 --self-test 退出 0。
