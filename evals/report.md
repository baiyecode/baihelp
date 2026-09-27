
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
