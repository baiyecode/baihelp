"""ch02 演示数据 + ch03 历史会话语料的幂等灌入:``uv run python -m app.db.seed``。

幂等规则(重复执行不产生重复数据):
- faq:表内 count>0 即整批跳过(八条种子视为一个整体,不逐条比对);
- 演示会话:user_id="seed-demo" 的会话存在即跳过,3 条消息随会话同批只建一次;
- 历史会话:user_id 前缀 "seed-hist-" 的会话存在即整批跳过(八通视为一个整体):
  六个可挖主题 + 两通噪音对照,给 ch03 挖知识 job 备语料,答案口径与
  data/knowledge/*.md 对齐,只造 user/assistant 纯文本行;
- 演示工单:固定号 T{当日YYYYMMDD}901 存在即跳过——序号取 901 高位段,
  避开 create_ticket 当日自增序号(001 起),给真实工单留出低位号段。

「邮费」「运费」刻意不进任何 question:验收判据⑥要求用户问「邮费是多少」时
query_faq 的 LIKE 落空(漏召回演示成立),完整回答照样生成;运费由谁承担只
写在 answer 列。退货政策一条的 answer 含「七天」,是 acceptance ⑤ 的判据词。

事务边界:整个 seed 在单个事务里完成,commit 归 session.begin() 上下文;
asyncio 陷阱(上游踩过):expire_on_commit=False 工厂里,新实例 flush 后即读
自增 id 属已加载属性,全程不发同步懒加载,不会触发 MissingGreenlet。
"""

import asyncio
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.engine import build_engine, build_session_factory
from app.db.models import Conversation, Faq, Message, Ticket

# 演示会话的固定 user_id(幂等判据,简报钉死)
DEMO_USER_ID = "seed-demo"
# 演示工单固定号尾段:T + 当日YYYYMMDD + 901
_DEMO_TICKET_SERIAL = "901"
# 演示会话里的工具调用申请单 id(assistant 行与 tool 行对号)
_DEMO_TOOL_CALL_ID = "call_seed_demo_001"

# FAQ 种子八条:(question, answer, category)。
# 注意:question 列一律不得出现「邮费」「运费」子串(漏召回验收前提);
# 第 ① 条 answer 含「七天」判据词,并可提及运费承担方。
_FAQ_ROWS: list[tuple[str, str, str]] = [
    (
        "退货政策是什么?",
        "亲,咱们支持七天无理由退货哦~收到货 7 天内、商品不影响二次销售"
        "(吊牌完整、没洗没穿)就能在「我的订单」里一键申请。质量问题导致的退货,"
        "运费由商家全额承担,您一分钱不用掏;非质量原因的退货,邮费得您自理哈~",
        "售后",
    ),
    (
        "怎么申请换货?",
        "很简单哒:打开「我的订单」→ 选中订单点「申请售后」→ 选「换货」,挑好"
        "想要的尺码或颜色提交就行。仓库收到退回的商品、质检通过后,新商品会在 "
        "48 小时内发出,全程物流信息在订单页都能看到~",
        "售后",
    ),
    (
        "一般多久发货?",
        "现货商品一般付款后 24 小时内发出(预售款、大促期间除外),发货后系统会"
        "自动推送物流单号给您。江浙沪隔天到,其他地区 2~4 天,偏远地区会稍慢一点,"
        "请您耐心等等哈~",
        "物流",
    ),
    (
        "保修政策是什么?",
        "电子产品自签收之日起享 12 个月官方保修;人为损坏、进水、私自拆机不在"
        "保修范围内哦。保修期内出问题,带上订单号找在线客服,我们会指导您寄修,"
        "来回邮费我们出~",
        "售后",
    ),
    (
        "怎么开发票?",
        "下单时在结算页勾选「需要发票」填好抬头就行;收货后 30 天内也可以到"
        "「我的订单」→「申请开票」补开。支持电子普通发票和增值税专用发票,"
        "电子票一般 1~3 个工作日开出到您预留的邮箱~",
        "售后",
    ),
    (
        "支持哪些支付方式?",
        "咱们支持微信支付、支付宝、银行卡直付,还有白条(先享后付)哦~大促期间"
        "用白条支付经常有免息活动,可以多关注首页公告~",
        "支付",
    ),
    (
        "会员积分怎么用?",
        "积分下单时能直接抵现金,100 积分抵 1 元;也能去「积分商城」换优惠券、"
        "换小礼品。积分自获得之日起一年内有效,记得常来用用呀~",
        "会员",
    ),
    (
        "优惠券怎么使用?",
        "结算时在「优惠明细」里勾选要用的券就行,大部分商品都能用(特殊标注的"
        "除外)。每单限用一张优惠券,不找零、不兑现;满减券没到门槛时用不了,"
        "系统会自动提示哒~",
        "优惠",
    ),
]

# 历史客服会话八通:(user_id, 会话状态, ((role, content), ...)),每通 3~5 轮。
# 话题钉死给 mine_qa 备料:01 退款到账时效 / 02 修改收货地址 / 03 优惠券过期 /
# 04 发票抬头修改 / 05 换货运费承担 / 06 预售发货时间六个可挖主题(每通至少
# 一轮「用户问真实问题 + 客服实质答案」),另有 07 纯寒暄、08 无答案转人工
# 两通噪音对照(挖矿应过滤,08 状态记「已转人工」)。
# 答案口径一律对齐 data/knowledge/*.md:退款 1~3 个工作日原路退回、偏远地区
# 运费 12 元、满 99 包邮/不满收 8 元(按券后实付)、优惠券十五日内有效过期
# 不可恢复、开票前订单页改抬头/已开可换开、非质量换货买家出退回运费且验收
# 合格三日内发新件、预售按商品页标注的预售期发货。只造 user/assistant
# 纯文本行,不造 tool 行。
_HISTORY_DIALOGUES: list[tuple[str, str, tuple[tuple[str, str], ...]]] = [
    (
        "seed-hist-01",
        "已结束",
        (
            ("user", "我昨天申请的退货退款,商家已经审核通过了,退款多久能到账啊?"),
            (
                "assistant",
                "亲,仓库验收商品合格后就会发起退款哒,退款会在一至三个工作日内"
                "按原支付路径退回哦~就是您用什么方式付款,就原路退回哪里哈~",
            ),
            ("user", "那要是过了三个工作日还没到账怎么办?"),
            (
                "assistant",
                "超时未到账您别着急,带上订单号来找在线客服,我们帮您核实退款"
                "流水、跟进到底哒~",
            ),
            ("user", "好的,那我再等等,谢谢啦。"),
            ("assistant", "不客气哒~退款到账后系统会给您发通知,有问题随时来问哦!"),
        ),
    ),
    (
        "seed-hist-02",
        "已结束",
        (
            (
                "user",
                "我刚下完单发现收货地址填错了,还能改吗?怕仓库发货太快来不及。",
            ),
            (
                "assistant",
                "亲,订单发货前都可以改地址哈~您在「我的订单」里找到订单点"
                "「修改地址」,或者把正确地址发我帮您改,发货前改都来得及哒~",
            ),
            ("user", "改成新疆乌鲁木齐的话,运费会不会变啊?"),
            (
                "assistant",
                "会的哦~新疆、西藏、内蒙古、甘肃、青海、宁夏这些偏远地区统一"
                "收取运费 12 元,改完地址后运费以结算页实时展示为准哒~",
            ),
            ("user", "这样啊,那我还是改到内地吧。对了,你们满多少包邮来着?"),
            (
                "assistant",
                "订单实付金额满 99 元包邮,不满 99 元收取运费 8 元,包邮门槛"
                "按券后实付金额计算哒~",
            ),
            ("user", "行,那我自己在订单页改了,谢谢提醒!"),
            ("assistant", "好嘞~改完记得核对一眼运费,有任何问题随时喊我哈!"),
        ),
    ),
    (
        "seed-hist-03",
        "已结束",
        (
            (
                "user",
                "我有张满 50 减 10 的优惠券,结算的时候怎么用不了,显示不可用?",
            ),
            (
                "assistant",
                "亲,优惠券自发放之日起十五日内有效,过期不可恢复也不能补发哦~"
                "您到「我的-优惠券」里看一下有效期哒~",
            ),
            ("user", "啊,我这张券是半年前领的,那确实是过期了,能再补发一张吗?"),
            (
                "assistant",
                "过期券补发不了哈~您可以关注首页的领券中心和大促活动,经常有"
                "新的满减券可以领哦~",
            ),
            ("user", "好吧,那我去领券中心看看。"),
            (
                "assistant",
                "好哒~结算页会自动展示您可用的优惠券,选中即抵扣,有需要随时"
                "来找我!",
            ),
        ),
    ),
    (
        "seed-hist-04",
        "已结束",
        (
            ("user", "你好,我下单时发票抬头把公司名打错了一个字,能改吗?"),
            (
                "assistant",
                "亲,分两种情况哈~开票前发现抬头错误,可以直接在订单页修改;"
                "已经开出的发票,可以申请换开哒~",
            ),
            ("user", "还没开呢,那我在哪里改?"),
            (
                "assistant",
                "在「我的订单」找到对应订单进发票详情就能改哦~本店默认开电子"
                "发票,确认收货后二十四小时内自动开票,开好后可以在订单详情页"
                "下载哒~",
            ),
            ("user", "那专票能开吗?我要报销用。"),
            (
                "assistant",
                "支持电子普通发票和增值税专用发票哈,专票记得把税号等开票信息"
                "填全哒~",
            ),
            ("user", "明白了,我先把抬头改过来。"),
            ("assistant", "好嘞~开票前改抬头都来得及,有问题随时来找我哦!"),
        ),
    ),
    (
        "seed-hist-05",
        "已结束",
        (
            ("user", "我买的卫衣拍大了想换个 M 码,换货运费谁来承担啊?"),
            (
                "assistant",
                "亲,尺码这种非质量原因的换货,退回的运费由您承担,新件发出的"
                "运费由我们商家承担哒~要是质量问题换货,来回运费都由我们出啦。",
            ),
            ("user", "那怎么算质量问题?"),
            (
                "assistant",
                "破损、漏发、错发、功能故障或者和页面描述严重不符,都算质量"
                "问题哈~您申请时拍照留证就行哒。",
            ),
            ("user", "明白了,那我在哪里申请换货?"),
            (
                "assistant",
                "在订单详情页选择换货,填好尺码和描述提交就行,签收后七天内"
                "支持同款换码或换色哦~客服确认后会把寄回地址和换货单号发给"
                "您哒。",
            ),
            ("user", "寄回去之后多久能收到新件?"),
            (
                "assistant",
                "仓库收到退件并验收合格后三日内就会发出新件哈,全程物流信息"
                "在订单页都能看到~",
            ),
            ("user", "行,那我今天就寄出去,谢谢啦。"),
            (
                "assistant",
                "不客气哒~寄回时记得吊牌完整、不影响二次销售,有问题随时找我"
                "哦!",
            ),
        ),
    ),
    (
        "seed-hist-06",
        "已结束",
        (
            (
                "user",
                "我拍了个预售的手办,页面上写着预售期 30 天,到底什么时候发货啊?",
            ),
            (
                "assistant",
                "亲,预售商品按商品页标注的预售期发货哒~您这款标注预售 30 天,"
                "会在预售期结束后的第一时间安排发出,发货后订单详情页就能查到"
                "物流单号哦~",
            ),
            ("user", "预售期能加急提前发货吗?我等着送人的。"),
            (
                "assistant",
                "预售款要按页面标注的预售期统一安排,暂时不支持加急提前发出哒~"
                "着急的话您看下同款现货商品哦。",
            ),
            ("user", "好吧,那我就再等等,谢谢。"),
            (
                "assistant",
                "不客气哒~发货后系统会自动推送物流单号给您,有问题随时来找我哦!",
            ),
        ),
    ),
    (
        "seed-hist-07",
        "已结束",
        (
            ("user", "在吗?"),
            ("assistant", "在的亲~请问有什么可以帮您?"),
            ("user", "哈哈没什么事,就是试试客服在不在。"),
            ("assistant", "一直在的呢~有任何问题都可以随时来问哈!"),
            ("user", "好嘞,那没事啦,拜拜。"),
            ("assistant", "感谢您的光临,祝您生活愉快,欢迎下次再来哦!"),
        ),
    ),
    (
        "seed-hist-08",
        "已转人工",
        (
            (
                "user",
                "客服你好,我上周退回去的包裹物流显示已签收,但退款一直没动静,"
                "问仓库又说没收到货,来回扯好几天了。",
            ),
            (
                "assistant",
                "亲,实在抱歉给您添麻烦了~这边帮您查一下退件签收记录和退款"
                "进度,麻烦提供一下订单号哦。",
            ),
            (
                "user",
                "订单号 2026091800123。仓库和快递互相踢皮球,我已经等了七天了,"
                "再不解决我就去投诉了。",
            ),
            (
                "assistant",
                "非常抱歉亲,您的情况我已经详细记录啦~退件包裹异常需要人工"
                "专员帮您核实处理,这里给您转接人工,处理进度会第一时间同步"
                "给您哒。",
            ),
            ("user", "行,那人工多久联系我?"),
            (
                "assistant",
                "已为您转接人工专员,会尽快与您联系处理,请您保持电话畅通~"
                "再次给您道歉哒。",
            ),
        ),
    ),
]


async def seed(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    """幂等灌入演示数据与挖矿语料,返回各表行数(表名 → 行数)。

    四条幂等判据各自独立判断:faq 以 count>0 跳过、演示会话以
    user_id="seed-demo" 存在跳过、历史会话以 user_id 前缀 "seed-hist-" 存在
    整批跳过、演示工单以固定号存在跳过;已存在的一方不会阻塞其余几方的补灌
    (例如历史库里只有 faq 时,本次只补会话与工单)。
    """
    async with session_factory() as session, session.begin():
        # 1. faq:表里已有数据(count>0)即整批跳过
        faq_count = await session.scalar(select(func.count()).select_from(Faq))
        if not faq_count:
            session.add_all(
                Faq(question=question, answer=answer, category=category)
                for question, answer, category in _FAQ_ROWS
            )
            await session.flush()

        # 2. 演示会话:user_id="seed-demo" 存在即跳过,消息随会话同批只建一次
        conversation = await session.scalar(
            select(Conversation)
            .where(Conversation.user_id == DEMO_USER_ID)
            .limit(1)
        )
        if conversation is None:
            conversation = Conversation(user_id=DEMO_USER_ID)
            session.add(conversation)
            await session.flush()  # 立刻拿自增 id,供消息/工单外键引用
            session.add_all(
                [
                    Message(
                        conversation_id=conversation.id,
                        role="user",
                        content="退货政策是什么?",
                    ),
                    Message(
                        conversation_id=conversation.id,
                        role="assistant",
                        content=None,  # assistant 纯工具调用,正文为空
                        tool_calls=[
                            {
                                "name": "query_faq",
                                "args": {"keyword": "退货"},
                                "id": _DEMO_TOOL_CALL_ID,
                            }
                        ],
                    ),
                    Message(
                        conversation_id=conversation.id,
                        role="tool",
                        # 镜像 query_faq 的返回格式:「问:…\n答:…」
                        content=(
                            f"问:{_FAQ_ROWS[0][0]}\n答:{_FAQ_ROWS[0][1]}"
                        ),
                        tool_call_id=_DEMO_TOOL_CALL_ID,
                    ),
                ]
            )
            await session.flush()

        # 3. 演示工单:固定号 T{当日YYYYMMDD}901 存在即跳过
        ticket_no = "T" + date.today().strftime("%Y%m%d") + _DEMO_TICKET_SERIAL
        if await session.get(Ticket, ticket_no) is None:
            session.add(
                Ticket(
                    ticket_no=ticket_no,
                    conversation_id=conversation.id,
                    description="演示工单:物流三天未更新,申请人工介入核实",
                    ticket_type="售后",
                )
            )
            await session.flush()

        # 4. 历史会话:user_id 前缀 "seed-hist-" 的会话存在即整批跳过
        #    (八通视为一个整体,与演示会话同型判据;前缀匹配靠 LIKE)
        history_conversation = await session.scalar(
            select(Conversation)
            .where(Conversation.user_id.like("seed-hist-%"))
            .limit(1)
        )
        if history_conversation is None:
            for user_id, status, turns in _HISTORY_DIALOGUES:
                conversation = Conversation(user_id=user_id, status=status)
                session.add(conversation)
                await session.flush()  # 立刻拿自增 id,供消息外键引用
                session.add_all(
                    Message(
                        conversation_id=conversation.id,
                        role=role,
                        content=content,
                    )
                    for role, content in turns
                )
            await session.flush()

        # 5. 统计各表行数(灌入后的最终状态),随事务提交一并返回
        return {
            "conversations": await session.scalar(
                select(func.count()).select_from(Conversation)
            ),
            "messages": await session.scalar(
                select(func.count()).select_from(Message)
            ),
            "faq": await session.scalar(select(func.count()).select_from(Faq)),
            "tickets": await session.scalar(select(func.count()).select_from(Ticket)),
            "history_conversations": await session.scalar(
                select(func.count())
                .select_from(Conversation)
                .where(Conversation.user_id.like("seed-hist-%"))
            ),
        }


if __name__ == "__main__":

    async def _main() -> None:
        """读配置建引擎 → 幂等灌数据 → 打印各表行数 → dispose 关停。"""
        engine = build_engine(get_settings().database_url)
        try:
            counts = await seed(build_session_factory(engine))
        finally:
            await engine.dispose()
        print("seed 完成,各表行数:")
        for table, count in counts.items():
            print(f"  {table}: {count}")

    asyncio.run(_main())
