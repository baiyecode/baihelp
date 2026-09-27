"""ch02 演示数据幂等灌入:``uv run python -m app.db.seed``。

幂等规则(重复执行不产生重复数据):
- faq:表内 count>0 即整批跳过(八条种子视为一个整体,不逐条比对);
- 演示会话:user_id="seed-demo" 的会话存在即跳过,3 条消息随会话同批只建一次;
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


async def seed(session_factory: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    """幂等灌入演示数据,返回各表行数(表名 → 行数)。

    三条幂等判据各自独立判断:faq 以 count>0 跳过、演示会话以
    user_id="seed-demo" 存在跳过、演示工单以固定号存在跳过;已存在的一方
    不会阻塞其余两方的补灌(例如历史库里只有 faq 时,本次只补会话与工单)。
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

        # 4. 统计各表行数(灌入后的最终状态),随事务提交一并返回
        return {
            "conversations": await session.scalar(
                select(func.count()).select_from(Conversation)
            ),
            "messages": await session.scalar(
                select(func.count()).select_from(Message)
            ),
            "faq": await session.scalar(select(func.count()).select_from(Faq)),
            "tickets": await session.scalar(select(func.count()).select_from(Ticket)),
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
