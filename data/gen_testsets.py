"""合成对话测试集:10 个人设,每个约 70 轮中文对话。

每个对话围绕一个 persona,预埋一组"可检索事实"(facts),自然散布在
user 发言中,并穿插日常闲聊把轮次凑到 ~70。facts 单独落盘到
data/testsets/<user_id>.facts.json,供检索评测(evaluate.py)比对。

对话 JSON 结构与 memory_framework.conversation.load_conversation 兼容:
    {"user_id": str, "messages": [{"role", "content"}, ...]}

用法:
    PYTHONPATH=. python data/gen_testsets.py
"""

import json
import os

OUT_DIR = "data/testsets"
TARGET_TURNS = 70  # 目标 user+assistant 消息总数(约 35 个来回)

# 每个 persona:一组结构化事实 + 用于承载事实的中文句子 + 主题闲聊句池。
# fact 的 key 用于评测时构造 query,value 是预期召回的关键词(任一命中即算召回)。
PERSONAS = [
    {
        "user_id": "chef_lin",
        "desc": "美食爱好者",
        "facts": {
            "饮食禁忌": (["不吃香菜", "对海鲜过敏"], "我不吃香菜,而且对海鲜过敏,点菜要注意"),
            "拿手菜": (["会做红烧肉", "红烧肉"], "我最拿手的菜是红烧肉,炖得软烂入味"),
            "厨房设备": (["买了铸铁锅", "铸铁锅"], "我最近入手了一口铸铁锅,煎牛排特别香"),
        },
        "chat": ["今天想研究一道新菜", "你觉得川菜和粤菜哪个更难", "周末打算去逛菜市场",
                 "昨天试了新的调味料", "看了个做面点的视频", "想学做甜点但总失败",
                 "高汤到底要熬多久", "刀工怎么练才快", "家里的调料柜又满了"],
    },
    {
        "user_id": "dev_zhao",
        "desc": "后端工程师",
        "facts": {
            "编程语言": (["主力语言是 Go", "用 Go", "Go"], "我是后端工程师,主力语言是 Go"),
            "学习方向": (["在学 Rust", "Rust"], "最近在自学 Rust,想搞懂它的所有权模型"),
            "工作城市": (["在杭州工作", "杭州"], "我在杭州一家做支付的公司上班"),
        },
        "chat": ["今天线上出了个诡异的 bug", "在看一篇讲分布式锁的文章", "同事推荐了一本讲架构的书",
                 "周会又开了两个小时", "想优化一下数据库查询", "在纠结微服务要不要拆",
                 "写单测写到怀疑人生", "云账单这个月又超了", "在研究怎么做限流"],
    },
    {
        "user_id": "runner_wang",
        "desc": "跑步爱好者",
        "facts": {
            "运动习惯": (["每天早上 6 点跑步", "跑 5 公里", "6 点"], "我习惯每天早上 6 点起床跑 5 公里"),
            "比赛目标": (["报名了全程马拉松", "马拉松"], "我报名了今年秋天的全程马拉松"),
            "装备偏好": (["穿某品牌的碳板鞋", "碳板鞋"], "我一直穿碳板跑鞋,提速明显"),
        },
        "chat": ["今天配速比昨天快", "膝盖有点不舒服", "在看马拉松的训练计划",
                 "天气转凉适合跑步", "补给到底吃什么好", "跑完拉伸很重要",
                 "想约人一起晨跑", "心率一直压不下来", "换了双新袜子"],
    },
    {
        "user_id": "reader_chen",
        "desc": "读书人",
        "facts": {
            "阅读偏好": (["喜欢科幻小说", "科幻"], "我最爱读科幻小说,尤其是硬科幻"),
            "最近在读": (["在读《三体》", "三体"], "我最近在重读《三体》三部曲"),
            "阅读习惯": (["睡前读纸质书", "纸质书"], "我习惯睡前读纸质书,不喜欢电子屏"),
        },
        "chat": ["今天读到一个精彩的设定", "在纠结要不要买新书架", "这本书的翻译有点生硬",
                 "想找个读书会", "作者的世界观好宏大", "读得太晚又熬夜了",
                 "在整理读书笔记", "图书馆借的书快到期了", "想重读一遍经典"],
    },
    {
        "user_id": "cat_owner_liu",
        "desc": "养猫的人",
        "facts": {
            "宠物": (["养了两只猫", "两只猫", "橘猫"], "我养了两只猫,一只橘猫一只狸花"),
            "猫的习惯": (["猫喜欢半夜跑酷", "半夜"], "我家猫喜欢半夜跑酷,吵得睡不着"),
            "过敏": (["对某种猫粮过敏", "猫粮过敏"], "有只猫对某种猫粮过敏,换了好几个牌子"),
        },
        "chat": ["今天猫又拆家了", "在研究猫的营养", "猫砂盆该换了",
                 "带猫去做体检", "猫抓板又被抓烂了", "想给猫买个爬架",
                 "两只猫打架了", "猫毛到处都是", "在给猫剪指甲"],
    },
    {
        "user_id": "traveler_sun",
        "desc": "旅行者",
        "facts": {
            "去过的地方": (["去过西藏", "西藏"], "我去年自驾去了西藏,风景太震撼了"),
            "旅行方式": (["喜欢穷游", "穷游"], "我喜欢穷游,住青旅背包走"),
            "下个目的地": (["想去冰岛", "冰岛"], "我一直想去冰岛看极光"),
        },
        "chat": ["在做下次旅行的攻略", "机票又涨价了", "想找个小众目的地",
                 "整理上次旅行的照片", "在学基础的当地语言", "背包又要换新的了",
                 "青旅遇到很多有趣的人", "想拍一组旅行 vlog", "签证办起来好麻烦"],
    },
    {
        "user_id": "musician_zhou",
        "desc": "音乐人",
        "facts": {
            "乐器": (["会弹吉他", "吉他"], "我弹了十年吉他,主要玩指弹"),
            "音乐偏好": (["喜欢爵士乐", "爵士"], "我特别喜欢爵士乐,尤其是钢琴三重奏"),
            "创作": (["在写一首原创", "原创歌"], "我最近在写一首原创歌,卡在副歌"),
        },
        "chat": ["今天练琴手指好疼", "在扒一首很难的曲子", "想组个乐队",
                 "耳机煲了很久", "在研究编曲软件", "去看了场 livehouse",
                 "琴弦该换了", "在学乐理", "录了段小样"],
    },
    {
        "user_id": "gardener_wu",
        "desc": "园艺爱好者",
        "facts": {
            "种的植物": (["种了多肉", "多肉"], "我阳台上种了几十盆多肉"),
            "园艺难题": (["总是浇水过多", "浇水"], "我最大的毛病是浇水过多,烂根好几次"),
            "梦想": (["想要个小花园", "花园"], "我梦想有个自己的小花园种月季"),
        },
        "chat": ["今天多肉又出状态了", "在研究怎么配土", "买了新的花盆",
                 "阳台光照不太够", "在扦插新的品种", "叶子有点发黄",
                 "想装个自动浇水", "虫害又来了", "在等种子发芽"],
    },
    {
        "user_id": "gamer_xu",
        "desc": "游戏玩家",
        "facts": {
            "游戏偏好": (["喜欢玩策略游戏", "策略游戏"], "我最爱玩策略游戏,回合制那种"),
            "平台": (["主要用 PC", "PC"], "我主要在 PC 上玩,配了台高配主机"),
            "在玩的游戏": (["在打某个开放世界", "开放世界"], "最近沉迷一个开放世界游戏,肝了几十小时"),
        },
        "chat": ["昨晚又通宵了", "这个 boss 太难了", "在等新游戏发售",
                 "显卡想升级了", "组队打副本", "剧情让我破防",
                 "在肝成就", "手柄该换了", "研究攻略研究到半夜"],
    },
    {
        "user_id": "parent_he",
        "desc": "新手爸妈",
        "facts": {
            "孩子": (["有个两岁女儿", "两岁", "女儿"], "我有个两岁的女儿,正是最闹的时候"),
            "育儿难题": (["孩子不好好吃饭", "吃饭"], "最头疼的是孩子不好好吃饭"),
            "期望": (["想培养阅读习惯", "阅读习惯"], "我想从小培养她的阅读习惯"),
        },
        "chat": ["今天娃又不睡觉", "在挑早教班", "买了一堆绘本",
                 "娃第一次叫爸爸", "辅食做起来好麻烦", "带娃去打疫苗",
                 "娃开始学走路了", "又被娃气笑了", "在研究儿童营养"],
    },
]


def _build_messages(persona: dict) -> list:
    facts = list(persona["facts"].values())  # [(keywords, sentence), ...]
    chat = persona["chat"]
    messages = []
    fact_i = 0
    chat_i = 0
    turn = 0
    # 交替:先抛一个事实句,再几句闲聊;循环直到达到目标轮数。
    while len(messages) < TARGET_TURNS:
        if fact_i < len(facts) and turn % 4 == 0:
            content = facts[fact_i][1]
            fact_i += 1
        else:
            content = chat[chat_i % len(chat)]
            chat_i += 1
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": "嗯,我记住了。" if turn % 4 == 0 else "了解~"})
        turn += 1
    # 保证所有事实都出现过(若前面没轮到,补在末尾)
    while fact_i < len(facts):
        messages.append({"role": "user", "content": facts[fact_i][1]})
        messages.append({"role": "assistant", "content": "好的,记下了。"})
        fact_i += 1
    return messages


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for persona in PERSONAS:
        uid = persona["user_id"]
        messages = _build_messages(persona)
        conv = {"user_id": uid, "messages": messages}
        with open(f"{OUT_DIR}/{uid}.json", "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)
        # facts:{query: [expected_keywords]}
        facts_out = {
            aspect: {"query": f"关于这个人的{aspect}", "expect_any": kws}
            for aspect, (kws, _sentence) in persona["facts"].items()
        }
        with open(f"{OUT_DIR}/{uid}.facts.json", "w", encoding="utf-8") as f:
            json.dump(facts_out, f, ensure_ascii=False, indent=2)
        print(f"{uid}: {len(messages)} 条消息({len(messages)//2} 来回), {len(persona['facts'])} 个预埋事实")
    print(f"\n共生成 {len(PERSONAS)} 个对话集 → {OUT_DIR}/")


if __name__ == "__main__":
    main()
