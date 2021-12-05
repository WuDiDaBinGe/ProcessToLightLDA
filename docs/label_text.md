## 标签工具

#### 1.输入文件格式

```python
# 一行一个文章
延寿县六团镇刘延龙最感谢的人就是同为残疾人并在残联工作的张荣华，有了张荣华的帮助，刘延龙一家于...
马里举行多国部队联合军事演习新华网达喀尔８月２０日电（记者陈顺）巴马科消息：马里、阿尔及利亚....
由万达影视集团投拍的电视剧《下班抓紧谈恋爱》，在将尽３个月的紧张拍摄后，即将杀青。集了朱雨辰....
金正恩为朝少年团代表安排宴会　学生发誓坚决跟随（图）中新网６月８日电　据朝中社报道......
公安机关销毁１０余万非法枪支　跨国武器走私渐起中广网唐山６月１２日消息（记者汤一亮　庄胜春）....
根据报道，二战结束后，美国在几年时间内陆续派德特里克堡基地细菌战专家前往日本，向“731部队”....
```

#### 2.输出格式

* 输出文章的主题分布，格式如下：

```python
# 文章序号 主题：主题次数
# K=18的结果
0  3:1 4:40 5:25 11:93 12:36 15:38
1  5:6 7:1 16:84
2  1:1 6:38 11:31 14:15 17:1
3  0:4 4:4 16:41
4  2:3 3:298 5:18 6:1 7:5 10:4 16:58 17:1
5  2:2 12:9 13:3 14:12 15:3 16:90
```

* 进一步处理上述文件，输出一个二维列表，每一行为每篇文章的`top5`主题序号

```python
# K=18的结果
[[11, 4, 15, 12, 5], [16, 5, 7, 17, 1], [6, 11, 14, 17, 1], [16, 0, 4, 1, 2], [3, 16, 5, 7, 10], [16, 14, 12, 15, 13]]
[[11, 12, 4, 10, 3], [16, 12, 5, 17, 1], [6, 11, 14, 17, 10], [16, 12, 0, 4, 7], [3, 16, 5, 10, 7], [16, 12, 15, 4, 13]]
# K=378的结果
[[102, 319, 64, 73, 20], [122, 105, 179, 255, 359], [209, 354, 326, 102, 352], [189, 165, 0, 170, 100], [351, 346, 192, 310, 364], [341, 263, 213, 336, 371]]

[[102, 20, 313, 64, 319], [122, 105, 113, 255, 18], [209, 352, 354, 270, 134], [189, 77, 237, 78, 224], [351, 346, 192, 226, 341], [341, 336, 263, 237, 212]]

```

#### 3.总结

1. 依赖训练集的`vocab.word_id.dict`文件，只用到了训练集中出现的词语。
2. 每次`infer`的结果可能不同，主题数目K少的时候大体相同，最高的主题也相同。但是K较大的时候每次`infer`的结果相差较大，最高的主题基本相同。
3. 结果准确性看，如测试集中的第2，4，6篇文档都被分配到了16主题，查看得知主题16有如下关键词，结果有一定的参考性。

```python
16   日 中国 和 美国 月 是 日本 说 表示 人 报道 年 他 称 这 国家 叙利亚 中 并 上 问题 其 总统 国际 俄罗斯 有 政府 合作 进行 但 举行 可能 日电 提供 后 伊朗 地区 记者 等 认为 新 媒体 该 计划 没有 时 钓鱼岛 权利 我们 前 印度 一 会议 英国 而 包括 组织 奥巴马 安全 会 宣布 委员会 目前 发生 关系 当地 埃及 内容 联合国 政治 韩国 希望 美 双方 支持 一个 新华社 事件 发言人 世界 当天 要求 国 官员 欧盟 他们 中方 应 之 到 调查 相关 保密 选举 菲律宾 俄 经济 以及 消息 军事
```

4. `python`脚本实现，调用了系统的命令。

### jun事数据集

1. 有一些短文本，以及一些词频不多的词汇，还有一些词频很高的词汇

2. 位于同一个大主题下，主题的区分度可能不高，主体之间的关键词
3. 数据集中分类大体上有：[政治、外交舆论、军事科技、经济金融]

K = 11

**原始数据**：

```
0   海军 中国 海上 美国 驱逐舰 舰 印度 航母 文章 舰艇 作战 是 潜艇 网站 攻击 军舰 新 核潜艇 上 有 能力 护卫舰 力量 司令部 计划 建造 阅兵 题为 舰队 编队 成立 两栖 中 包括 导弹 执行 说 美海军 水下 发表 关注 北极 太平洋 海岸 可能 拥有 部队 称 数量 战区 组成 会 范围 认为 D 任务 独立 实力 水面 活动 之 服役 战舰 海洋 开展 世纪 规模 参加 辽宁 利益 地区 舰载 行动 大型 资料 等 下水 成为 舰船 开始 护航 艇 使 超过 来说 问题 庆祝 图片 印度洋 情报 全面 警卫队 舷号 苏联 全文 中国人民解放军海军 目的 军事 影响 弹道导弹
1   战略 态势 政治 外交 舆论 军事 科技 美 美盟 香港 美美 伊朗 特朗普 华为 作者 朝鲜 宣布 港 示威者 G 访问 发布 警方 非法 加 集结 加拿大 韩国 表示 美蓬 主席 政府 佩奥 论坛 印 引渡 发表 人权 外交部 等 拘捕 一带 暴力 媒体 孟 金正恩 人 中 独 任正非 会见 委内瑞拉 提醒 习 接受 孟晚舟 记者 失败 遭 新疆 暴徒 晚 提出 赴 警察 白皮书 英 制裁 参与 名 舟 采访 文件 经济 新西兰 瑜 所 事件 习近平 问题 讲话 访华 参选 民主 击落 立法会 分子 俄 涉嫌 总理 批准 列为 欧盟 国 香港机场 法 巴基斯坦 金特 进行 赖清德
2   台湾 战略 外交 美国 舆论 态势 美 是 作者 南海 表示 蔡 安全 政治 英文 台 支持 和平 稳定 副 国际 中国 回应 问题 总统 法案 出席 政府 承诺 有 合作 法 军售 美盟 测试数据 美国国务院 会 国家 外交部 国务卿 发表 特朗普 印太 时 维护 坚决 在台协会 助理 言论 委员会 美美 会议 发言人 部长 涉台 政策 关系 两岸 上 王毅 文明 强调 持续 地区 活动 打 组织 台海 关系法 加强 听证会 前 事务 自由 所谓 坚定 宣言 亚太 一国两制 批评 访台 众议院 高官 蓬佩奥 提出 之 保证 反制 亚洲 民主 行为 人民 主任 会见 履行 积极 挑衅 美国众议院 驻 美国国防部
3   称 报道 战机 战斗机 作战 中国 空军 是 歼 进行 飞机 演习 航母 海域 训练 特种 中 隐身 网站 部署 轰炸机 解放军 说 飞行 国产 有 任务 空军基地 苏 中队 到 中国空军 目标 F 部队 视频 东海 具备 等 联合 进入 巡航 可能 发布 装备 空中 实施 运 援引 美 南华早报 图片 南部 时间 之后 指出 美军 发动机 航行 南海 飞行员 B 军演 集团 海军陆战队 大规模 黄海 运输机 公布 香港 验证 完成 机 很 人士 兵力 后 媒称 支援 设备 演练 平台 新型 分析 消息 港 编队 表示 媒体 改装 内 搭载 军事行动 返回 台军 先进 测试 开产 空域始 量
4   战略 经济 态势 金融 中国 美国 华为 企业 美美 美 宣布 征收 发布 进口 清单 公司 国家 相关 中 G 作者 正式 反倾销税 产品 美国商务部 安全 实体 调查 决定 人民币 达 要求 列入 最高 出口 稀土 启动 旅行 政治 生产 机构 继续 出售 有限公司 汇率 联邦 发布公告 美国政府 警告 德国 工业 技术 反倾销 服务 等 批准 华尔街日报 使用 科技 允许 豁免 发展 取消 大豆 价值 禁止 商务部 设备 组织 芯片 原产 作出 发改委 包括 商品 美盟 影响 操纵 进行 制裁 指标 个人 起 公民 所有 反补贴 互联网 措施 产业 风险 声明 欧盟 控制 增长 征 网络攻击 税 购买 华为公司 管制
5   关税 美 战略 态势 中 中国 经济 中方 加征 磋商 美国 贸易 金融 双方 特朗普 经贸 商品 作者 举行 表示 代表 协议 问题 中美 国务院 宣布 高级别 部分 会晤 实施 达成 对话 副 政治 进行 机制 总统 谈判 刘鹤 等 华盛顿 北京 委员会 上 重要 落实 会议 国 取得 两国 继续 排除 有关 讨论 全面 办公室 税则 中华人民共和国 定期 商务部 共识 计划 关切 进口商品 提出 进展 决定 反制 解决 提高 采购 措施 人 原则 利益 美中 立场 团队 发言人 时 事件 对华 美美 作出 争端 新闻 会谈 后 总理 知识产权 莱特 希泽 新 成立 峰会 税率 世界 习近平 到 中国外交部
6   导弹 武器 系统 报道 是 俄罗斯 称 上 发射 卫星 研发 太空 进行 技术 中国 目标 可 美国 防空 新型 网站 高超音速 测试 题为 装备 弹道导弹 中 射程 无人 发表 防务 来 成功 远程 新 研制 试验 能力 配备 雷达 利用 平台 显示 图片 设计 使用 大 导弹系统 美 轨道 直升机 能够 国家 英国 周刊 展示 研究 计划 说 领域 大量 本网 注 时 地面 飞行 前 拦截 会 反 制造 认为 东风 公司 工作 先进 用于 简氏 全文 航空 采用 距离 主要 军用 类似 空中 消息人士 约合 生产 常规 苏联 传感器 能 打击 后 出现 性能 资料 超 工业
7   日本 报道 称 俄罗斯 中国 坦克 国家 下 中 举行 包括 直升机 上 完成 国际 防卫 最大 等 青岛 后 参加 俄 年度 成为 地区 能 有 网站 测试 市场 公司 时 来自 炮 小时 进行 港口 战车 外媒 抵达 到 情况 步兵 军人 表示 在内 男子 VT 比赛 条件 火力 活动 T 登陆 重型 使用 机场 电磁炮 相关 两栖 攻击 时间 消息 人 新加坡 车辆 射击 环境 超过 省 名 章鱼 提供 资料 仪式 实施 白俄罗斯 口径 世界 吸引 S 没有 作为 组织 日本政府 上海 基地 特殊 自卫队 制造 费用 具有 北方 月球 设备 有效 珠海 期间 莫斯科 SDM1
8   起飞 基地 开展 编号 行动 侦察 南海 冲绳 嘉手纳 侦察机 美海军 空军 上空 反潜巡逻机 P RC C A 平泽 附近 美 美陆军 旅 E 驻 韩 监视 军事情报 提供 加油机 X 美国 飞机 KC R 巴士海峡 飞行 W 台湾 韩国 仁川 空中加油 EP 飞越 CL 高度 公司 训练 菲律宾 进入 航空航天 战场 分 西南部 N9191 克拉克 指挥 纳克斯 海上 国民 警卫队 空中 佐治亚州 美军 机场 飞往 加油 统计 关岛 ROGUE15 文莱 月份 台湾海峡 南口 安德森 B 澳大利亚 S 出动 鹰 A47 EO 期间 东北部 RQ 数量 要 全球 PEARL48 远 无法 派出 ADS 实际 东南部 声明 下 列入 人 经
9   中国 是 军事 发展 军队 报道 报告 联合 说 专家 称 合作 国防 指出 等 作战 全球 部队 新 实现 现代化 举行 问题 力量 地 能力 发表 北京 发布 高 网站 提高 巴基斯坦 认为 要 希望 努力 中国人民解放军 新华社 习近平 取得 军迷 获得 世界 意味着 交流 训练 重点 挑战 推动 以来 竞争 地区 国 旨在 工作 大国 到 提升 大 军 英国 对抗 俄罗斯 下 研究所 领导 改革 演练 建设 重大 雷达 自主 记者 年度 作为 视距 实战 介绍 行动 武器装备 之际 媒体报道 论坛 基本 体系 水平 有关 参加 解放军 方式 中央军委 起 联合作战 国际 战斗 限制 美军 文化 确保
10   美国 是 说 无人机 中 称 报道 技术 有 军事 计划 可能 威胁 文章 上 能力 国家 重要 五角大楼 防务 官员 报告 战争 会 研究 安全 应对 人员 使用 新闻 要 伊朗 国防部 能够 美军 系统 信息 陆军 让 冲突 提供 网站 摘编 伊拉克 能 评估 时间 得到 到 题为 国防 下 阿富汗 成为 方面 新 军队 要求 使 通信 来 需要 内 购买 对手 俄罗斯 透露 部署 战场 量子 土耳其 情况 数据 士兵 预算 发表 北约 袭击 领域 网络 人工智能 国际 所有 强大 没有 全文 进行 发生 美国空军 项目 后 打击 任务 资金 人 负责 无法 作战 认为 变化
```

**进一步数据预处理：**

1.去掉低频词，设置低频词的阈值为1

2.去掉分词后单个的字

```python
0   战略 态势 经济 金融 中国 科技 华为 美国 军事 美美 政治 美盟 发布 企业 宣布 征收 伊朗 进口 作者 公司 反倾销税 正式 清单 特朗普 调查 实体 美国商务部 相关 决定 要求 进行 旅行 服务 产品 安全 制裁 发布公告 最高 取消 欧盟 表示 反倾销 继续 警告 措施 列入 商务部 汇率 设备 德国 人民币 联邦 报告 使用 允许 芯片 原产 技术 机构 启动 请求 操纵 华为公司 加拿大 反补贴 国家 公民 作出 出口 提供 列为 豁免 产业 发展 个人 条约 美国政府 下调 管制 禽肉 政府 购买 起诉 英国 包括 实施 峰会 复审 访华 部分 申请 在内 裁定 联邦快递 有限公司 风险 日本 法院 涉嫌 提醒
1   起飞 基地 开展 编号 侦察 行动 南海 冲绳 嘉手纳 侦察机 美海军 空军 上空 反潜巡逻机 RC 平泽 附近 美陆军 军事情报 监视 美国 飞机 加油机 KC 提供 巴士海峡 飞行 台湾 仁川 空中加油 韩国 EP 训练 高度 飞越 CL 公司 进入 航空航天 西南部 N9191 菲律宾 克拉克 战场 指挥 警卫队 纳克斯 海上 空中 国民 佐治亚州 美军 加油 飞往 统计 机场 ROGUE15 关岛 月份 安德森 澳大利亚 文莱 南口 台湾海峡 出动 A47 EO 东北部 无法 PEARL48 全球 RQ 期间 列入 实际 声明 东南部 ADS 无人机 开启 数量 设备 N488CR 预警机 无人 MQ 派出 飞行高度 私人 AE687E 高空 海神 PEARL44 详情 PEARL43 RONIN61 自由行动 防务 AE67DC COMIC66
2   导弹 俄罗斯 武器 印度 卫星 报道 进行 发射 计划 太空 美国 测试 司令部 高超音速 系统 日本 网站 弹道导弹 试验 北极 成功 研发 领域 目标 轨道 导弹系统 全文 工作 研究 发表 射程 题为 雷达 相关 摘编 防卫 飞行 军方 开始 拦截 部署 年度 空军 成为 增加 距离 获得 电磁炮 独立 俄军 宣布 国家 官员 接受 试射 设备 资金 情报 专家 装备 超过 俄国防部 指出 技术 地面 消息人士 开发 评估 完成 实现 炮弹 激光 任务 用于 包括 约合 采购 的话 最大 苏联 项目 预算 表示 战区 潜射 可能 英里 以来 合作 陆基 预计 洲际 称为 配备 认为 达到 本网 解决 负责 大量
3   战略 态势 外交 舆论 政治 美盟 南海 中国 英文 外交部 问题 美美 作者 中方 论坛 访问 会见 在台协会 发言人 国家 举行 香港 反制 合作 表示 军迷 王毅 回应 发表 特朗普 会面 批评 主任 稳定 美国 高官 佩奥 美蓬 国防 宣言 接受 和平 文化 任正非 主席 维护 政策 泰国 人民日报 韩国 采访 失败 原则 防长 华为 敦促 选举 金正恩 台独 大势 宣布 东盟 落实 完成 希望 海上 提交 中国军力 中国外交部 出访 越南 行为准则 警告 联合国 引渡 两岸 政府 坚决 过境 会议 行为 朝鲜 国台办 参选 外长 外交部长 分歧 网络 所罗门群岛 利益 发文 信心 表态 亚洲 参加 方案 访美 新华 联合 金特
4   报道 中国 国家 坦克 导弹 直升机 防务 图片 公司 网站 工业 生产 资料 俄罗斯 显示 目标 使用 集团 发表 测试数据 国际 武装 英国 平台 飞机 土耳其 机器人 士兵 简氏 周刊 发射 最大 设计 功能 媒体 制造 测试 武器 介绍 军用 步兵 技术 情况 火炮 世界 战车 比赛 市场 VT 新闻 携带 空地导弹 交付 题为 条件 发射器 作为 有限公司 巴基斯坦 来自 控制 具有 指标 集团公司 车辆 过程 表示 武直 企业 出口 配备 苏联 工厂 射击 珠海航展 成为 小时 章鱼 消息人士 火力 科工 人民币 环境 发现 签署 完成 首飞 出现 愿景 QN 协议 能力 截至 特殊 出售 重型 透露 越南 主要 时间
5   关税 战略 中国 经济 美国 态势 加征 磋商 金融 贸易 商品 双方 经贸 中方 作者 代表 举行 协议 宣布 问题 特朗普 国务院 高级别 部分 北京 会晤 表示 进展 办公室 对话 委员会 排除 达成 全面 刘鹤 机制 谈判 进行 提高 稀土 重要 继续 实施 会议 税则 决定 华盛顿 定期 成立 产品 进口商品 关切 团队 商务部 有关 共识 措施 总理 美中 讨论 保护 中美 要求 美美 会谈 两国 中华人民共和国 委员 部长 争端 清单 采购 公布 莱特 希泽 知识产权 税率 取得 批准 总统 实质性 平衡 税目 原则 落实 基础 企业 姆努钦 计划 牵头 作为 深入 有效 财政部长 符合 事件 结束 开展 农业 议题
6   战略 台湾 态势 政治 美国 香港 外交 作者 习近平 国际 支持 发表 舆论 法案 表示 特朗普 安全 总统 政府 美国国务院 主席 军售 国务卿 军事 印太 部长 助理 出席 涉台 言论 承诺 民主 批准 上将 示威者 美美 警方 非法 美盟 台海 集结 持续 组织 关系法 强化 会议 司令 亚太 包括 讲话 访台 强调 回应 保证 听证会 参与 委员会 拘捕 事务 众议院 暴力 美众议院 美国众议院 科技 攻击 召开 人权 官员 两岸 委内瑞拉 问题 使用 分子 国家 指出 新西兰 美盟台 拜登 防卫 一国两制 媒体 太平洋 议长 暴徒 蓬佩奥 台湾地区 统一 活动 执行 美国国防部 呼吁 伙伴 授权 提供 合作 事务局 一带 警察 部分 可靠
7   美国 中国 军事 发展 军队 报道 国家 威胁 安全 报告 新闻 五角大楼 国防 伊朗 应对 国防部 美军 地区 表示 关系 能力 问题 官员 计划 阿富汗 重要 和平 行动 保持 可能 要求 时间 情况 成为 需要 建立 工作 北约 冲突 重大 组织 伊拉克 继续 采取 建设 确保 中美 机构 加强 有关 特朗普 获得 得到 全球 世界 稳定 文明 发布 政策 军种 政府 领导 华盛顿 改变 积极 国防部长 导致 总统 面临 记者 中国人民解放军 推动 拜登 所有 支持 坚决 发言人 活动 发生 武器装备 没有 制定 不会 北京 领域 敌人 提出 对手 准备 竞争 坚定 这份 美国国防部 关键 作出 挑衅 当地 明确 两国 领导人
8   中国 海军 航母 海上 驱逐舰 美国 力量 舰艇 潜艇 作战 军舰 文章 网站 核潜艇 可能 护卫舰 攻击 报道 建造 阅兵 军事 美海军 舰队 参加 题为 辽宁 专家 水下 导弹 成立 两栖 活动 包括 图片 编队 海岸 计划 资料 弹道导弹 能力 拥有 举行 青岛 发展 大型 进行 战舰 水面 搭载 军队 现代化 海洋 舰载 港口 国产 组成 世界 护航 数量 下水 全面 庆祝 规模 需要 发布 舷号 超过 影响 舰船 直升机 认为 执行 北京 中国人民解放军海军 保护 退役 巨大 猛虎 上海 服役 升级 战斗群 新加坡 水面舰艇 国际 世纪 近海 防御 利益 取得 仪式 排水量 试航 编译 A型 发表 警卫队 的话 国海军 核动力
9   中国 系统 无人机 技术 报道 美国 战斗机 能力 网站 新型 能够 作战 俄罗斯 文章 装备 认为 战机 武器 防空 研究 隐身 使用 题为 研发 防务 无人 利用 方面 远程 打击 发表 视频 研制 先进 报告 战争 部队 网络 指出 摘编 进行 最近 全文 空军 可能 雷达 英国 双月刊 拥有 优势 人员 量子 通信 制造 展示 军队 平台 陆军 东风 迅速 人工智能 用于 小型 空中 飞行 购买 信息 亮相 包括 意味着 提升 分析 目标 现代化 任务 轰炸机 提供 利益 公司 重要 中国空军 实现 帮助 战场 隐形 服役 装置 性能 战术 类似 关注 成功 导弹 无法 程度 敌方 媒称 重点 美国陆军 达到
10   报道 作战 中国 训练 联合 演习 战机 部队 解放军 特种 空军 海域 部署 任务 日本 举行 进行 空军基地 飞机 演练 行动 中队 网站 实施 航行 东海 战斗机 编队 指出 南海 地区 南部 太平洋 台湾海峡 执行 开展 空中 轰炸机 袭击 军演 大国 美军 巡航 进入 验证 军事 黄海 人员 目标 合作 援引 男子 军人 来自 人士 兵力 发布 支援 开始 以来 大规模 参加 附近 媒称 战区 战斗 地点 海军陆战队 部分 台军 运输机 返回 装备 媒体 打击 机场 巴基斯坦 组织 实战 消息 展开 声明 之后 新加坡 基地 空域 试飞 时间 印度洋 提供 海上 防空 大队 信号 救援 中国空军 宣布 年度 登陆 值得
```

按照4类标签：

```
0   战略 态势 美国 中国 经济 金融 关税 无人机 华为 军事 美美 政治 科技 加征 伊朗 宣布 企业 商品 香港 国家 作者 坦克 特朗普 要求 公司 发布 部分 产品 继续 报道 清单 征收 组织 美盟 进行 决定 使用 贸易 政府 进口 安全 计划 相关 工业 机构 最高 批准 伊拉克 袭击 协议 出口 控制 反倾销税 技术 武装 人民币 排除 美国商务部 办公室 表示 调查 量子 设备 实体 土耳其 威胁 有限公司 实施 包括 报告 市场 价值 委员会 作出 取消 制裁 签署 原产 示威者 列入 商务部 非法 购买 汇率 出售 提高 德国 稀土 正式 提供 警方 发布公告 生产 服务 风险 发展 公布 小时 方面 集结
1   战略 态势 外交 舆论 美国 政治 中国 台湾 问题 美盟 作者 军事 表示 中方 磋商 特朗普 双方 安全 举行 南海 经贸 经济 支持 总统 美美 发表 科技 习近平 国防 国家 英文 部长 贸易 国际 和平 主席 会议 合作 发言人 外交部 金融 北京 法案 政策 华盛顿 承诺 会见 稳定 美国国务院 回应 中美 代表 发展 香港 政府 出席 论坛 反制 高级别 对话 原则 访问 拜登 重要 会晤 持续 测试数据 提出 军售 维护 国务卿 关系 有关 解决 印太 强调 取得 美国国防部 继续 地区 讨论 坚决 亚洲 记者 助理 在台协会 发布 要求 达成 言论 两国 进展 国防部长 警告 涉台 上将 宣布 机制 交流 积极
2   起飞 基地 开展 编号 行动 侦察 南海 冲绳 嘉手纳 侦察机 美海军 空军 上空 反潜巡逻机 RC 附近 平泽 飞机 美陆军 监视 军事情报 美国 提供 加油机 飞行 KC 巴士海峡 韩国 台湾 训练 空中加油 EP 仁川 进入 高度 飞越 公司 CL 特种 菲律宾 警卫队 空中 航空航天 西南部 空军基地 N9191 指挥 克拉克 战场 中队 纳克斯 美军 海上 澳大利亚 国民 机场 佐治亚州 台湾海峡 飞往 部署 军迷 作战 加油 统计 出动 验证 返回 关岛 ROGUE15 月份 文莱 空域 之后 南口 大队 支援 愿景 安德森 期间 A47 D328 穿越 无人机 东北部 Artu 中东 降落 运输机 实施 设备 轰炸机 EO 副驾驶 N645HM 联队 AG600 预警机 RQ 主管 派出
3   中国 报道 海军 美国 导弹 网站 俄罗斯 作战 系统 武器 进行 能力 航母 海上 技术 题为 文章 战机 发表 战斗机 可能 军事 计划 驱逐舰 目标 发射 部队 国家 装备 卫星 日本 军队 印度 认为 联合 任务 测试 防务 包括 图片 研究 研发 指出 报告 使用 司令部 力量 太空 新型 舰艇 训练 开始 能够 空军 潜艇 发展 防空 举行 弹道导弹 演习 成为 部署 攻击 直升机 领域 资料 摘编 雷达 全文 时间 地区 没有 军舰 海域 美军 最大 解放军 需要 发布 英国 执行 新闻 编队 打击 拥有 平台 高超音速 成功 隐身 飞行 主要 国际 两栖 专家 核潜艇 五角大楼 射程 远程 超过 世界
```

**主题塌陷**：出现很多意思相近的主题

使用WTM模型

```shell
['航空航天公司', '佐治亚州', '加油', '早上', '飞往', 'RONIN', '东北部', '点钟', '东南部', '预警机', '关岛', '高级别', '南口', '轮', '下午', '美陆军', '落实', '宣言', '文本', '各方', '务实', '节', '平衡'', '非关税壁垒', '文明', '牵头', '进一步', '信心', '指示', '高效', '创造', '极限', '步', '阿根廷', '交流', '上海', '农业', '委员', '副总理', '凌晨', '刘鹤', '东盟']
['反倾销税', '反倾销', '大豆', '反补贴', '调查', '裁定', '复审', '征', '税', '期终', '中国商务部', '联邦', '排除', '豁免', '商务部', '吨', '欧盟', '船', '客户', '国务院关税税则委员会', '输', '越南', '受首批', '人民币', '还包括', '同日', '重量', '姓名', '不到', '国内', '国土', '基础设施', '这项', '予以', '初步', '详情', '深化', '部门', '成果', '这场']
['全球鹰', '击落', '立场', 'RQ', '破坏', '新华社', '遭', '高空', '白皮书', '在台协会', '美国在台协会', '发布会', '区', '证实', '现役', '亚太', '影响力', '美国众议院', '宣言', '仪式', '举办', '大陆', '长, '涉及', '位', '部门', '承认', '国庆', '澳大利亚', '边境', '断交', '表达', '日前', '赴', '指责', '副主任', '这项', '交流', '各国', '跨', '美国国会', '进攻']
['欧盟', '加拿大', '发起', '法国', '调查', '细节', '大使', '加剧', '此举', '公民', '会面', '秘密', '境外', '德国', '条', '达成协议', '管控', '推特', '澳大利亚', '•', '发', '拟', '边境', '期待', '发文', , '凌晨', '日本政府', '届', '互联网', '访', '时刻', '推出', '华尔街日报', '位', '议员', '连续', '金正恩', '解释', '文', '韩', '现场', '进口商品', '中心', '推', '高级']
['万亿', '破', '大跌', '中俄', '赖清德', '打造', '调查', '施压', '元首', '参选', '暴徒', '美国财政部', '供', '访', '减少', '提', '越南', '芯片', '祝贺', '评', '境外', '企', '产业', '投资', '议长', '访美 '边境', '自主', '依赖', '表态', '现场', '国土', '传递', '金正恩', '主持', '原计划', '峰会', '中央', '台湾地区', '一带', '潜在', '秘密']
['核', '伊', '豁免', '讲话', '石油', '封锁', '纪念', '中俄', '台湾同胞', '核武器', '请求', '登陆舰', '欧洲', '巡洋舰', '霸权', '美国众议院', '履行', '大国', '拒绝', '展开', '强化', '停止', '副总统', '策', '习', '综合', '体现', '今日', '捍卫', '关心', '发文', '指控', '香港机场', '声称', '压力', '一国两制', '示威者', '围绕', '暴徒', '空间', '机场']
['提醒', '巴基斯坦', '赴', '击落', '印', '文化', '加拿大', '释放', '飞行员', '开发', '稀土', '恢复', '阿联酋', '委内瑞拉', '分子', '原计划', '愿', '自由', '表现', '敦促', '断交', '越南', '委', '沙特', '回答', '扣押', '观察', '案', '届', '非法', '习', '推', '携带', '美国政府', '访', '峰会', '访华', '各国', '美众议院', '人权', '民主', '华为公司', '部', '失败', '干涉', '英', '具']
['机场', '澳大利亚', '地点', '至少', '凌晨', '沙特', '周边', '南部', '电视台', '击中', '发动', '指责', '死亡', '小时', '爆炸', '刚刚', '击落', '部', '新疆', '抵达', '量子', '科学家', '反恐', '获胜', '污', '并未', '打算', '台当局', '苏', '星', '军演', '注意到', '海', '人权', '社交', '中俄', '疑似', '台海', '新时代', '颗', '巡航', '为期']
['平泽', '军事情报', '韩美', '架次', '坦克', '特种', '高超音速', '仁川', 'ROGUE', '苏', '中国空军', '北极', '导弹系统', '轨道', '辽宁', '中队', '武装', '巡航', '小型', '舰载', '演练', '消息人士', '空军基', '北约', '运输机', '战区', '量子', '东风', '美国空军', '俄军', '式', '战', '潜', '军用', '巴基斯坦', '世纪', '功能', '意味着', '发动机', '大国', '敌人', '战车']
['稀土', '部', '指标', '发改委', '非法', '吨', '成员', '管理', '高级别', '资源', '鼓励', '分别为', '知识产权', '下达', '暂停', '强化', '产业', '落实', '联盟', '制度', '有限公司', '庞大', '卡', '获悉', ', '至关重要', '改善', '初步', '步', '议员', '远海', '石油', '电子', '标准', '边境', '在内', '安排', '管控', '牵头', '签署', '建', '盟国']
['护航', '亿元', '战区', '人民币', '挑衅', '舰船', '军人', '航', '海空', '海岸', '增长', '南部', '疫情', '近海', '引发', '吴谦', '军事战略', '指标', '士兵', '霸权', '演练', '今日', '此类', '集团', '危机 '兵', '警卫队', '远海', '海洋', '欧盟', '吨', '予以', '地方', '船', '有限公司', '互联网', '国家主权', '军力', '主权', '特别', '坚定', '机']


c_v:0.4762312739517499, c_w2v:None, c_uci:-11.313135863035184, c_npmi:-0.3415606147027925
mimno topic coherence:-125.06825856876094
```

### 标签工具2.0

**训练方面：**

1. 去掉了单字，语料库中的低频词

2. 去掉了语料库中的**标签词**

3. 去掉了**中文常用停用词**

```shell
0   美国 伊朗 特朗普 军事 时间 伊拉克 美军 阿富汗 总统 朝鲜 情况 航母 接受 没有 透露 计划 国防部长 行动 新冠 采访 华盛顿 购买 进行 领导人 之后 政府 当地 官员 金正恩 任正非 五角大楼 章鱼 埃斯 消息人士 代理 伊斯兰 主席 打击 危机 可能 疫情 军人 进入 要求 官网 国家 巴格达 导致 尼米兹 组织 实验室 负责人 继续 SDM1 资料 期间 发起 极端 访华 参谋长 细节 路透社 陆军 上将 军方 原因 议员 出现 图片 指出 访问 马克 士兵 间谍 法国 英军 表示 发生 引发 拒绝 叙利亚 旨在 苏莱曼 加剧 愿意 整个 英里 乘坐 事件 萨拉米 致命 价值 疫苗 最后 恐怖主义 现场 会面 期待 用途 战斗群
1   中国 图片 直升机 显示 防务 公司 制造 网站 生产 集团 工业 人民币 土耳其 周刊 最大 英国 简氏 飞机 导弹 平台 展示 视频 企业 资料 安装 采购 国家 有限公司 签署 空地导弹 发表 武装 合同 协议 武直 编译 国际 出口 新型 科工 珠海航展 航天 透露 工厂 出现 合作 涉及 携带 吸引 题为 首飞 最近 消息人士 乌克兰 交付 最高 清楚 精确 集团公司 互联网 航空 中航 人们 搭载 组件 价值 机身 蓝箭 NM 市场 类似 巴基斯坦 设计 明显 增长 中国航天 疑似 金额 举行 简称 成为 世界 反坦克 速度 出售 来自 证实 产品 尺寸 制导 重型 内容 军工 枭龙 轻型 商用 客机 介绍 防务展 变化
2   印度 海上 海军 战区 司令部 参加 联合 举行 训练 时间 地区 两栖 作战 比赛 抵达 来自 成立 成为 计划 独立 当地 国际 东部 开展 军事 军人 猛虎 图片 国防部 以来 白俄罗斯 运输机 在内 演习 中国国防部 会谈 珠海 活动 最大 综合 消息 海洋 登陆 参赛 使用 匕首 进行 宣布 陆军 俄军 干扰 加尔各答 取得 军舰 AG600 外交 发表 海战 岛礁 地点 水域 完成 驻扎 人道主义 救援 岸防 西部 网站 军区 研究院 波罗的海 地缘 表现 拥有 规模 沿海 侵犯 旨在 空军 注意 设备 后勤 目光 击沉 科考船 阅兵式 特殊 文章 电视台 舰队 自主 造成 排水量 巨大 采购 赛事 纪录 法国 同类 安排
3   无人机 使用 系统 无人 目标 进行 装备 袭击 美国 公司 设备 人工智能 机器人 美国空军 网站 利用 安全 空中 功能 提供 战术 飞行 装置 组织 指出 士兵 通信 远程 发生 能够 研发 敌方 国际 击落 环境 媒称 马丁 迅速 伊朗 武装 直接 移动 设施 集群 发动 愿景 测试 洛克希德 直升机 战争 传感器 战场 联军 负责 发表 识别 指挥官 亚美尼亚 作用 任务 防务 Artu 后者 帮助 现实 做好 陆军 西班牙 导弹 人类 修复 军用 阿塞拜疆 军事行动 机载 专门 地面 MQ 爆炸 以色列 隐形 战机 远距离 革命 红外 副驾驶 皮肤 战士 卫队 大型 发现 执行 有效 死亡 L115A3 狙击步枪 受到 搭载 应用 发挥
4   作战 空军 部队 特种 部署 任务 美国 联合 北极 实施 执行 中队 空军基地 海军陆战队 演练 太平洋 司令部 人员 海岸 战斗机 地区 航行 提供 范围 能力 验证 步兵 海域 支援 时报 训练 军事 警卫队 东海 战斗 目标 大规模 陆军 飞机 平台 对抗 合作 战车 大国 设备 亚洲 增加 火力 机场 印度洋 大队 概念 司令 港口 兵力 以下 双方 美军 特种部队 反映 拥有 D328 日益 水兵 人士 军方 空中 模型 运输机 协同 指挥 特战 目的 上述 行动 远征 穿越 之后 反恐 降落 主要 联队 N645HM 环境 太平洋地区 中东 内容 中东地区 突破 信息 水面 派遣 空域 消息 重型 军事行动 飞往 之间 文件 隶属于
5   台湾 美国 香港 作者 法案 国际 表示 出席 美美 国务卿 军售 印太 安全 涉台 助理 支持 民主 美国国务院 部长 示威者 言论 英文 警方 承诺 集结 总统 批准 美盟 委员会 发表 外交 关系法 亚太 外交部 组织 访台 听证会 佩奥 美蓬 蓬佩奥 拘捕 上将 保证 众议院 暴力 官员 访问 美国众议院 会见 亚洲 美众议院 非法 合作 中国 事务 提供 特朗普 参与 断交 回应 持续 美盟台 政策 会面 两岸 履行 呼吁 暴徒 出访 统一 主任 定期 强调 警察 议长 论坛 事件 邦交国 问题 司令 对话 人权 一带 事务局 政府 代表 高官 北京 台美 台北 经济 立法会 伙伴 举办 所罗门群岛 确认 方式 防卫 太平洋 美国国防部
6   关税 加征 美国 贸易 磋商 商品 经贸 中国 特朗普 双方 代表 作者 举行 中方 协议 宣布 高级别 国务院 会晤 问题 美美 刘鹤 排除 北京 达成 委员会 部分 实施 团队 办公室 税则 经济 华盛顿 进口商品 习近平 共识 进展 全面 提高 美中 取得 谈判 总理 清单 总统 讨论 中华人民共和国 表示 进行 继续 重要 税率 希泽 莱特 美国政府 产品 两国 措施 国家 会见 农业 保护 税目 牵头 基本 委员 财政部长 对话 知识产权 姆努钦 方面 结束 主席 计划 会议 通话 关切 大阪 事件 采购 技术转让 机制 服务业 原产 中共中央政治局 文本 恢复 阿根廷 非关税壁垒 工作 上调 符合 平衡 金特 平等 推迟 援助 带领 申请 重启
7   日本 武器 高超音速 美国 发表 计划 国防 获得 摘编 研究 全文 英国 题为 预算 防卫 文章 对手 能够 拥有 目标 美军 网站 拦截 相关 项目 攻击 增加 情报 年度 利益 意味着 可能 用于 双月刊 部分 速度 激光 程度 战争 约合 加快 射程 加速 研发 没有 打击 防务 敌方 无法 本网 担忧 人士 冷战 达到 实现 作者 自卫队 日本政府 费用 超级 利用 至关重要 反舰导弹 资金 希望 财年 强调 预计 旨在 授权 弹药 部署 庞大 威慑 距离 时间 现有 来说 减少 方式 改装 时说 杂志 持续 防御 最近 远程 出现 看到 方向 制导 全球 搭载 吉尔 世纪 开支 数量 威力 改善 之外
8   南海 起飞 开展 侦察 基地 行动 编号 嘉手纳 冲绳 美海军 空军 反潜巡逻机 侦察机 上空 加油机 监视 飞机 KC 提供 巴士海峡 美国 台湾 空中加油 飞越 CL EP 高度 菲律宾 进入 公司 西南部 航空航天 N9191 克拉克 纳克斯 海上 战场 指挥 台湾海峡 国民 警卫队 美军 加油 空中 佐治亚州 机场 统计 关岛 飞往 月份 澳大利亚 安德森 南口 文莱 无人机 A47 东北部 数量 期间 PEARL48 ADS 开启 实际 飞行高度 派出 防务 RQ 无法 列入 声明 设备 预警机 N488CR MQ AE687E 私人 AE6784 返回 PEARL44 AE6819 无人 PEARL43 AE67DC 海神 COMIC66 南部 AE67D6 RONIN61 PEARL41 之后 详情 APS AE67B3 东南部 弹道导弹 情况 AE6836 指挥舰 LCC PEARL40
9   俄罗斯 战机 战斗机 中国 系统 防空 雷达 隐身 装备 网站 研发 飞行 轰炸机 中国空军 新型 先进 飞机 进行 具备 完成 能力 工作 巡航 专家 测试 题为 研制 成功 方面 视频 国家 技术 远程 训练 服役 打击 升级 香港 空军 具有 南华早报 作战 改进 发动机 部队 空中 项目 国产 媒称 俄国防部 飞行员 提升 导弹系统 可能 苏联 量产 视距 进入 接受 设计 没有 武器 试飞 以来 发布 取得 性能 改装 组成 表示 列装 指出 发表 现役 涂装 资料 战备 采用 俄军 实现 阶段 航空 努力 编队 莫斯科 预警机 媒体报道 规模 配备 大量 消息人士 俄空天军 两国 标识 电子 传感器 军事 期间 结束 取代
10   海军 中国 航母 驱逐舰 美国 舰艇 潜艇 海上 网站 核潜艇 军舰 文章 护卫舰 力量 建造 阅兵 攻击 导弹 编队 美海军 可能 辽宁 水下 作战 两栖 舰队 能力 舰载 题为 参加 组成 国产 战舰 大型 资料 新加坡 青岛 下水 附近 海域 世界 保护 活动 弹道导弹 舷号 舰船 护航 中国人民解放军海军 图片 关注 媒体 水面舰艇 近海 区域 退役 水面 试航 行动 成立 数量 直升机 之前 A型 打击 搭载 上海 世纪 国海军 服役 起降 海盗 造船 仪式 利益 现代化 反潜 具备 迅速 检阅 展示 甲板 编译 举行 进行 全面 发表 发布 补给舰 核动力 没有 实力 可靠 人民解放军 期间 海洋 登陆舰 国家 改造 发现 主要
11   中国 发展 军队 军事 报告 发布 国防 现代化 力量 地区 问题 测试数据 专家 经济 解放军 军迷 发表 论坛 重点 国际 政治 改革 中国人民解放军 建设 保持 研究所 领导 重要 竞争 指出 北京 文化 联合作战 实现 核心 安全 体系 以来 军力 武器装备 举行 评估 威胁 介绍 基本 情报局 重大 努力 推动 时代 政策 保障 核力量 中国军力 自主 武装力量 白皮书 社会 反恐 三军 纲要 交流 地位 这场 关键 产业 专业 层面 经验 解决 世界 大国 行动 令人 主要 活动 年度 制造 普京 标准 自信 联合 体制 外国 组织 意识 国家 美国国防部 依赖 确保 能力 转型 印象 意义 原则 围绕 这份 高级官员 不会 制度
12   导弹 发射 卫星 武器 太空 进行 俄罗斯 测试 美国 弹道导弹 系统 射程 试验 轨道 领域 目标 能力 成功 网站 技术 部署 电磁炮 使用 试射 东风 装备 火炮 地面 国家 苏联 炮弹 研制 导弹系统 潜射 发射器 条件 战略 洲际 司令部 配备 过程 达到 军方 任务 电磁 射击 口径 QN 设计 战斗 距离 提升 巨浪 用于 中心 宣布 ASAT 滑翔 证明 最大 指出 月球 追踪 实现 DA 开发 研究所 空军 完成 符合 探测 反导 非洲 投入 官员 火箭 努力 操作 称为 陆基 军用 跟踪 小型 利用 新型 宇宙 正式 具体 模式 便携式 红箭 登陆 中央 天基 展示 击中 应该 卢旺达 嫦娥 探测器
13   中国 中方 南海 表示 和平 发言人 稳定 中美 作者 美国 问题 维护 合作 外交部 坚决 国家 磋商 反制 文明 王毅 关系 世界 原则 挑衅 地区 发展 宣言 坚定 举行 利益 威胁 严重 立场 落实 双方 人民 安全 泰国 积极 人民日报 回应 中美关系 行径 中国人民解放军 提出 施压 记者会 加强 坚持 行为准则 中国外交部 会议 两军 对话 管控 行为 吴谦 东盟 例行 香港 希望 手段 两国 商务部 要求 明确 继续 失败 升级 对华 态度 承诺 予以 冲突 两岸 贸易战 交换意见 关切 国台办 海空 外交部长 外长 选举 全面 峰会 解决 高官 主义 信心 摩擦 谈判 发布会 所谓 防长 损害 违反 面对 国务委员 准则 推进
14   美盟 政府 拜登 解放军 在台协会 美国 警告 华为 发表 问题 大陆 英文 台湾 台军 特朗普 声明 攻击 韩国 台海 政策 美美 支持 会议 批评 表示 连续 台湾海峡 委内瑞拉 轰炸机 关注 日本 所谓 引发 台当局 担任 美国国务院 值得 演习 参选 岛内 欧洲 总统 援助 部门 台湾地区 新西兰 主席 选举 当选 访美 展开 赖清德 安倍 理查德 盟国 空域 海峡 高度 宣布 社论 中央社 沙特 主任 AIT 民进党 遭到 岛屿 当局 总理 突然 台军方 排除 势力 大使 台独 华尔街日报 声称 战备 媒体 执行 物资 消息 会见 之外 证实 千方百计 罕见 炒作 加强 联盟 大规模 反潜机 幻影 言论 疑似 台北 上任 主持 航行 更名
15   基地 起飞 侦察机 RC 开展 编号 平泽 附近 美陆军 军事情报 行动 飞行 侦察 韩国 上空 仁川 训练 冲绳 ROGUE15 全球 嘉手纳 空军 出动 EO 国民 佐治亚州 警卫队 飞机 空中 监视 战场 高空 EP 美海军 指挥 LEVET21 ROGUE11 N158CL ROGUE06 ROGUE83 ROGUE18 东南部 西侧 N59AG ROGUE22 沿海地区 AE67C2 华南 到达 SOGGY46 反潜巡逻机 宫崎 JEDI01 AE67DE LEVET22 JEDI03 美国 JEDI07 ROGUE68 电子 星球大战 PEARL59 ROGUE44 共用 JEDI08 JEDI04 CRAZY07 全权负责 AE67F7 JEDI05 CRAZY02 东沙群岛 LEVET30 RONIN33 卢克 准备 主角 ROXIE37 美台 级别 安保 AE6899 主办 AE6883 统计 般的 军事设施 RQ 网友 西沙群岛 禁飞区 发表谈话 TUNED01 巴赫 顺序 总共 AE682D 受伤 因素 撤离
16   美国 战略 技术 能力 文章 研究 重要 威胁 五角大楼 计划 国家 系统 成为 方面 领域 应对 网络 俄罗斯 国防部 防御 情况 能够 报告 量子 可能 问题 数据 面临 信息 人员 防务 支持 官员 军队 作战 题为 全球 控制 所有 关注 网站 北约 军种 联合 建立 进展 生产 方式 之间 战争 评估 全文 周刊 战场 摘编 小型 冲突 优势 机构 加强 无法 要求 无人机 继续 制定 敌人 工作 安全 变化 存在 采取 指标 中心 扩大 强大 改变 对抗 考虑 刊发 协调 美国陆军 处理 平台 澳大利亚 公布 取得 美国国防部 上述 行动 影响 部长 资金 不会 负责 架构 整合 试图 通信 重点 确保
17   中国 美国 企业 美美 征收 进口 相关 作者 正式 反倾销税 调查 最高 商务部 继续 发布公告 措施 反倾销 宣布 进行 取消 允许 大豆 原产 美国商务部 反补贴 欧盟 部分 解决 商品 启动 规则 发布 条约 出口 禽肉 裁定 复审 影响 产品 体系 美企 涉嫌 猪肉 公开 价值 争端 鲶鱼 义务 声明 行为 诉讼 构建 公布 边境 GDP 确认 海关 世界贸易组织 WTO 离开 海运 规避 大跌 钢铁 介入 裁决 保证金 机密 要求 驻华 最后 来自 股指 原料 退出 关口 韩国 质疑 南通 世贸组织 窃取 联名 强硬 可达 学校 美国农业部 未能 行长 等效 提出 监管 出台 对外 全线 对华 法律法规 苯酚 日本 墨西哥 收报
18   中国 演习 美盟 习近平 军事 举行 进行 海域 巴基斯坦 要求 发布 新华社 南海 提出 援引 关系 军演 工作 安全 指出 方案 黄海 提高 旅行 网站 发出 香港 一国两制 讲话 部分 强化 加拿大 提醒 宣布 表示 作者 国家 主席 准备 声明 持续 接受 活动 训练 引渡 东海 压力 发表 回应 实战 南部 孟晚舟 请求 之前 警告 中央军委 文件 英文 加强 挑战 台湾 媒体 任务 全军 所有 公民 承诺 打仗 会议 渤海 实弹射击 执行 老人 高度 评估 位于 充分 备战 达成协议 解放军报 斗争 反制 联训 定期 雄鹰 南华早报 推动 联合演习 境外 赴美 解放军 驻华大使 司法部 放弃 船只 启动 紧张 释放 军事演习 长沙
19   华为 中国 坦克 美美 美国 公司 国家 清单 实体 使用 宣布 列入 制裁 德国 服务 稀土 技术 美盟 表示 机构 VT 媒体 设备 安全 产品 美国商务部 越南 购买 实施 男子 发布 发改委 组织 出售 联邦快递 要求 芯片 操纵 伊朗 市场 警告 政府 客户 联邦 豁免 起诉 主战 下调 禁止 俄罗斯 汇率 华尔街日报 个人 提供 管制 西方 违反 减少 石油 需求 华为公司 安全局 法院 委员会 阿联酋 召开 美国财政部 东南亚 外媒 海外 有限公司 电信 武汉 在内 污蔑 限制 取消 销售 可靠 执法 通知 全部 工业 新疆 非法 发表声明 美国政府 认定 出口 企业 商业 城市 安全部 停止 无法 通信 软件 谷歌 国土 有效
```

推断方面：

对文章中的词进行了两次筛选

1. 利用**百度LAC**分词，进行词性、**重要性**和停用词筛选，**重要性**大于1的词才用。
2. 利用得到的`topic:wordsList(Top100)` 找到**每个主题下面前100个关键词**，组成单词列表。上一步筛选后，在该列表中的词才会被统计。

```python
# 输出是根据每篇文章每个主题占比多少 排序输出
[(4, {'隶属于', '降落', '执行', '中队', '作战', '海军陆战队', '联队', '战斗'}), (9, {'系统', '打击', '战机', '防空', '具备', '巡航', '战斗机'}), (3, {'设施', '隐形', '做好', '敌方'}), (10, {'航母', '核动力', '海军', '甲板'}), (8, {'指挥', '起飞', '飞机'}), (16, {'美国', '能力'}), (1, {'安装', '直升机'}), (18, {'准备'}), (7, {'计划'}), (2, {'驻扎'}), (15, {'飞行'})]

[(9, {'隐身', '中国空军', '雷达', '量产', '打击', '规模', '先进', '战斗机', '服役', '南华早报'}), (11, {'中国', '解放军'}), (8, {'空军', '行动'}), (15, {'训练', '飞行'}), (7, {'意味着', '武器'}), (10, {'编队', '文章'}), (1, {'机身', '最大'}), (18, {'境外'}), (19, {'媒体'}), (5, {'香港'})]

[(10, {'力量', '国产', '航母', '海军', '建造', '护卫舰', '核潜艇'}), (7, {'计划', '年度', '日本', '发表'}), (9, {'俄罗斯', '装备', '研发', '服役'}), (16, {'美国', '能力', '强大', '成为'}), (2, {'印度', '司令部'}), (4, {'联合', '司令'}), (13, {'发布会', '合作'}), (11, {'中国', '自主'}), (0, {'时间'}), (17, {'相关'}), (1, {'制造'}), (8, {'开展'}), (6, {'团队'}), (18, {'演习'})]

[(10, {'青岛', '舰艇', '登陆舰', '舰队', '力量', '近海', '驱逐舰', '阅兵', '反潜', '补给舰', '编队', '航母', '海军', '护卫舰', '区域', '核潜艇', '潜艇', '攻击'}), (4, {'反映', '港口', '实施', '执行', '作战', '协同', '火力'}), (11, {'核心', '交流', '中国', '建设', '发展', '关键'}), (9, {'取代', '远程', '打击', '具备', '防空'}), (2, {'海上', '消息', '综合'}), (16, {'战略', '能力', '强大'}), (13, {'世界', '人民'}), (19, {'海外', '国土'}), (6, {'全面'}), (17, {'公开'}), (12, {'导弹'}), (8, {'行动'}), (3, {'空中'})]

[(19, {'电信', '起诉', '美国商务部'}), (3, {'设备'}), (12, {'测试'}), (0, {'实验室'})]
[(19, {'华为公司', '电信'}), (1, {'金额', '最大'}), (16, {'美国'}), (7, {'费用'})]

[(5, {'外交', '会面', '所罗门群岛', '主任'}), (16, {'战略', '美国'})]
[(5, {'外交', '访问', '言论', '亚洲', '美国国务院', '台湾', '支持'}), (16, {'战略', '不会', '国家', '美国'}), (7, {'发表', '计划', '日本'}), (19, {'美美'}), (13, {'回应'}), (6, {'美中'})]
```

**文章相似度计算**：

使用推理后会得到一个`doc_topic`记录了每篇文章对应的主题分布，将其读入后，转成文章—Topic的矩阵，每一行就是一篇文章的主题分布，计算文章的相似度，就是计算文章对应主题向量的cos相似度。

```shell
# 相似度计算结果 序号：相似度最大文章序号
0 3
1 0
2 3
3 2
4 5
5 4
6 7
7 6
```

​         