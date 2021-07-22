# -*- coding: utf-8 -*-
# @Time    : 2021/7/22 上午11:05
# @Author  : WuDiDaBinGe
# @FileName: test.py
# @Software: PyCharm
import requests
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.base_url = 'http://127.0.0.1:8000/'
        self.data = {
            1: "美媒称，美国海军陆战队宣布，第一个配备F-35C“闪电”Ⅱ隐形战斗机的中队已经做好战斗准备。据美国《福布斯》双周刊网站12月3日报道，洛克希德-马丁公司为F-35C安装了比常规的F-35A和垂直起降的F-35B更大的机翼，使C型“闪电”Ⅱ能够在美国海军核动力航母1000英尺（约合305米）的甲板上起飞和降落。报道称，作为舰载机联队的一部分，F-35C中队负责执行纵深打击任务，突破防空系统打击敌方基地、补给线和指挥设施。加利福尼亚州米拉马的海军陆战队第3航空联队周二宣布，海军陆战队第314战斗攻击机中队（VMFA-314）——隶属于第3航空联队——已具备初步作战能力。海军计划2022年初将VMFA-314中队部署在一艘西海岸的航母上。报道称，美国海军首个F-35C中队——VFA-147——将于明年跟随驻扎在圣迭戈的“卡尔·文森”号航母进行首次巡航。报道指出，到本世纪30年代，美国海军所有9个航母舰载机联队都将拥有F-35C战斗机中队。这种大翼隐形战机将与舰载机联队的F/A-18E/F战斗机、EA-18G电子战机、E-2预警机、MQ-25无人加油机、CMV-22倾转旋翼飞机和MH-60直升机一同飞行。",
            2: "境外媒体称，近期解放军空军展示的7架歼-20战斗机同时飞行的视频，意味着这款先进隐身战斗机已进入量产阶段。据香港《南华早报》网站9月4日发布的题为《人民解放军空军编队是中国“隐身战斗机批量生产的标志”》的文章称，中国空军发布了一段7架歼-20隐身战斗机同时飞行的视频，这是迄今为止该机型规模最大的编队飞行。这表明这款第5代战斗机或已进入量产阶段。港媒称，在这段最新的宣传视频中，7架歼-20战斗机参与了一次训练行动。中国人民解放军空军3日通过官方社交账号发布了这段视频。港媒指出，歼-20隐身战斗机于2017年服役，是中国最先进的隐身战斗机。由于其机身拥有较低的雷达剖面，表面涂有吸收能量的隐身材料，其武器可实现超视距打击，歼-20可与F-35隐身战斗机相匹敌。中国人民解放军空军3日发布宣传片展现7架歼-20战斗机编队飞行训练",
            3: "《印度教徒报》网站12月4日发表了题为《印度东部海军司令部司令说，“维克兰特”号航空母舰可能在2022年至2023年入列》的报道称，印度东部海军司令部司令阿图尔·库马尔·贾殷说，科钦造船有限公司正在建造的“维克兰特”号航空母舰可能在2022年至2023年入列印度海军。全文摘编如下：贾殷中将当地时间周四表示，这艘印度国产航母将在东部海军司令部麾下服役，它将与相关计划提议的核潜艇等装备一起，在印太地区组成一支令人生畏的力量。贾殷在年度新闻发布会上说，印度海军很好地契合了“印度制造”计划，即将成为一支强大的远海力量。他说：“目前，印度正在建造国产航母，之后还会建造什瓦里克级多用途护卫舰和卡莫尔塔级反潜护卫舰。今后5到10年，我们希望打造一支具有自主研发能力的优势海军力量。要做到这一点，我们需要国有企业、海军、中小微企业和初创企业共同开展团队合作。”他认为，双边和多边海军演习取得了良好的效果。与日本、俄罗斯和美国的联合演习对遏制中国很有用",
            4: "4月23日下午，中国海军在青岛举行庆祝人民海军成立70周年海上阅兵活动，中国海军32艘各型舰艇与来自13国海军的18艘舰艇列队受阅，西宁舰担任检阅舰。这次海上阅兵全面展现了中国海军建设的最新成果及和平之师的良好形象，也加深了与世界各国海军的友好交流。那么，参加这次阅兵的中国海军舰艇有哪些特点，又反映出中国海上力量发展所取得的哪些成果呢？本文将结合各方发布的公开信息做一简要解析。根据央视新闻客户端23日发布的消息，受阅中国海军舰艇由潜艇群、驱逐舰群、护卫舰群、登陆舰群、辅助舰群和航母群6个舰艇群组成。其中，潜艇群包括2艘战略导弹核潜艇、2艘攻击型核潜艇和4艘常规动力潜艇；驱逐舰群由包括1艘055型驱逐舰在内的7艘驱逐舰组成（另有1艘052D型担任检阅舰）；护卫舰群由4艘054A型、4艘056型护卫舰组成；登陆舰群由4艘登陆舰组成；辅助舰群由2艘大型补给舰、1艘潜艇支援舰、1艘医院船（即名冠中外的“和平方舟”号医院船）组成；航母群则由001型航母辽宁舰组成。从上述受阅舰艇的阵容来看，中国海军已具备构成综合远洋作战编队的关键要素。航母是一支远洋舰艇编队的核心力量，可为在远离国土的海外独自行动的舰队提供空中掩护及对海、对岸攻击力量。055、052D和052C型驱逐舰则拥有强大的区域防空、远程火力打击能力，可与舰载航空兵协同，为编队提供“防护网”和火力杀伤带，充当航母的“带刀护卫”。054A型护卫舰具备中近程区域防空和反舰能力，可单独或在编队内执行中低烈度打击任务，或在大型远洋作战编队中充当补充性作战力量。056型护卫舰则可在近海海域充任“全能舰”，担负近海防御、屏护港口和民用船只，实施近海反潜和扫布雷作战等任务，在未来其可能取代现有的400至1000吨级的近海作战舰艇。资料图片：",
            5: "中国华为公司在美国哥伦比亚特区联邦地区法院对美国商务部提出起诉，理由是美商务部于2017年7月将一套包括服务器和以太网交换机在内的电信设备发往位于加利福尼亚的测试实验室的途中将其扣押在阿拉斯加。",
            6: "华为公司已要求美国最大电信运营商威瑞森通信支付230多项专利的费用，总计金额超过10亿美元",
            7: "2019-03-10 战略态势 外交舆论 美盟台美高官在所罗门群岛会面作者:台湾外交部政务次长徐斯俭和美国国安会亚洲资深主任博明、大洋洲与印太安全主任葛瑞（Alexander Gray）在所罗门群岛会面。",
        }

    def test_extractKeyDocs(self):
        r = requests.post(self.base_url + 'extractDocsKeyWord/', json=self.data)
        print(r.json())

    def test_inferDocTopic(self):
        r = requests.post(self.base_url + 'inferDocsTopic/', json=self.data)
        print(r.json())


if __name__ == '__main__':
    unittest.main()