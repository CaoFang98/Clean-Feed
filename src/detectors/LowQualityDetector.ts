import { Detector } from './Detector';
import { DetectionResult } from '../core/types';

export class LowQualityDetector extends Detector {
  readonly id = 'low-quality';
  readonly name = '低质量内容检测';

  private static instance: LowQualityDetector;

  // 低质量关键词（中文）
  private readonly lowQualityKeywords = [
    '不看后悔', '震惊', '快看', '必看', '转疯了', '收藏了',
    '干货', '纯干货', '太有用了', '绝了', 'yyds', '永远的神',
    '家人们', '谁懂啊', '救命', '我真的', '谁懂', '咱就是说',
    '一整个', '大动作', '狠狠', '绝绝子', '集美', '兄dei',
    '99%', '100%', '赶紧', '马上', '立刻', '速度',
    '免费领', '红包', '福利', '点击领取', '扫码', '加微信',
    '微商', '代购', '赚钱', '副业', '兼职', '日赚',
    '爆款', '热销', '限时', '秒杀', '抢购', '最后一天',
  ];

  // 低质量短语模式
  private readonly lowQualityPatterns = [
    /^[0-9]+[.、]\s*[^\n]{0,10}$/gm,  // 短句列表
    /[!！]{2,}/g,                       // 连续感叹号
    /[?？]{2,}/g,                       // 连续问号
    /[a-zA-Z]{5,}/g,                    // 过长英文（可能是乱码）
    /[0-9]{8,}/g,                       // 过长数字（可能是广告）
  ];

  public static getInstance(): LowQualityDetector {
    if (!LowQualityDetector.instance) {
      LowQualityDetector.instance = new LowQualityDetector();
    }
    return LowQualityDetector.instance;
  }

  async detect(content: string): Promise<DetectionResult> {
    let score = 0;
    const reasons: string[] = [];

    // 1. 检查文本长度
    const length = content.length;
    if (length < 30) {
      score += 0.3;
      reasons.push('内容过短');
    } else if (length > 2000) {
      score -= 0.1;  // 长文通常质量较高
    }

    // 2. 检查低质量关键词
    let keywordCount = 0;
    for (const keyword of this.lowQualityKeywords) {
      const regex = new RegExp(keyword, 'gi');
      const matches = content.match(regex);
      if (matches) {
        keywordCount += matches.length;
      }
    }
    if (keywordCount > 0) {
      score += Math.min(keywordCount * 0.1, 0.4);
      reasons.push(`包含 ${keywordCount} 个低质量关键词`);
    }

    // 3. 检查低质量模式
    let patternCount = 0;
    for (const pattern of this.lowQualityPatterns) {
      const matches = content.match(pattern);
      if (matches) {
        patternCount += matches.length;
      }
    }
    if (patternCount > 0) {
      score += Math.min(patternCount * 0.05, 0.3);
      reasons.push(`匹配 ${patternCount} 个低质量模式`);
    }

    // 4. 检查平均句长
    const sentences = content.split(/[。！？.!?]/).filter(s => s.trim().length > 0);
    if (sentences.length > 0) {
      const avgSentenceLength = content.length / sentences.length;
      if (avgSentenceLength < 10) {
        score += 0.2;
        reasons.push('句子过短');
      }
    }

    // 5. 检查特殊字符比例
    const specialChars = content.match(/[^\p{L}\p{N}\s]/gu) || [];
    const specialRatio = specialChars.length / content.length;
    if (specialRatio > 0.3) {
      score += 0.2;
      reasons.push('特殊字符过多');
    }

    // 归一化分数到 0-1
    const confidence = Math.max(0, Math.min(1, score));

    console.log(`[CleanFeed] 📉 低质量检测: 分数=${confidence.toFixed(2)}, 原因=${reasons.join(', ')}`);

    return {
      detectorId: this.id,
      shouldFilter: confidence > 0.5,
      confidence,
      metadata: { reasons },
    };
  }
}
