import { Detector } from './Detector';
import { DetectionResult } from '../core/types';
import { pipeline, env } from '@xenova/transformers';

// 配置 Transformers.js 使用本地模型
env.allowLocalModels = false;
env.allowRemoteModels = true;

export class AIDetector extends Detector {
  readonly id = 'ai-content';
  readonly name = 'AI内容检测';

  private static instance: AIDetector;
  private classifier: any = null;
  private isLoading = false;
  private initPromise: Promise<void> | null = null;

  public static getInstance(): AIDetector {
    if (!AIDetector.instance) {
      AIDetector.instance = new AIDetector();
    }
    return AIDetector.instance;
  }

  private async init() {
    if (this.classifier) return;
    if (this.isLoading && this.initPromise) return this.initPromise;

    this.isLoading = true;
    console.log('[CleanFeed] 🧠 正在加载 AI 检测模型...');

    this.initPromise = (async () => {
      try {
        console.log('[CleanFeed] 📥 开始加载模型...');
        this.classifier = await pipeline(
          'text-classification',
          'Xenova/roberta-base-openai-detector',
          {
            progress_callback: (progress: any) => {
              if (progress.status === 'downloading') {
                console.log(`[CleanFeed] 📥 下载进度: ${(progress.progress * 100).toFixed(1)}%`);
              } else if (progress.status === 'loaded') {
                console.log('[CleanFeed] ✅ 模型加载完成！');
              }
            }
          }
        );
        console.log('[CleanFeed] ✅ AI 检测模型加载成功！');
      } catch (error) {
        console.error('[CleanFeed] ❌ 模型加载失败:', error);
        throw error;
      } finally {
        this.isLoading = false;
      }
    })();

    return this.initPromise;
  }

  async detect(content: string): Promise<DetectionResult> {
    await this.init();

    if (!this.classifier) {
      console.warn('[CleanFeed] ⚠️ 检测器未初始化，返回默认结果');
      return {
        detectorId: this.id,
        shouldFilter: false,
        confidence: 0,
      };
    }

    try {
      const result = await this.classifier(content);
      const aiResult = result.find((r: any) => r.label === 'AI');
      const humanResult = result.find((r: any) => r.label === 'Human');

      const aiScore = aiResult?.score || 0;
      const humanScore = humanResult?.score || 0;

      console.log(`[CleanFeed] 🔍 AI检测结果: AI=${(aiScore * 100).toFixed(1)}%, Human=${(humanScore * 100).toFixed(1)}%`);

      return {
        detectorId: this.id,
        shouldFilter: aiScore > humanScore,
        confidence: aiScore,
        metadata: { aiScore, humanScore },
      };
    } catch (error) {
      console.error('[CleanFeed] ❌ 检测过程出错:', error);
      return {
        detectorId: this.id,
        shouldFilter: false,
        confidence: 0,
      };
    }
  }
}
