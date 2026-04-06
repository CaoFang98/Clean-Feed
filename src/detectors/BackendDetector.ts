import { Detector } from './Detector';
import { DetectionResult } from '../core/types';

interface BackendResponse {
  task_results?: Record<string, {
    label: boolean | null;
    confidence: number;
    reason: string;
  }>;
  is_low_quality: boolean;
  is_ai_generated: boolean;
  confidence: number;
  label: string;
  reason: string;
  rag_example?: string;
}

export class BackendDetector extends Detector {
  readonly id: string;
  readonly name: string;
  readonly taskId: 'is_low_quality' | 'is_ai_generated';
  readonly apiUrl: string = 'http://127.0.0.1:8765/classify';

  private static instances: Map<string, BackendDetector> = new Map();

  private constructor(id: string, name: string, taskId: 'is_low_quality' | 'is_ai_generated') {
    super();
    this.id = id;
    this.name = name;
    this.taskId = taskId;
  }

  public static getInstance(taskId: 'is_low_quality' | 'is_ai_generated'): BackendDetector {
    const id = taskId === 'is_low_quality' ? 'low-quality-backend' : 'ai-content-backend';
    const name = taskId === 'is_low_quality' ? '低质量内容检测(后端)' : 'AI内容检测(后端)';
    
    if (!BackendDetector.instances.has(id)) {
      BackendDetector.instances.set(id, new BackendDetector(id, name, taskId));
    }
    return BackendDetector.instances.get(id)!;
  }

  async detect(content: string): Promise<DetectionResult> {
    try {
      console.log(`[CleanFeed] 🔍 调用后端 API (${this.taskId})...`);
      
      const platform = this.detectPlatform();
      
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: content.substring(0, 3000),
          platform: platform,
          task_ids: [this.taskId],
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend API error: ${response.status}`);
      }

      const data: BackendResponse = await response.json();
      
      console.log(`[CleanFeed] 📊 后端检测结果:`, {
        label: data.label,
        confidence: data.confidence,
        reason: data.reason,
      });

      // 根据检测器类型返回对应的结果
      let shouldFilter = false;
      let confidence = 0;
      const taskResult = data.task_results?.[this.taskId];

      if (taskResult) {
        shouldFilter = taskResult.label === true;
        confidence = taskResult.confidence;
      } else if (this.taskId === 'is_low_quality') {
        shouldFilter = data.is_low_quality;
        confidence = data.is_low_quality ? data.confidence : 1 - data.confidence;
      } else {
        shouldFilter = data.is_ai_generated;
        confidence = data.is_ai_generated ? data.confidence : 1 - data.confidence;
      }

      return {
        detectorId: this.id,
        shouldFilter: shouldFilter,
        confidence: confidence,
        metadata: {
          label: data.label,
          reason: taskResult?.reason || data.reason,
          rag_example: data.rag_example,
          taskResult,
        },
      };
    } catch (error) {
      console.warn(`[CleanFeed] ⚠️ 后端 API 调用失败，使用本地规则:`, error);
      
      // Fallback to simple local rules
      return this.fallbackDetect(content);
    }
  }

  private detectPlatform(): string | undefined {
    const url = window.location.href;
    if (url.includes('xiaohongshu.com')) return 'xiaohongshu';
    if (url.includes('zhihu.com')) return 'zhihu';
    return undefined;
  }

  private fallbackDetect(content: string): DetectionResult {
    const lowerContent = content.toLowerCase();
    
    // Low quality keywords
    const lowQualityKeywords = ['家人们', '谁懂啊', '绝绝子', 'yyds', '宝子们', '挖到宝'];
    const aiKeywords = ['随着...的发展', '在当今社会', '综上所述', '由此可见'];
    
    let lqCount = 0;
    let aiCount = 0;
    
    for (const kw of lowQualityKeywords) {
      if (content.includes(kw)) lqCount++;
    }
    for (const kw of aiKeywords) {
      if (content.includes(kw)) aiCount++;
    }

    let shouldFilter = false;
    let confidence = 0.5;

    if (this.taskId === 'is_low_quality') {
      shouldFilter = lqCount > 0;
      confidence = Math.min(0.9, 0.5 + lqCount * 0.15);
    } else {
      shouldFilter = aiCount > 0;
      confidence = Math.min(0.9, 0.5 + aiCount * 0.2);
    }

    return {
      detectorId: this.id,
      shouldFilter: shouldFilter,
      confidence: confidence,
      metadata: { fallback: true },
    };
  }
}
