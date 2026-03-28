import { DetectionResult } from '../core/types';

export abstract class Detector {
  abstract readonly id: string;
  abstract readonly name: string;

  /**
   * 检测内容
   */
  abstract detect(content: string): Promise<DetectionResult>;
}
