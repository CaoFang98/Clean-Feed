import { DetectionResult } from '../core/types';

export abstract class Effect {
  abstract readonly id: string;
  abstract readonly name: string;

  /**
   * 应用效果
   */
  abstract apply(element: HTMLElement, result: DetectionResult, options?: Record<string, any>): void;

  /**
   * 移除效果
   */
  abstract remove(element: HTMLElement): void;
}
