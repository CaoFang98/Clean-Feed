import { PlatformAdapter } from './PlatformAdapter';
import { Post } from '../core/types';

export class ZhihuAdapter extends PlatformAdapter {
  readonly id = 'zhihu';
  readonly name = '知乎';
  private readonly previewMaxChars = 200;

  private static instance: ZhihuAdapter;

  public static getInstance(): ZhihuAdapter {
    if (!ZhihuAdapter.instance) {
      ZhihuAdapter.instance = new ZhihuAdapter();
    }
    return ZhihuAdapter.instance;
  }

  matchUrl(url: string): boolean {
    return url.includes('zhihu.com');
  }

  getPosts(): Post[] {
    const posts: Post[] = [];
    
    // 尽量聚焦回答/文章卡片，避免把整页容器或纯问题卡片也送进检测器
    const selectors = [
      '.Feed-item',
      '.AnswerItem',
      '.ContentItem',
      'article',
    ];
    
    const elements = new Set<Element>();
    selectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => elements.add(el));
    });

    console.log(`[CleanFeed] 📝 知乎找到 ${elements.size} 个潜在帖子元素`);

    elements.forEach((element) => {
      const postContent = this.extractPostContent(element as HTMLElement);
      
      // 只处理能稳定提取出“回答预览”的卡片
      if (!postContent || postContent.answerPreview.length < 20) {
        return;
      }
      
      // 检查是否已经被处理过
      if ((element as HTMLElement).dataset.feedProcessed === 'true') {
        return;
      }
      
      // 标记为已处理
      (element as HTMLElement).dataset.feedProcessed = 'true';
      
      const post: Post = {
        id: this.generateId(element as HTMLElement),
        element: element as HTMLElement,
        text: postContent.modelInput,
        metadata: {
          question: postContent.question,
          answerPreview: postContent.answerPreview,
          answerLength: postContent.answerPreview.length,
        },
        detectionResults: new Map(),
      };

      console.log('[CleanFeed] 🧾 知乎提取结果:', {
        question: postContent.question || '(无问题标题)',
        answerPreview: postContent.answerPreview,
        modelInput: postContent.modelInput,
        answerLength: postContent.answerPreview.length,
        modelInputLength: postContent.modelInput.length,
      });
      
      posts.push(post);
    });

    console.log(`[CleanFeed] ✅ 知乎筛选出 ${posts.length} 个有效帖子`);
    return posts;
  }

  extractText(element: HTMLElement): string {
    const postContent = this.extractPostContent(element);
    if (postContent) {
      return postContent.modelInput;
    }
    return this.normalizeText(element.innerText || element.textContent || '');
  }

  private extractPostContent(element: HTMLElement): { question: string; answerPreview: string; modelInput: string } | null {
    const question = this.extractQuestion(element);
    const answerPreview = this.extractAnswerPreview(element);

    if (!answerPreview) {
      return null;
    }

    return {
      question,
      answerPreview,
      modelInput: this.buildModelInput(question, answerPreview),
    };
  }

  private extractQuestion(element: HTMLElement): string {
    const questionSelectors = [
      '.ContentItem-title',
      '.QuestionItem-title',
      'h2 a',
      'h2',
    ];

    for (const selector of questionSelectors) {
      const questionEl = element.querySelector(selector);
      const text = this.normalizeText(questionEl?.textContent || '');
      if (text.length >= 4) {
        return text;
      }
    }

    const detailTitle = document.querySelector('.QuestionHeader-title');
    return this.normalizeText(detailTitle?.textContent || '');
  }

  private extractAnswerPreview(element: HTMLElement): string {
    const answerSelectors = [
      '.RichContent-inner',
      '.RichText',
      '[itemprop="text"]',
    ];

    for (const selector of answerSelectors) {
      const contentEl = element.querySelector(selector) as HTMLElement | null;
      const text = this.extractCleanText(contentEl);
      if (text.length >= 20) {
        return this.buildPreview(text);
      }
    }

    return '';
  }

  private extractCleanText(element: HTMLElement | null): string {
    if (!element) {
      return '';
    }

    const textParts: string[] = [];
    
    const extract = (node: Node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = node.textContent?.trim() || '';
        if (text.length > 0) {
          textParts.push(text);
        }
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        const elem = node as Element;
        const tagName = elem.tagName.toLowerCase();
        if (tagName === 'script' || tagName === 'style' || tagName === 'svg' || tagName === 'noscript') {
          return;
        }
        elem.childNodes.forEach(extract);
      }
    };
    
    element.childNodes.forEach(extract);
    return this.normalizeText(textParts.join(' '));
  }

  private normalizeText(text: string): string {
    return text.replace(/\s+/g, ' ').trim();
  }

  private buildPreview(text: string): string {
    if (text.length <= this.previewMaxChars) {
      return text;
    }
    return `${text.slice(0, this.previewMaxChars).trimEnd()}...`;
  }

  private buildModelInput(question: string, answerPreview: string): string {
    if (question) {
      return `问题：${question}\n回答预览：${answerPreview}`;
    }
    return answerPreview;
  }

  getContainer(element: HTMLElement): HTMLElement {
    return element;
  }
}
