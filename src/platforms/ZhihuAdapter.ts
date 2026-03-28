import { PlatformAdapter } from './PlatformAdapter';
import { Post } from '../core/types';

export class ZhihuAdapter extends PlatformAdapter {
  readonly id = 'zhihu';
  readonly name = '知乎';

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
    
    // 知乎常见的帖子/回答容器
    const selectors = [
      '.Feed-item',              // 信息流
      '.AnswerItem',             // 回答列表
      '.QuestionItem',           // 问题列表
      '[class*="Card"]',         // 卡片
      '[class*="List-item"]',    // 列表项
      '[data-zop-feedlist]',     // 信息流标记
      'article',                 // 文章
    ];
    
    const elements = new Set<Element>();
    selectors.forEach(selector => {
      document.querySelectorAll(selector).forEach(el => elements.add(el));
    });

    console.log(`[CleanFeed] 📝 知乎找到 ${elements.size} 个潜在帖子元素`);

    elements.forEach((element) => {
      const text = this.extractText(element as HTMLElement);
      
      // 过滤掉太短的内容
      if (text.length < 20) {
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
        text,
        detectionResults: new Map(),
      };
      
      posts.push(post);
    });

    console.log(`[CleanFeed] ✅ 知乎筛选出 ${posts.length} 个有效帖子`);
    return posts;
  }

  extractText(element: HTMLElement): string {
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
    return textParts.join(' ').replace(/\s+/g, ' ').trim();
  }

  getContainer(element: HTMLElement): HTMLElement {
    return element;
  }
}
