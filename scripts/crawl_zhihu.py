#!/usr/bin/env python3
"""
知乎数据爬取脚本
- 非无头模式，首次运行需手动登录
- DOM 选择器结构化提取问题、回答、作者
- 两阶段：列表页收集链接 → 详情页爬完整内容
- 同时保存完整回答(用于标注)和截断回答(用于训练，模拟插件实际看到的内容)
- 自动保存/加载 cookies
"""
import asyncio
import json
import random
import re
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeout

# 路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "zhihu_raw_data.jsonl"
COOKIES_FILE = PROJECT_ROOT / "data" / "zhihu_cookies.json"

# 配置
TARGET_COUNT = 2000  # 目标爬取条数
DELAY_MIN = 3
DELAY_MAX = 6
SCROLL_TIMES = 8  # 每个列表页滚动次数
TRUNCATE_LEN = 200  # 模拟知乎 Feed 页截断长度（字符数）

# 种子页面：用于收集问题链接（话题覆盖多领域，确保数据多样性）
SEED_URLS = [
    "https://www.zhihu.com/hot",
    # 科技
    "https://www.zhihu.com/topic/19550517/top-answers",  # 互联网
    "https://www.zhihu.com/topic/19554298/top-answers",  # 历史
    "https://www.zhihu.com/topic/19551147/top-answers",  # 心理学
    "https://www.zhihu.com/topic/19550825/top-answers",  # 电影
    "https://www.zhihu.com/topic/19552332/top-answers",  # 经济
    "https://www.zhihu.com/topic/19551275/top-answers",  # 教育
    "https://www.zhihu.com/topic/19554791/top-answers",  # 科技
    "https://www.zhihu.com/topic/19552330/top-answers",  # 法律
    # 生活
    "https://www.zhihu.com/topic/19551137/top-answers",  # 美食
    "https://www.zhihu.com/topic/19550374/top-answers",  # 健康
    "https://www.zhihu.com/topic/19551388/top-answers",  # 旅行
    "https://www.zhihu.com/topic/19553632/top-answers",  # 职场
    "https://www.zhihu.com/topic/19550453/top-answers",  # 体育
    # 人文
    "https://www.zhihu.com/topic/19550429/top-answers",  # 音乐
    "https://www.zhihu.com/topic/19554804/top-answers",  # 文学
    "https://www.zhihu.com/topic/19550344/top-answers",  # 哲学
    "https://www.zhihu.com/topic/19553298/top-answers",  # 社会学
    # 理工
    "https://www.zhihu.com/topic/19554535/top-answers",  # 数学
    "https://www.zhihu.com/topic/19551432/top-answers",  # 物理
    "https://www.zhihu.com/topic/19556432/top-answers",  # 生物学
    "https://www.zhihu.com/topic/19552520/top-answers",  # 编程
]


async def random_delay(lo=None, hi=None):
    delay = random.uniform(lo or DELAY_MIN, hi or DELAY_MAX)
    await asyncio.sleep(delay)


async def save_cookies(context):
    cookies = await context.cookies()
    COOKIES_FILE.parent.mkdir(exist_ok=True)
    with open(COOKIES_FILE, "w", encoding="utf-8") as f:
        json.dump(cookies, f, ensure_ascii=False)
    print(f"[cookies] 已保存 {len(cookies)} 个 cookies")


async def load_cookies(context):
    if COOKIES_FILE.exists():
        with open(COOKIES_FILE, "r", encoding="utf-8") as f:
            cookies = json.load(f)
        await context.add_cookies(cookies)
        print(f"[cookies] 已加载 {len(cookies)} 个 cookies")
        return True
    return False


async def wait_for_login(page):
    """等待用户手动登录，检测登录状态"""
    print("\n" + "=" * 60)
    print("请在浏览器中完成以下操作：")
    print("  1. 如果出现验证码，请手动完成验证")
    print("  2. 如果未登录，请登录你的知乎账号")
    print("  3. 登录完成后，等待页面正常加载即可")
    print("脚本会自动检测登录状态...")
    print("=" * 60 + "\n")

    for i in range(120):  # 最多等 2 分钟
        try:
            # 检测是否有登录后才有的元素（头像/个人中心入口）
            logged_in = await page.query_selector(
                'button[aria-label="个人中心"], .AppHeader-profile, img.Avatar'
            )
            if logged_in:
                print("[login] 检测到已登录！")
                return True

            # 检测是否至少能正常看到内容（未登录但能访问）
            content = await page.query_selector(
                '.HotList-item, .TopicFeedList, .QuestionHeader-title, .ContentItem'
            )
            if content:
                print("[login] 检测到页面内容可正常访问")
                return True
        except Exception:
            pass
        await asyncio.sleep(1)

    print("[login] 等待超时，尝试继续...")
    return False


async def collect_question_urls(page, url):
    """从列表页收集问题链接"""
    print(f"[collect] 正在访问: {url}")
    try:
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
    except PlaywrightTimeout:
        print(f"[collect] 页面加载超时，跳过: {url}")
        return []

    await random_delay(2, 4)

    # 滚动加载更多内容
    for i in range(SCROLL_TIMES):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        await asyncio.sleep(1.5)

    # 提取所有问题链接
    links = await page.eval_on_selector_all(
        'a[href*="/question/"]',
        """elements => {
            const seen = new Set();
            const results = [];
            for (const el of elements) {
                const href = el.href;
                // 提取 /question/xxxxx 格式的链接
                const match = href.match(/\\/question\\/(\\d+)/);
                if (match && !seen.has(match[1])) {
                    seen.add(match[1]);
                    results.push({
                        url: 'https://www.zhihu.com/question/' + match[1],
                        question_id: match[1],
                        title: el.textContent.trim().substring(0, 200)
                    });
                }
            }
            return results;
        }"""
    )

    print(f"[collect] 从 {url} 收集到 {len(links)} 个问题链接")
    return links


async def extract_answers_from_question_page(page, question_url):
    """从问题详情页提取完整回答，带整体超时保护"""
    try:
        return await asyncio.wait_for(
            _extract_answers_impl(page, question_url),
            timeout=60  # 单个问题页最多 60 秒
        )
    except asyncio.TimeoutError:
        print(f"[extract] 整体超时 (60s): {question_url}")
        return []
    except Exception as e:
        print(f"[extract] 未知异常: {e}")
        return []


async def _extract_answers_impl(page, question_url):
    """实际的提取逻辑"""
    try:
        await page.goto(question_url, timeout=20000, wait_until="domcontentloaded")
    except PlaywrightTimeout:
        print(f"[extract] 页面加载超时: {question_url}")
        return []

    await random_delay(2, 4)

    # 等待内容加载
    try:
        await page.wait_for_selector(
            '.QuestionHeader-title, .QuestionPage',
            timeout=8000
        )
    except PlaywrightTimeout:
        print(f"[extract] 内容加载超时: {question_url}")
        return []

    # 点击所有 "展开阅读全文" 按钮
    try:
        expand_buttons = await page.query_selector_all('button:has-text("展开阅读全文"), button:has-text("阅读全文")')
        for btn in expand_buttons[:10]:
            try:
                await btn.click()
                await asyncio.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass

    # 滚动几次加载更多回答
    for _ in range(3):
        await page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        await asyncio.sleep(1)

    # 再次点击展开按钮（滚动后可能出现新的）
    try:
        expand_buttons = await page.query_selector_all('button:has-text("展开阅读全文"), button:has-text("阅读全文")')
        for btn in expand_buttons[:10]:
            try:
                await btn.click()
                await asyncio.sleep(0.3)
            except Exception:
                pass
    except Exception:
        pass

    # 用 JS 一次性提取所有结构化数据
    data = await page.evaluate("""() => {
        // 提取问题标题
        const titleEl = document.querySelector('.QuestionHeader-title');
        const question = titleEl ? titleEl.textContent.trim() : '';

        // 提取问题详情
        const detailEl = document.querySelector('.QuestionRichText .RichText');
        const questionDetail = detailEl ? detailEl.textContent.trim() : '';

        // 提取回答列表
        const answerItems = document.querySelectorAll('.AnswerItem, .List-item[tabindex]');
        const answers = [];

        for (const item of answerItems) {
            // 回答正文
            const contentEl = item.querySelector('.RichContent-inner, .RichText');
            if (!contentEl) continue;

            const answerText = contentEl.textContent.trim();
            // 跳过太短的回答
            if (answerText.length < 10) continue;

            // 作者
            const authorEl = item.querySelector(
                '.AuthorInfo-name .UserLink-link, .AuthorInfo-name a, meta[itemprop="name"]'
            );
            const author = authorEl
                ? (authorEl.textContent || authorEl.getAttribute('content') || '').trim()
                : '匿名用户';

            // 赞同数
            const voteEl = item.querySelector(
                'button[aria-label*="赞同"] .Button-label, .VoteButton--up .Button-label'
            );
            let votes = 0;
            if (voteEl) {
                const voteText = voteEl.textContent.trim();
                const m = voteText.match(/[\\d,]+/);
                if (m) votes = parseInt(m[0].replace(/,/g, ''), 10) || 0;
            }

            // 评论数
            const commentEl = item.querySelector(
                'button:has(.css-1ybo4re), button[aria-label*="评论"]'
            );
            let commentCount = 0;
            if (commentEl) {
                const cText = commentEl.textContent.trim();
                const cm = cText.match(/(\\d+)\\s*条?评论/);
                if (cm) commentCount = parseInt(cm[1], 10) || 0;
            }

            answers.push({
                answer: answerText,
                author: author,
                votes: votes,
                comment_count: commentCount,
            });
        }

        return { question, questionDetail, answers };
    }""")

    if not data or not data.get("question"):
        print(f"[extract] 未提取到问题标题: {question_url}")
        return []

    results = []
    for ans in data.get("answers", []):
        full_answer = ans["answer"]
        # 生成截断版本：模拟知乎 Feed 页用户实际看到的内容
        if len(full_answer) > TRUNCATE_LEN:
            truncated = full_answer[:TRUNCATE_LEN] + "..."
        else:
            truncated = full_answer

        results.append({
            "question": data["question"],
            "question_detail": data.get("questionDetail", ""),
            "answer_full": full_answer,
            "answer_truncated": truncated,
            "author": ans["author"],
            "votes": ans["votes"],
            "comment_count": ans["comment_count"],
            "platform": "zhihu",
            "url": question_url,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    print(f"[extract] {data['question'][:30]}... → {len(results)} 条回答")
    return results


def load_existing_data():
    """加载已有数据，用于去重"""
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 兼容旧格式(answer)和新格式(answer_full)
                    answer = item.get("answer_full") or item.get("answer", "")
                    key = item.get("url", "") + "|" + answer[:100]
                    existing.add(key)
                except Exception:
                    pass
    return existing


def save_items(items):
    """追加保存数据"""
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedup_key(item):
    answer = item.get("answer_full") or item.get("answer", "")
    return item.get("url", "") + "|" + answer[:100]


async def main():
    print(f"目标爬取: {TARGET_COUNT} 条回答")
    print(f"输出文件: {OUTPUT_FILE}")

    existing_keys = load_existing_data()
    print(f"已有数据: {len(existing_keys)} 条\n")

    # 清掉之前的无效数据（验证码页面内容）
    if OUTPUT_FILE.exists():
        valid_lines = []
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # 过滤掉验证码/异常页面的数据
                    answer = item.get("answer_full") or item.get("answer", "")
                    if "系统监测" in answer or "验证按钮" in answer or len(answer) < 10:
                        continue
                    if not item.get("question"):
                        continue
                    valid_lines.append(line)
                except Exception:
                    pass
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
        print(f"清理后保留 {len(valid_lines)} 条有效数据\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # 非无头，方便处理验证码/登录
            args=[
                "--disable-blink-features=AutomationControlled",
            ]
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
        )

        # 注入反检测脚本
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)

        # 加载 cookies
        has_cookies = await load_cookies(context)
        page = await context.new_page()

        # 先访问知乎首页，检测登录状态
        print("[init] 正在打开知乎...")
        await page.goto("https://www.zhihu.com", timeout=30000)
        await asyncio.sleep(3)

        # 等待登录
        await wait_for_login(page)
        await save_cookies(context)

        # ===== 阶段一：收集问题链接 =====
        print("\n" + "=" * 40)
        print("阶段一：收集问题链接")
        print("=" * 40)

        all_questions = {}  # question_id -> {url, title}
        for seed_url in SEED_URLS:
            links = await collect_question_urls(page, seed_url)
            for link in links:
                qid = link["question_id"]
                if qid not in all_questions:
                    all_questions[qid] = link
            await random_delay()

            # 如果已经收集了足够多的链接就停止
            if len(all_questions) >= TARGET_COUNT * 2:
                break

        question_list = list(all_questions.values())
        random.shuffle(question_list)  # 打乱顺序，避免按热度顺序爬
        print(f"\n共收集到 {len(question_list)} 个问题链接\n")

        # ===== 阶段二：逐个问题页提取回答 =====
        print("=" * 40)
        print("阶段二：爬取问题详情")
        print("=" * 40)

        total_new = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5  # 连续失败 5 次则重启页面

        for i, q in enumerate(question_list):
            if total_new >= TARGET_COUNT:
                print(f"\n已达到目标 {TARGET_COUNT} 条，停止爬取")
                break

            print(f"\n[{i+1}/{len(question_list)}] (已爬: {total_new}/{TARGET_COUNT})")

            try:
                answers = await extract_answers_from_question_page(page, q["url"])
            except Exception as e:
                print(f"[error] 爬取出错: {e}")
                answers = []

            if not answers:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[recover] 连续 {consecutive_failures} 次失败，重启页面...")
                    try:
                        await page.close()
                    except Exception:
                        pass
                    page = await context.new_page()
                    await random_delay(5, 10)
                    consecutive_failures = 0
                else:
                    await random_delay(3, 6)
                continue

            consecutive_failures = 0

            # 去重并保存
            new_items = []
            for item in answers:
                key = dedup_key(item)
                if key not in existing_keys:
                    existing_keys.add(key)
                    new_items.append(item)

            if new_items:
                save_items(new_items)
                total_new += len(new_items)
                print(f"[save] 新增 {len(new_items)} 条 (总计: {total_new})")

            await random_delay()

        # 保存最终 cookies
        await save_cookies(context)
        await browser.close()

    print(f"\n{'=' * 40}")
    print(f"爬取完成！新增 {total_new} 条数据")
    print(f"数据保存在: {OUTPUT_FILE}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    asyncio.run(main())
