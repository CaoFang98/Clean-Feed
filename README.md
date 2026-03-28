# CleanFeed - 多平台内容过滤器

一个浏览器插件，支持在多个平台检测并过滤不同类型的内容。

## 功能特性

### 支持的平台
- 📕 小红书
- 📘 知乎

### 检测器
- 🤖 AI内容检测
- ⚠️ 低质量内容检测

### 效果
- 🌫️ 虚化效果
- 🏷️ 标签提示

## 技术栈

### 前端
- TypeScript + React
- Vite + CRXJS
- Tailwind CSS

### 后端
- FastAPI
- Chroma DB（向量库）
- BGE-Small-Chinese（Embedding）

## 快速开始

### 1. 安装后端依赖

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 启动后端服务

```bash
python app.py
```

后端服务将在 `http://127.0.0.1:8765` 启动

### 3. 安装浏览器插件

```bash
npm install
npm run dev
```

然后在浏览器中加载 `dist` 目录作为扩展

## 项目结构

```
clean-feed/
├── src/                      # 浏览器插件源代码
│   ├── core/                 # 核心引擎
│   ├── platforms/            # 平台适配器
│   ├── detectors/            # 检测器
│   └── effects/             # 效果器
├── backend/                  # 后端服务
├── docs/                     # 文档
├── scripts/                  # 脚本
└── config/                   # 配置文件
```

## 开发

### 开发模式

```bash
npm run dev
```

### 生产构建

```bash
npm run build
```

## License

MIT
