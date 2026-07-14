# 感染性葡萄膜炎分类工程交付包

这是项目结项使用的可运行工程包。系统根据 1–4 张眼底图像，以及可选的血常规文本或结构化数值，输出“感染性葡萄膜炎”或“非感染性葡萄膜炎”的辅助分类结果。

交付包包含：

- 单病例命令行推理。
- 中文 Web 演示页面。
- JSONL 批量推理与输入预检查。
- 已训练模型、固定依赖、合成 smoke test 和 SHA-256 交付清单。

本仓库不包含原始病例、患者标识、训练数据、训练脚本或实验过程文件。

## 1. 环境安装

已验证环境为 Python 3.10。推荐使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

首次运行时，`torchvision` 可能需要下载 MobileNetV3-Large 的 ImageNet 预训练权重。GuangXi 上已验证的环境为：

```text
/home/yehaocun23s/venvs/qwen_vllm310/bin/python
```

## 2. 交付自检

```bash
python tools/verify_delivery.py
python tests/smoke_delivery.py --device cpu
```

smoke test 使用程序生成的合成图像，不需要真实患者数据。测试内容包括模型加载、图像 backbone、full/image-only 两个分支、Web API、JSONL 预检查和批量推理。

## 3. 单病例推理

### 图像 + 血常规

```bash
python infer_final_uveitis.py \
  --image /path/to/image1.jpg \
  --image /path/to/image2.jpg \
  --image /path/to/image3.jpg \
  --image /path/to/image4.jpg \
  --prompt "WBC 6.10; NEUT% 61.5%; LYMPH# 1.20; HGB 128; PLT 230"
```

### 仅图像

```bash
python infer_final_uveitis.py --image-dir /path/to/case_images
```

输入了文本但没有识别到支持的血常规时，CLI 默认报错。确实需要退回仅图像分支时添加：

```bash
--allow-image-only-fallback
```

关键输出字段：

- `prediction`：分类标签。
- `fusion_branch`：实际使用 `full` 或 `image_only`。
- `fusion_raw_probability`：参与阈值判断的感染性分数。
- `fusion_calibrated_probability`：用于展示的校准分数。
- `recognized_lab_count`：识别到的血常规项目数。
- `input_warning_codes`：输入质量提示。

## 4. 中文 Web

```bash
python -m uvicorn demo.app:app --host 127.0.0.1 --port 7860
```

打开 `http://127.0.0.1:7860`。页面支持上传 1–4 张 JPG/PNG；填写可解析血常规时走 full 分支，否则走 image-only 分支。

## 5. JSONL 批量推理

病例格式见 `examples/external_cases.example.jsonl`。图片相对路径按 `--data-root` 解析，文本/JSON 侧车相对路径按 JSONL 文件所在目录解析。

先检查输入：

```bash
python validate_external_cases_jsonl.py \
  --jsonl /path/to/cases.jsonl \
  --data-root /path/to/image_root \
  --mode auto \
  --output external-output/preflight_report.json
```

再执行批量推理：

```bash
python evaluate_final_external_jsonl.py \
  --jsonl /path/to/cases.jsonl \
  --data-root /path/to/image_root \
  --output-dir external-output \
  --mode auto \
  --device cpu
```

输出包括 `predictions.jsonl` 和 `summary.json`。如果服务器需要使用 GPU，将 `--device cpu` 改为 `--device cuda`。

## 6. 输入约定

- 推荐每例提供 4 张眼底图像，程序支持 1–4 张。
- full 分支建议至少识别到 5 项血常规。
- 支持 WBC、NEUT%、LYMPH#、HGB、PLT 等常见中文名称和缩写。
- CRP、降钙素原、ESR 等非血常规指标不会进入当前模型。
- 模型 bundle 为可信交付文件；不要加载来源不明的 `.joblib` 文件。

模型版本和工程验收摘要见 `artifacts/model_metadata.json`。

> 本系统用于项目演示和辅助分类，不作为独立临床诊断依据。

