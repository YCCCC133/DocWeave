# DocWeave · 本地文档转换

基于 MinerU 的本地 PDF 转换工具。Swift 原生界面管理转换队列，Python 后端提供文字识别、公式识别和 Markdown 输出。

## 功能

- 不同识别档位与本地模型推理
- 已保存页面和当前识别阶段分别显示进度
- MLX、Torch 后端与本地批量处理
- 包含已安装版本的本地运行库源码及性能补丁

## 从源码构建

要求 macOS 14 或更新版本，以及 Xcode Command Line Tools。

```bash
./build.sh --compile-only
```

这会编译 Swift 程序到 `build/`。运行完整应用还需要以下依赖：

Apple Silicon、Python 3.11、MinerU 4.0.10、MLX-VLM 0.7.4，具体版本见 `requirements.txt`。模型权重需自行下载，参考 `Resources/recognition-manifest.json` 和上游模型说明。

`Sources/` 存放原生界面源码，`Resources/` 存放应用资源和后端脚本。准备好依赖后运行 `./build.sh` 生成应用包。构建脚本执行本地 ad-hoc 签名；正式发布需另行签名和公证。

如已有对应应用的完整依赖，可以指定其资源目录重新打包：

```bash
APP_RESOURCES="/path/to/Application.app/Contents/Resources" ./build.sh
```

此方式复用第三方运行时，重新编译本仓库的原生程序；构建结果在 `build/`，不提交到源码仓库。脚本不会打包已有的文件索引或用户转换结果。

## Python 后端与模型

`Resources/convert.py` 和 `Resources/fast_ocr.py` 是转换入口。运行环境采用以下应用资源布局：

```text
python/Frameworks/Python.framework/Versions/3.11/bin/python3.11
venv/lib/python3.11/site-packages/
mlx-runtime/mlx_vlm/
models/
```

可用 Python 3.11 创建依赖环境并安装 `requirements.txt`，之后把 `vendor/mineru`、`vendor/mineru_vl_utils` 覆盖到对应 site-packages，把 `vendor/mlx_vlm` 放到 `mlx-runtime/`。依赖复制脚本会自动应用仓库内的运行库源码。

`vendor/` 是当前应用使用的源码快照，包含本地补丁，不应当用上游安装包直接覆盖。模型、Python 可执行文件及编译扩展未收入 Git，需按上述布局准备；单独运行 Swift 编译不代表识别环境已配置完成。

## 许可证与第三方组件

本项目原创界面、脚本和构建文件使用 [MIT License](LICENSE)。第三方源码、图标和运行时遵循其原有许可，根目录许可不覆盖第三方材料。

[MinerU](https://github.com/opendatalab/MinerU)、[MLX-VLM](https://github.com/Blaizzy/mlx-vlm)。`vendor/mineru` 使用 MinerU 附加条款许可；`vendor/mineru_vl_utils` 使用 Apache-2.0；`vendor/mlx_vlm` 使用 MIT。各目录保留原始许可证。
