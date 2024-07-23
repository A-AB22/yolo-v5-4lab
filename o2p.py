import tensorrt as trt

# 初始化 TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = [1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
# 创建 TensorRT 构建器和网络定义
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    # 载入 ONNX 模型
    with open("model.onnx", 'rb') as model:
        parser.parse(model.read())

    # 配置构建器以优化和构建 TensorRT 引擎
    builder.max_workspace_size = 1 << 30  # 设置工作空间大小
    builder.max_batch_size = your_batch_size  # 你的批量大小
    engine = builder.build_cuda_engine(network)

    # 序列化 TensorRT 引擎到文件
    with open("model_trt.engine", "wb") as f:
        f.write(engine.serialize())