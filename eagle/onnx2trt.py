import tensorrt as trt
import sys

# ONNX文件路径
onnx_file_path = "trt-org/eagle_small.onnx"
# 生成的TensorRT引擎文件路径
engine_file_path = "trt/small_engine.engine"

# 创建一个TensorRT logger对象
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 创建builder和network
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 解析ONNX模型
with open(onnx_file_path, 'rb') as model:
    if not parser.parse(model.read()):
        print ('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print (parser.get_error(error))
        sys.exit(-1)

# 建立优化配置
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (8 << 30)) # 修改为所需的工作空间大小

profile = builder.create_optimization_profile()
# --minShapes=past_key_values:2x1x20x1x128 --optShapes=past_key_values:2x1x20x800x128 --maxShapes=past_key_values:2x1x20x1024x128
# profile.set_shape("past_key_values", (2, 1, 20, 1, 128), (2, 1, 20, 800, 128), (2, 1, 20, 900, 128)) 
profile.set_shape("past_key_values", (2, 1, 20, 1, 128), (2, 1, 20, 800, 128), (2, 1, 20, 2048, 128)) 
config.add_optimization_profile(profile)

# 选择FP16模式
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# 创建TensorRT引擎
engine = builder.build_engine(network, config)

# 将引擎保存到文件
with open(engine_file_path, "wb") as f:
    f.write(engine.serialize())

print("ONNX模型已成功转换为TensorRT模型并保存至：", engine_file_path)