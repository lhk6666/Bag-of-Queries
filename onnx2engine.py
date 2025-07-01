import tensorrt as trt

onnx_path = "models/test.onnx"
engine_path = "models/test.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_path, "rb") as model:
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# ===== 动态输入处理，添加 optimization profile =====
profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    input_name = network.get_input(i).name
    # 这里假设输入 shape 是 [batch, 3, 224, 224]
    # 你可以根据实际情况修改
    profile.set_shape(input_name, min=(1, 3, 224, 224), opt=(1, 3, 224, 224), max=(1, 3, 224, 224))
config.add_optimization_profile(profile)
# ===============================================

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    print("Engine build failed!")
    exit(1)

with open(engine_path, "wb") as f:
    f.write(serialized_engine)

print("Engine saved to", engine_path)
