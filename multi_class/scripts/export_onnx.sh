model_path=$1
optimum-cli export onnx --task text-classification --model $model_path  $model_path/onnx
optimum-cli onnxruntime quantize --avx512 --onnx_model  $model_path/onnx  -o  $model_path/onnx_quantized
