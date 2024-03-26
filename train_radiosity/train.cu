#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>
#include <filesystem/path.h>
#include <filesystem/directory.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace tcnn;
using precision_t = network_precision_t;

struct TrainingMetadata {
	GPUMemory<float> posW;
	GPUMemory<float> normalW;
	GPUMemory<float> wiW;
	GPUMemory<float> diff;
	GPUMemory<float> color;

	ivec2 resolution = ivec2(0);

	TrainingMetadata(const uint32_t width, const uint32_t height) {
		resolution.x = width;
		resolution.y = height;
		posW.resize(width * height * 4 * 4);	// 4 channels * 4 bytes
		normalW.resize(width * height * 4 * 4);
		wiW.resize(width * height * 4 * 4);
		diff.resize(width * height * 4 * 4);
		color.resize(width * height * 4 * 4);
	}
};

struct TrainingTexture {
	cudaTextureObject_t posW;
	cudaTextureObject_t normalW;
	cudaTextureObject_t wiW;
	cudaTextureObject_t diff;
	cudaTextureObject_t color;
};

GPUMemory<float> load_image(const filesystem::path& path, int& width, int& height) {
	// width * height * RGBA
	float* out;
	const char* err = nullptr;
	int ret = LoadEXR(&out, &width, &height, path.str().c_str(), &err);
	FreeEXRErrorMessage(err);
	GPUMemory<float> result(width * height * 4);
	result.copy_from_host(out);
	free(out);
	return result;
}

TrainingMetadata load_metadata(filesystem::path& folder_path) {
	std::vector<filesystem::path> paths;
	for (auto& path: filesystem::directory(folder_path)) {
		if (path.is_file() && path.extension() == "exr") {
			paths.push_back(path);
		}
	}
	std::sort(paths.begin(), paths.end(), [](const filesystem::path& a, const filesystem::path& b) {
		return a.str() < b.str();
	});

	int width, height;

	load_image(paths[0], width, height);

	TrainingMetadata result(width, height);

	uint32_t img_size = width * height * 4 * 4;

	for (auto& path: paths) {
		size_t lastDot = path.str().rfind('.');
		size_t secondLastDot = path.str().rfind('.', lastDot - 1);
		size_t thirdLastDot = path.str().rfind('.', secondLastDot - 1);
		std::string buffer_name = path.str().substr(thirdLastDot + 1, secondLastDot - thirdLastDot - 1);

		if (buffer_name == "posW") {
			cudaMemcpy(result.posW.data(), load_image(path, width, height).data(), img_size, cudaMemcpyDeviceToDevice);
		}
		if (buffer_name == "normalW") {
			cudaMemcpy(result.normalW.data(), load_image(path, width, height).data(), img_size, cudaMemcpyDeviceToDevice);
		}
		if (buffer_name == "wiW") {
			cudaMemcpy(result.wiW.data(), load_image(path, width, height).data(), img_size, cudaMemcpyDeviceToDevice);
		}
		if (buffer_name == "diff") {
			cudaMemcpy(result.diff.data(), load_image(path, width, height).data(), img_size, cudaMemcpyDeviceToDevice);
		}
		if (buffer_name == "color") {
			cudaMemcpy(result.color.data(), load_image(path, width, height).data(), img_size, cudaMemcpyDeviceToDevice);
		}
	}
	return result;
}

void create_cuda_texture(GPUMemory<float>& image, uint32_t width, uint32_t height, cudaTextureObject_t& texture) {
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = image.data();
	resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	resDesc.res.pitch2D.width = width;
	resDesc.res.pitch2D.height = height;
	resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = true;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;

	CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
}

TrainingTexture create_training_texture(TrainingMetadata& metadata) {
	TrainingTexture result;
	create_cuda_texture(metadata.posW, metadata.resolution.x, metadata.resolution.y, result.posW);
	create_cuda_texture(metadata.normalW, metadata.resolution.x, metadata.resolution.y, result.normalW);
	create_cuda_texture(metadata.wiW, metadata.resolution.x, metadata.resolution.y, result.wiW);
	create_cuda_texture(metadata.diff, metadata.resolution.x, metadata.resolution.y, result.diff);
	create_cuda_texture(metadata.color, metadata.resolution.x, metadata.resolution.y, result.color);

	return result;
}

void destroyTexture(TrainingTexture texture) {
	cudaDestroyTextureObject(texture.posW);
	cudaDestroyTextureObject(texture.normalW);
	cudaDestroyTextureObject(texture.wiW);
	cudaDestroyTextureObject(texture.diff);
	cudaDestroyTextureObject(texture.color);
}

template <typename T, uint32_t input_stride, uint32_t output_stride>
__global__ void sample_input_output(uint32_t n_elements, TrainingTexture texture,
									T* xs_and_ys, T* input, T* output) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	uint32_t texture_offset = i * 2;
	uint32_t input_offset = i * input_stride;
	uint32_t output_offset = i * output_stride;

	float4 posW_val 	= tex2D<float4>(texture.posW, xs_and_ys[texture_offset + 0], xs_and_ys[texture_offset + 1]);
	float4 normalW_val 	= tex2D<float4>(texture.normalW, xs_and_ys[texture_offset + 0], xs_and_ys[texture_offset + 1]);
	float4 wiW_val 		= tex2D<float4>(texture.wiW, xs_and_ys[texture_offset + 0], xs_and_ys[texture_offset + 1]);
	float4 diff_val 	= tex2D<float4>(texture.diff, xs_and_ys[texture_offset + 0], xs_and_ys[texture_offset + 1]);
	float4 color_val 	= tex2D<float4>(texture.diff, xs_and_ys[texture_offset + 0], xs_and_ys[texture_offset + 1]);

	input[input_offset + 0] = posW_val.x;		input[input_offset + 1] = posW_val.y;		input[input_offset + 2] = posW_val.z;
	// input[input_offset + 3] = normalW_val.x;	input[input_offset + 4] = normalW_val.y;	input[input_offset + 5] = normalW_val.z;
	// input[input_offset + 6] = wiW_val.x;		input[input_offset + 7] = wiW_val.y;		input[input_offset + 8] = wiW_val.z;
	// input[input_offset + 9] = diff_val.x;		input[input_offset + 10] = diff_val.y;		input[input_offset + 11] = diff_val.z;

	output[output_offset + 0] = color_val.x;	output[output_offset + 1] = color_val.y;	output[output_offset + 2] = color_val.z;
}

int main(int argc, char* argv[]) {
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		if (argc != 3) {
			std::cerr << "Usage: " << argv[0] << " <dataset folder> <json path>" << std::endl;
			return EXIT_FAILURE;
		}

		filesystem::path dataset_folder(argv[1]);
		std::vector<filesystem::path> frame_folders;
		if (!dataset_folder.empty()) {
			if (dataset_folder.is_directory()) {
				for (const auto& path: filesystem::directory(dataset_folder)) {
					if (path.is_directory() && path.str().find("frame") != std::string::npos) {
						frame_folders.push_back(path);
					}
				}
			}
		}

		if (frame_folders.empty()) {
			std::cerr << "Error: No frame folders found in dataset folder." << std::endl;
			return EXIT_FAILURE;
		}

		std::ifstream f{argv[2]};
		json config = json::parse(f, nullptr, true, true);

		const uint32_t n_training_steps = 5000;
		const uint32_t n_input_dims = 12;
		const uint32_t n_texture_dims = 2;
		const uint32_t n_output_dims = 3;
		const uint32_t batch_size = 1 << 12;
		const uint32_t n_frames = frame_folders.size();

		const uint32_t log_interval = 100;
		float tmp_loss = 0;
		uint32_t tmp_loss_counter = 0;

		cudaStream_t inference_stream;
		CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));
		cudaStream_t training_stream = inference_stream;

		default_rng_t rng{1337};

		json encoding_opts = config.value("encoding", json::object());
		json loss_opts = config.value("loss", json::object());
		json optimizer_opts = config.value("optimizer", json::object());
		json network_opts = config.value("network", json::object());

		std::shared_ptr<Loss<precision_t>> loss{create_loss<precision_t>(loss_opts)};
		std::shared_ptr<Optimizer<precision_t>> optimizer{create_optimizer<precision_t>(optimizer_opts)};
		std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = std::make_shared<NetworkWithInputEncoding<precision_t>>(n_input_dims, n_output_dims, encoding_opts, network_opts);

		auto trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(network, optimizer, loss);

		GPUMatrix<float> training_input(n_input_dims, batch_size);
		GPUMatrix<float> training_output(n_output_dims, batch_size);
		GPUMatrix<float> sample_coords(n_texture_dims, batch_size);

		for (uint32_t i = 0; i < n_training_steps; i++) {
			uint32_t training_set_idx = (uint32_t)(rng.next_float() * n_frames);
			TrainingMetadata metadata = load_metadata(frame_folders[training_set_idx]);
			TrainingTexture training_texture = create_training_texture(metadata);

			generate_random_uniform<float>(training_stream, rng, batch_size * n_texture_dims, sample_coords.data());
			linear_kernel(sample_input_output<float, n_input_dims, n_output_dims>, 0, training_stream,
				batch_size, training_texture, sample_coords.data(), training_input.data(), training_output.data());

			auto ctx = trainer->training_step(training_stream, training_input, training_output);

			tmp_loss += trainer->loss(training_stream, *ctx);
			tmp_loss_counter++;

			if (i % log_interval == 0) {
				std::cout << "Step#" << i << ": " << "loss=" << tmp_loss/(float)tmp_loss_counter << std::endl;

				tmp_loss = 0;
				tmp_loss_counter = 0;
			}

			destroyTexture(training_texture);
		}

		json network_config;
		std::string network_config_save_path = "network_weights.json";
		network_config = trainer->serialize(false);
		std::ofstream of(network_config_save_path);
		of << network_config.dump(4);
		of.close();

		free_all_gpu_memory_arenas();

	} catch (const std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}
