#include "Network.h"

#include <fstream>
#include <iostream>
#include <filesystem/path.h>

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/config.h>
#include <json/json.hpp>

using namespace tcnn;
using precision_t = network_precision_t;

namespace
{

struct NetworkComponents {
    std::shared_ptr<Loss<precision_t>> loss = nullptr;
    std::shared_ptr<Optimizer<precision_t>> optimizer = nullptr;
    std::shared_ptr<NetworkWithInputEncoding<precision_t>> network = nullptr;
    std::shared_ptr<Trainer<float, precision_t, precision_t>> trainer = nullptr;
};

struct IOData {
    GPUMatrix<float>* input_mat = nullptr;
    GPUMatrix<float>* output_mat = nullptr;
};

cudaStream_t inference_stream = nullptr;

NetworkComponents* mNetworkComponents = nullptr;

IOData* mIOData = nullptr;

}

template <typename T, uint32_t stride>
__global__ void formatInput(uint32_t n_elements, Falcor::RadiosityQuery* queries, T* input)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Falcor::RadiosityQuery query = queries[i];

    input[i * stride + 0] = query.posW.x;           input[i * stride + 1] = query.posW.y;           input[i * stride + 2] = query.posW.z;
    input[i * stride + 3] = query.normalW.x;        input[i * stride + 4] = query.normalW.y;        input[i * stride + 5] = query.normalW.z;
    input[i * stride + 6] = query.wiW.x;            input[i * stride + 7] = query.wiW.y;            input[i * stride + 8] = query.wiW.z;
    input[i * stride + 9] = query.diff.x;           input[i * stride + 10] = query.diff.y;          input[i * stride + 11] = query.diff.z;
}


template <typename T, uint32_t stride>
__global__ void mapToOutSurf(uint32_t n_elements, uint32_t width, T* output, cudaSurfaceObject_t outSurf)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    uint32_t x = i % width;
    uint32_t y = i / width;

    float4 color = { 0, 0, 0, 1 };

    color.x = output[i * stride + 0];
    color.y = output[i * stride + 1];
    color.z = output[i * stride + 2];

    surf2Dwrite(color, outSurf, x * sizeof(float4), y);
}


RadiosityNetwork::RadiosityNetwork(const uint32_t width, const uint32_t height)
{
    CUDA_CHECK_THROW(cudaStreamCreate(&inference_stream));

    mNetworkComponents = new NetworkComponents();
    mIOData = new IOData();

    filesystem::path c_path(NetConfig::netConfigPath);
    if (!c_path.exists()) {
        std::cout << "Cannot find the network config!" << std::endl;
        return;
    } else {
        std::cout << "Successfully find the network config!" << std::endl;
    }

    std::ifstream f(c_path.str());
    json config = json::parse(f, nullptr, true, true);

    json encoding_opts = config.value("encoding", json::object());
	json loss_opts = config.value("loss", json::object());
	json optimizer_opts = config.value("optimizer", json::object());
	json network_opts = config.value("network", json::object());

    mNetworkComponents->loss = std::shared_ptr<Loss<precision_t>>(create_loss<precision_t>(loss_opts));
    mNetworkComponents->optimizer = std::shared_ptr<Optimizer<precision_t>>(create_optimizer<precision_t>(optimizer_opts));
    mNetworkComponents->network = std::make_shared<NetworkWithInputEncoding<precision_t>>(NetConfig::n_input_dims, NetConfig::n_output_dims, encoding_opts, network_opts);
    mNetworkComponents->trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(mNetworkComponents->network, mNetworkComponents->optimizer, mNetworkComponents->loss);

    filesystem::path w_path(NetConfig::weightsPath);
    if (!w_path.exists()) {
        std::cout << "Cannot find the weights!" << std::endl;
        return;
    } else {
        std::cout << "Successfully find the weights!" << std::endl;
    }
    std::ifstream wf(w_path.str());
    json loaded_weights = json::parse(wf, nullptr, true, true);

    mNetworkComponents->trainer->deserialize(loaded_weights);

    mIOData->input_mat = new GPUMatrix<float>(NetConfig::n_input_dims, width * height);
    mIOData->output_mat = new GPUMatrix<float>(NetConfig::n_output_dims, width * height);

    frame_width = width;
    frame_height = height;
}


RadiosityNetwork::~RadiosityNetwork()
{
    delete mNetworkComponents;
    delete mIOData;
}


void RadiosityNetwork::forward(Falcor::RadiosityQuery* queries, cudaSurfaceObject_t output)
{
    uint32_t n_elements = frame_width * frame_height;

    linear_kernel(formatInput<float, NetConfig::n_input_dims>, 0, inference_stream, n_elements, queries, mIOData->input_mat->data());

    mNetworkComponents->network->inference(inference_stream, *mIOData->input_mat, *mIOData->output_mat);

    linear_kernel(mapToOutSurf<float, NetConfig::n_output_dims>, 0, inference_stream, n_elements, frame_width, mIOData->output_mat->data(), output);
}
