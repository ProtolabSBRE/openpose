#include <numeric> // std::accumulate

// Caffe
#include <atomic>
#include <mutex>
#include <caffe/net.hpp>
#include <glog/logging.h> // google::InitGoogleLogging

// Cuda
#include <openpose/gpu/cuda.hpp>

#include <openpose/utilities/fileSystem.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/net/netRT.hpp>

// To manually create a caffe::blob
#include <boost/make_shared.hpp>

namespace op
{
    std::mutex sMutexNetRT;
    std::atomic<bool> sGoogleLoggingInitializedRT{true};

    struct NetRT::ImplNetRT
    {
        // Init with constructor
        const int mGpuId;
        const std::string mRTPlan;
        const std::string mLastBlobName;
        std::vector<int> mNetInputSize4D;

        // Init with thread
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* nvengine; 
        nvinfer1::IExecutionContext* context;
        void* buffers[NB_BINDINGS];
        int inputBlobIndex;
        int outputBlobIndex;

        boost::shared_ptr<caffe::Blob<float>> spOutputBlob;

        ImplNetRT(const std::string& RTPlan, const int gpuId,
                  const bool enableGoogleLogging, const std::string& lastBlobName) :
            mGpuId{gpuId},
            mRTPlan{RTPlan},
            mLastBlobName{lastBlobName}
        {
            if (!existFile(mRTPlan))
                error("TensorRT plan not found: " + RTPlan, __LINE__, __FUNCTION__, __FILE__);
        
            if (enableGoogleLogging && !sGoogleLoggingInitializedRT)
            {
                std::lock_guard<std::mutex> lock{sMutexNetRT};
                if (enableGoogleLogging && !sGoogleLoggingInitializedRT)
                {
                    google::InitGoogleLogging("OpenPose");
                    sGoogleLoggingInitializedRT = true;
                }
            }
        }

        ~ImplNetRT()
        {
            cudaFree(buffers[inputBlobIndex]);
            cudaFree(buffers[outputBlobIndex]);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        }
    };

    NetRT::NetRT(const std::string& RTPlan, const int gpuId,
                    const bool enableGoogleLogging, const std::string& lastBlobName) : 
        upImpl(new ImplNetRT(RTPlan, gpuId, enableGoogleLogging, lastBlobName)),
        gLogger()  
    {}

    NetRT::~NetRT()
    {}

    void NetRT::initializationOnThread()
    {
        // Load the TensorRT plan from file
        std::ifstream modelFile(upImpl->mRTPlan, std::ios::binary | std::ios::ate); // open model at the end
        std::streamsize modelSize = modelFile.tellg(); // get the size of the model
        modelFile.seekg(0, std::ios::beg); // go back to the beginning for reading

        std::vector<char> modelBuffer(modelSize);
        if(!modelFile.read(modelBuffer.data(), modelSize))
        {
            error("Could not load tensorrt engine.", __LINE__, __FUNCTION__, __FILE__);
        }

        // Create the engine from the serialized plan
        upImpl->runtime = nvinfer1::createInferRuntime(gLogger);
        upImpl->nvengine = upImpl->runtime->deserializeCudaEngine(modelBuffer.data(), modelBuffer.size(), nullptr);
        upImpl->context = upImpl->nvengine->createExecutionContext();

        // Check number of bindings
        const auto& engine = upImpl->context->getEngine();
        if(engine.getNbBindings() != NB_BINDINGS)
            error("The engine should have 2 binding (One input and one output)", __LINE__, __FUNCTION__, __FILE__);
        
        // Bind the output and input tensor name to indexes
        upImpl->inputBlobIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        upImpl->outputBlobIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // Allocate memory for the input and output buffers on the gpu
        cudaMalloc(&(upImpl->buffers[upImpl->inputBlobIndex]), 
                    1 * INPUT_IMAGE_NBR_CHANNELS * INPUT_IMAGE_HEIGHT * INPUT_IMAGE_WIDTH * sizeof(float));
        cudaMalloc(&(upImpl->buffers[upImpl->outputBlobIndex]),
                    1 * OUTPUT_HEATMAP_NB_CHANNELS * OUTPUT_HEATMAP_HEIGHT * OUTPUT_HEATMAP_WIDTH * sizeof(float));

        // Redirect to a caffe blob for compatibility with the current pipeline
        upImpl->spOutputBlob = boost::make_shared<caffe::Blob<float>>(std::vector<int>{1, OUTPUT_HEATMAP_NB_CHANNELS, OUTPUT_HEATMAP_HEIGHT, OUTPUT_HEATMAP_WIDTH});
        upImpl->spOutputBlob->set_gpu_data(static_cast<float*>(upImpl->buffers[upImpl->outputBlobIndex]));

        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
    }

    void NetRT::forwardPass(const Array<float>& inputData) const
    {
        // Security checks
        if (inputData.empty())
            error("The Array inputData cannot be empty.", __LINE__, __FUNCTION__, __FILE__);
        if (inputData.getNumberDimensions() != 4 || inputData.getSize(1) != 3)
            error("The Array inputData must have 4 dimensions: [batch size, 3 (RGB), height, width].",
                __LINE__, __FUNCTION__, __FILE__);
        const auto& size = inputData.getSize();
        int totalSize = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
        if(totalSize != (1 * INPUT_IMAGE_NBR_CHANNELS * INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT))
        {
            std::string hardcodedSize("[total size = " + std::to_string(1 * 3 * INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT) + "]");
            std::string inputSizeDetail("total size = " + std::to_string(totalSize) + "[batch size = " + std::to_string(size[0]) + ", channel (RGB) = " + std::to_string(size[1]) + ", height = " + std::to_string(size[2]) + ", width = " + std::to_string(size[3])+ "]");
            error("Dimension conflict " + hardcodedSize + " vs " + inputSizeDetail,
                    __LINE__, __FUNCTION__, __FILE__);
        }

        // Keep to avoid something exploding somewhere else
        if (!vectorsAreEqual(upImpl->mNetInputSize4D, inputData.getSize()))
        {
            upImpl->mNetInputSize4D = inputData.getSize();
        }
        
        // Copy the image to gpu memory
        cudaMemcpy(upImpl->buffers[upImpl->inputBlobIndex], inputData.getConstPtr(),
                   inputData.getVolume() * sizeof(float), cudaMemcpyHostToDevice);

        // Infere on the image
        if(!upImpl->context->execute(1, upImpl->buffers))
            error("Inference execution failed", __LINE__, __FUNCTION__, __FILE__);
    
        // The output blob is directly mapped to the output gpu memory so nothing else to do
    }

    boost::shared_ptr<caffe::Blob<float>> NetRT::getOutputBlob() const
    {
        try
        {
            #ifdef USE_CAFFE
                return upImpl->spOutputBlob;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
