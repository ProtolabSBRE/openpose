#ifndef OPENPOSE_NET_NET_RT_HPP
#define OPENPOSE_NET_NET_RT_HPP

#include <openpose/core/common.hpp>
#include <openpose/net/net.hpp>

// TensorRT
#include <NvInfer.h>

namespace op
{

    class OP_API RTLogger : public nvinfer1::ILogger
    {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) override
        {
            switch(severity)
            {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << "\n";
        }
    };

    class OP_API NetRT : public Net
    {
    public:
        const char* INPUT_BLOB_NAME = "image";
        const char* OUTPUT_BLOB_NAME = "net_output";

        static constexpr int NB_BINDINGS = 2; // One input and one output
        const int INPUT_IMAGE_NBR_CHANNELS = 3;
        const int INPUT_IMAGE_HEIGHT = 320;
        const int INPUT_IMAGE_WIDTH = 240;

        const int OUTPUT_HEATMAP_NB_CHANNELS = 57;
        const int OUTPUT_HEATMAP_HEIGHT = 40;
        const int OUTPUT_HEATMAP_WIDTH = 30;

        NetRT(const std::string& RTPlan, const int gpuId = 0,
              const bool enableGoogleLogging = true, const std::string& lastBlobName = "net_output");

        virtual ~NetRT();

        void initializationOnThread();

        void forwardPass(const Array<float>& inputNetData) const;

        boost::shared_ptr<caffe::Blob<float>> getOutputBlob() const;

    private:
        RTLogger gLogger;

        // PIMPL idiom
        // http://www.cppsamples.com/common-tasks/pimpl.html
        struct ImplNetRT;
        std::unique_ptr<ImplNetRT> upImpl;

        // PIMP requires DELETE_COPY & destructor, or extra code
        // http://oliora.github.io/2015/12/29/pimpl-and-rule-of-zero.html
        DELETE_COPY(NetRT);
    };
}

#endif
