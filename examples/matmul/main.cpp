#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#define ASSERT(Expr, Msg) _assert(#Expr, Expr, __FILE__, __LINE__, Msg)

void _assert(
    const char* expr_str,
    bool expr,
    const char* file,
    int line,
    const char* msg)
{
    if (!expr) {
        std::cerr << "Assert failed:\t" << msg << "\n"
                  << "Expected:\t" << expr_str << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        std::exit(EXIT_FAILURE);
    }
}

namespace fs = std::filesystem;

std::string loadKernelSource(const fs::path& path)
{
    std::ifstream file(path.string());
    ASSERT(file.is_open(), "File not open.");
    std::ostringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

cl::Device getDevice()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Platforms: " << platforms.size() << std::endl;
    ASSERT(platforms.size() > 0, "No platform found");

    std::vector<cl::Device> devices;
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0) {
        std::cerr << "No GPU found, falling back on CPU";
        platforms.front().getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }
    std::cout << "Devices: " << devices.size() << std::endl;
    ASSERT(devices.size() > 0, "No devices found");

    return devices.front();
}

struct Matrix {
    size_t rows;
    size_t cols;
    std::vector<float> elements;
};

Matrix matrixZero(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.elements = std::vector<float>(rows * cols, 0.0f);
    return m;
}

size_t matrixByteCount(const Matrix& m)
{
    return m.rows * m.cols * sizeof(m.elements.front());
}

std::pair<size_t, size_t>
multiplicationShape(const Matrix& lhs, const Matrix& rhs)
{
    ASSERT(lhs.cols == rhs.rows, "Shape mismatch");
    return {lhs.rows, rhs.cols};
}

int main(int argc, char* argv[])
{
    auto device = getDevice();
    auto context = cl::Context(device);
    auto queue = cl::CommandQueue(context, device);
    Matrix lhs = {2, 3, {1, 2, 3, 4, 5, 6}};
    Matrix rhs = {3, 2, {7, 8, 9, 10, 11, 12}};

    const auto [rows, cols] = multiplicationShape(lhs, rhs);

    Matrix dst = matrixZero(rows, cols);

    cl::Buffer lhs_gpu(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixByteCount(lhs),
        lhs.elements.data());
    cl::Buffer rhs_gpu(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        matrixByteCount(rhs),
        rhs.elements.data());
    cl::Buffer dst_gpu(context, CL_MEM_READ_WRITE, matrixByteCount(dst));

    const std::string program_source = loadKernelSource("matmul.cl");
    cl::Program program = cl::Program(context, program_source);
    program.build();

    cl::Kernel kernel(program, "matmul");
    kernel.setArg(0, lhs_gpu);
    kernel.setArg(1, rhs_gpu);
    kernel.setArg(2, dst_gpu);
    kernel.setArg(3, static_cast<int>(lhs.cols));
    kernel.setArg(4, static_cast<int>(rhs.cols));

    const auto global_range = cl::NDRange(rows, cols);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range);
    queue.enqueueReadBuffer(
        dst_gpu, CL_TRUE, 0, matrixByteCount(dst), dst.elements.data());
    queue.finish();

    for (size_t row = 0; row < dst.rows; ++row) {
        std::cout << '\n';
        for (size_t col = 0; col < dst.cols; ++col) {
            std::cout << dst.elements[row * dst.cols + col] << ' ';
        }
    }
    std::cout << std::endl;

    return 0;
}
