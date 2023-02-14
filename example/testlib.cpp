#include <CL/sycl.hpp>
#include <iostream>
#include <memory>
#include <vector>

namespace
{
void print_device_info(const sycl::device &device)
{
    using namespace sycl::info::device;

    std::cout << "Max work group size: "
              << device.get_info<max_work_group_size>() << std::endl;

    std::flush(std::cout);
}

void *
_foo(sycl::queue *queue, float *in, float *out, int64_t size, void *events)
{
    print_device_info(queue->get_device());
    std::vector<float> in_cpu(size);
    queue->copy(in, in_cpu.data(), size).wait();

    for (auto &&v : in_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    float result = 0;
    for (int i = 0; i < size; ++i) {
        result += in_cpu[i];
        in_cpu[i] = result;
    }

    for (auto &&v : in_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    auto ev = queue->copy(in_cpu.data(), out, size);

    ev.wait();
    auto ev1 = queue->copy(in_cpu.data(), in, size);

    // returning event to numba makes result incorrect
    // return new sycl::event(std::move(ev1));

    ev1.wait();

    return 0;

}
} // namespace

extern "C"
{
    void *foo(void *queue, void *in, void *out, int64_t size, void *events)
    {
        return _foo((sycl::queue *)queue, (float *)in, (float *)out, size,
                    events);
    }

    int64_t get_foo_ptr() { return (int64_t)foo; }
}
