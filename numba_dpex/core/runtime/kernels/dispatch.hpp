#ifndef __DISPATCH_HPP__
#define __DISPATCH_HPP__

namespace ndpx
{
namespace runtime
{
namespace kernel
{
namespace dispatch
{

template <typename funcPtrT,
          template <typename fnT, typename T>
          typename factory,
          int _num_types>
class DispatchVectorBuilder
{
private:
    template <typename Ty> const funcPtrT func_per_type() const
    {
        funcPtrT f = factory<funcPtrT, Ty>{}.get();
        return f;
    }

public:
    DispatchVectorBuilder() = default;
    ~DispatchVectorBuilder() = default;

    void populate_dispatch_vector(funcPtrT vector[]) const
    {
        const auto fn_map_by_type = {func_per_type<bool>(),
                                     func_per_type<int8_t>(),
                                     func_per_type<uint8_t>(),
                                     func_per_type<int16_t>(),
                                     func_per_type<uint16_t>(),
                                     func_per_type<int32_t>(),
                                     func_per_type<uint32_t>(),
                                     func_per_type<int64_t>(),
                                     func_per_type<uint64_t>(),
                                     func_per_type<sycl::half>(),
                                     func_per_type<float>(),
                                     func_per_type<double>(),
                                     func_per_type<std::complex<float>>(),
                                     func_per_type<std::complex<double>>()};
        assert(fn_map_by_type.size() == _num_types);
        int ty_id = 0;
        for (auto &fn : fn_map_by_type) {
            vector[ty_id] = fn;
            ++ty_id;
        }
    }
};

} // namespace dispatch
} // namespace kernel
} // namespace runtime
} // namespace ndpx

#endif
