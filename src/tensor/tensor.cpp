#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1; // 最后一个维度的步长默认为 1
    // 从最后一个维度向前遍历
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride; // 记录当前维度的步长
        stride *= shape[ndim_ - i];  // 累乘形状，为下一个维度做准备
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;             // 上个循环结束后的 stride 正好是所有维度的乘积
    size_t dtype_size = utils::dsize(dtype); // 单个元素的大小（字节）
    // 需要的总字节数 = total_elems * dtype_size

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t z = 1;

    for (int i = this->ndim() - 1; i >= 0; --i) {
        size_t current_dim = this->shape()[i];

        if (current_dim == 0) {
            return true;
        }
        // 这种维度不会引起内存地址的跳变，也不需要更新累积步长 z
        if (current_dim == 1) {
            continue;
        }

        // 普通维度，步长必须等于累积值
        if (this->strides()[i] != (ptrdiff_t)z) {
            return false;
        }

        // 更新期望的步长，供前一个维度检查使用
        z *= current_dim;
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 维度检查
    if (order.size() != this->ndim()) {
        throw std::runtime_error("Permute dimension mismatch: order size must match tensor dimension.");
    }

    size_t ndim_ = this->ndim();
    std::vector<size_t> new_shape(ndim_);
    std::vector<ptrdiff_t> new_strides(ndim_);

    // 用于检查 order 是否包含重复维度
    std::vector<bool> seen(ndim_, false);

    // 根据 order 重排 shape 和 strides
    for (size_t i = 0; i < ndim_; ++i) {
        size_t old_dim_index = order[i];

        // 检查是否越界
        if (old_dim_index >= ndim_) {
            throw std::runtime_error("Permute index out of bounds.");
        }

        // 检查是否出现重复项
        if (seen[old_dim_index]) {
            throw std::runtime_error("Permute order contains duplicate dimensions.");
        }
        seen[old_dim_index] = true;

        // 核心变换逻辑：直接映射
        new_shape[i] = this->shape()[old_dim_index];
        new_strides[i] = this->strides()[old_dim_index];
    }

    // 构建新的 Meta
    TensorMeta new_meta{
        this->dtype(),
        new_shape,
        new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查形状是否一致
    size_t new_numel = 1;
    for (auto s : shape) {
        new_numel *= s;
    }
    if (new_numel != this->numel()) {
        // 形状不匹配, 先 throw runtime_error
        throw std::runtime_error("View shape mismatch: numel must be the same.");
    }

    // 检查连续性
    if (!this->isContiguous()) {
        throw std::runtime_error("View is only supported on contiguous tensors.");
    }

    // 计算新的strides
    size_t ndim_ = shape.size(), stride = 1;
    std::vector<ptrdiff_t> new_strides(ndim_);
    for (size_t i = 1; i <= ndim_; ++i) {
        new_strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    // 构建新的TensorMeta
    TensorMeta new_meta{
        this->dtype(),
        shape,
        new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim_ = this->ndim();

    // 检查参数
    if (dim >= ndim_) {
        throw std::runtime_error("Slice dimension out of bounds.");
    }

    size_t dim_size = this->shape()[dim];
    // 处理 start/end 超出范围的情况 (也可以选择抛异常，
    // 这里选择 clamp 到有效范围类似于 Python)
    if (start > dim_size) {
        start = dim_size;
    }
    if (end > dim_size) {
        end = dim_size;
    }

    if (start >= end) {
        // 抛出异常
        throw std::runtime_error("Slice range empty or invalid (start >= end).");
    }

    // 构建新的 shape
    // 只有切片维度的 shape 变了
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;

    // 构建新的 strides（保持不变）
    std::vector<ptrdiff_t> new_strides = this->strides();

    // 计算新的 offset
    size_t elem_size = this->elementSize(), stride_val = this->strides()[dim];
    size_t new_offset = this->_offset + start * elem_size * stride_val;

    // 构建新的 meta
    TensorMeta new_meta{
        this->dtype(),
        new_shape,
        new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 总字节数
    size_t size_in_bytes = this->numel() * this->elementSize();

    // 设置上下文
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 确认拷贝类型：H2H or H2D
    // src_ 是 const void*，默认指代 CPU 上的主机内存
    llaisysMemcpyKind_t copy_kind;

    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        copy_kind = LLAISYS_MEMCPY_H2H;
    } else {
        copy_kind = LLAISYS_MEMCPY_H2D;
    }

    // 调用 Runtime API 执行同步拷贝
    // dst: this->data() 获取 tensor 内部存储的起始地址 + 偏移量
    // src: src_ 用户传入的源数据指针
    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        size_in_bytes,
        copy_kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
