#include <torch/extension.h>
#include <thread>
#include <future>
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

class AsyncFetcher {
public:
    AsyncFetcher(const torch::Tensor& item_idx_batch, const torch::Tensor& train_data_pinned, const torch::Device& device)
        : promise_(std::make_shared<std::promise<torch::Tensor>>()), device_(device) {
        future_ = promise_->get_future();
        std::thread([this, item_idx_batch, train_data_pinned]() {
            try {
                torch::Tensor result = fetch(item_idx_batch, train_data_pinned);
                promise_->set_value(result);
            } catch (...) {
                promise_->set_exception(std::current_exception());
            }
        }).detach();
    }

    torch::Tensor get() {
        return future_.get();
    }

private:
    torch::Tensor fetch(const torch::Tensor& item_idx_batch, const torch::Tensor& train_data_pinned) {
        // Create a list of optional tensors for the indices
        c10::List<c10::optional<at::Tensor>> indices_list;
        indices_list.reserve(train_data_pinned.dim());
        indices_list.emplace_back(item_idx_batch);  // Directly add the item_idx_batch tensor

        // Use aten::index to perform the indexing
        torch::Tensor buffer = at::index(train_data_pinned, indices_list).contiguous().pin_memory();

        // Transfer to device asynchronously
        return buffer.to(device_, /*non_blocking=*/true);
    }

    std::shared_ptr<std::promise<torch::Tensor>> promise_;
    std::future<torch::Tensor> future_;
    torch::Device device_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<AsyncFetcher>(m, "AsyncFetcher")
        .def(pybind11::init<const torch::Tensor&, const torch::Tensor&, const torch::Device&>())  // Correct initialization
        .def("get", &AsyncFetcher::get);
}
