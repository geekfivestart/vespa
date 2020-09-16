// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "packed_mixed_tensor.h"

namespace vespalib::eval {

class PackedMixedTensorIndexView : public NewValue::Index::View
{
private:
    const PackedMappings& _mappings;
    std::vector<size_t> _view_dims;
    std::vector<vespalib::stringref> _lookup_addr;
    std::vector<vespalib::stringref> _full_address;
    size_t _index;
public:
    PackedMixedTensorIndexView(const PackedMappings& mappings,
                               const std::vector<size_t> &dims)
        : _mappings(mappings),
          _view_dims(dims),
          _full_address(),
          _index(0)
    {}

    void lookup(const std::vector<const vespalib::stringref*> &addr) override;
    bool next_result(const std::vector<vespalib::stringref*> &addr_out, size_t &idx_out) override;
    ~PackedMixedTensorIndexView() override = default;
};

void
PackedMixedTensorIndexView::lookup(const std::vector<const vespalib::stringref*> &addr)
{
    size_t d = _view_dims.size();
    assert(addr.size() == d);
    _lookup_addr.clear();
    _lookup_addr.reserve(d);
    for (const vespalib::stringref * label_ptr : addr) {
        _lookup_addr.push_back(*label_ptr);
    }
    _full_address.resize(_mappings.num_sparse_dims());
    _index = 0;
}

bool
PackedMixedTensorIndexView::next_result(const std::vector<vespalib::stringref*> &addr_out, size_t &idx_out)
{
    for (; _index < _mappings.size(); ++_index) {
        idx_out = _mappings.fill_by_sortid(_index, _full_address);
        bool couldmatch = true;
        size_t vd_idx = 0;
        size_t ao_idx = 0;
        for (size_t i = 0; i < _mappings.num_sparse_dims(); ++i) {
            size_t vd = _view_dims[vd_idx];
            if (i == vd) {
                if (_lookup_addr[vd] != _full_address[i]) {
                    couldmatch = false;
                    break;
                }
                ++vd_idx;
            } else {
                *addr_out[ao_idx] = _full_address[i];
                ++ao_idx;
            }
        }
        if (couldmatch) {
            return true;
        }
    }
    return false;
}

PackedMixedTensor::~PackedMixedTensor() = default;

std::unique_ptr<NewValue::Index::View>
PackedMixedTensor::create_view(const std::vector<size_t> &dims) const
{
    for (size_t i = 1; i < dims.size(); ++i) {
        assert(dims[i-1] < dims[i]);
        assert(dims[i] < _mappings.num_sparse_dims());
    }
    return std::make_unique<PackedMixedTensorIndexView>(_mappings, dims);
}

} // namespace
