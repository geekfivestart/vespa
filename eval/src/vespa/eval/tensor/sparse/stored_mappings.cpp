// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "stored_mappings.h"
#include <assert.h>

namespace vespalib::tensor {

int32_t
StoredMappings::index_of_mapping(const std::vector<vespalib::stringref> &address) const
{
    std::vector<uint32_t> to_find;
    to_find.reserve(_num_dims);
    for (const auto & label_value : address) {
        int32_t label_idx = _label_store.find_label(label_value);
        if (label_idx < 0) {
            return -1;
        }
        to_find.push_back(label_idx);
    }
    uint32_t lo = 0;
    uint32_t hi = _num_mappings;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (enums_of_index(mid) < to_find) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    assert(lo == hi);
    if (lo < _num_mappings && enums_of_index(lo) == to_find) {
        uint32_t offset = lo * (1 + _num_dims);
        return _mapping_store[offset + _num_dims];
    }
    return -1;
}

std::vector<vespalib::stringref>
StoredMappings::mapping_of_index(uint32_t index) const
{
    std::vector<vespalib::stringref> result;
    if (index < _num_mappings) {
        auto mapping_enums = enums_of_index(index);
        assert(mapping_enums.size() == _num_dims);
        for (uint32_t label_idx : mapping_enums) {
            result.push_back(_label_store.label_value(label_idx));
        }
    }
    return result;
}

void
StoredMappings::validate_mappings()
{
    assert((_num_mappings * (1 + _num_dims)) == _mapping_store.size());
    for (uint32_t label_index : _mapping_store) {
        assert(label_index < _label_store.num_labels());
    }
    for (uint32_t i = 0; i+1 < _num_mappings; ++i) {
        // XXX needlessly expensive
        auto prev = enums_of_index(i);
        auto next = enums_of_index(i+1);
        assert(prev < next);
    }
}

std::vector<uint32_t>
StoredMappings::enums_of_index(uint32_t index) const
{
    std::vector<uint32_t> result;
    result.reserve(_num_dims);
    assert(index < _num_mappings);
    uint32_t offset = index * (1 + _num_dims);
    for (uint32_t i = 0; i < _num_dims; ++i) {
        result.push_back(_mapping_store[i + offset]);
    }
    return result;
}

} // namespace
