// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "stored_mappings_builder.h"
#include <assert.h>

namespace vespalib::tensor {

uint32_t
StoredMappingsBuilder::add_mapping_for(SparseAddress address)
{
    assert(address.size() == _num_dims);
    for (auto & label_value : address) {
        // store label string in our own set:
        auto iter = _labels.insert(label_value).first;
        label_value = *iter;
    }

    uint32_t next_index = _mappings.size();
    IndexMap::value_type new_val(address, next_index);
    auto iter = _mappings.insert(new_val).first;
    return iter->second;
}


std::unique_ptr<StoredMappings>
StoredMappingsBuilder::build_mappings() const
{
    size_t meta_size = sizeof(StoredMappings);
    size_t enums_cnt = (1 + _num_dims) * _mappings.size();
    size_t enums_size = enums_cnt * sizeof(uint32_t);
    size_t label_cnt = _labels.size();
    size_t offsets_size = label_cnt * sizeof(uint32_t);
    size_t label_bytes = 0;
    for (const auto & label_value : _labels) {
        label_bytes += (label_value.size() + 1);
    }
    size_t total_size = meta_size + enums_size + offsets_size + label_bytes;
    char * mem = (char *) malloc(total_size);
    uint32_t * enums_mem = (uint32_t *) (mem + meta_size);
    uint32_t * offsets_mem = (uint32_t *) (mem + meta_size + enums_size);
    char * labels_mem = mem + meta_size + enums_size + offsets_size;

    ArrayRef<uint32_t> enums_data(enums_mem, enums_cnt);
    ArrayRef<uint32_t> label_offsets(offsets_mem, label_cnt);
    ArrayRef<char> labels_data(labels_mem, label_bytes);
    
    size_t byte_idx = 0;
    size_t i = 0;
    for (const auto & label_value : _labels) {
        label_offsets[i++] = byte_idx;
        size_t len_with_zero = label_value.size() + 1;
        memcpy(&labels_data[byte_idx], label_value.data(), len_with_zero);
        byte_idx += len_with_zero;
    }
    StoredLabels stored_labels(label_cnt, label_offsets, labels_data);

    size_t enum_idx = 0;
    for (const auto & kv : _mappings) {
        const SparseAddress & k = kv.first;
        uint32_t v = kv.second;
        for (const auto & label_value : k) {
            int32_t label_idx = stored_labels.find_label(label_value);
            assert(label_idx >= 0);
            enums_data[enum_idx++] = label_idx;
        }
        enums_data[enum_idx++] = v;
    }
    assert(enum_idx == enums_cnt);
    StoredMappings * built = new (mem) StoredMappings(_num_dims, _mappings.size(), enums_data, stored_labels);
    return std::unique_ptr<StoredMappings>(built);
}

} // namespace

