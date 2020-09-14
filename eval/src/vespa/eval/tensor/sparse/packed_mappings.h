// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "packed_labels.h"
#include <vector>

namespace vespalib::tensor {

class PackedMappings {
public:
    using Address = std::vector<vespalib::stringref>;

    uint32_t size() const { return _num_mappings; }

    // returns -1 if mapping does not contain address 
    int32_t subspace_of_address(const Address &address) const;

    // returns -1 if mapping does not contain address 
    int32_t sortid_of_address(const std::vector<vespalib::stringref> &address) const;

    std::vector<vespalib::stringref> address_of_subspace(uint32_t subspace_index) const;

    /** returns sortid */
    uint32_t fill_by_subspace(uint32_t subspace_index, Address &address) const;

    /** returns subspace_index */
    uint32_t fill_by_sortid(uint32_t sortid, Address &address) const;

    uint32_t num_sparse_dims() const { return _num_dims; }

private:
    PackedMappings(uint32_t num_dims, uint32_t num_mappings,
                   ConstArrayRef<uint32_t> int_store,
                   PackedLabels label_store)
      : _num_dims(num_dims),
        _num_mappings(num_mappings),
        _int_store(int_store),
        _label_store(label_store)
    {
        validate();
    }
    friend class PackedMappingsBuilder;
private:
    void validate() const;

    std::vector<uint32_t> enums_of_index(uint32_t internal_index) const;

    const uint32_t _num_dims;
    const uint32_t _num_mappings;
    /*
       _int_store contains data corresponding to this model:
       struct IntStore {
           // map to index in next table:
           uint32_t index_of_subspace[num_mappings];
           // sorted lexicographically by label_enums:
           struct MappingData {
               uint32_t label_enums[num_dims];
               uint32_t subspace_index;
           } mappings[num_mappings];
       };
     */
    const ConstArrayRef<uint32_t> _int_store;
    const PackedLabels _label_store;

    uint32_t offset_of_mapping_data(uint32_t idx) const {
        return (idx * (1 + _num_dims)) + _num_mappings;
    }
    uint32_t subspace_of_map(uint32_t idx) const {
        uint32_t offset = offset_of_mapping_data(idx);
        return _int_store[offset + _num_dims];
    }
    uint32_t index_of_subspace(uint32_t idx) const {
        return _int_store[idx];
    }
};

} // namespace
