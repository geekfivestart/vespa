// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "stored_labels.h"
#include <vector>

namespace vespalib::tensor {

class StoredMappings {
public:
    uint32_t size() const { return _num_mappings; }

    // returns -1 if mapping does not contain address 
    int32_t index_of_mapping(const std::vector<vespalib::stringref> &address) const;

    std::vector<vespalib::stringref> mapping_of_index(uint32_t index) const;

    StoredMappings(uint32_t num_dims, uint32_t num_mappings,
                   ConstArrayRef<uint32_t> mapping_store,
                   StoredLabels label_store)
      : _num_dims(num_dims),
        _num_mappings(num_mappings),
        _mapping_store(mapping_store),
        _label_store(label_store)
    {
        validate_mappings();
    }
private:
    void validate_mappings();
    std::vector<uint32_t> enums_of_index(uint32_t index) const;

    const uint32_t _num_dims;
    const uint32_t _num_mappings;
    const ConstArrayRef<uint32_t> _mapping_store;
    const StoredLabels _label_store;
};

} // namespace
