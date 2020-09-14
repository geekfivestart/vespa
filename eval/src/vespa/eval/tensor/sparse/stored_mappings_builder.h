// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include "stored_mappings.h"
#include <vespa/vespalib/stllike/string.h>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace vespalib::tensor {

class StoredMappingsBuilder {
public:
    using SparseAddress = std::vector<vespalib::stringref>;

    StoredMappingsBuilder(uint32_t num_dims)
      : _num_dims(num_dims),
        _labels(),
        _mappings()
    {}

    uint32_t add_mapping_for(SparseAddress address);

    std::unique_ptr<StoredMappings> build_mappings() const;
private:
    uint32_t _num_dims;
    std::set<vespalib::string> _labels;
    using IndexMap = std::map<SparseAddress, uint32_t>;
    IndexMap _mappings;
};

} // namespace
