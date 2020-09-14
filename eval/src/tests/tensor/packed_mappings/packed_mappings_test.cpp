// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#include <vespa/eval/tensor/sparse/packed_mappings_builder.h>
#include <vespa/eval/tensor/sparse/packed_mappings.h>
#include <vespa/eval/tensor/sparse/packed_labels.h>
#include <vespa/vespalib/gtest/gtest.h>

using namespace vespalib::tensor;

class MappingsBuilderTest : public ::testing::Test {
public:
    std::unique_ptr<PackedMappingsBuilder> builder;
    std::unique_ptr<PackedMappings> built;

    MappingsBuilderTest() = default;

    virtual ~MappingsBuilderTest() = default;

    void build_and_compare() {
        ASSERT_TRUE(builder);
        built = builder->build_mappings();
        ASSERT_TRUE(built);
        EXPECT_EQ(builder->num_sparse_dims(), built->num_sparse_dims());
        EXPECT_EQ(builder->size(), built->size());
        for (size_t idx = 0; idx < built->size(); ++idx) {
            auto got = built->address_of_subspace(idx);
            printf("Got address:");
            for (auto ref : got) {
                printf(" '%s'", ref.data());
            }
            uint32_t original = builder->add_mapping_for(got);
            printf(" -> %u\n", original);
            EXPECT_EQ(idx, original);
        }
    }
};

TEST_F(MappingsBuilderTest, empty_mapping)
{
    for (uint32_t dims : { 0, 1, 2, 3 }) {
        builder = std::make_unique<PackedMappingsBuilder>(dims);
        build_and_compare();
    }
}

TEST_F(MappingsBuilderTest, just_one)
{
    vespalib::string label("foobar");
    for (uint32_t dims : { 0, 1, 2, 3, 7 }) {
        builder = std::make_unique<PackedMappingsBuilder>(dims);
        std::vector<vespalib::stringref> foo(dims, label);
        uint32_t idx = builder->add_mapping_for(foo);
        EXPECT_EQ(idx, 0);
        build_and_compare();
    }
}

TEST_F(MappingsBuilderTest, some_random)
{
    vespalib::string label1(""),
                     label2("foo"),
                     label3("bar");
    vespalib::string label4("foobar"),
                     label5("barfoo"),
                     label6("other");
    vespalib::string label7("long text number one"),
                     label8("long text number two"),
                     label9("long text number three");
    size_t rnd = 123456789;
    for (uint32_t dims : { 1, 2, 5 }) {
        builder = std::make_unique<PackedMappingsBuilder>(dims);
        rnd *= 17;
        size_t cnt = 3 + (rnd % 20);
        for (size_t i = 0; i < cnt; ++i) {
            std::vector<vespalib::stringref> foo(dims, label1);
            for (auto & ref : foo) {
                rnd *= 17;
                size_t pct = (rnd % 100);
                     if (pct <  5) { ref = label1; }
                else if (pct < 30) { ref = label2; }
                else if (pct < 55) { ref = label3; }
                else if (pct < 65) { ref = label4; }
                else if (pct < 75) { ref = label5; }
                else if (pct < 85) { ref = label6; }
                else if (pct < 90) { ref = label7; }
                else if (pct < 95) { ref = label8; }
                else               { ref = label9; }
            }
            uint32_t idx = builder->add_mapping_for(foo);
            EXPECT_LE(idx, i);
        }
        build_and_compare();
    }
}

GTEST_MAIN_RUN_ALL_TESTS()
