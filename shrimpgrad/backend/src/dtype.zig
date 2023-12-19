fn half() type {
    return struct { .name = "half", .size = 2, .type = f16 };
}

fn float() type {
    return struct { .name = "float", .size = 4, .type = f32 };
}

fn double() type {
    return struct { .name = "double", .size = 8, .type = f64 };
}

const DType = union(enum) {
    float16: half(),
};
