#include <iostream>
#include <torch/csrc/jit/ir/type_hashing.h>

#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/hash.h>
#include <torch/csrc/jit/ir/ir.h>

namespace std {

template<>
struct hash<c10::Stride> {
  bool operator()(const c10::Stride& s) const {
    return at::hash_combine(at::hash_combine(c10::get_hash(s.stride_index_), c10::get_hash(s.contiguous_)), c10::get_hash(s.stride_));
  }
};

} // namespace std

namespace torch::jit {

namespace {
template<typename T>
size_t get_hash_varying_shape(const c10::VaryingShape<T>& v) {
  if (!v.sizes()) {
    return 0;
  }
  auto ref = ArrayRef<c10::optional<T>>(*v.sizes());
  return get_hash(ref);
}

size_t hashTensorType(const TensorType& type) {
  size_t hash = c10::get_hash(type.kind());
  /*
  hash = at::hash_combine(hash, c10::get_hash(type.scalarType()));
  hash = at::hash_combine(hash, get_hash_varying_shape(type.sizes()));
  hash = at::hash_combine(hash, get_hash_varying_shape(type.stride_properties()));
  hash = at::hash_combine(hash, c10::get_hash(type.device()));
  hash = at::hash_combine(hash, c10::get_hash(type.requiresGrad()));
  hash = at::hash_combine(hash, c10::get_hash(type.undefined()));
  */
  return hash;
}

size_t hashType(const Type& type) {
  if (auto named_type = type.castRaw<ClassType>()) {
    return c10::get_hash(named_type->name().value(), named_type->compilation_unit());
  } else if (auto tensor_type = type.castRaw<TensorType>()) {
    return hashTensorType(*tensor_type);
  }

  size_t hash = 0;
  for (const auto& containedType : type.containedTypes()) {
    hash = at::hash_combine(hash, hashType(*containedType));
  }
  hash = at::hash_combine(hash, get_hash(type.kind()));
  return hash;
}

size_t hashType_logging(const Type& type) {
  auto res = hashType(type);
  std::cerr << " ht log| " << type.repr_str() << " : " << res << std::endl;
  return res;
}
} // namespace

size_t HashType::operator()(const TypePtr& type) const {
  return hashType_logging(*type);
}

size_t HashType::operator()(const c10::ConstTypePtr& type) const {
  return hashType_logging(*type);
}

bool EqualType::operator()(const TypePtr& a, const TypePtr& b) const {
  return *a == *b;
}

bool EqualType::operator()(
    const c10::ConstTypePtr& a,
    const c10::ConstTypePtr& b) const {
  return *a == *b;
}

} // namespace torch::jit
