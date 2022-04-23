#include <torch/csrc/jit/tensorexpr/ir_cloner.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <
    typename Op,
    typename std::enable_if<std::is_same<
        decltype(detail::bin_op_deducer(std::declval<Op>())),
        void>::value>::type* = nullptr>
static ExprPtr mutate_binary_op(
    NodePtr<Op> v,
    IRCloner* cloner,
    bool option = false) {
  ExprPtr lhs_new = v->lhs()->accept_mutator(cloner);
  ExprPtr rhs_new = v->rhs()->accept_mutator(cloner);
  IRNodeType expr_type = v->expr_type();
  switch (expr_type) {
    case IRNodeType::kAdd:
      return alloc<Add>(lhs_new, rhs_new);
    case IRNodeType::kSub:
      return alloc<Sub>(lhs_new, rhs_new);
    case IRNodeType::kMul:
      return alloc<Mul>(lhs_new, rhs_new);
    case IRNodeType::kDiv:
      return alloc<Div>(lhs_new, rhs_new);
    case IRNodeType::kMod:
      return alloc<Mod>(lhs_new, rhs_new);
    case IRNodeType::kMax:
      return alloc<Max>(lhs_new, rhs_new, option);
    case IRNodeType::kMin:
      return alloc<Min>(lhs_new, rhs_new, option);
    case IRNodeType::kAnd:
      return alloc<And>(lhs_new, rhs_new);
    case IRNodeType::kOr:
      return alloc<Or>(lhs_new, rhs_new);
    case IRNodeType::kXor:
      return alloc<Xor>(lhs_new, rhs_new);
    case IRNodeType::kLshift:
      return alloc<Lshift>(lhs_new, rhs_new);
    case IRNodeType::kRshift:
      return alloc<Rshift>(lhs_new, rhs_new);
    default:
      throw unimplemented_lowering(v);
  }
}

template <typename T>
StmtPtr IRCloner::get_stmt_from_cache(const std::shared_ptr<T>& ptr) {
  auto it = stmt_cache_.find((void*) ptr.get());
  if (it == stmt_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template <typename T>
ExprPtr IRCloner::get_expr_from_cache(const std::shared_ptr<T>& ptr) {
  auto it = expr_cache_.find((void*) ptr.get());
  if (it == expr_cache_.end()) {
    return nullptr;
  }
  return it->second;
}

template <typename T>
StmtPtr IRCloner::cache_stmt(const std::shared_ptr<T>& ptr, StmtPtr copy) {
  stmt_cache_[(void*) ptr.get()] = copy;
  return copy;
}

template <typename T>
ExprPtr IRCloner::cache_expr(const std::shared_ptr<T>& ptr, ExprPtr copy) {
  expr_cache_[(void*) ptr.get()] = copy;
  return copy;
}

#define EXPR_TRY_RETURN_CACHED(v)             \
if (auto cached = get_expr_from_cache(v))   { \
  return cached;                              \
}

#define STMT_TRY_RETURN_CACHED(v)             \
if (auto cached = get_stmt_from_cache(v))   { \
  return cached;                              \
}

ExprPtr IRCloner::mutate(AddPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(SubPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(MulPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(DivPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(ModPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(AndPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(OrPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(XorPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(LshiftPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(RshiftPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this));
}

ExprPtr IRCloner::mutate(MaxPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this, v->propagate_nans()));
}

ExprPtr IRCloner::mutate(MinPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, mutate_binary_op(v, this, v->propagate_nans()));
}

ExprPtr IRCloner::mutate(CompareSelectPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr lhs_new = v->lhs()->accept_mutator(this);
  ExprPtr rhs_new = v->rhs()->accept_mutator(this);
  ExprPtr retval1_new = v->ret_val1()->accept_mutator(this);
  ExprPtr retval2_new = v->ret_val2()->accept_mutator(this);
  return cache_expr(v, alloc<CompareSelect>(
      lhs_new,
      rhs_new,
      retval1_new,
      retval2_new,
      v->compare_select_op(),
      v->bias()));
}

// NOLINTNEXTLINE
#define IMM_MUTATE_DEFINE(_1, Name)          \
  ExprPtr IRCloner::mutate(Name##ImmPtr v) { \
    EXPR_TRY_RETURN_CACHED(v);               \
    return cache_expr(v, v);                 \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DEFINE);
#undef IMM_MUTATE_DEFINE

ExprPtr IRCloner::mutate(CastPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  return cache_expr(v, alloc<Cast>(v->dtype(), src_value_new));
}

ExprPtr IRCloner::mutate(BitCastPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr src_value_new = v->src_value()->accept_mutator(this);
  return cache_expr(v, alloc<BitCast>(v->dtype(), src_value_new));
}

ExprPtr IRCloner::mutate(RampPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr base_new = v->base()->accept_mutator(this);
  ExprPtr stride_new = v->stride()->accept_mutator(this);
  return cache_expr(v, alloc<Ramp>(base_new, stride_new, v->lanes()));
}

ExprPtr IRCloner::mutate(LoadPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (ExprPtr ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return cache_expr(v, alloc<Load>(v->dtype(), buf_new, indices_new));
}

// We do not clone Vars since the original IR and cloned IR are expected to
// share the underlying variables.
ExprPtr IRCloner::mutate(VarPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, v);
}

// We do not clone Bufs since the original IR and cloned IR are expected to
// share the underlying Bufs. In spite of Bufs having expressions as dims and
// initializers, this is the expected usage of clone at this point.
//
// TODO: Revisit this if Bufs need to be cloned as well.
ExprPtr IRCloner::mutate(BufPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, v);
}

ExprPtr IRCloner::mutate(BroadcastPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  int lanes = v->lanes();
  ExprPtr value_new = v->value()->accept_mutator(this);
  return cache_expr(v, alloc<Broadcast>(value_new, lanes));
}

ExprPtr IRCloner::mutate(IfThenElsePtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr condition_new = v->condition()->accept_mutator(this);
  ExprPtr true_value_new = v->true_value()->accept_mutator(this);
  ExprPtr false_value_new = v->false_value()->accept_mutator(this);

  return cache_expr(v, alloc<IfThenElse>(condition_new, true_value_new, false_value_new));
}

ExprPtr IRCloner::mutate(IntrinsicsPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  std::vector<ExprPtr> params_new;
  params_new.reserve(v->nparams());
  for (auto param : v->params()) {
    params_new.push_back(param->accept_mutator(this));
  }
  return cache_expr(v, alloc<Intrinsics>(v->op_type(), v->dtype(), params_new));
}

ExprPtr IRCloner::mutate(TermPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return cache_expr(v, alloc<Term>(v->hasher(), scalar_new, variables_new));
}

ExprPtr IRCloner::mutate(PolynomialPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr scalar_new = v->scalar()->accept_mutator(this);

  std::vector<TermPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(static_to<Term>(t->accept_mutator(this)));
  }
  return cache_expr(v, alloc<Polynomial>(v->hasher(), scalar_new, variables_new));
}

ExprPtr IRCloner::mutate(RoundOffPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  return cache_expr(v, alloc<RoundOff>(
      v->lhs()->accept_mutator(this), v->rhs()->accept_mutator(this)));
}

ExprPtr IRCloner::mutate(MaxTermPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return cache_expr(v, alloc<MaxTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new));
}

ExprPtr IRCloner::mutate(MinTermPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr scalar_new =
      v->scalar() ? v->scalar()->accept_mutator(this) : nullptr;

  std::vector<ExprPtr> variables_new;
  variables_new.reserve(v->variables().size());
  for (auto t : v->variables()) {
    variables_new.push_back(t->accept_mutator(this));
  }
  return cache_expr(v, alloc<MinTerm>(
      v->hasher(), scalar_new, v->propagate_nans(), variables_new));
}

ExprPtr IRCloner::mutate(ReduceOpPtr v) {
  EXPR_TRY_RETURN_CACHED(v);
  ExprPtr body_new = v->body()->accept_mutator(this);

  std::vector<VarPtr> reduce_args_new;
  reduce_args_new.reserve(v->reduce_args().size());
  for (auto r : v->reduce_args()) {
    reduce_args_new.push_back(static_to<Var>(r->accept_mutator(this)));
  }

  return cache_expr(v, alloc<ReduceOp>(body_new, reduce_args_new, v->reducer()));
}

StmtPtr IRCloner::mutate(ForPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  auto start_new = v->start()->accept_mutator(this);
  auto stop_new = v->stop()->accept_mutator(this);
  auto body_new = v->body()->accept_mutator(this);

  return cache_stmt(v, alloc<For>(v->var(), start_new, stop_new, body_new, v->loop_options()));
}

StmtPtr IRCloner::mutate(BlockPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  std::vector<StmtPtr> stmts_new;
  stmts_new.reserve(v->nstmts());
  for (StmtPtr stmt : *v) {
    stmts_new.push_back(stmt->accept_mutator(this));
  }
  return cache_stmt(v, alloc<Block>(stmts_new));
}

StmtPtr IRCloner::mutate(StorePtr v) {
  STMT_TRY_RETURN_CACHED(v);
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return cache_stmt(v, alloc<Store>(buf_new, indices_new, value_new));
}

StmtPtr IRCloner::mutate(AtomicAddPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  std::vector<ExprPtr> indices_new;
  indices_new.reserve(v->indices().size());
  for (auto ind : v->indices()) {
    indices_new.push_back(ind->accept_mutator(this));
  }
  auto value_new = v->value()->accept_mutator(this);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return cache_stmt(v, alloc<AtomicAdd>(buf_new, indices_new, value_new));
}

StmtPtr IRCloner::mutate(AllocatePtr v) {
  STMT_TRY_RETURN_CACHED(v);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return cache_stmt(v, alloc<Allocate>(buf_new));
}

StmtPtr IRCloner::mutate(FreePtr v) {
  STMT_TRY_RETURN_CACHED(v);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));
  return cache_stmt(v, alloc<Free>(buf_new));
}

StmtPtr IRCloner::mutate(SyncThreadsPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  return cache_stmt(v, alloc<SyncThreads>());
}

StmtPtr IRCloner::mutate(ExternalCallPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  BufPtr buf_new = to<Buf>(v->buf()->accept_mutator(this));

  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (BufPtr buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (ExprPtr arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  return cache_stmt(v, alloc<ExternalCall>(buf_new, v->func_name(), buf_args_new, args_new));
}

StmtPtr IRCloner::mutate(ExternalCallWithAllocPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  std::vector<BufPtr> buf_out_args_new;
  buf_out_args_new.reserve(v->buf_out_args().size());
  for (const auto& buf_out_arg : v->buf_out_args()) {
    buf_out_args_new.push_back(to<Buf>(buf_out_arg->accept_mutator(this)));
  }

  std::vector<BufPtr> buf_args_new;
  buf_args_new.reserve(v->buf_args().size());
  for (const auto& buf_arg : v->buf_args()) {
    buf_args_new.push_back(to<Buf>(buf_arg->accept_mutator(this)));
  }
  std::vector<ExprPtr> args_new;
  args_new.reserve(v->args().size());
  for (const auto& arg : v->args()) {
    args_new.push_back(arg->accept_mutator(this));
  }

  return cache_stmt(v, alloc<ExternalCallWithAlloc>(
      v->func_name(), buf_out_args_new, buf_args_new, args_new));
}

StmtPtr IRCloner::mutate(LetPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  auto value_new = v->value()->accept_mutator(this);
  return cache_stmt(v, alloc<Let>(v->var(), value_new));
}

StmtPtr IRCloner::mutate(CondPtr v) {
  STMT_TRY_RETURN_CACHED(v);
  auto condition_new = v->condition()->accept_mutator(this);
  StmtPtr true_old = v->true_stmt();
  StmtPtr false_old = v->false_stmt();
  StmtPtr true_new = true_old ? true_old->accept_mutator(this) : true_old;
  StmtPtr false_new = false_old ? false_old->accept_mutator(this) : false_old;
  return cache_stmt(v, alloc<Cond>(condition_new, true_new, false_new));
}

StmtPtr Stmt::clone(StmtPtr s) {
  IRCloner cloner;
  StmtPtr cloned = s->accept_mutator(&cloner);
  set_parent(cloned, nullptr);
  return cloned;
}

ExprPtr Expr::clone(ExprPtr e) {
  IRCloner cloner;
  return e->accept_mutator(&cloner);
}

#undef EXPR_TRY_RETURN_CACHED
#undef STMT_TRY_RETURN_CACHED

} // namespace tensorexpr
} // namespace jit
} // namespace torch
