#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/testing/file_check.h>

namespace torch {
namespace jit {
TEST(EliminateDeadCodeTest, Basic) {
  auto graph = std::make_shared<Graph>();

  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  // We want to check that b[0] and b are properly marked as live and thus not
  // DCE'd.
  const std::string input =
      R"IR(
graph():
  %48 : None = prim::Constant()
  %50 : bool = prim::Constant[value=1]()
  %0 : int = prim::Constant[value=2]()
  %12 : int = prim::Constant[value=1]()
  %24 : int = prim::Constant[value=3]()
  %31 : int = prim::Constant[value=0]()
  %2 : int[] = prim::ListConstruct(%0, %0)
  %a.1 : Tensor = prim::MakeTestTensor()
  %14 : int[] = prim::ListConstruct(%12)
  %tot.1 : Tensor = prim::MakeTestTensor()
  %tot : Tensor = prim::Loop(%24, %50, %tot.1)
    block0(%i : int, %tot.6 : Tensor):
      %33 : Tensor = aten::select(%a.1, %31, %31)
      %35 : Tensor = aten::select(%33, %31, %31)
      # CHECK: add_
      %tot.3 : Tensor = aten::add_(%tot.6, %35, %12)
      %b.1 : Tensor = aten::select(%a.1, %31, %31)
      %44 : Tensor = aten::select(%b.1, %31, %31)
      # CHECK: add_
      %46 : Tensor = aten::add_(%44, %12, %12)
      -> (%50, %tot.3)
  return (%tot)
)IR";
  parseIR(input, graph.get());
  EliminateDeadCode(graph);
  // Check that dead code elimin
  testing::FileCheck().run(input, *graph);
}

TEST(EliminateDeadCodeTest, TestAliasThing) {


  auto graph = std::make_shared<Graph>();

  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  // We want to check that b[0] and b are properly marked as live and thus not
  // DCE'd.
  const std::string input =
      R"IR(
graph(%x : Tensor, %y : Tensor):
  %1 : int = prim::Constant[value=1]()
  %2 : int = prim::Constant[value=0]()
  %3 : Tensor = aten::add(%x, %y, %1)
  %4 : Tensor[] = aten::split(%3, %1, %2)
  return (%4)
)IR";
  parseIR(input, graph.get());

  AliasDb aliasdb(graph);

  for (Node* n : graph->nodes()) {
    if (n->kind() == aten::split) {
      for (size_t i : c10::irange(n->inputs().size())) {
        for (size_t j : c10::irange(n->outputs().size())) {
          auto may_contain = aliasdb.mayContainAlias(n->inputs()[i], n->outputs()[j]);
          auto may = aliasdb.mayAlias(n->inputs()[i], n->outputs()[j]);
          std::cerr << " INPUTS " << i << " and output " << j << " : " << may_contain << " and mayalias: " << may << std::endl;
        }
      }
    }
  }

  EXPECT_EQ(true, false);
/*
  // Consider the following loop:
  //   for i in range(3):
  //     tot += a[0][0]
  //     b = a[0]
  //     b[0] += 1
  //   print(tot)
  // We want to check that b[0] and b are properly marked as live and thus not
  // DCE'd.
  c10::FunctionSchema schema = torch::jit::parseSchema(
      "split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]");
  auto print_set = [](const std::unordered_set<Symbol>& s) {
    for (const Symbol& ss: s) {
      std::cerr << ss.toQualString() << ' ';
    }
    std::cerr << '\n';
  };
  for (const auto& argument: schema.arguments()) {
    std::cerr << argument.name();
    if (argument.alias_info()) {
      print_set(argument.alias_info()->beforeSets());
      print_set(argument.alias_info()->afterSets());
    }
  }
    for (const auto& argument: schema.returns()) {
    std::cerr << argument.name();
    if (argument.alias_info()) {
      print_set(argument.alias_info()->beforeSets());
      print_set(argument.alias_info()->afterSets());
    }
  }
  */
}
} // namespace jit
} // namespace torch
