package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const (
	huberDelta      = 1.0
	bceEpsilon      = 1e-7
	klEpsilon       = 1e-12
)

func RenderPairLoss(
	moduleName string,
	elementFormat dtype.DType,
	vectorShape tensor.Shape,
	lossKind string,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := reductionInputLiteral(elementType, vectorShape)
	entryLayout := fmt.Sprintf("%s,%s->%s[]", vectorLiteral, vectorLiteral, elementType)
	count := vectorShape.Dims()[0]

	body, err := pairLossBody(elementType, vectorLiteral, lossKind)

	if err != nil {
		return "", err
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
%s
  init = %s[] constant(0)
  total = %s[] reduce(per_elem, init), dimensions={0}, to_apply=%%add
  count_c = %s[] constant(%d)
  ROOT result = %s[] divide(total, count_c)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		body, elementType, elementType, elementType, count, elementType), nil
}

func pairLossBody(elementType, vectorLiteral, lossKind string) (string, error) {
	switch lossKind {
	case "mse":
		return fmt.Sprintf(`  pred = %s parameter(0)
  targ = %s parameter(1)
  diff = %s subtract(pred, targ)
  per_elem = %s multiply(diff, diff)
`, vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral), nil
	case "mae":
		return fmt.Sprintf(`  pred = %s parameter(0)
  targ = %s parameter(1)
  diff = %s subtract(pred, targ)
  per_elem = %s abs(diff)
`, vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral), nil
	case "huber":
		return fmt.Sprintf(`  pred = %s parameter(0)
  targ = %s parameter(1)
  diff = %s subtract(pred, targ)
  abs_diff = %s abs(diff)
  delta = %s[] constant(%g)
  delta_b = %s broadcast(delta), dimensions={}
  half = %s[] constant(0.5)
  half_b = %s broadcast(half), dimensions={}
  half_sq = %s multiply(half_b, multiply(diff, diff))
  half_delta = %s[] constant(%g)
  half_delta_b = %s broadcast(half_delta), dimensions={}
  linear = %s multiply(delta_b, subtract(abs_diff, half_delta_b))
  pred_le = %s compare(abs_diff, delta_b), direction=LE
  per_elem = %s select(pred_le, half_sq, linear)
`, vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral,
			elementType, huberDelta, vectorLiteral,
			elementType, vectorLiteral,
			vectorLiteral,
			elementType, 0.5*huberDelta, vectorLiteral,
			vectorLiteral,
			vectorLiteral,
			vectorLiteral), nil
	case "bce":
		return fmt.Sprintf(`  pred = %s parameter(0)
  targ = %s parameter(1)
  eps = %s[] constant(%g)
  eps_b = %s broadcast(eps), dimensions={}
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  one_minus_eps = %s subtract(one_b, eps_b)
  clamped_hi = %s minimum(pred, one_minus_eps)
  clamped = %s maximum(clamped_hi, eps_b)
  log_pred = %s log(clamped)
  one_minus_pred = %s subtract(one_b, clamped)
  log_one_minus = %s log(one_minus_pred)
  term_a = %s multiply(targ, log_pred)
  term_b = %s multiply(subtract(one_b, targ), log_one_minus)
  per_elem = %s subtract(negate(term_a), term_b)
`, vectorLiteral, vectorLiteral,
			elementType, bceEpsilon, vectorLiteral,
			elementType, vectorLiteral,
			vectorLiteral,
			vectorLiteral, vectorLiteral,
			vectorLiteral,
			vectorLiteral, vectorLiteral,
			vectorLiteral, vectorLiteral,
			vectorLiteral), nil
	case "kl":
		return fmt.Sprintf(`  pred = %s parameter(0)
  targ = %s parameter(1)
  eps = %s[] constant(%g)
  eps_b = %s broadcast(eps), dimensions={}
  pred_c = %s maximum(pred, eps_b)
  targ_c = %s maximum(targ, eps_b)
  ratio = %s divide(targ_c, pred_c)
  per_elem = %s multiply(targ_c, log(ratio))
`, vectorLiteral, vectorLiteral,
			elementType, klEpsilon, vectorLiteral,
			vectorLiteral, vectorLiteral,
			vectorLiteral, vectorLiteral), nil
	default:
		return "", fmt.Errorf("unsupported XLA pair loss: %s", lossKind)
	}
}

func RenderCrossEntropy(
	moduleName string,
	elementFormat dtype.DType,
	batchSize, classes int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	logitsLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, batchSize, classes)
	targetLiteral := fmt.Sprintf("s32[%d]{0}", batchSize)
	rowLiteral := fmt.Sprintf("%s[%d]{0}", elementType, batchSize)
	entryLayout := fmt.Sprintf("%s,%s->%s[]", logitsLiteral, targetLiteral, elementType)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] maximum(lhs, rhs)
}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  logits = %s parameter(0)
  targets = %s parameter(1)
  neg_inf = %s[] constant(-inf)
  row_max = %s reduce(logits, neg_inf), dimensions={1}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={0}
  shifted = %s subtract(logits, row_max_b)
  exp_val = %s exponential(shifted)
  zero = %s[] constant(0)
  row_sum = %s reduce(exp_val, zero), dimensions={1}, to_apply=%%add
  row_sum_b = %s broadcast(row_sum), dimensions={0}
  log_probs = %s subtract(shifted, log(row_sum_b))
  target_idx = s32[%d,1]{1,0} reshape(targets), dimensions={0}
  neg_log_prob = %s negate(gather(log_probs, target_idx, offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=1, slice_sizes={1}))
  init = %s[] constant(0)
  total = %s[] reduce(neg_log_prob, init), dimensions={0}, to_apply=%%add
  batch_c = %s[] constant(%d)
  ROOT result = %s[] divide(total, batch_c)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		logitsLiteral, targetLiteral,
		elementType, rowLiteral, logitsLiteral, logitsLiteral, logitsLiteral,
		elementType, rowLiteral, logitsLiteral, logitsLiteral,
		batchSize, logitsLiteral,
		elementType, elementType, elementType, batchSize, elementType), nil
}

func PairLossOperationName(lossKind string) string {
	return "loss_" + lossKind
}

func PairLossKindFromOperation(operationName string) (string, bool) {
	const prefix = "loss_"

	if len(operationName) <= len(prefix) || operationName[:len(prefix)] != prefix {
		return "", false
	}

	return operationName[len(prefix):], true
}
